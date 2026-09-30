"""Stage 4 regression tests: MIL formulations and the latent-label store
(docs/MIL_REFRAME_PLAN.md Stage 4).

Covers the plan's Stage 4 test bullets:
  1. Fold-leakage assertion: a deliberately mismatched fold must raise.
  2. Alignment checksum rejects a stale posterior file after re-export.
  3. Degeneracy diagnostics: pi_b distribution, shrinkage check, null-component
     cosines.
  4. mi-SVM / EM recover a sensible posterior on a clearly-separable synthetic
     MIL dataset (a basic correctness sanity-check both formulations need
     before either is trusted on real data).
"""
import numpy as np
import pytest

from src.mil.common import (
    allowed_states_mask, null_component_collapse_diagnostic, pi_b_per_bag,
    posterior_to_K1, shrinkage_diagnostic,
)
from src.mil.em import fit_em, score_em
from src.mil.misvm import _subsample_for_svm, fit_misvm, predict_misvm
from src.mil.store import (
    alignment_checksum, assert_no_fold_leakage, merge_out_of_sample_posterior,
    read_posteriors, write_posteriors,
)


# ---------------------------------------------------------------------------
# Synthetic constrained-MIL dataset: 2 classes, 3 null components, well
# separated in feature space so both formulations have something real to
# recover (a sanity floor, not a claim about real-data performance).
# ---------------------------------------------------------------------------

def _synthetic_constrained_mil(n_bags_per_class=8, n_per_bag=40, n_features=10,
                               pi_true=0.3, seed=0):
    rng = np.random.default_rng(seed)
    # Component means, well separated: null components near 0, diagnostic
    # classes far apart along the first feature.
    null_means = rng.normal(0, 0.3, size=(3, n_features))
    class_means = np.zeros((2, n_features))
    class_means[0, 0] = -6.0
    class_means[1, 0] = 6.0

    X, y_bag, bag_ids, z_true = [], [], [], []
    for cls in (0, 1):
        for b in range(n_bags_per_class):
            bag_id = f"bag_{cls}_{b}"
            n_diag = int(round(pi_true * n_per_bag))
            n_null_each = (n_per_bag - n_diag) // 3
            for _ in range(n_diag):
                X.append(rng.normal(class_means[cls], 0.5))
                z_true.append(3 + cls)
            for m in range(3):
                for _ in range(n_null_each):
                    X.append(rng.normal(null_means[m], 0.5))
                    z_true.append(m)
            n_bag_actual = n_diag + n_null_each * 3
            y_bag += [cls] * n_bag_actual
            bag_ids += [bag_id] * n_bag_actual

    return (np.array(X, dtype=np.float32), np.array(y_bag), np.array(bag_ids, dtype=object),
           np.array(z_true))


# ---------------------------------------------------------------------------
# 1. src/mil/common.py -- pure helpers
# ---------------------------------------------------------------------------

def test_allowed_states_mask_only_permits_own_class_and_nulls():
    y_bag = np.array([0, 1, 0])
    mask = allowed_states_mask(y_bag, n_classes=2)
    assert mask.shape == (3, 5)  # M=3 nulls + K=2 classes
    np.testing.assert_array_equal(mask[:, :3], True)  # nulls always allowed
    # instance 0/2 (bag class 0) may take state 3 (class 0) but not state 4
    assert mask[0, 3] and not mask[0, 4]
    assert mask[2, 3] and not mask[2, 4]
    # instance 1 (bag class 1) may take state 4 but not state 3
    assert mask[1, 4] and not mask[1, 3]


def test_posterior_to_K1_collapses_nulls_and_sums_to_one():
    state_scores = np.array([
        [0.1, 0.1, 0.1, 0.7, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    out = posterior_to_K1(state_scores, n_classes=2)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(out[0], [0.3, 0.7, 0.0], atol=1e-8)
    np.testing.assert_allclose(out[1], [0.0, 0.0, 1.0], atol=1e-8)


def test_pi_b_per_bag_is_mean_own_class_mass():
    posterior = np.array([
        [0.2, 0.8, 0.0],  # bag A, class 0 -> own-class mass 0.8
        [0.5, 0.5, 0.0],  # bag A
        [0.1, 0.0, 0.9],  # bag B, class 1 -> own-class mass 0.9
    ])
    y_bag = np.array([0, 0, 1])
    bag_ids = np.array(['A', 'A', 'B'])
    pi_b = pi_b_per_bag(posterior, y_bag, bag_ids)
    assert pi_b['A'] == pytest.approx(0.65)
    assert pi_b['B'] == pytest.approx(0.9)


def test_shrinkage_diagnostic_flags_total_collapse():
    # Every bag returned exactly mu (in probability space) -> zero spread ->
    # collapsed regardless of the prior's own implied spread.
    mu = 0.0
    pi_b = {f'b{i}': 0.5 for i in range(10)}
    diag = shrinkage_diagnostic(pi_b, mu=mu, sigma=1.0)
    assert diag['observed_spread'] == pytest.approx(0.0, abs=1e-9)
    assert diag['collapsed'] is True


def test_shrinkage_diagnostic_passes_with_real_spread():
    rng = np.random.default_rng(1)
    pi_b = {f'b{i}': float(np.clip(rng.normal(0.15, 0.08), 0.01, 0.9)) for i in range(30)}
    diag = shrinkage_diagnostic(pi_b, mu=np.log(0.15 / 0.85), sigma=0.6)
    assert diag['observed_spread'] > 0
    assert diag['shrinkage_ratio'] > 0.1
    assert diag['collapsed'] is False


def test_null_component_collapse_diagnostic_detects_distinct_components():
    rng = np.random.default_rng(2)
    n, n_features = 300, 8
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    # Make responsibility columns clearly favour disjoint, well-separated
    # regions of X so the 3 "components" are genuinely distinct.
    X[:100, 0] += 10
    X[100:200, 1] += 10
    X[200:, 2] += 10
    gamma = np.zeros((n, 5))
    gamma[:100, 0] = 1.0
    gamma[100:200, 1] = 1.0
    gamma[200:, 2] = 1.0
    diag = null_component_collapse_diagnostic(X, gamma, top_frac=0.2, min_n=20)
    assert diag['pairwise_cosine'].shape == (3, 3)
    assert diag['effective_n_components'] > 2.0  # close to fully distinct (3)


def test_null_component_collapse_diagnostic_detects_collapse():
    rng = np.random.default_rng(3)
    n, n_features = 300, 8
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    # All three "components" point at the SAME region -> should collapse.
    X[:, 0] += 10
    gamma = rng.dirichlet(np.ones(3), size=n)
    diag = null_component_collapse_diagnostic(X, gamma, top_frac=0.3, min_n=20)
    assert diag['effective_n_components'] < 1.5


# ---------------------------------------------------------------------------
# 2. mi-SVM / EM recover a sensible posterior on separable synthetic data
# ---------------------------------------------------------------------------

def test_misvm_recovers_class_separation():
    X, y_bag, bag_ids, z_true = _synthetic_constrained_mil(seed=10)
    fit = fit_misvm(X, bag_ids, y_bag, n_classes=2, pi_min=0.1, max_iters=5,
                    seed=10, verbose=False)
    posterior = fit['posterior']
    is_diagnostic = z_true >= 3
    # True diagnostic instances should get more of their OWN class's posterior
    # mass than the true null instances get of that same class, on average.
    for cls in (0, 1):
        own_class_mass_diag = posterior[is_diagnostic & (y_bag == cls), 1 + cls]
        own_class_mass_null = posterior[~is_diagnostic & (y_bag == cls), 1 + cls]
        assert own_class_mass_diag.mean() > own_class_mass_null.mean()


def test_em_recovers_class_separation_and_reasonable_pi_b():
    X, y_bag, bag_ids, z_true = _synthetic_constrained_mil(pi_true=0.3, seed=11)
    fit = fit_em(X, bag_ids, y_bag, n_classes=2, pi_min=0.05, pi_init=0.2,
                max_iters=15, seed=11, verbose=False)
    posterior = fit['posterior']
    is_diagnostic = z_true >= 3
    for cls in (0, 1):
        own_class_mass_diag = posterior[is_diagnostic & (y_bag == cls), 1 + cls]
        own_class_mass_null = posterior[~is_diagnostic & (y_bag == cls), 1 + cls]
        assert own_class_mass_diag.mean() > own_class_mass_null.mean()
    # Fitted pi_b should land somewhere in the right ballpark of the true 0.3,
    # not collapse to 0 or 1 (a coarse sanity bound, not a precision claim).
    mean_pi_b = np.mean(list(fit['pi_b'].values()))
    assert 0.05 < mean_pi_b < 0.9


def test_em_shrinkage_stronger_for_smaller_bags():
    """Empirical-Bayes property this implementation is supposed to have: a
    smaller bag's fitted pi_b should sit closer to the population mean mu than
    a larger bag's, all else equal (docs/MIL_REFRAME_PLAN.md §1: 'partial
    pooling borrows strength across bags')."""
    X, y_bag, bag_ids, z_true = _synthetic_constrained_mil(
        n_bags_per_class=6, n_per_bag=60, pi_true=0.3, seed=12)

    # Shrink one specific bag down to very few instances (weak own evidence),
    # leaving the rest at full size (strong own evidence).
    target_bag = bag_ids[0]
    target_idx = np.where(bag_ids == target_bag)[0][:6]
    other_idx = np.where(bag_ids != target_bag)[0]
    keep = np.concatenate([target_idx, other_idx])
    X_s, y_s, bag_s, z_s = X[keep], y_bag[keep], bag_ids[keep], z_true[keep]

    fit = fit_em(X_s, bag_s, y_s, n_classes=2, pi_min=0.05, pi_init=0.2,
                max_iters=15, seed=12, verbose=False)
    mu_prob = 1.0 / (1.0 + np.exp(-fit['mu']))
    small_bag_pi = fit['pi_b'][target_bag]
    large_bag_pis = [v for b, v in fit['pi_b'].items() if b != target_bag]

    small_bag_dist_to_mu = abs(small_bag_pi - mu_prob)
    large_bag_dist_to_mu = np.mean([abs(v - mu_prob) for v in large_bag_pis])
    assert small_bag_dist_to_mu <= large_bag_dist_to_mu + 0.05  # allow small noise margin


def test_subsample_for_svm_caps_every_bag_without_starving_any():
    """Apollo's RBF SVM scales badly with sample count; Stage 4 retired the
    global patient_cap, so this per-bag re-cap (for the SVM fit only) is what
    keeps a single fit tractable at full fold size. Every bag must still
    contribute at least min(bag_size, cap) instances -- none dropped
    entirely -- and no bag may exceed the cap."""
    rng = np.random.default_rng(30)
    bag_ids = np.concatenate([
        np.full(50, 'big', dtype=object),
        np.full(10, 'small', dtype=object),   # below the cap already
        np.full(500, 'huge', dtype=object),
    ])
    idx = _subsample_for_svm(bag_ids, cap_per_bag=20, seed=1)
    kept = bag_ids[idx]
    counts = {b: int((kept == b).sum()) for b in np.unique(bag_ids)}
    assert counts['big'] == 20
    assert counts['small'] == 10  # under the cap -> kept whole, not padded
    assert counts['huge'] == 20
    assert len(idx) == len(set(idx.tolist()))  # no duplicate rows


def test_misvm_subsampling_still_recovers_class_separation():
    """The whole point of the cap: fitting on a subsample must not break
    mi-SVM's actual job. Uses a bag size comfortably above a tiny cap so the
    subsampling path is definitely exercised, not accidentally skipped."""
    X, y_bag, bag_ids, z_true = _synthetic_constrained_mil(
        n_bags_per_class=6, n_per_bag=80, seed=31)
    fit = fit_misvm(X, bag_ids, y_bag, n_classes=2, pi_min=0.1, max_iters=5,
                    seed=31, verbose=False, svm_subsample_per_bag=30)
    posterior = fit['posterior']
    is_diagnostic = z_true >= 3
    for cls in (0, 1):
        own_class_mass_diag = posterior[is_diagnostic & (y_bag == cls), 1 + cls]
        own_class_mass_null = posterior[~is_diagnostic & (y_bag == cls), 1 + cls]
        assert own_class_mass_diag.mean() > own_class_mass_null.mean()


def test_misvm_and_em_score_functions_work_on_held_out_data():
    X, y_bag, bag_ids, _ = _synthetic_constrained_mil(seed=13)
    n_train = len(y_bag) * 3 // 4
    X_tr, y_tr, bag_tr = X[:n_train], y_bag[:n_train], bag_ids[:n_train]
    X_va, y_va, bag_va = X[n_train:], y_bag[n_train:], bag_ids[n_train:]

    misvm_fit = fit_misvm(X_tr, bag_tr, y_tr, n_classes=2, max_iters=3, seed=13, verbose=False)
    val_posterior_misvm = predict_misvm(misvm_fit, X_va)
    assert val_posterior_misvm.shape == (len(y_va), 3)
    np.testing.assert_allclose(val_posterior_misvm.sum(axis=1), 1.0, atol=1e-6)

    em_fit = fit_em(X_tr, bag_tr, y_tr, n_classes=2, max_iters=5, seed=13, verbose=False)
    val_posterior_em = score_em(em_fit, X_va, bag_va, y_va, n_classes=2)
    assert val_posterior_em.shape == (len(y_va), 3)
    np.testing.assert_allclose(val_posterior_em.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. src/mil/store.py -- leakage assertion + alignment checksum
# ---------------------------------------------------------------------------

def test_store_write_read_roundtrip(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    n = 20
    posterior = np.random.default_rng(0).dirichlet(np.ones(3), size=n).astype(np.float32)
    in_sample = np.array([True] * 15 + [False] * 5)
    sample_ids = np.array([f"slide_{i // 4}" for i in range(n)])
    coords = np.stack([np.arange(n), np.arange(n)], axis=1)

    write_posteriors(store_path, 'em', 'constrained', 'fold_0', posterior, in_sample,
                     sample_ids, coords)
    data = read_posteriors(store_path, 'em', 'constrained', 'fold_0')
    np.testing.assert_allclose(data['posterior'], posterior)
    np.testing.assert_array_equal(data['in_sample'], in_sample)
    assert data['attrs']['method'] == 'em'
    assert data['attrs']['K'] == 2


def test_store_alignment_checksum_rejects_stale_reexport(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    sample_ids = np.array(['a', 'b', 'c'])
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    posterior = np.random.default_rng(0).dirichlet(np.ones(2), size=3).astype(np.float32)
    write_posteriors(store_path, 'em', 'constrained', 'fold_0', posterior,
                     np.array([True, True, False]), sample_ids, coords)

    correct_checksum = alignment_checksum(sample_ids, coords)
    read_posteriors(store_path, 'em', 'constrained', 'fold_0',
                    expected_checksum=correct_checksum)  # should not raise

    stale_checksum = alignment_checksum(np.array(['a', 'b', 'X']), coords)
    with pytest.raises(AssertionError, match="alignment_checksum mismatch"):
        read_posteriors(store_path, 'em', 'constrained', 'fold_0',
                        expected_checksum=stale_checksum)


def test_fold_leakage_assertion_passes_when_correctly_out_of_sample(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    bag_ids_all = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    # Fold 0's own validation bag is 'C' -> correctly marked in_sample=False.
    in_sample = np.array([True, True, True, True, False, False])
    posterior = np.random.default_rng(0).dirichlet(np.ones(2), size=6).astype(np.float32)
    write_posteriors(store_path, 'em', 'constrained', 'fold_0', posterior, in_sample,
                     bag_ids_all, None)

    assert_no_fold_leakage(store_path, 'em', 'constrained', 'fold_0',
                           val_bag_ids_this_fold=['C'], bag_ids_all=bag_ids_all)  # no raise


def test_merge_out_of_sample_posterior_averages_duplicated_bag(tmp_path):
    """A bag that is a validation (out-of-sample) bag in TWO folds -- §7.1's
    documented case -- must get the AVERAGE of both folds' estimates, not
    whichever fold happened to be read first, and ordinary single-fold bags
    must pass through unchanged."""
    store_path = tmp_path / 'latent_labels.h5'
    bag_ids_all = np.array(['A', 'A', 'B', 'B'])  # A: ordinary, B: duplicated

    # Fold 0: A in-sample (train), B out-of-sample (val) with posterior [0.2, 0.8]
    write_posteriors(
        store_path, 'em', 'constrained', 'fold_0',
        posterior=np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.2, 0.8]], dtype=np.float32),
        in_sample=np.array([True, True, False, False]),
        sample_ids=bag_ids_all, coords=None)

    # Fold 1: A out-of-sample (val) with posterior [0.7, 0.3]; B ALSO
    # out-of-sample here (the duplicated case) with posterior [0.4, 0.6].
    write_posteriors(
        store_path, 'em', 'constrained', 'fold_1',
        posterior=np.array([[0.7, 0.3], [0.7, 0.3], [0.4, 0.6], [0.4, 0.6]], dtype=np.float32),
        in_sample=np.array([False, False, False, False]),
        sample_ids=bag_ids_all, coords=None)

    merged, counts = merge_out_of_sample_posterior(
        store_path, 'em', 'constrained', ['fold_0', 'fold_1'])

    np.testing.assert_array_equal(counts, [1, 1, 2, 2])
    np.testing.assert_allclose(merged[0], [0.7, 0.3], atol=1e-5)  # A: only fold_1 was OOS
    np.testing.assert_allclose(merged[2], [0.3, 0.7], atol=1e-5)  # B: mean of [.2,.8] and [.4,.6]
    np.testing.assert_allclose(merged.sum(axis=1), 1.0, atol=1e-5)


def test_fold_leakage_assertion_raises_on_mismatched_fold(tmp_path):
    """The plan's own required negative test: deliberately asking about a bag
    that IS marked in_sample=True (i.e. treating a training bag as if it were
    this fold's held-out validation bag) must raise."""
    store_path = tmp_path / 'latent_labels.h5'
    bag_ids_all = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    in_sample = np.array([True, True, True, True, False, False])
    posterior = np.random.default_rng(0).dirichlet(np.ones(2), size=6).astype(np.float32)
    write_posteriors(store_path, 'em', 'constrained', 'fold_0', posterior, in_sample,
                     bag_ids_all, None)

    with pytest.raises(AssertionError, match="in_sample=True"):
        assert_no_fold_leakage(store_path, 'em', 'constrained', 'fold_0',
                               val_bag_ids_this_fold=['A'],  # wrong: A was in-sample
                               bag_ids_all=bag_ids_all)


# ---------------------------------------------------------------------------
# pi_b external validity (docs/MIL_REFRAME_PLAN.md §1, backlog §13):
# scripts/stage4_pi_b_external_validity.py's per-bag averaging
# ---------------------------------------------------------------------------

def test_load_pi_b_per_bag_averages_across_folds(tmp_path):
    """A bag in TRAINING for several folds should get its pi_b averaged, not
    overwritten by whichever fold's diagnostics happened to load last."""
    import json

    from scripts.stage4_pi_b_external_validity import _load_pi_b_per_bag

    path = tmp_path / 'em_diagnostics.json'
    with open(path, 'w') as f:
        json.dump({
            'fold_0': {'pi_b': {'A': 0.10, 'B': 0.20}},
            'fold_1': {'pi_b': {'A': 0.20, 'C': 0.30}},
            'fold_2': {'pi_b': {'A': 0.30}},
        }, f)
    pi_b, counts = _load_pi_b_per_bag(path)
    assert pi_b['A'] == pytest.approx(0.20)  # mean of 0.10, 0.20, 0.30
    assert counts['A'] == 3
    assert pi_b['B'] == pytest.approx(0.20)
    assert counts['B'] == 1
    assert pi_b['C'] == pytest.approx(0.30)
    assert counts['C'] == 1
