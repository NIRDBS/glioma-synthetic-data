"""MIL formulations and the latent-label store:
  1. Fold-leakage assertion: a deliberately mismatched fold must raise.
  2. Alignment checksum rejects a stale posterior file after re-export.
  3. mi-SVM recovers a sensible posterior on a clearly-separable synthetic
     MIL dataset (a basic correctness sanity-check before it is trusted on
     real data).
"""
import numpy as np
import pytest

from src.mil.misvm import _subsample_for_svm, fit_misvm, predict_misvm
from src.mil.store import (
    alignment_checksum, assert_no_fold_leakage, config_hash, merge_out_of_sample_posterior,
    read_posteriors, write_posteriors,
)


# ---------------------------------------------------------------------------
# Synthetic constrained-MIL dataset: 2 classes, 3 null components, well
# separated in feature space so the formulation has something real to
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
# 1. mi-SVM recovers a sensible posterior on separable synthetic data
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


def test_subsample_for_svm_caps_every_bag_without_starving_any():
    """Apollo's RBF SVM scales badly with sample count; mi-SVM does not use the
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


def test_misvm_score_function_works_on_held_out_data():
    X, y_bag, bag_ids, _ = _synthetic_constrained_mil(seed=13)
    n_train = len(y_bag) * 3 // 4
    X_tr, y_tr, bag_tr = X[:n_train], y_bag[:n_train], bag_ids[:n_train]
    X_va, y_va = X[n_train:], y_bag[n_train:]

    misvm_fit = fit_misvm(X_tr, bag_tr, y_tr, n_classes=2, max_iters=3, seed=13, verbose=False)
    val_posterior_misvm = predict_misvm(misvm_fit, X_va)
    assert val_posterior_misvm.shape == (len(y_va), 3)
    np.testing.assert_allclose(val_posterior_misvm.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. src/mil/store.py -- leakage assertion + alignment checksum
# ---------------------------------------------------------------------------

def test_store_write_read_roundtrip(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    n = 20
    posterior = np.random.default_rng(0).dirichlet(np.ones(3), size=n).astype(np.float32)
    in_sample = np.array([True] * 15 + [False] * 5)
    sample_ids = np.array([f"slide_{i // 4}" for i in range(n)])
    coords = np.stack([np.arange(n), np.arange(n)], axis=1)

    write_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0', posterior, in_sample,
                     sample_ids, coords)
    data = read_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0')
    np.testing.assert_allclose(data['posterior'], posterior)
    np.testing.assert_array_equal(data['in_sample'], in_sample)
    assert data['attrs']['method'] == 'misvm'
    assert data['attrs']['K'] == 2


def test_store_alignment_checksum_rejects_stale_reexport(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    sample_ids = np.array(['a', 'b', 'c'])
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    posterior = np.random.default_rng(0).dirichlet(np.ones(2), size=3).astype(np.float32)
    write_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0', posterior,
                     np.array([True, True, False]), sample_ids, coords)

    correct_checksum = alignment_checksum(sample_ids, coords)
    read_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0',
                    expected_checksum=correct_checksum)  # should not raise

    stale_checksum = alignment_checksum(np.array(['a', 'b', 'X']), coords)
    with pytest.raises(AssertionError, match="alignment_checksum mismatch"):
        read_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0',
                        expected_checksum=stale_checksum)


def test_fold_leakage_assertion_passes_when_correctly_out_of_sample(tmp_path):
    store_path = tmp_path / 'latent_labels.h5'
    bag_ids_all = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
    # Fold 0's own validation bag is 'C' -> correctly marked in_sample=False.
    in_sample = np.array([True, True, True, True, False, False])
    posterior = np.random.default_rng(0).dirichlet(np.ones(2), size=6).astype(np.float32)
    write_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0', posterior, in_sample,
                     bag_ids_all, None)

    assert_no_fold_leakage(store_path, 'misvm', 'mi_svm', 'fold_0',
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
        store_path, 'misvm', 'mi_svm', 'fold_0',
        posterior=np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.2, 0.8]], dtype=np.float32),
        in_sample=np.array([True, True, False, False]),
        sample_ids=bag_ids_all, coords=None)

    # Fold 1: A out-of-sample (val) with posterior [0.7, 0.3]; B ALSO
    # out-of-sample here (the duplicated case) with posterior [0.4, 0.6].
    write_posteriors(
        store_path, 'misvm', 'mi_svm', 'fold_1',
        posterior=np.array([[0.7, 0.3], [0.7, 0.3], [0.4, 0.6], [0.4, 0.6]], dtype=np.float32),
        in_sample=np.array([False, False, False, False]),
        sample_ids=bag_ids_all, coords=None)

    merged, counts = merge_out_of_sample_posterior(
        store_path, 'misvm', 'mi_svm', ['fold_0', 'fold_1'])

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
    write_posteriors(store_path, 'misvm', 'mi_svm', 'fold_0', posterior, in_sample,
                     bag_ids_all, None)

    with pytest.raises(AssertionError, match="in_sample=True"):
        assert_no_fold_leakage(store_path, 'misvm', 'mi_svm', 'fold_0',
                               val_bag_ids_this_fold=['A'],  # wrong: A was in-sample
                               bag_ids_all=bag_ids_all)


def test_config_hash_is_stable_and_distinguishes_configs():
    """The store's config_hash attribute must identify a config across runs:
    order-independent for dict keys, different for different configs."""
    a = {'name': 'exp_608', 'mil': {'pi_floor': 0.02, 'svm_subsample_per_bag': 500}}
    b = {'mil': {'svm_subsample_per_bag': 500, 'pi_floor': 0.02}, 'name': 'exp_608'}
    c = {'name': 'exp_608', 'mil': {'pi_floor': 0.2, 'svm_subsample_per_bag': 500}}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 64


def test_config_hash_is_identical_across_processes():
    """Regression: the previous str(hash(...)) changed with every interpreter
    start (string hashing is salted per process), so the stored value could not
    identify a config. Two processes with different hash seeds must agree."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("from src.mil.store import config_hash; "
            "print(config_hash({'name': 'exp_608', 'mil': {'pi_floor': 0.02}}))")
    root = Path(__file__).resolve().parents[1]
    out = []
    for seed in ('1', '2'):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out.append(subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                                  capture_output=True, text=True, check=True).stdout.strip())
    assert out[0] == out[1] != ''
