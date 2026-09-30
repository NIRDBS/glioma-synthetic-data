"""Regression tests for src/mil/counterfactual.py (the CVAE counterfactual
probe -- see that module's docstring and docs/counterfactual_probe.md).

All synthetic, no real CVAE weights needed -- reuses
tests/test_stage6_generative_mil.py's FakeEncoder/FakeDecoder pattern
(z_mean == X, true class = sign of the first feature, decoder reconstructs
exactly on a class match and += 5.0 off on mismatch), which makes
Delta(z) an exact, hand-checkable constant (+5 or -5) depending on an
instance's true class -- see _fake_X/FakeEncoder/FakeDecoder below.
"""
import numpy as np
import pytest

from src.mil.counterfactual import (
    IDH_MUTANT_CLASS_IDX,
    IDH_WILDTYPE_CLASS_IDX,
    band_assignment_report,
    bootstrap_ci_mean,
    compute_counterfactual_reconstructions,
    compute_delta,
    cosine_to_reference,
    delta_effect_size,
    fit_z_class_probe,
    independent_classifier_flip_rate,
    mean_delta_per_bag,
    merge_out_of_sample_arrays,
    normalized_distance_to_reference,
    pairwise_cosine_stability,
    random_direction_null,
    score_z_class_probe,
    within_class_reconstruction_spread,
)


def test_class_index_constants_match_project_convention():
    """Every reported sign in the paper rides on this one constant --
    verified from three independent sources this session
    (src/evaluation/synthetic_realism.py's CLASS_NAMES/binary mapping,
    pipeline/_common.py::prepare_fold's docstring, paper/main.tex's
    LGm-to-IDH correspondence). A loud regression guard is worth it."""
    assert IDH_MUTANT_CLASS_IDX == 0
    assert IDH_WILDTYPE_CLASS_IDX == 1


# ---------------------------------------------------------------------------
# Deterministic fake encoder/decoder -- same construction as
# tests/test_stage6_generative_mil.py's FakeEncoder/FakeDecoder
# ---------------------------------------------------------------------------

class FakeEncoder:
    def predict(self, X_exp, batch_size=256, verbose=0):
        n = X_exp.shape[0]
        z_mean = X_exp.reshape(n, -1).astype(np.float32)
        z_log_var = np.zeros_like(z_mean)
        z = z_mean.copy()
        true_class = (z_mean[:, 0] >= 0).astype(int)
        class_pred = np.zeros((n, 2), dtype=np.float32)
        class_pred[np.arange(n), true_class] = 1.0
        return z_mean, z_log_var, z, class_pred


class FakeDecoder:
    def predict(self, inputs, batch_size=256, verbose=0):
        z, label = inputs
        requested = np.argmax(label, axis=1)
        true_class = (z[:, 0] >= 0).astype(int)
        match = requested == true_class
        rec = z.copy()
        rec[~match] += 5.0
        return rec


def _fake_X(n_per_class=4, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.uniform(0.1, 1.0, size=(n_per_class, n_features)).astype(np.float32)  # true_class=1
    X_neg = rng.uniform(-1.0, -0.1, size=(n_per_class, n_features)).astype(np.float32)  # true_class=0
    X = np.concatenate([X_pos, X_neg])
    true_class = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)]).astype(int)
    return X, true_class


# ---------------------------------------------------------------------------
# compute_counterfactual_reconstructions / compute_delta
# ---------------------------------------------------------------------------

def test_compute_counterfactual_reconstructions_shapes_and_values():
    X, true_class = _fake_X(seed=1)
    out = compute_counterfactual_reconstructions(X, FakeEncoder(), FakeDecoder(), num_classes=2)
    assert out['z_mean'].shape == X.shape
    assert out['recon'].shape == (len(X), 2, X.shape[1])
    np.testing.assert_allclose(out['z_mean'], X, atol=1e-6)
    # Per the fake's rule: reconstruction under the MATCHING class-condition
    # equals z_mean exactly; under the mismatching one it's z_mean + 5.
    for i, tc in enumerate(true_class):
        np.testing.assert_allclose(out['recon'][i, tc, :], X[i], atol=1e-6)
        np.testing.assert_allclose(out['recon'][i, 1 - tc, :], X[i] + 5.0, atol=1e-6)


def test_compute_delta_matches_hand_computed_value_for_fake_decoder():
    """Delta = recon[:,mutant,:] - recon[:,wildtype,:]. true_class=1 rows:
    mutant(0)-condition is a mismatch (+5), wildtype(1)-condition matches (0)
    -> Delta = (z+5) - z = +5. true_class=0 rows: mutant matches (0),
    wildtype mismatches (+5) -> Delta = z - (z+5) = -5. Constant, exact."""
    X, true_class = _fake_X(seed=2)
    out = compute_counterfactual_reconstructions(X, FakeEncoder(), FakeDecoder(), num_classes=2)
    delta = compute_delta(out['recon'])
    assert delta.shape == X.shape
    sign = np.where(true_class == 1, 5.0, -5.0)
    expected = np.broadcast_to(sign[:, None], delta.shape)
    np.testing.assert_allclose(delta, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# merge_out_of_sample_arrays
# ---------------------------------------------------------------------------

def test_merge_out_of_sample_arrays_averages_duplicate_bag_and_leaves_uncovered_nan():
    full_bag_ids = np.array(['a', 'b', 'c', 'd'])  # 4 rows total
    # fold 0 covers rows 0,1 (a,b) with array values 10,20
    mask0 = np.array([True, True, False, False])
    arr0 = np.array([[10.0], [20.0]])
    # fold 1 also covers row 0 (a duplicated bag) plus row 2 (c)
    mask1 = np.array([True, False, True, False])
    arr1 = np.array([[30.0], [40.0]])
    # row 3 (d) covered by neither fold

    merged, counts = merge_out_of_sample_arrays([(mask0, arr0), (mask1, arr1)], full_bag_ids)
    assert merged.shape == (4, 1)
    np.testing.assert_allclose(merged[0], [20.0], atol=1e-6)  # avg(10,30)=20
    np.testing.assert_allclose(merged[1], [20.0], atol=1e-6)  # only fold 0
    np.testing.assert_allclose(merged[2], [40.0], atol=1e-6)  # only fold 1
    assert np.isnan(merged[3]).all()
    np.testing.assert_array_equal(counts, [2, 1, 1, 0])


# ---------------------------------------------------------------------------
# mean_delta_per_bag -- the bag-of-bags convention, pinned against the naive
# pooled-instance alternative on a case where they visibly differ
# ---------------------------------------------------------------------------

def test_mean_delta_per_bag_grand_mean_is_bag_of_bags_not_pooled_instances():
    # Bag 'big': 3 instances all at value 0.0. Bag 'small': 1 instance at 12.0.
    # Bag-of-bags grand mean = mean(0.0, 12.0) = 6.0.
    # Pooled-instance mean = (0+0+0+12)/4 = 3.0 -- a different number, showing
    # the naive convention would let the big bag dominate.
    bag_ids = np.array(['big', 'big', 'big', 'small'])
    delta = np.array([[0.0], [0.0], [0.0], [12.0]])
    bags_sorted, per_bag, grand_mean = mean_delta_per_bag(delta, bag_ids)
    assert list(bags_sorted) == ['big', 'small']
    np.testing.assert_allclose(per_bag, [[0.0], [12.0]], atol=1e-6)
    np.testing.assert_allclose(grand_mean, [6.0], atol=1e-6)


def test_mean_delta_per_bag_excludes_bags_with_zero_covered_instances():
    bag_ids = np.array(['a', 'a', 'b'])
    delta = np.array([[np.nan], [np.nan], [4.0]])  # bag 'a' entirely uncovered
    bags_sorted, per_bag, grand_mean = mean_delta_per_bag(delta, bag_ids)
    assert list(bags_sorted) == ['b']
    np.testing.assert_allclose(per_bag, [[4.0]], atol=1e-6)
    np.testing.assert_allclose(grand_mean, [4.0], atol=1e-6)


# ---------------------------------------------------------------------------
# cosine_to_reference / normalized_distance_to_reference
# ---------------------------------------------------------------------------

def test_cosine_to_reference_hand_computed():
    reference = np.array([1.0, 0.0])
    vectors = np.array([
        [1.0, 0.0],    # identical direction -> cosine 1
        [0.0, 1.0],    # orthogonal -> cosine 0
        [-1.0, 0.0],   # opposite -> cosine -1
    ])
    out = cosine_to_reference(vectors, reference)
    np.testing.assert_allclose(out, [1.0, 0.0, -1.0], atol=1e-6)


def test_normalized_distance_to_reference_hand_computed():
    reference = np.array([3.0, 4.0])  # ||reference|| = 5
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = normalized_distance_to_reference(vectors, reference)
    # row 0: ||0||/5 = 0 ; row 1: ||(-3,-4)||/5 = 5/5 = 1
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# pairwise_cosine_stability / random_direction_null
# ---------------------------------------------------------------------------

def test_pairwise_cosine_stability_identical_vectors_gives_effective_n_one():
    vectors = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    out = pairwise_cosine_stability(vectors)
    np.testing.assert_allclose(out['pairwise_cosine'], np.ones((5, 5)), atol=1e-6)
    assert out['effective_n_components'] == pytest.approx(1.0, abs=1e-6)
    assert out['mean_offdiag_cosine'] == pytest.approx(1.0, abs=1e-6)


def test_pairwise_cosine_stability_orthogonal_vectors_gives_effective_n_equal_m():
    vectors = np.eye(4)  # 4 orthogonal one-hot vectors
    out = pairwise_cosine_stability(vectors)
    np.testing.assert_allclose(out['pairwise_cosine'], np.eye(4), atol=1e-6)
    assert out['effective_n_components'] == pytest.approx(4.0, abs=1e-6)
    assert out['mean_offdiag_cosine'] == pytest.approx(0.0, abs=1e-6)


def test_random_direction_null_returns_near_zero_mean_for_large_d():
    out = random_direction_null(d=200, m=5, n_trials=500, seed=42)
    assert abs(out['null_mean']) < 0.05  # ~1/sqrt(200) ~= 0.07-ish scale, generous tolerance
    assert out['null_std'] > 0.0


def test_random_direction_null_handles_single_fold_gracefully():
    """A fold-name-subset smoke test (e.g. --folds fold_0) has m=1 and no
    off-diagonal pair at all -- must return NaN cleanly, not raise/warn."""
    out = random_direction_null(d=50, m=1, n_trials=100, seed=42)
    assert np.isnan(out['null_mean'])
    assert np.isnan(out['null_std'])


# ---------------------------------------------------------------------------
# within_class_reconstruction_spread / delta_effect_size
# ---------------------------------------------------------------------------

def test_within_class_reconstruction_spread_hand_computed():
    # 2 instances per class, 1 feature. mutant(0) factual recons: [1.0, 3.0] -> std=1.0.
    # wildtype(1) factual recons: [10.0, 10.0] -> std=0.0.
    recon = np.zeros((4, 2, 1))
    y_true = np.array([0, 0, 1, 1])
    recon[0, 0, 0] = 1.0
    recon[1, 0, 0] = 3.0
    recon[2, 1, 0] = 10.0
    recon[3, 1, 0] = 10.0
    out = within_class_reconstruction_spread(recon, y_true)
    np.testing.assert_allclose(out['mutant'], [1.0], atol=1e-6)
    np.testing.assert_allclose(out['wildtype'], [0.0], atol=1e-6)
    np.testing.assert_allclose(out['pooled'], [0.5], atol=1e-6)


def test_delta_effect_size_hand_computed_including_zero_spread_guard():
    mean_delta = np.array([2.0, -3.0, 5.0])
    spread = np.array([1.0, 1.5, 0.0])  # last feature has zero spread
    out = delta_effect_size(mean_delta, spread)
    np.testing.assert_allclose(out[:2], [2.0, 2.0], atol=1e-6)
    assert np.isnan(out[2])


# ---------------------------------------------------------------------------
# fit_z_class_probe / score_z_class_probe
# ---------------------------------------------------------------------------

def test_fit_and_score_z_class_probe_recovers_perfectly_separable_classes():
    rng = np.random.default_rng(0)
    z_train = np.concatenate([rng.normal(5, 0.1, size=(30, 2)), rng.normal(-5, 0.1, size=(30, 2))])
    y_train = np.array([1] * 30 + [0] * 30)
    z_val = np.concatenate([rng.normal(5, 0.1, size=(10, 2)), rng.normal(-5, 0.1, size=(10, 2))])
    y_val = np.array([1] * 10 + [0] * 10)

    scaler, model = fit_z_class_probe(z_train, y_train, seed=42)
    proba = score_z_class_probe(scaler, model, z_val)
    pred = proba.argmax(axis=1)
    assert (pred == y_val).mean() > 0.95


def test_fit_and_score_z_class_probe_near_chance_on_class_independent_noise():
    """The 'doesn't falsely trigger' companion: pure noise z_mean, independent
    of class, must NOT let the probe claim strong separability."""
    rng = np.random.default_rng(1)
    z_train = rng.normal(0, 1, size=(200, 2))
    y_train = rng.integers(0, 2, size=200)
    z_val = rng.normal(0, 1, size=(200, 2))
    y_val = rng.integers(0, 2, size=200)

    scaler, model = fit_z_class_probe(z_train, y_train, seed=42)
    proba = score_z_class_probe(scaler, model, z_val)
    pred = proba.argmax(axis=1)
    acc = (pred == y_val).mean()
    assert 0.35 < acc < 0.65  # near chance (0.5), generous band for sampling noise


# ---------------------------------------------------------------------------
# independent_classifier_flip_rate
# ---------------------------------------------------------------------------

def test_independent_classifier_flip_rate_hand_computed(monkeypatch):
    """A deterministic stand-in 'classifier' (sign of feature 0), on
    hand-built X_real/recon_swap pairs designed so flip_to_target_rate and
    flip_from_own_rate come out DIFFERENT -- including one instance where
    the baseline was already wrong on the real spectrum (instance index 1),
    the exact case that makes the two rates diverge."""
    import src.mil.counterfactual as cf_module

    def fake_load_and_predict(path, model_type, X):
        return (X[:, 0] >= 0).astype(int)

    monkeypatch.setattr(
        'src.utils.statistical_evaluation.load_and_predict', fake_load_and_predict)

    y_true = np.array([0, 0, 1, 1])
    # pred_real via sign rule: [0, 1, 1, 1] -- instance 1 is WRONG (true=0, pred=1)
    X_real = np.array([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    # pred_cf via sign rule: [1, 1, 0, 1]
    recon_swap = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])

    out = independent_classifier_flip_rate(
        X_real, y_true, recon_swap, model_path='unused', model_type='sklearn')

    np.testing.assert_array_equal(out['pred_real'], [0, 1, 1, 1])
    np.testing.assert_array_equal(out['pred_cf'], [1, 1, 0, 1])
    # target = 1-y_true = [1,1,0,0]; pred_cf==target -> [T,T,T,F] -> 0.75
    assert out['flip_to_target_rate'] == pytest.approx(0.75, abs=1e-6)
    # pred_cf != pred_real -> [T,F,T,F] -> 0.50
    assert out['flip_from_own_rate'] == pytest.approx(0.50, abs=1e-6)
    assert out['flip_to_target_rate'] != out['flip_from_own_rate']


# ---------------------------------------------------------------------------
# bootstrap_ci_mean
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean_contains_the_true_mean_and_is_seeded_deterministic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out1 = bootstrap_ci_mean(values, n_boot=500, seed=42)
    out2 = bootstrap_ci_mean(values, n_boot=500, seed=42)
    assert out1 == out2  # deterministic given the same seed
    assert out1['mean'] == pytest.approx(3.0, abs=1e-6)
    assert out1['ci_lo'] <= out1['mean'] <= out1['ci_hi']


# ---------------------------------------------------------------------------
# band_assignment_report
# ---------------------------------------------------------------------------

def test_band_assignment_report_maps_to_nearest_wavenumber_and_reports_correct_sign():
    wavenumbers = np.array([998.0, 1005.0, 1650.0, 1658.0])
    mean_delta = np.array([0.1, 2.0, -3.0, -3.5])
    per_fold_mean_delta = np.array([
        [0.1, 2.0, -3.0, -3.5],   # agrees in sign at every band
        [0.1, -1.0, -3.0, -3.5],  # disagrees at "Phe" band (index 1)
    ])
    spread = np.array([1.0, 1.0, 1.0, 0.0])  # last band: zero spread -> effect_size None
    peaks = {'Phe': 1003, 'Amide I': 1655}

    out = band_assignment_report(mean_delta, wavenumbers, per_fold_mean_delta=per_fold_mean_delta,
                                 spread=spread, peaks=peaks)

    assert set(out.keys()) == {'Phe', 'Amide I'}
    # 1003 nearest to 1005.0 (index 1)
    assert out['Phe']['wavenumber_matched'] == pytest.approx(1005.0)
    assert out['Phe']['delta'] == pytest.approx(2.0)
    assert out['Phe']['direction'] == 'higher in IDH-Mutant'
    assert out['Phe']['effect_size'] == pytest.approx(2.0)
    assert out['Phe']['fold_sign_agreement_frac'] == pytest.approx(0.5)  # 1/2 folds agree

    # 1655 nearest to 1658.0 (index 3; 1650.0 is equidistant-minus-2, not a tie)
    assert out['Amide I']['delta'] == pytest.approx(-3.5)
    assert out['Amide I']['direction'] == 'higher in IDH-Wildtype'
    assert out['Amide I']['effect_size'] is None  # zero spread guard
    assert out['Amide I']['fold_sign_agreement_frac'] == pytest.approx(1.0)


def test_band_assignment_report_uses_project_raman_peaks_by_default():
    """Confirms the default `peaks=None` path really does import the
    project's own in-crop RAMAN_PEAKS table, not an empty/placeholder one."""
    from src.evaluation.synthetic_realism import RAMAN_PEAKS

    wavenumbers = np.linspace(700, 1800, 571)
    mean_delta = np.zeros(571)
    out = band_assignment_report(mean_delta, wavenumbers)
    assert set(out.keys()) == set(RAMAN_PEAKS.keys())
