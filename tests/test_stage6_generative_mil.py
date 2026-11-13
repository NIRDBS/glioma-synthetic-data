"""Stage 6 (Tier 4) regression tests: generative MIL from the CVAE.

All synthetic, no real CVAE weights or exp_501 export needed -- the encoder/
decoder are lightweight fakes with a fully deterministic, hand-checkable
input/output relationship (see ``FakeEncoder``/``FakeDecoder`` below), in the
same spirit as ``tests/test_stage4_mil.py``'s synthetic constrained-MIL
dataset: a correctness sanity floor, not a claim about real-data performance.

Covers:
  1. ``src/mil/generative.py``'s pure-numpy pieces (``mse_to_posterior``,
     ``class_probs_to_posterior``, ``_to_K1_posterior``'s shared convention)
     on hand-built arrays.
  2. ``instance_posteriors_from_cvae`` end-to-end against the fakes, checking
     both variants recover the right answer and the non-diagnostic-mass
     convention holds.
  3. ``pipeline/mil_fit_stage.py``'s ``_score_one_fold_cbr`` -- the scatter-
     back onto the full dataset's row order and the in_sample bookkeeping,
     monkeypatching ``load_encoder_decoder`` so no real weights are touched.
"""
import numpy as np
import pytest

from src.mil.generative import (
    _to_K1_posterior, class_probs_to_posterior, instance_posteriors_from_cvae,
    mse_to_posterior,
)


# ---------------------------------------------------------------------------
# 1. Pure-numpy pieces
# ---------------------------------------------------------------------------

def test_to_K1_posterior_matches_misvm_convention_and_sums_to_one():
    """Same convention as src/mil/misvm.py's combination rule: non-diagnostic
    mass = 1 - max_k p_k, renormalised."""
    class_probs = np.array([
        [0.9, 0.1],   # confident class 0 -> low non-diagnostic mass
        [0.5, 0.5],   # uncertain -> non-diagnostic mass = 0.5 before renorm
    ])
    out = _to_K1_posterior(class_probs)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    # row 0: raw = [0.1, 0.9, 0.1], renormalised (sums to 1.1 before dividing)
    raw0 = np.array([0.1, 0.9, 0.1])
    np.testing.assert_allclose(out[0], raw0 / raw0.sum(), atol=1e-6)
    # row 1: raw = [0.5, 0.5, 0.5] / 1.5 -- fully uniform across all K+1 states
    np.testing.assert_allclose(out[1], [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_mse_to_posterior_favours_lowest_mse_class():
    mse = np.array([
        [0.001, 0.010],   # class 0 reconstructs far better
        [0.010, 0.001],   # class 1 reconstructs far better
    ])
    out = mse_to_posterior(mse, temperature=0.001)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    assert out[0, 1] > out[0, 2]   # row 0 favours class 0 (column 1)
    assert out[1, 2] > out[1, 1]   # row 1 favours class 1 (column 2)


def test_mse_to_posterior_temperature_controls_sharpness():
    """Higher T flattens the posterior toward uniform; lower T sharpens it
    toward the hard argmin -- exactly the real-data behaviour the default
    (0.001) was chosen from."""
    mse = np.array([[0.001, 0.003]])
    sharp = mse_to_posterior(mse, temperature=0.0001)
    flat = mse_to_posterior(mse, temperature=10.0)
    assert sharp[0].max() > flat[0].max()
    # At a very large T, the posterior should be close to uniform (1/3 each).
    np.testing.assert_allclose(flat[0], [1 / 3, 1 / 3, 1 / 3], atol=1e-3)


def test_mse_to_posterior_numerically_stable_on_large_mse():
    """The max-subtraction before exp must prevent overflow even when MSE
    values are large relative to a small temperature."""
    mse = np.array([[100.0, 200.0, 300.0]])
    out = mse_to_posterior(mse, temperature=0.001)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_class_probs_to_posterior_is_a_pure_K1_lift():
    class_probs = np.array([[0.2, 0.3, 0.5]])
    out = class_probs_to_posterior(class_probs)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
    # non-diagnostic mass = 1 - 0.5 = 0.5 before renormalisation
    raw = np.array([0.5, 0.2, 0.3, 0.5])
    np.testing.assert_allclose(out[0], raw / raw.sum(), atol=1e-6)


# ---------------------------------------------------------------------------
# Deterministic fake encoder/decoder: z_mean == X (an "identity encoder"),
# true class = sign of the first feature, decoder reconstructs exactly when
# the requested class-condition matches that true class and badly otherwise.
# This makes every downstream number hand-checkable without any real CVAE.
# ---------------------------------------------------------------------------

class FakeEncoder:
    def __init__(self, num_classes=2, head_confidence=0.8):
        self.num_classes = num_classes
        self.head_confidence = head_confidence

    def predict(self, X_exp, batch_size=256, verbose=0):
        n = X_exp.shape[0]
        z_mean = X_exp.reshape(n, -1).astype(np.float32)
        z_log_var = np.zeros_like(z_mean)
        z = z_mean.copy()
        true_class = (z_mean[:, 0] >= 0).astype(int)
        class_pred = np.full((n, self.num_classes),
                             (1.0 - self.head_confidence) / (self.num_classes - 1),
                             dtype=np.float32)
        class_pred[np.arange(n), true_class] = self.head_confidence
        return z_mean, z_log_var, z, class_pred


class FakeDecoder:
    def predict(self, inputs, batch_size=256, verbose=0):
        z, label = inputs
        requested = np.argmax(label, axis=1)
        true_class = (z[:, 0] >= 0).astype(int)
        match = requested == true_class
        rec = z.copy()
        rec[~match] += 5.0   # badly wrong reconstruction on a class mismatch
        return rec


def _fake_X(n_per_class=6, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    X_pos = rng.uniform(0.1, 1.0, size=(n_per_class, n_features)).astype(np.float32)
    X_neg = rng.uniform(-1.0, -0.1, size=(n_per_class, n_features)).astype(np.float32)
    X = np.concatenate([X_pos, X_neg])
    true_class = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)]).astype(int)
    return X, true_class


# ---------------------------------------------------------------------------
# 2. instance_posteriors_from_cvae, end-to-end against the fakes
# ---------------------------------------------------------------------------

def test_instance_posteriors_from_cvae_reconstruction_recovers_true_class():
    X, true_class = _fake_X(seed=1)
    out = instance_posteriors_from_cvae(
        X, FakeEncoder(), FakeDecoder(), num_classes=2, temperature=0.01)
    assert set(out.keys()) == {'reconstruction', 'encoder_head'}
    recon = out['reconstruction']
    assert recon.shape == (len(X), 3)
    np.testing.assert_allclose(recon.sum(axis=1), 1.0, atol=1e-5)
    # Perfect reconstruction (mse=0) under the true class vs. mse=5**2*n_features
    # under the other -- softmax should assign almost all mass to the true class.
    pred = recon[:, 1:].argmax(axis=1)
    np.testing.assert_array_equal(pred, true_class)


def test_instance_posteriors_from_cvae_encoder_head_matches_fake_confidence():
    X, true_class = _fake_X(seed=2)
    out = instance_posteriors_from_cvae(
        X, FakeEncoder(head_confidence=0.8), FakeDecoder(), num_classes=2, temperature=0.01)
    head = out['encoder_head']
    pred = head[:, 1:].argmax(axis=1)
    np.testing.assert_array_equal(pred, true_class)
    # class_probs_to_posterior's exact non-diagnostic-mass arithmetic for a
    # confident-0.8 row: raw = [0.2, 0.8, 0.2] (or symmetric) -> /1.2
    own_class_mass = head[np.arange(len(X)), 1 + true_class]
    np.testing.assert_allclose(own_class_mass, 0.8 / 1.2, atol=1e-5)


def test_instance_posteriors_from_cvae_shapes_match_num_classes():
    X, _ = _fake_X(n_per_class=3, seed=3)
    out = instance_posteriors_from_cvae(
        X, FakeEncoder(), FakeDecoder(), num_classes=2, temperature=1.0)
    for variant in ('reconstruction', 'encoder_head'):
        assert out[variant].shape == (len(X), 3)
        assert out[variant].dtype == np.float32


# ---------------------------------------------------------------------------
# 3. pipeline/mil_fit_stage.py::_score_one_fold_cbr -- scatter-back + in_sample
# ---------------------------------------------------------------------------

def test_score_one_fold_cbr_scatters_back_and_marks_in_sample(monkeypatch):
    from pipeline import mil_fit_stage

    monkeypatch.setattr(
        'pipeline._cvae_gen.load_encoder_decoder',
        lambda cvae_config, weights_dir, fold_name: (FakeEncoder(), FakeDecoder()))
    monkeypatch.setattr(
        'pipeline._cvae_gen.build_cvae_config',
        lambda config, n_features: {})

    full_bag_ids = np.array(['train_a', 'train_b', 'val_c', 'val_d'], dtype=object)
    fold_data = {
        'fold_name': 'fold_0',
        'train': (np.array([[1.0, 0.5, 0.2], [1.0, 0.5, 0.2]], dtype=np.float32), np.array([2, 2])),
        'val': (np.array([[-1.0, -0.5, -0.2]], dtype=np.float32), np.array([1])),
        'train_groups': np.array(['train_a', 'train_b'], dtype=object),
        'val_groups': np.array(['val_c'], dtype=object),
        'train_coords': None, 'val_coords': None,
        'train_sample_ids': np.array(['train_a', 'train_b'], dtype=object),
        'val_sample_ids': np.array(['val_c'], dtype=object),
    }
    config = {'models': {'cnn': {'num_classes': 2}}, 'paths': {}}

    fold_name, variant_results = mil_fit_stage._score_one_fold_cbr(
        fold_data, config, n_classes=2, full_bag_ids=full_bag_ids, full_coords=None,
        cvae_weights_dir='unused', temperature=0.01)

    assert fold_name == 'fold_0'
    variants = {v for v, _, _ in variant_results}
    assert variants == {'reconstruction', 'encoder_head'}

    for variant, posterior_full, in_sample_full in variant_results:
        assert posterior_full.shape == (4, 3)
        # train_a/train_b in-sample (True); val_c out-of-sample (False);
        # val_d never appeared in this fold at all -> NaN, untouched.
        np.testing.assert_array_equal(in_sample_full, [True, True, False, False])
        assert np.isnan(posterior_full[3]).all(), "val_d (not in this fold) must stay NaN"
        assert not np.isnan(posterior_full[:3]).any()
        np.testing.assert_allclose(posterior_full[:3].sum(axis=1), 1.0, atol=1e-5)


def test_score_one_fold_cbr_not_in_default_methods_but_in_all_methods():
    """METHODS (mi-SVM) is the default run_suite.py relies on;
    'cbr' is available but must be requested explicitly."""
    from pipeline.mil_fit_stage import ALL_METHODS, METHODS
    assert 'cbr' not in METHODS
    assert METHODS == ('misvm',)
    assert 'cbr' in ALL_METHODS


# ---------------------------------------------------------------------------
# 4. scripts/bag_level_eval.py's circularity guard: a method that hard-restricts every instance to {nulls, own-bag-class}
# throughout scoring makes K-diagnostic-column bag classification trivially,
# vacuously correct. This must stay caught, not silently regress.
# ---------------------------------------------------------------------------

def test_diagnostic_column_competition_frac_detects_hard_masked_posterior():
    """A hard-masked Model (C) shape: every instance has exactly one nonzero diagnostic
    column (its own bag's known class) -- zero real competition."""
    from scripts.bag_level_eval import diagnostic_column_competition_frac

    fold_data = [{
        'proba': np.array([
            [1.0, 0.0],    # bag class 0 -> mass only on column 0
            [0.9999, 1e-8],
            [0.0, 1.0],    # bag class 1 -> mass only on column 1
        ]),
        'y_true': np.array([0, 0, 1]),
        'bag_ids': np.array(['a', 'a', 'b']),
    }]
    frac = diagnostic_column_competition_frac(fold_data)
    assert frac == pytest.approx(0.0, abs=1e-6)


def test_diagnostic_column_competition_frac_detects_real_competition():
    """mi-SVM / Tier 4's shape: independent per-class scores, both columns
    routinely nonzero -- genuine competition."""
    from scripts.bag_level_eval import diagnostic_column_competition_frac

    fold_data = [{
        'proba': np.array([[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]),
        'y_true': np.array([0, 1, 0]),
        'bag_ids': np.array(['a', 'b', 'c']),
    }]
    frac = diagnostic_column_competition_frac(fold_data)
    assert frac == pytest.approx(1.0, abs=1e-6)


def test_diagnostic_column_competition_frac_handles_empty_input():
    from scripts.bag_level_eval import diagnostic_column_competition_frac
    assert np.isnan(diagnostic_column_competition_frac([]))
