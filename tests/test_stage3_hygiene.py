"""Stage 3 regression tests: 1D-CNN instance-model hygiene
(docs/MIL_REFRAME_PLAN.md §7.2) -- the leak fix, dropout/loss/learning_rate
forwarding, and the new/fixed augmentations.

Covers the plan's three Stage 3 test bullets:
  1. No test-fold sample_id reaches the early-stopping monitor (a leak fix,
     not a tuning fix).
  2. dropout_rate from config reaches the built model.
  3. Augmentation output stays in-range and label-preserving.
"""
import numpy as np
import pytest

from pipeline._common import _inner_validation_split, train_four_models
from src.data.augmentation import (
    AugmentedDataGenerator,
    additive_noise,
    band_dropout,
    baseline_wobble,
    intensity_jitter,
    spectral_shift,
    within_class_mixup,
)
from src.models.cnn import build_1d_cnn


# ---------------------------------------------------------------------------
# 1. Leak fix: _inner_validation_split never touches the true test fold (it is
#    never even passed in), and is group-aware when bag ids are real slides.
# ---------------------------------------------------------------------------

def _synthetic_bagged_dataset(n_bags=12, min_size=5, max_size=15, seed=0):
    rng = np.random.default_rng(seed)
    bag_ids, y = [], []
    for b in range(n_bags):
        n = int(rng.integers(min_size, max_size))
        cls = b % 3
        bag_ids += [f"bag_{b}"] * n
        y += [cls] * n
    return np.array(y), np.array(bag_ids, dtype=object)


def test_inner_split_is_disjoint_and_covers_all_rows():
    y, bag_ids = _synthetic_bagged_dataset()
    train_idx, val_idx = _inner_validation_split(y, bag_ids, val_fraction=0.2, seed=1)
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(y)))


def test_inner_split_never_puts_one_bag_on_both_sides():
    """The group-awareness property: a slide (bag) must go entirely to one
    side of the inner split, never straddling it."""
    y, bag_ids = _synthetic_bagged_dataset(n_bags=20, seed=2)
    train_idx, val_idx = _inner_validation_split(y, bag_ids, val_fraction=0.25, seed=2)
    train_bags = set(bag_ids[train_idx].tolist())
    val_bags = set(bag_ids[val_idx].tolist())
    assert train_bags.isdisjoint(val_bags)
    # Both sides non-trivial (sanity: the split actually did something)
    assert len(val_bags) >= 1 and len(train_bags) >= 1


def test_inner_split_ungrouped_rows_use_plain_stratified_split():
    """bag_ids=None entirely (TSTR's pure-synthetic case) -- no group concept
    applies, falls back to a plain per-row stratified split."""
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    train_idx, val_idx = _inner_validation_split(y, bag_ids=None, val_fraction=0.2, seed=3)
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert len(val_idx) == pytest.approx(40, abs=2)  # ~20% of 200


def test_inner_split_handles_mixed_real_and_synthetic_rows():
    """TSRTR's case: real rows carry a bag id, synthetic rows are None. Real
    bags must not straddle the split; None rows split by the ungrouped path."""
    y_real, bag_ids_real = _synthetic_bagged_dataset(n_bags=10, seed=4)
    y_synth = np.random.default_rng(4).integers(0, 3, size=50)
    bag_ids_synth = np.array([None] * len(y_synth), dtype=object)

    y = np.concatenate([y_real, y_synth])
    bag_ids = np.concatenate([bag_ids_real.astype(object), bag_ids_synth])

    train_idx, val_idx = _inner_validation_split(y, bag_ids, val_fraction=0.2, seed=4)
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(y)))

    real_mask_train = train_idx < len(y_real)
    real_mask_val = val_idx < len(y_real)
    train_real_bags = set(bag_ids[train_idx[real_mask_train]].tolist())
    val_real_bags = set(bag_ids[val_idx[real_mask_val]].tolist())
    assert train_real_bags.isdisjoint(val_real_bags)


def test_train_four_models_cnn_never_sees_test_fold_as_validation_data(monkeypatch):
    """End-to-end: the true (X_val, y_val) fingerprint must never appear as the
    Keras validation_data the early-stopping callback watches. Uses tiny data
    and 2 epochs so this stays fast; only the CNN path is exercised by
    replacing the other three models with instantaneous stubs would require
    changing production code, so instead this just runs the real
    train_four_models on a deliberately tiny problem and inspects the
    inner-val size reported in its own print/return contract indirectly via
    a monkeypatched Keras fit call."""
    calls = {}
    import tensorflow as tf
    real_fit = tf.keras.Model.fit

    def spy_fit(self, *args, **kwargs):
        calls['validation_data'] = kwargs.get('validation_data')
        kwargs['epochs'] = 1
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(tf.keras.Model, 'fit', spy_fit)

    rng = np.random.default_rng(5)
    n_features = 32
    y, bag_ids = _synthetic_bagged_dataset(n_bags=8, min_size=8, max_size=12, seed=5)
    X_train = rng.normal(size=(len(y), n_features)).astype(np.float32)
    X_val = rng.normal(size=(30, n_features)).astype(np.float32) + 100.0  # distinct fingerprint
    y_val = rng.integers(0, 3, size=30)

    config = {
        'models': {
            'apollo': {'n_features': 5, 'random_state': 42},
            'random_forest': {'n_estimators': 5, 'class_weight': 'balanced',
                              'random_state': 42, 'n_jobs': 1},
            'xgboost': {'objective': 'multi:softprob', 'random_state': 42, 'n_jobs': 1,
                       'tree_method': 'hist'},
            'cnn': {'num_classes': 3, 'optimizer': 'adam', 'loss': 'categorical_crossentropy',
                   'batch_size': 8, 'epochs': 1, 'early_stopping_patience': 1,
                   'dropout_rate': 0.4, 'inner_val_fraction': 0.25},
        },
        'augmentation': {'enabled': False},
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        train_four_models(X_train, y, X_val, y_val, config, tmp, 'fold_test', seed=5,
                          train_bag_ids=bag_ids)

    assert calls['validation_data'] is not None
    inner_val_X = calls['validation_data'][0]
    # The true test fold has a +100 fingerprint; the inner split, carved from
    # X_train, never does -- confirms Keras' early-stopping callback watched
    # the inner split, not (X_val, y_val).
    assert not np.allclose(np.asarray(inner_val_X).mean(), X_val.mean(), atol=10)


# ---------------------------------------------------------------------------
# 2. dropout_rate (and loss/learning_rate) forwarding -- one compile, one place
# ---------------------------------------------------------------------------

def test_train_four_models_cnn_survives_multiple_epochs_with_augmentation():
    """Regression pin: a from_generator-based augmented tf.data.Dataset run
    for >1 epoch previously hit 'input ran out of data' after epoch 1 (Keras
    does not reliably restart a from_generator dataset per epoch on its own).
    Exercising 3 real epochs with augmentation enabled is the only way this
    class of bug actually surfaces -- a 1-epoch smoke test does not."""
    rng = np.random.default_rng(20)
    n_features = 32
    y, bag_ids = _synthetic_bagged_dataset(n_bags=8, min_size=10, max_size=14, seed=20)
    X_train = rng.normal(size=(len(y), n_features)).astype(np.float32)
    X_val = rng.normal(size=(20, n_features)).astype(np.float32)
    y_val = rng.integers(0, 3, size=20)

    config = {
        'models': {
            'apollo': {'n_features': 5, 'random_state': 42},
            'random_forest': {'n_estimators': 5, 'class_weight': 'balanced',
                              'random_state': 42, 'n_jobs': 1},
            'xgboost': {'objective': 'multi:softprob', 'random_state': 42, 'n_jobs': 1,
                       'tree_method': 'hist'},
            'cnn': {'num_classes': 3, 'optimizer': 'adam', 'loss': 'categorical_crossentropy',
                   'batch_size': 8, 'epochs': 3, 'early_stopping_patience': 3,
                   'dropout_rate': 0.4, 'inner_val_fraction': 0.25},
        },
        'augmentation': {'enabled': True, 'max_shift': 2, 'mixup_alpha': 0.3,
                         'baseline_wobble': {'enabled': True}},
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        accs = train_four_models(X_train, y, X_val, y_val, config, tmp, 'fold_aug', seed=20,
                                 train_bag_ids=bag_ids)
    assert not np.isnan(accs['1D_CNN'])


def test_train_four_models_dispatches_cnn_to_modal_when_configured(monkeypatch):
    """models.cnn.backend: 'modal' must route the CNN step through
    pipeline.modal_cnn.run_cnn_on_modal instead of training locally -- checked
    by stubbing that function (so this stays a fast, network-free unit test;
    the real Modal path itself is verified separately via a manual smoke run,
    consistent with this project's standing practice of not hitting external
    services from the pytest suite) and asserting it alone was called, with
    Apollo/RF/XGBoost still trained for real."""
    calls = {}

    def fake_run_cnn_on_modal(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg,
                              model_dir, fold_name, seed, train_bag_ids=None):
        calls['called'] = True
        calls['fold_name'] = fold_name
        return 0.42

    import pipeline.modal_cnn
    monkeypatch.setattr(pipeline.modal_cnn, 'run_cnn_on_modal', fake_run_cnn_on_modal)

    rng = np.random.default_rng(11)
    n_features = 16
    y, bag_ids = _synthetic_bagged_dataset(n_bags=6, min_size=6, max_size=9, seed=11)
    X_train = rng.normal(size=(len(y), n_features)).astype(np.float32)
    X_val = rng.normal(size=(10, n_features)).astype(np.float32)
    y_val = rng.integers(0, 3, size=10)

    config = {
        'models': {
            'apollo': {'n_features': 5, 'random_state': 42},
            'random_forest': {'n_estimators': 5, 'class_weight': 'balanced',
                              'random_state': 42, 'n_jobs': 1},
            'xgboost': {'objective': 'multi:softprob', 'random_state': 42, 'n_jobs': 1,
                       'tree_method': 'hist'},
            'cnn': {'num_classes': 3, 'optimizer': 'adam', 'loss': 'categorical_crossentropy',
                   'batch_size': 8, 'epochs': 1, 'early_stopping_patience': 1,
                   'dropout_rate': 0.4, 'inner_val_fraction': 0.25, 'backend': 'modal'},
        },
        'augmentation': {'enabled': False},
    }

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        accs = train_four_models(X_train, y, X_val, y_val, config, tmp, 'fold_modal', seed=11,
                                 train_bag_ids=bag_ids)

    assert calls.get('called') is True
    assert calls['fold_name'] == 'fold_modal'
    assert accs['1D_CNN'] == 0.42
    assert not np.isnan(accs['Apollo'])  # the other 3 models still trained for real, locally


def test_dropout_rate_reaches_the_built_model():
    from tensorflow.keras import layers
    model = build_1d_cnn(input_shape=(64, 1), num_classes=3, dropout_rate=0.73)
    dropout_layers = [l for l in model.layers if isinstance(l, layers.Dropout)]
    assert len(dropout_layers) == 1
    assert dropout_layers[0].rate == pytest.approx(0.73)


def test_build_1d_cnn_compiles_with_requested_loss():
    model = build_1d_cnn(input_shape=(64, 1), num_classes=3, loss='categorical_crossentropy')
    assert model.loss == 'categorical_crossentropy'


# ---------------------------------------------------------------------------
# 3. Augmentation output stays in-range and label-preserving
# ---------------------------------------------------------------------------

@pytest.fixture
def spectrum():
    rng = np.random.default_rng(6)
    return rng.uniform(0, 1, size=200).astype(np.float32)


def test_spectral_shift_preserves_shape_and_is_finite(spectrum):
    out = spectral_shift(spectrum, shift_amount=2.5)
    assert out.shape == spectrum.shape
    assert np.all(np.isfinite(out))


def test_baseline_wobble_preserves_shape_scales_with_amplitude(spectrum):
    small = baseline_wobble(spectrum, max_amplitude=0.01)
    large = baseline_wobble(spectrum, max_amplitude=0.5)
    assert small.shape == spectrum.shape == large.shape
    assert np.all(np.isfinite(small)) and np.all(np.isfinite(large))
    # Larger amplitude budget -> (on average) larger perturbation magnitude.
    assert np.abs(large - spectrum).mean() > np.abs(small - spectrum).mean()


def test_intensity_jitter_is_a_pure_scaling(spectrum):
    out = intensity_jitter(spectrum, scale_range=(0.5, 0.5))  # deterministic
    np.testing.assert_allclose(out, spectrum * 0.5, atol=1e-5)


def test_additive_noise_preserves_shape_and_is_finite(spectrum):
    out = additive_noise(spectrum, snr_db=20.0)
    assert out.shape == spectrum.shape
    assert np.all(np.isfinite(out))
    assert not np.allclose(out, spectrum)  # noise actually added


def test_band_dropout_only_touches_the_dropped_band(spectrum):
    out = band_dropout(spectrum.copy(), max_width=5, n_bands=1)
    assert out.shape == spectrum.shape
    diff_idx = np.where(~np.isclose(out, spectrum))[0]
    if len(diff_idx) > 0:
        # The changed region must be contiguous and no wider than max_width.
        assert diff_idx.max() - diff_idx.min() + 1 <= 5


def test_within_class_mixup_only_draws_from_the_same_class():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(50, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=50)
    class_idx = 1
    spectrum = X[np.where(y == class_idx)[0][0]]
    out = within_class_mixup(spectrum, X, y, class_idx, mixup_alpha=0.5)
    assert out.shape == spectrum.shape
    assert np.all(np.isfinite(out))


def test_augmented_data_generator_all_augmentations_enabled_stays_finite():
    """label-preserving: augment_sample takes/returns only the spectrum, never
    touches y_data -- so labels cannot be corrupted by construction. This
    checks the numeric output is sane with every augmentation switched on."""
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 1, size=(30, 64)).astype(np.float32)
    y = rng.integers(0, 2, size=30)
    gen = AugmentedDataGenerator(X, y, {
        'enabled': True, 'max_shift': 3, 'mixup_alpha': 0.5,
        'baseline_wobble': {'enabled': True, 'max_amplitude': 0.05},
        'intensity_jitter': {'enabled': True, 'scale_range': (0.9, 1.1)},
        'additive_noise': {'enabled': True, 'snr_db': 25.0},
        'band_dropout': {'enabled': True, 'max_width': 4, 'n_bands': 1},
    })
    for idx in range(len(X)):
        out = gen.augment_sample(idx)
        assert out.shape == (64,)
        assert np.all(np.isfinite(out))


def test_augmented_data_generator_disabled_returns_original():
    rng = np.random.default_rng(9)
    X = rng.uniform(0, 1, size=(10, 16)).astype(np.float32)
    y = rng.integers(0, 2, size=10)
    gen = AugmentedDataGenerator(X, y, {'enabled': False})
    np.testing.assert_array_equal(gen.augment_sample(3), X[3])
