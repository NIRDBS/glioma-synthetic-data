"""
Regression test for pipeline/cvae_stage.py's resume-from-checkpoint
capability (_train_one_fold's resume_from_dir/epoch_offset, added to support
continuing exp_501's stale 70-epoch CVAE to 100 epochs without retraining
from scratch -- Suite 600's CVAE beta-tuning sweep).

Real (tiny, fast) CVAE on synthetic data, not fakes -- the thing under test
is specifically Keras .load_weights() actually restoring a prior training
state, which a mocked encoder/decoder (as used in
tests/test_stage6_generative_mil.py) can't exercise meaningfully.
"""
import numpy as np

from pipeline.cvae_stage import _train_one_fold


def _tiny_cvae_config():
    return {
        'input_shape': (16, 1), 'num_classes': 2, 'latent_dim': 4,
        'loss_weights': {'beta': 1.0, 'alpha': 0.2, 'gamma': 1.0, 'delta': 0.0},
        'encoder': {'filters': [8], 'kernels': [3], 'strides': [2],
                   'use_global_pooling': False, 'use_skip_gap': False},
        'decoder': {'initial_filters': 8, 'use_dense_projection': True,
                   'filters': [8], 'kernels': [3], 'strides': [2]},
    }


def _tiny_fold_data(seed=0, n=40, n_features=16):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    groups = np.array([f'bag_{i}' for i in range(n)], dtype=object)
    return {'fold_name': 'fold_0', 'train': (X, y), 'train_groups': groups}


def _minimal_config():
    return {'models': {'cvae': {}}, 'augmentation': {'enabled': False},
           'cv_strategy': {'patient_cap': 10_000}}


def test_resume_continues_training_rather_than_restarting(tmp_path):
    """Train 2 epochs, save; resume for 2 more epochs from that checkpoint;
    the resumed run's starting loss should be close to where the first run
    left off, not a fresh-random-init-sized loss -- the direct, observable
    signature of a genuine resume vs. an accidental restart.

    Explicitly seeds python/numpy/TF before each sub-run (tf.keras.utils.
    set_random_seed) so the comparison is fully reproducible regardless of
    what other tests ran earlier in the same process and left global random
    state in -- found necessary after this test flaked (and, separately, an
    unrelated pre-existing test also flaked the same way) when run as part
    of the full suite but not in isolation, both traced to the same root
    cause: comparing two differently-seeded runs without pinning global
    state first.
    """
    import tensorflow as tf

    cvae_config = _tiny_cvae_config()
    config = _minimal_config()
    tf.keras.utils.set_random_seed(100)
    fold_data = _tiny_fold_data(seed=1)

    stage1_dir = tmp_path / 'stage1_model'
    stage1_synth_dir = tmp_path / 'stage1_synth'
    _train_one_fold(
        fold_data, config, cvae_config, epochs=2, batch_size=8, samples_per_class=4,
        learning_rate=0.001, num_classes=2,
        model_out_dir=str(stage1_dir), synthetic_out_dir=str(stage1_synth_dir),
    )
    assert (stage1_dir / 'encoder_fold_0.weights.h5').exists()
    assert (stage1_dir / 'decoder_fold_0.weights.h5').exists()

    import pandas as pd
    hist1 = pd.read_csv(stage1_dir / 'history_fold_0.csv')
    assert list(hist1['epoch']) == [1, 2]
    final_loss_stage1 = float(hist1['total_loss'].iloc[-1])

    # A completely fresh run, for comparison -- what a FAILED/accidental-restart
    # resume would look like at its very first epoch. Explicitly re-seeded
    # (a different seed than stage1's) so this is a genuine, reproducible
    # fresh initialisation, not whatever global RNG state stage1's training
    # happened to leave behind.
    tf.keras.utils.set_random_seed(200)
    fresh_dir = tmp_path / 'fresh_model'
    fresh_synth_dir = tmp_path / 'fresh_synth'
    _train_one_fold(
        _tiny_fold_data(seed=2), config, cvae_config, epochs=1, batch_size=8,
        samples_per_class=4, learning_rate=0.001, num_classes=2,
        model_out_dir=str(fresh_dir), synthetic_out_dir=str(fresh_synth_dir),
    )
    fresh_hist = pd.read_csv(fresh_dir / 'history_fold_0.csv')
    fresh_first_epoch_loss = float(fresh_hist['total_loss'].iloc[0])

    # Resume stage: 2 MORE epochs from stage1's checkpoint, epoch_offset=2.
    # Re-seeded back to stage1's own seed for a reproducible comparison.
    tf.keras.utils.set_random_seed(100)
    stage2_dir = tmp_path / 'stage2_model'
    stage2_synth_dir = tmp_path / 'stage2_synth'
    _train_one_fold(
        fold_data, config, cvae_config, epochs=2, batch_size=8, samples_per_class=4,
        learning_rate=0.001, num_classes=2,
        model_out_dir=str(stage2_dir), synthetic_out_dir=str(stage2_synth_dir),
        resume_from_dir=str(stage1_dir), epoch_offset=2,
    )
    hist2 = pd.read_csv(stage2_dir / 'history_fold_0.csv')

    # Epoch numbers continue from the offset, not restart at 1.
    assert list(hist2['epoch']) == [3, 4]

    resumed_first_epoch_loss = float(hist2['total_loss'].iloc[0])

    # The real, decisive check: resuming from a converging-ish checkpoint
    # starts much closer to where training left off than a fresh random
    # initialisation does, on the SAME tiny architecture/data scale.
    assert abs(resumed_first_epoch_loss - final_loss_stage1) < abs(fresh_first_epoch_loss - final_loss_stage1), (
        f"resumed run's first-epoch loss ({resumed_first_epoch_loss:.4f}) should be closer to "
        f"stage1's final loss ({final_loss_stage1:.4f}) than a fresh run's first-epoch loss "
        f"({fresh_first_epoch_loss:.4f}) is -- otherwise this isn't really resuming."
    )


def test_resume_without_resume_from_dir_is_unaffected():
    """epoch_offset/resume_from_dir both default to None/0 -- confirms the
    existing (non-resume) call path's history still starts at epoch 1,
    unchanged by this feature's addition."""
    import pandas as pd
    import tempfile
    from pathlib import Path

    cvae_config = _tiny_cvae_config()
    config = _minimal_config()
    fold_data = _tiny_fold_data(seed=3)

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / 'model'
        synth_dir = Path(tmp) / 'synth'
        _train_one_fold(
            fold_data, config, cvae_config, epochs=2, batch_size=8, samples_per_class=4,
            learning_rate=0.001, num_classes=2,
            model_out_dir=str(model_dir), synthetic_out_dir=str(synth_dir),
        )
        hist = pd.read_csv(model_dir / 'history_fold_0.csv')
        assert list(hist['epoch']) == [1, 2]
