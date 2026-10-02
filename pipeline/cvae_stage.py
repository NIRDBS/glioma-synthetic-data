"""
Stage: CVAE training + synthetic generation.

Ports the canonical manual training loop from train_CVAE.py / notebook 03:
config-driven CVAE, dynamic per-epoch patient capping, on-the-fly augmentation,
optional beta/alpha annealing, then generation of a balanced synthetic pool
(samples_per_class per class) saved as fold_{name}_X_synth.npy / _y_synth.npy.

Local by default. With backend='modal' the heavy training is offloaded to a GPU
on Modal and the resulting weights + synthetic pool are downloaded back into the
local results directory, so downstream stages run identically.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline._cvae_gen import build_cvae_config, generate_synthetic


def _make_epoch_dataset(X_epoch, y_epoch, aug_cfg, num_classes, batch_size):
    """
    Build a TF dataset for one epoch using TF-native augmentation ops.

    Shuffling is done in numpy BEFORE creating the dataset so TF never needs to
    fill a large shuffle buffer (which was the dominant bottleneck: 15-25 s/epoch
    just to materialise 22k elements into the buffer).  Augmentation:
    - Within-class mixup: partners pre-computed in numpy, blending in TF graph.
    - Spectral shift: one random integer shift PER SAMPLE (docs/MIL_REFRAME_PLAN.md
      §7.2's named defect was a single shared shift for the whole batch -- fixed
      here via a vectorised per-row tf.gather, not a Python-level loop).
    - Baseline wobble: a vectorised TF-native stand-in for
      src/data/augmentation.py's baseline_wobble (smooth random perturbation,
      amplitude scaled to each sample's own dynamic range) -- "the most valuable"
      of §7.2's suggested augmentations, motivated by the Stage 0 baseline study.
      Not literally src/data/augmentation.py's cubic-spline-through-control-points
      version: that is a per-row scipy.interp1d call, and at ~22k rows x 70
      epochs x 5 folds (~7.7M calls) it would materially slow CVAE training,
      the single most expensive stage in the suite. This achieves the same
      qualitative effect (smooth, random, amplitude-scaled) at batch-vectorised
      cost via a couple of random-amplitude/phase low-frequency sinusoids.
    """
    import tensorflow as tf

    # Shuffle in numpy — free, avoids TF shuffle buffer fill cost entirely
    perm = np.random.permutation(len(X_epoch))
    X    = X_epoch[perm].astype('float32')
    y_s  = y_epoch[perm]
    y_oh = tf.keras.utils.to_categorical(y_s, num_classes=num_classes).astype('float32')

    enabled = aug_cfg.get('enabled', False)
    if not enabled:
        return (tf.data.Dataset
                .from_tensor_slices((X, y_oh))
                .batch(batch_size, drop_remainder=True)
                .map(lambda x, y: (tf.expand_dims(x, -1), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))

    mixup_alpha = float(aug_cfg.get('mixup_alpha', 0.5))
    max_shift   = int(aug_cfg.get('max_shift', 3))
    wobble_cfg  = aug_cfg.get('baseline_wobble', {'enabled': True})
    wobble_enabled = bool(wobble_cfg.get('enabled', True))
    wobble_amplitude = float(wobble_cfg.get('max_amplitude', 0.05))

    # Pre-compute within-class mixup partners (numpy, once per epoch, O(n))
    partners = np.empty_like(X)
    for c in np.unique(y_s):
        idx_c = np.where(y_s == c)[0]
        p     = np.random.permutation(len(idx_c))
        partners[idx_c] = X[idx_c[p]]

    n_features = X.shape[1]

    @tf.function
    def _augment(x, partner, y):
        batch = tf.shape(x)[0]

        # 1. Within-class mixup
        lam = tf.random.uniform((batch, 1), 0.0, mixup_alpha)
        x   = lam * partner + (1.0 - lam) * x

        # 2. Spectral shift, per sample (see docstring above)
        shifts = tf.random.uniform((batch,), -max_shift, max_shift + 1, dtype=tf.int32)
        pos = tf.range(n_features)[tf.newaxis, :] - shifts[:, tf.newaxis]
        idx = tf.math.floormod(pos, n_features)
        x = tf.gather(x, idx, batch_dims=1)

        # 3. Baseline wobble (see docstring above)
        if wobble_enabled:
            pos_f = tf.cast(tf.range(n_features), tf.float32)[tf.newaxis, :] / tf.cast(n_features, tf.float32)
            amp = wobble_amplitude * (
                tf.reduce_max(x, axis=1, keepdims=True) - tf.reduce_min(x, axis=1, keepdims=True) + 1e-8)
            freq1 = tf.random.uniform((batch, 1), 1.0, 3.0)
            freq2 = tf.random.uniform((batch, 1), 3.0, 6.0)
            phase1 = tf.random.uniform((batch, 1), 0.0, 2 * np.pi)
            phase2 = tf.random.uniform((batch, 1), 0.0, 2 * np.pi)
            a1 = tf.random.uniform((batch, 1), 0.3, 1.0)
            wobble = amp * (a1 * tf.sin(2 * np.pi * freq1 * pos_f + phase1)
                            + (1.0 - a1) * tf.sin(2 * np.pi * freq2 * pos_f + phase2))
            x = x + wobble

        return tf.expand_dims(x, -1), y

    return (tf.data.Dataset
            .from_tensor_slices((X, partners.astype('float32'), y_oh))
            .batch(batch_size, drop_remainder=True)
            .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def _train_one_fold(fold_data, config, cvae_config, epochs, batch_size, samples_per_class,
                    learning_rate, num_classes, model_out_dir, synthetic_out_dir,
                    resume_from_dir=None, epoch_offset=0):
    """
    Train the CVAE on one fold and generate + save its synthetic pool.

    ``resume_from_dir``: if given, loads that directory's
    ``encoder_{fold_name}.weights.h5``/``decoder_{fold_name}.weights.h5`` into
    the freshly-built CVAE BEFORE training starts, continuing from wherever
    that checkpoint left off rather than from random initialisation. The
    optimizer (Adam moments) is NOT restored -- only compile() is called
    after, so it starts fresh; this is the standard, usually-acceptable
    caveat of weight-only resume (a brief re-warm-up, not a correctness
    issue). ``epochs`` is the number of NEW epochs to run from that
    checkpoint (e.g. 30, not 100); ``epoch_offset`` (e.g. 70) is used only
    for the printed/logged epoch numbers, so the history CSV correctly shows
    epochs 71-100 rather than restarting the label at 1. None/0 (the
    defaults) reproduce prior behaviour exactly -- fresh random init, no
    resume, no offset.
    """
    import tensorflow as tf
    from src.models.cvae import CVAE

    fold_name = fold_data['fold_name']

    out_x = Path(synthetic_out_dir) / f"{fold_name}_X_synth.npy"
    out_y = Path(synthetic_out_dir) / f"{fold_name}_y_synth.npy"
    dec_path = Path(model_out_dir) / f"decoder_{fold_name}.weights.h5"
    if out_x.exists() and out_y.exists() and dec_path.exists():
        print(f"  [skip] {fold_name}: synthetic + decoder weights already exist.")
        return

    print(f"\n{'='*60}\n Processing {fold_name}\n{'='*60}")

    # 1. Unpack + label prep
    X_train, y_train = fold_data['train']
    train_groups = fold_data['train_groups']
    if np.min(y_train) == 1:
        y_train = y_train - 1
    if num_classes == 2:
        y_train = np.where(y_train < 3, 0, 1)

    X_train_2d = X_train.squeeze() if X_train.ndim == 3 else X_train

    # 2. Build + compile CVAE
    tf.keras.backend.clear_session()
    cvae = CVAE(cvae_config)
    if resume_from_dir is not None:
        enc_path = Path(resume_from_dir) / f"encoder_{fold_name}.weights.h5"
        resume_dec_path = Path(resume_from_dir) / f"decoder_{fold_name}.weights.h5"
        print(f"  [resume] loading {enc_path} / {resume_dec_path} "
              f"(continuing from epoch_offset={epoch_offset})")
        cvae.encoder.load_weights(str(enc_path))
        cvae.decoder.load_weights(str(resume_dec_path))
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    # Annealing config (exp_201 has none -> constant weights)
    cvae_cfg = config['models']['cvae']
    anneal_cfg = cvae_cfg.get('annealing', {'enabled': False})
    is_annealing = anneal_cfg.get('enabled', False)
    max_beta = cvae_config['loss_weights'].get('beta', 1.0)
    max_alpha = cvae_config['loss_weights'].get('alpha', 0.2)

    # Windowed beta schedule: an explicit, arbitrary (not necessarily
    # monotonic) per-window beta value -- e.g. an alternating
    # [1.25,1,1.25,1,...] pattern, which the start/warmup linear-ramp
    # annealing above has no way to express. Takes precedence over
    # ``annealing`` if both are present (not expected to co-occur in
    # practice). Windows are relative to THIS run's own local epoch loop
    # (0-indexed from 0, not abs_epoch) -- i.e. "window 1" is this run's
    # own epochs 1-10, matching how the schedule was specified, regardless
    # of epoch_offset (which only affects history/annealing labelling).
    beta_window_schedule = cvae_cfg.get('beta_window_schedule')
    if beta_window_schedule:
        window_size = int(beta_window_schedule['window_size'])
        window_values = [float(v) for v in beta_window_schedule['values']]

    aug_cfg = config.get('augmentation', {})
    patient_cap = config.get('cv_strategy', {}).get('patient_cap', 500)

    print(f" Training CVAE on {len(X_train_2d)} real samples "
          f"(epochs={epochs}, batch={batch_size}, cap={patient_cap}/epoch)")

    fold_history = []
    for epoch in range(epochs):
        abs_epoch = epoch_offset + epoch  # absolute epoch index, so a resumed run's
        # annealing schedule (start_epoch/warmup_epochs, both defined in absolute
        # epoch terms) picks up where it should, not restarted at the local loop's
        # own epoch 0 -- irrelevant for a constant (non-annealed) beta/alpha, where
        # epoch_offset defaults to 0 and this is a no-op.
        # --- weight schedules ---
        if beta_window_schedule:
            window_idx = min(epoch // window_size, len(window_values) - 1)
            cvae.beta = window_values[window_idx]
            cvae.alpha = max_alpha  # this schedule type only varies beta
        elif is_annealing:
            beta_cfg = anneal_cfg.get('beta', {})
            start_beta_ep = beta_cfg.get('start_epoch', 10)
            warmup_beta = beta_cfg.get('warmup_epochs', 80)
            cvae.beta = 0.0 if abs_epoch < start_beta_ep else min(
                max_beta, max_beta * ((abs_epoch - start_beta_ep) / max(warmup_beta, 1)))

            alpha_cfg = anneal_cfg.get('alpha', {})
            start_alpha_ep = alpha_cfg.get('start_epoch', 90)
            cooldown_alpha = alpha_cfg.get('warmup_epochs', 60)
            if abs_epoch < start_alpha_ep:
                cvae.alpha = max_alpha
            else:
                decay = 1.0 - ((abs_epoch - start_alpha_ep) / max(cooldown_alpha, 1))
                cvae.alpha = max(0.0, max_alpha * decay)
        else:
            cvae.beta = max_beta
            cvae.alpha = max_alpha

        # --- dynamic per-epoch patient cap (fresh subsample each epoch) ---
        keep_indices = []
        for group_id in np.unique(train_groups):
            indices = np.where(train_groups == group_id)[0]
            if len(indices) > patient_cap:
                keep_indices.extend(np.random.choice(indices, size=patient_cap, replace=False))
            else:
                keep_indices.extend(indices)
        keep_indices = np.sort(np.array(keep_indices))
        X_epoch = X_train_2d[keep_indices]
        y_epoch = y_train[keep_indices]

        ds = _make_epoch_dataset(X_epoch, y_epoch, aug_cfg, num_classes, batch_size)

        # model.fit() compiles train_step via make_train_function() into a single
        # @tf.function — one optimised kernel graph per epoch instead of ~100
        # eager CUDA kernel launches × 82 batches (which is what the manual loop did).
        h = cvae.fit(ds, epochs=1, verbose=0)
        hist = h.history
        avg_loss = float(hist['total_loss'][0])
        acc = float(hist['accuracy'][0]) * 100
        print(f"  Epoch {epoch_offset+epoch+1}/{epoch_offset+epochs}: loss={avg_loss:.4f} acc={acc:.2f}% "
              f"(beta={cvae.beta:.3f} alpha={cvae.alpha:.2f})")
        fold_history.append({
            'epoch': epoch_offset + epoch + 1, 'total_loss': avg_loss,
            'mse_loss': float(hist.get('mse_loss', [0])[0]),
            'cos_loss': float(hist.get('cos_loss', [0])[0]),
            'kl_loss': float(hist.get('kl_loss', [0])[0]),
            'class_loss': float(hist.get('class_loss', [0])[0]),
            'accuracy': acc, 'beta': float(cvae.beta), 'alpha': float(cvae.alpha),
        })

    # 3. Save weights + history
    Path(model_out_dir).mkdir(parents=True, exist_ok=True)
    cvae.encoder.save_weights(str(Path(model_out_dir) / f"encoder_{fold_name}.weights.h5"))
    cvae.decoder.save_weights(str(dec_path))
    pd.DataFrame(fold_history).to_csv(Path(model_out_dir) / f"history_{fold_name}.csv", index=False)

    # 4. Generate + save balanced synthetic pool
    print(f" Generating {samples_per_class * num_classes} synthetic spectra...")
    X_synth, y_synth = generate_synthetic(
        cvae.decoder, samples_per_class, num_classes, cvae_config['latent_dim'])
    Path(synthetic_out_dir).mkdir(parents=True, exist_ok=True)
    np.save(out_x, X_synth)
    np.save(out_y, y_synth)
    print(f" [saved] {out_x.name} {X_synth.shape}, decoder weights, history")

    del cvae, X_synth, y_synth
    tf.keras.backend.clear_session()
    gc.collect()


def run_local(config, epochs=None, permutation_seed=None, model_out_dir=None,
             synthetic_out_dir=None, resume_from_dir=None, epoch_offset=0):
    """
    Train the CVAE locally for every fold and generate synthetic pools.

    ``permutation_seed``: if given, substitutes the loader's own y_class with
    an exact permutation of the true label multiset (src/mil/shuffle_control.py
    -- the same technique already used for Tiers 1-3's shuffled-bag-label
    controls, docs/MIL_REFRAME_PLAN.md backlog §13) BEFORE any fold splitting,
    then trains completely unchanged from there. This is what Tier 4's own
    control needs: a full CVAE retrain under permuted labels, since (unlike
    Tiers 1-3) Tier 4 conditions directly on the CVAE's own weights rather
    than on a separate, frozen-trunk-based model.

    ``model_out_dir``/``synthetic_out_dir``: override the config's own output
    paths -- unused by the real training run (unchanged behaviour), used by a
    shuffled-label run so it can never overwrite the real CVAE weights that
    Tier 3's frozen trunk and Tier 4's CbR both depend on.

    ``resume_from_dir``/``epoch_offset``: see _train_one_fold's docstring --
    load an existing per-fold checkpoint and continue training ``epochs``
    MORE epochs from there (not retrain ``epochs`` from scratch), with
    ``epoch_offset`` only affecting the epoch numbers written to history.
    """
    import tensorflow as tf
    from pipeline._common import make_loader, num_classes_of, set_seeds

    set_seeds(config)
    print(f"\n{'='*80}\nCVAE STAGE (local): {config['name']}\n{'='*80}")
    print(f"TensorFlow {tf.__version__} | devices: {tf.config.list_physical_devices()}")

    loader = make_loader(config)
    if permutation_seed is not None:
        from src.mil.shuffle_control import apply_permuted_labels, permuted_bag_label_map
        label_map = permuted_bag_label_map(loader.groups, loader.y_class, seed=permutation_seed)
        n_moved = sum(1 for b in label_map
                     if label_map[b] != loader.y_class[loader.groups == b][0])
        print(f"  [shuffled-label control] permutation_seed={permutation_seed}  "
              f"{n_moved}/{len(label_map)} bags received a label different from their true one")
        loader.y_class = apply_permuted_labels(loader.groups, label_map)
    n_features = loader.X.shape[1]
    num_classes = num_classes_of(config)

    cvae_cfg = config['models']['cvae']
    cvae_config = build_cvae_config(config, n_features)
    n_epochs = int(epochs) if epochs is not None else int(cvae_cfg['epochs'])
    batch_size = int(cvae_cfg.get('batch_size', 256))
    samples_per_class = int(cvae_cfg['samples_per_class'])
    learning_rate = float(cvae_cfg['learning_rate'])

    model_out_dir = model_out_dir or config['paths']['models_cvae_dir']
    synthetic_out_dir = synthetic_out_dir or config['paths']['synthetic_dir']

    for fold_data in loader:
        _train_one_fold(
            fold_data, config, cvae_config, n_epochs, batch_size, samples_per_class,
            learning_rate, num_classes, model_out_dir, synthetic_out_dir,
            resume_from_dir=resume_from_dir, epoch_offset=epoch_offset,
        )

    print(f"\n CVAE stage complete. Weights: {model_out_dir} | Synthetic: {synthetic_out_dir}")


def cvae_stage(config, output_dir=None, backend='local', epochs=None):
    """
    Run the CVAE stage.

    backend='local' (default): train on this machine.
    backend='modal': offload training to a Modal GPU, then download the weights +
                     synthetic pool back into the local results dirs.
    """
    if backend == 'modal':
        from pipeline.modal_cvae import run_on_modal
        return run_on_modal(config['name'], epochs=epochs)
    return run_local(config, epochs=epochs)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="CVAE training + synthetic generation")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_201")
    parser.add_argument('--backend', default='local', choices=['local', 'modal'])
    parser.add_argument('--epochs', type=int, default=None, help="override epochs (smoke tests)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    cvae_stage(cfg, out_dir, backend=args.backend, epochs=args.epochs)
