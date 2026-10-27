"""
Shared helpers for the unified pipeline stages.

These functions factor out the canonical logic used by notebook 02
(02_Train_Classifiers.ipynb): config-driven loading, per-fold preparation
(0-indexing fix, binary mapping, train-only patient cap), and the 4-model
benchmark (Apollo / RandomForest / XGBoost / 1D-CNN). The stage modules
(trtr/tstr/tsrtr) all delegate here so the training behaviour stays identical
across regimens.
"""

import gc
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def set_seeds(config):
    """Set numpy/random (and TF later) seeds from config['training']['random_seed']."""
    seed = config.get('training', {}).get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    return seed


def num_classes_of(config):
    """The classification task size lives in models.cnn.num_classes (2 or 6)."""
    return config['models']['cnn']['num_classes']


def make_loader(config):
    """Instantiate the config-driven loader (applies crop + renormalisation)."""
    from src.data.loader import GliomaDataLoader
    return GliomaDataLoader.from_config(
        h5_data_path=config['paths']['processed_file'],
        config=config,
        memory_mapped=False,
    )


@dataclass
class FoldArrays:
    """Return type of :func:`prepare_fold`.

    Iterates as the historical 4-tuple ``(X_train, y_train, X_val, y_val)`` so every
    existing call site's ``X_train, y_train, X_val, y_val = prepare_fold(...)`` keeps
    working unchanged. New code (Stage 4+ MIL fitting) reaches the per-instance
    coords/bag-ids by attribute instead of threading more positional returns:
    ``fd = prepare_fold(...); fd.train_coords, fd.train_bag_ids``.

    Coords/bag-ids are ``None`` when the source HDF5 predates Stage 1 (no
    ``coords`` dataset) -- callers that need them should check, not assume.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    train_coords: Optional[np.ndarray] = None
    val_coords: Optional[np.ndarray] = None
    train_bag_ids: Optional[np.ndarray] = None
    val_bag_ids: Optional[np.ndarray] = None

    def __iter__(self):
        return iter((self.X_train, self.y_train, self.X_val, self.y_val))


def prepare_fold(fold_data, config, cap_train=True):
    """
    Prepare one CV fold exactly as notebook 02 does.

    Order of operations (must match the notebook):
      1. 0-indexing fix (labels stored 1..6 -> 0..5)
      2. Binary mapping when num_classes == 2  (y<3 -> 0 Mutant, else 1 Wildtype)
      3. Train-only per-slide cap (validation is never capped). This is a single
         FIXED subsample drawn once per fold. (The CVAE stage instead re-samples
         the cap fresh every epoch — see pipeline/cvae_stage.py. The config key
         ``cap_strategy`` documents that CVAE behaviour and is not read here.)

    Returns a :class:`FoldArrays`. The train arrays are the capped REAL training
    data; regimen-specific stages decide what to actually train on (real,
    synthetic, or a mix). Coords/bag-ids (when the loader supplied them) are
    subset by the exact same index arrays as X/y at every step, so
    ``fd.train_coords[i]`` always describes ``fd.X_train[i]``.
    """
    nc = num_classes_of(config)

    X_val = fold_data['val'][0].astype(np.float32)
    y_val = fold_data['val'][1].copy()
    X_train = fold_data['train'][0].astype(np.float32)
    y_train = fold_data['train'][1].copy()
    train_coords = fold_data.get('train_coords')
    val_coords = fold_data.get('val_coords')
    train_bag_ids = fold_data.get('train_groups')
    val_bag_ids = fold_data.get('val_groups')

    # 1. 0-index fix
    if y_train.size and np.min(y_train) == 1:
        y_train = y_train - 1
    if y_val.size and np.min(y_val) == 1:
        y_val = y_val - 1

    # 2. Binary mapping (before the cap, so the cap sees binary classes)
    if nc == 2:
        y_train = np.where(y_train < 3, 0, 1)
        y_val = np.where(y_val < 3, 0, 1)

    # 3. Train-only patient cap
    if cap_train:
        patient_cap = config.get('cv_strategy', {}).get('patient_cap')
        if patient_cap:
            groups = fold_data['train_groups']
            keep = []
            for slide_id in np.unique(groups):
                idx = np.where(groups == slide_id)[0]
                keep.extend(
                    np.random.choice(idx, size=patient_cap, replace=False)
                    if len(idx) > patient_cap else idx
                )
            keep = np.sort(np.array(keep))
            X_train = X_train[keep]
            y_train = y_train[keep]
            if train_coords is not None:
                train_coords = train_coords[keep]
            if train_bag_ids is not None:
                train_bag_ids = train_bag_ids[keep]
            print(f"  Patient cap {patient_cap}/slide -> {len(keep)} train spectra")

    return FoldArrays(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        train_coords=train_coords, val_coords=val_coords,
        train_bag_ids=train_bag_ids, val_bag_ids=val_bag_ids,
    )


def _inner_validation_split(y, bag_ids=None, val_fraction=0.15, seed=42):
    """
    Carve an inner train/val split out of a fold's TRAINING data only, for the
    CNN's early-stopping monitor -- never the true held-out CV test fold
    (docs/MIL_REFRAME_PLAN.md §7.2: "early-stops on the test fold itself... not
    merely optimism -- restore_best_weights selects on held-out data").

    Group-aware when ``bag_ids`` identifies real slides (whole slides go
    entirely to one side, so the leakage this project's whole CV design exists
    to prevent doesn't creep back in at this smaller scale). Rows with no group
    (``bag_ids`` entry is ``None`` -- synthetic instances have no slide to leak
    across) get a plain per-row stratified split instead; TSTR's purely
    synthetic training set (``bag_ids=None`` entirely) falls through to a plain
    split for every row.

    Returns (train_idx, val_idx), positional indices into ``y`` (and whatever
    else the caller subsets the same way).
    """
    n = len(y)
    idx = np.arange(n)

    if bag_ids is None:
        return train_test_split(idx, test_size=val_fraction, stratify=y, random_state=seed)

    bag_ids = np.asarray(bag_ids, dtype=object)
    has_group = np.array([b is not None for b in bag_ids])
    grouped_idx = idx[has_group]
    ungrouped_idx = idx[~has_group]

    train_parts, val_parts = [], []
    if len(grouped_idx) > 0:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        g_train, g_val = next(gss.split(grouped_idx, y[grouped_idx], groups=bag_ids[grouped_idx]))
        train_parts.append(grouped_idx[g_train])
        val_parts.append(grouped_idx[g_val])
    if len(ungrouped_idx) > 0:
        u_y = y[ungrouped_idx]
        stratify = u_y if len(np.unique(u_y)) > 1 else None
        u_train, u_val = train_test_split(
            ungrouped_idx, test_size=val_fraction, stratify=stratify, random_state=seed)
        train_parts.append(u_train)
        val_parts.append(u_val)

    return np.concatenate(train_parts), np.concatenate(val_parts)


def train_four_models(X_train, y_train, X_val, y_val, config, model_dir, fold_name, seed=42,
                      train_bag_ids=None):
    """
    Train the 4 benchmark models on (X_train, y_train), evaluate on (X_val, y_val),
    and save each to ``model_dir`` using the canonical filename scheme:
        apollo_{fold}.joblib, rf_{fold}.joblib, xgb_{fold}.json, cnn_{fold}.keras

    Returns a dict {model_name: val_accuracy}. TensorFlow is imported lazily inside
    the CNN block (matches the notebook; avoids Windows DLL load issues up front).

    ``train_bag_ids`` (optional): per-row slide id for ``X_train`` (``None`` for
    rows with no real slide, e.g. synthetic instances). Used only by the CNN's
    inner validation split (see ``_inner_validation_split``); the other three
    models are unaffected.
    """
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from src.models.baselines import ApolloPipeline

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Defensive 0-index fix for synthetic / mixed labels
    if y_train.size and np.min(y_train) == 1:
        y_train = y_train - 1
    if y_val.size and np.min(y_val) == 1:
        y_val = y_val - 1

    apollo_cfg = config['models']['apollo']
    rf_cfg = config['models']['random_forest']
    xgb_cfg = config['models']['xgboost']
    cnn_cfg = config['models']['cnn']

    accs = {}
    print(f"  Train: {X_train.shape[0]} spectra, Val: {X_val.shape[0]} spectra")

    # --- MODEL 1: Apollo (RF feature selection + SVM) ---
    print(f"  [1/4] Apollo (n_features={apollo_cfg['n_features']})...")
    apollo = ApolloPipeline(n_features=apollo_cfg['n_features'], random_state=apollo_cfg['random_state'])
    apollo.fit(X_train, y_train)
    accs['Apollo'] = float(accuracy_score(y_val, apollo.predict(X_val)))
    joblib.dump(apollo, model_dir / f"apollo_{fold_name}.joblib")
    print(f"        Val Acc: {accs['Apollo']:.4f}")
    del apollo
    gc.collect()

    # --- MODEL 2: Random Forest ---
    print(f"  [2/4] RandomForest (n_est={rf_cfg['n_estimators']})...")
    rf = RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        class_weight=rf_cfg['class_weight'],
        random_state=rf_cfg['random_state'],
        n_jobs=rf_cfg['n_jobs'],
    )
    rf.fit(X_train, y_train)
    accs['RF_Standard'] = float(accuracy_score(y_val, rf.predict(X_val)))
    joblib.dump(rf, model_dir / f"rf_{fold_name}.joblib")
    print(f"        Val Acc: {accs['RF_Standard']:.4f}")
    del rf
    gc.collect()

    # --- MODEL 3: XGBoost ---
    print(f"  [3/4] XGBoost...")
    xgb_model = XGBClassifier(
        objective=xgb_cfg['objective'],
        num_class=xgb_cfg.get('num_class', None),
        random_state=xgb_cfg['random_state'],
        n_jobs=xgb_cfg['n_jobs'],
        tree_method=xgb_cfg['tree_method'],
    )
    # XGBClassifier has no class_weight param (unlike RF/SVM above) -- the
    # inverse-frequency weighting still needs to happen, just via sample_weight.
    xgb_sample_weight = compute_sample_weight('balanced', y_train)
    xgb_model.fit(X_train, y_train, sample_weight=xgb_sample_weight)
    accs['XGBoost'] = float(accuracy_score(y_val, xgb_model.predict(X_val)))
    xgb_model.save_model(str(model_dir / f"xgb_{fold_name}.json"))
    print(f"        Val Acc: {accs['XGBoost']:.4f}")
    del xgb_model
    gc.collect()

    # --- MODEL 4: 1D-CNN ---
    # GLIOMA_DEFER_CNN=1: skip the CNN here because it is trained separately and
    # concurrently across folds (scripts/train_cnn_modal_parallel.py). The config
    # still says backend=modal; the deferral is a per-run scheduling choice.
    if os.environ.get('GLIOMA_DEFER_CNN') == '1':
        print("  [4/4] 1D-CNN deferred (GLIOMA_DEFER_CNN=1; trained by scripts/train_cnn_modal_parallel.py)")
        accs['1D_CNN'] = float('nan')
        return accs
    cnn_backend = cnn_cfg.get('backend', 'local')
    print(f"  [4/4] 1D-CNN (epochs={cnn_cfg['epochs']}, backend={cnn_backend})...")
    try:
        if cnn_backend == 'modal':
            from pipeline.modal_cnn import run_cnn_on_modal
            aug_cfg = dict(config.get('augmentation', {}))
            acc = run_cnn_on_modal(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg,
                                   model_dir, fold_name, seed, train_bag_ids)
        else:
            acc = train_cnn_only(X_train, y_train, X_val, y_val, cnn_cfg,
                                 dict(config.get('augmentation', {})),
                                 model_dir, fold_name, seed, train_bag_ids)
        accs['1D_CNN'] = acc
    except Exception as e:
        print(f"        CNN training failed: {str(e)[:120]}")
        accs['1D_CNN'] = float('nan')

    return accs


def train_cnn_only(X_train, y_train, X_val, y_val, cnn_cfg, aug_cfg, model_dir, fold_name,
                   seed=42, train_bag_ids=None, progress_every=None):
    """
    Train just the 1D-CNN (extracted from train_four_models so it can be reused
    unchanged by pipeline/modal_cnn.py's remote worker -- Apollo/RF/XGBoost
    always train locally, only the CNN benefits from a GPU and was the
    dominant per-fold wall-clock cost of the 4-model benchmark locally).

    Saves ``model_dir/cnn_{fold_name}.keras`` and returns the held-out
    (X_val, y_val) accuracy as a float (NaN on failure, swallowed by the
    caller -- unchanged contract from the original inline block).

    ``progress_every`` (None by default -- no change to the original local
    behaviour/log volume): if set to an int, prints a flushed one-line
    heartbeat ("epoch e/N loss=... val_loss=...") every that many epochs.
    pipeline/modal_cnn.py's remote worker passes this, since Keras's own
    ``verbose=0`` (kept here for parity with pre-Modal behaviour) leaves a
    long remote job with zero progress signal in ``modal app logs`` --
    silent-but-working becomes indistinguishable from silently stuck.
    """
    import tensorflow as tf
    from src.models.cnn import build_1d_cnn
    from src.data.augmentation import AugmentedDataGenerator

    model_dir = Path(model_dir)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)

    n_features = X_train.shape[1]
    num_classes = cnn_cfg['num_classes']

    # Inner split for early stopping -- NEVER the true held-out test fold
    # (fold's own X_val/y_val, used only for the final accuracy below).
    inner_val_fraction = cnn_cfg.get('inner_val_fraction', 0.15)
    tr_idx, inner_val_idx = _inner_validation_split(
        y_train, train_bag_ids, inner_val_fraction, seed)
    X_fit, y_fit = X_train[tr_idx], y_train[tr_idx]
    X_inner_val, y_inner_val = X_train[inner_val_idx], y_train[inner_val_idx]

    # Single authoritative compile (dropout_rate/learning_rate/loss all
    # forwarded here, and nowhere else) -- src/models/cnn.py's docstring.
    cnn_model = build_1d_cnn(
        input_shape=(n_features, 1), num_classes=num_classes,
        learning_rate=cnn_cfg.get('learning_rate', 0.001),
        dropout_rate=cnn_cfg.get('dropout_rate', 0.4),
        loss=cnn_cfg['loss'],
    )

    X_inner_val_cnn = X_inner_val.reshape(-1, n_features, 1)
    y_inner_val_oh = tf.keras.utils.to_categorical(y_inner_val, num_classes=num_classes)

    # Inverse-frequency class weighting, computed on the fitting split only
    # (never X_inner_val/X_val -- those must stay the real, unweighted
    # distribution). Keras' class_weight is applied correctly for BOTH
    # branches below, including the augmented tf.data.Dataset one -- Keras 3
    # applies it via a per-batch argmax-on-y map (make_class_weight_map_fn),
    # confirmed directly against the installed version rather than assumed
    # from older TF/Keras docs that (incorrectly, for this version) say
    # class_weight is Dataset-incompatible.
    fit_classes = np.unique(y_fit)
    fit_class_weights = compute_class_weight('balanced', classes=fit_classes, y=y_fit)
    class_weight = dict(zip(fit_classes.tolist(), fit_class_weights.tolist()))

    if aug_cfg.get('enabled', False):
        gen = AugmentedDataGenerator(X_fit, y_fit, aug_cfg)
        # get_tf_dataset(epochs=1) yields exactly one epoch's worth of
        # (shuffled, augmented) batches from a from_generator dataset --
        # .fit(dataset, epochs=N) does NOT reliably re-invoke the
        # underlying Python generator fresh each outer epoch (confirmed:
        # without .repeat(), a real training run hit "input ran out of
        # data" after epoch 1). .repeat() makes the wrap-around explicit
        # and documented instead of relying on .fit()'s own dataset-
        # restart behaviour; steps_per_epoch bounds each epoch so Keras
        # still sees the same amount of data per epoch as the unaugmented
        # path. Each wrap-around still reshuffles/re-augments, since
        # AugmentedDataGenerator draws fresh randomness every time its
        # generator function itself is (re-)invoked.
        steps_per_epoch = int(np.ceil(len(X_fit) / cnn_cfg['batch_size']))
        train_ds = (
            gen.get_tf_dataset(batch_size=cnn_cfg['batch_size'], epochs=1, shuffle=True)
            .repeat()
            .map(lambda x, y: (tf.expand_dims(x, -1), tf.one_hot(y, num_classes)),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )
        fit_data_kwargs = dict(x=train_ds, steps_per_epoch=steps_per_epoch)
    else:
        X_fit_cnn = X_fit.reshape(-1, n_features, 1)
        y_fit_oh = tf.keras.utils.to_categorical(y_fit, num_classes=num_classes)
        fit_data_kwargs = dict(x=X_fit_cnn, y=y_fit_oh, batch_size=cnn_cfg['batch_size'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=cnn_cfg['early_stopping_patience'],
        restore_best_weights=True,
    )
    callbacks = [early_stop]
    if progress_every:
        total_epochs = cnn_cfg['epochs']

        def _heartbeat(epoch, logs):
            if epoch == 0 or (epoch + 1) % progress_every == 0:
                print(f"        epoch {epoch + 1}/{total_epochs}  "
                     f"loss={logs.get('loss'):.4f}  val_loss={logs.get('val_loss'):.4f}",
                     flush=True)
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_heartbeat))

    history = cnn_model.fit(
        **fit_data_kwargs,
        class_weight=class_weight,
        validation_data=(X_inner_val_cnn, y_inner_val_oh),
        epochs=cnn_cfg['epochs'],
        callbacks=callbacks,
        verbose=0,
    )
    X_val_cnn = X_val.reshape(-1, n_features, 1)
    y_pred_cnn = np.argmax(cnn_model.predict(X_val_cnn, verbose=0), axis=1)
    acc = float(accuracy_score(y_val, y_pred_cnn))
    model_dir.mkdir(parents=True, exist_ok=True)
    cnn_model.save(str(model_dir / f"cnn_{fold_name}.keras"))
    print(f"        Val Acc: {acc:.4f}  ({len(history.history['loss'])} epochs, "
          f"inner val={len(inner_val_idx)}/{len(y_train)})")
    del cnn_model, X_inner_val_cnn, X_val_cnn, y_inner_val_oh
    tf.keras.backend.clear_session()
    gc.collect()
    return acc


def print_benchmark_summary(per_fold_accs, title="BENCHMARK COMPLETE"):
    """per_fold_accs: list of {model_name: acc} dicts (one per fold)."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    models = ['Apollo', 'RF_Standard', 'XGBoost', '1D_CNN']
    for name in models:
        vals = [d[name] for d in per_fold_accs if name in d and not np.isnan(d[name])]
        if vals:
            print(f"{name:>15}: {np.mean(vals):.4f} +/- {np.std(vals):.4f} ({len(vals)} folds)")
        else:
            print(f"{name:>15}: SKIPPED")
    print("=" * 60)
