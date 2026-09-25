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


def train_four_models(X_train, y_train, X_val, y_val, config, model_dir, fold_name, seed=42):
    """
    Train the 4 benchmark models on (X_train, y_train), evaluate on (X_val, y_val),
    and save each to ``model_dir`` using the canonical filename scheme:
        apollo_{fold}.joblib, rf_{fold}.joblib, xgb_{fold}.json, cnn_{fold}.keras

    Returns a dict {model_name: val_accuracy}. TensorFlow is imported lazily inside
    the CNN block (matches the notebook; avoids Windows DLL load issues up front).
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
    xgb_model.fit(X_train, y_train)
    accs['XGBoost'] = float(accuracy_score(y_val, xgb_model.predict(X_val)))
    xgb_model.save_model(str(model_dir / f"xgb_{fold_name}.json"))
    print(f"        Val Acc: {accs['XGBoost']:.4f}")
    del xgb_model
    gc.collect()

    # --- MODEL 4: 1D-CNN ---
    print(f"  [4/4] 1D-CNN (epochs={cnn_cfg['epochs']})...")
    try:
        import tensorflow as tf
        from src.models.cnn import build_1d_cnn

        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)

        n_features = X_train.shape[1]
        X_train_cnn = X_train.reshape(-1, n_features, 1)
        X_val_cnn = X_val.reshape(-1, n_features, 1)
        y_train_cnn = tf.keras.utils.to_categorical(y_train, num_classes=cnn_cfg['num_classes'])
        y_val_cnn = tf.keras.utils.to_categorical(y_val, num_classes=cnn_cfg['num_classes'])

        cnn_model = build_1d_cnn(input_shape=(n_features, 1), num_classes=cnn_cfg['num_classes'])
        cnn_model.compile(optimizer=cnn_cfg['optimizer'], loss=cnn_cfg['loss'], metrics=['accuracy'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=cnn_cfg['early_stopping_patience'],
            restore_best_weights=True,
        )
        history = cnn_model.fit(
            X_train_cnn, y_train_cnn,
            validation_data=(X_val_cnn, y_val_cnn),
            epochs=cnn_cfg['epochs'],
            batch_size=cnn_cfg['batch_size'],
            callbacks=[early_stop],
            verbose=0,
        )
        y_pred_cnn = np.argmax(cnn_model.predict(X_val_cnn, verbose=0), axis=1)
        accs['1D_CNN'] = float(accuracy_score(y_val, y_pred_cnn))
        cnn_model.save(str(model_dir / f"cnn_{fold_name}.keras"))
        print(f"        Val Acc: {accs['1D_CNN']:.4f}  ({len(history.history['loss'])} epochs)")
        del cnn_model, X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn
        tf.keras.backend.clear_session()
        gc.collect()
    except Exception as e:
        print(f"        CNN training failed: {str(e)[:120]}")
        accs['1D_CNN'] = float('nan')

    return accs


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
