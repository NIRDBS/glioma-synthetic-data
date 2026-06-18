"""
Statistical Evaluation Utilities for TRTR vs TSTR Benchmark

This module provides functions for:
1. Model loading and prediction extraction (sklearn, XGBoost, Keras)
2. Slide-level stratified bootstrap for statistical comparison
3. Classification by Reconstruction (CbR) via CVAE encoder/decoder
4. Confidence interval computation with no data leakage

Key Design Principles:
- Slide-level (not spectrum-level) resampling
- No data leakage: each slide predicted only by fold where they were in validation set
- Fast bootstrap: pre-computes predictions once, resamples slides across iterations
- CVAE integration: supports both standard bootstrap and CbR-specific bootstrap
"""

import os
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import accuracy_score

from src.models.cvae import CVAE


# ============================================================================
# SECTION 1: Helper Functions (Model Loading & Prediction Extraction)
# ============================================================================

def find_model_path(base_dir: str, prefix: str, ext: str, fold_idx: int) -> str:
    """
    Find model file for a specific fold by checking multiple naming conventions.
    
    Args:
        base_dir: Directory containing models
        prefix: Model filename prefix (e.g., 'apollo', 'rf', 'xgb', 'cnn')
        ext: File extension (e.g., '.joblib', '.json', '.keras')
        fold_idx: Cross-validation fold index
    
    Returns:
        str: Full path to model file
    
    Raises:
        FileNotFoundError: If model not found under any naming convention
    """
    candidates = [
        f"{base_dir}/{prefix}_fold_{fold_idx}{ext}",
        f"{base_dir}/{prefix}_fold{fold_idx}{ext}",
        f"{base_dir}/{prefix}_{fold_idx}{ext}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model '{prefix}' fold {fold_idx} not found in {base_dir}")


def load_and_predict(path: str, model_type: str, X: np.ndarray) -> np.ndarray:
    """
    Load a trained model and return class label predictions.
    
    Supports sklearn, XGBoost, and Keras model types with appropriate
    serialization formats (.joblib, .json, .keras respectively).
    
    Args:
        path: Full path to model file
        model_type: One of 'sklearn', 'xgboost', 'keras'
        X: Input features array (n_samples, n_features)
    
    Returns:
        np.ndarray: (n_samples,) array of integer class labels
    
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "sklearn":
        model = joblib.load(path)
        return model.predict(X).astype(int)
    
    elif model_type == "xgboost":
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model.predict(X).astype(int)
    
    elif model_type == "keras":
        X_3d = X.reshape(-1, X.shape[1], 1)
        model = tf.keras.models.load_model(path, compile=False)
        preds = np.argmax(model.predict(X_3d, verbose=0), axis=1)
        tf.keras.backend.clear_session()
        return preds.astype(int)
    
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


def get_fold_specific_predictions(preds_by_fold, patient_ids, patient_to_fold):
    """
    Extract fold-specific predictions with no data leakage.
    
    For each patient, uses ONLY the predictions from the fold where that patient
    was in the validation set (i.e., the fold whose model was trained WITHOUT that patient).
    
    This ensures honest evaluation: no patient is evaluated by a model that trained on them.
    
    Args:
        preds_by_fold: (n_folds, n_samples) array of predictions from each fold's model
        patient_ids: (n_samples,) array of patient IDs
        patient_to_fold: dict mapping patient_id -> fold_idx (patient's validation fold)
    
    Returns:
        np.ndarray: (n_samples,) array where preds[i] comes from the model trained on
                    patient_ids[i]'s validation fold (not their training folds)
    """
    preds = np.zeros(len(patient_ids), dtype=int)
    
    for i, patient_id in enumerate(patient_ids):
        fold_j = patient_to_fold[patient_id]
        preds[i] = preds_by_fold[fold_j][i]
    
    return preds


def stratified_slide_accuracy(preds, y_true, slide_ids):
    """
    Compute accuracy as the mean of per-(slide, class) accuracies — exactly the
    statistic that stratified_slide_bootstrap accumulates in each iteration.

    Using this for the observed point estimate ensures the reported delta and its
    bootstrap CI are derived from the same statistic, so the CI is centred around
    the delta.  Slides contribute equally (not weighted by spectrum count), and
    classes contribute equally (macro-averaged over classes), matching the
    clinical motivation of treating each patient slide as one observation.
    """
    unique_classes = np.unique(y_true)

    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        slide_to_indices.setdefault(sid, []).append(idx)
    for sid in slide_to_indices:
        slide_to_indices[sid] = np.array(slide_to_indices[sid])

    class_to_slides = {
        cls: np.unique(slide_ids[y_true == cls]) for cls in unique_classes
    }

    accs = []
    for cls in unique_classes:
        for slide in class_to_slides[cls]:
            idx = slide_to_indices[slide]
            cls_idx = idx[y_true[idx] == cls]
            if len(cls_idx) > 0:
                accs.append(accuracy_score(y_true[cls_idx], preds[cls_idx]))
    return float(np.mean(accs)) if accs else 0.0


def stratified_slide_bootstrap(preds_A, preds_B, y_true, slide_ids, n_boot=1000, seed=42):
    """
    Fast slide-level stratified bootstrap for comparing two models.
    
    Compares two models (e.g., TRTR vs TSTR) by resampling slides (not spectra)
    within each class and computing bootstrap accuracy distributions.
    
    Speed optimization: Pre-computes slide-to-indices mapping to avoid
    repeated lookups in the bootstrap loop.
    
    Args:
        preds_A: (n_samples,) array of Model A predictions
        preds_B: (n_samples,) array of Model B predictions
        y_true: (n_samples,) array of true labels
        slide_ids: (n_samples,) array of slide IDs
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: Three arrays of shape (n_boot,)
            - acc_A_boot: Bootstrap accuracies for Model A
            - acc_B_boot: Bootstrap accuracies for Model B
            - boot_diffs: Bootstrap differences (Model B - Model A)
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_true)
    
    # 1. PRE-COMPUTE SLIDE INDICES (Speed optimization)
    # Maps 'Slide_ID' -> array of row indices for that slide
    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        if sid not in slide_to_indices:
            slide_to_indices[sid] = []
        slide_to_indices[sid].append(idx)
    
    for sid in slide_to_indices:
        slide_to_indices[sid] = np.array(slide_to_indices[sid])
    
    # 2. Map class -> unique slides in that class
    class_to_slides = {}
    for cls in unique_classes:
        class_to_slides[cls] = np.unique(slide_ids[y_true == cls])
    
    acc_A_boot, acc_B_boot, boot_diffs = [], [], []
    
    # 3. FAST BOOTSTRAP LOOP
    print("Running optimized slide-level bootstrap...")
    for i in range(n_boot):
        boot_acc_A = []
        boot_acc_B = []
        
        for cls in unique_classes:
            slides_in_cls = class_to_slides[cls]
            # Resample slides with replacement
            sampled_slides = rng.choice(slides_in_cls, size=len(slides_in_cls), replace=True)
            
            for slide in sampled_slides:
                s_idx = slide_to_indices[slide]
                # Get indices where slide's samples belong to this class
                cls_idx = s_idx[y_true[s_idx] == cls]
                
                # Calculate accuracy for this slide in this class
                slide_acc_A = accuracy_score(y_true[cls_idx], preds_A[cls_idx])
                slide_acc_B = accuracy_score(y_true[cls_idx], preds_B[cls_idx])
                
                boot_acc_A.append(slide_acc_A)
                boot_acc_B.append(slide_acc_B)
        
        # Average slide accuracies (cluster bootstrap at slide level)
        acc_A = np.mean(boot_acc_A)
        acc_B = np.mean(boot_acc_B)
        
        acc_A_boot.append(acc_A)
        acc_B_boot.append(acc_B)
        boot_diffs.append(acc_B - acc_A)
    
    return np.array(acc_A_boot), np.array(acc_B_boot), np.array(boot_diffs)


# ============================================================================
# SECTION 2: CVAE Classification by Reconstruction (CbR) Functions
# ============================================================================

def compute_cbr_predictions_per_fold(loader, cvae_config, model_dir, n_folds):
    """
    Extract per-fold Classification by Reconstruction predictions.
    
    For each fold:
    1. Load validation data (X_val, y_val, sample_names)
    2. Load CVAE encoder/decoder for that fold
    3. Encode X_val -> z_mean
    4. For each class k: decode(z_mean, class_k) -> reconstruct X_k
    5. Calculate MSE for each class -> argmin(MSE) = predicted class
    6. Return predictions and metadata
    
    Args:
        loader: GliomaDataLoader iterator yielding fold_data dicts
        cvae_config: Configuration dict for CVAE architecture
        model_dir: Directory containing CVAE weights (encoder_fold_X.weights.h5, decoder_fold_X.weights.h5)
        n_folds: Number of cross-validation folds
    
    Returns:
        tuple:
            - cbr_preds_all: (n_samples,) array of CbR predictions across all folds
            - cbr_sample_names: (n_samples,) array of sample names (for patient ID extraction)
            - cbr_classes_all: (n_samples,) array of true class labels
    """
    cbr_preds_all = []
    cbr_sample_names_all = []
    cbr_classes_all = []
    
    print("\n" + "="*70)
    print("PHASE 1: Computing Classification by Reconstruction per-fold")
    print("="*70)
    
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        print(f"\n{fold_name}...")
        
        # Unpack validation data
        X_val = fold_data['val'][0].astype(np.float32)
        y_val = fold_data['val'][1]
        sample_names = fold_data['val'][2] if len(fold_data['val']) > 2 else None
        
        # Fix 1-indexed to 0-indexed if necessary
        if np.min(y_val) == 1:
            y_val = y_val - 1
        
        X_val_exp = np.expand_dims(X_val, axis=-1)  # (N, 1738, 1)
        n_samples = X_val.shape[0]
        
        # Initialize CVAE
        tf.keras.backend.clear_session()
        cvae = CVAE(cvae_config)
        cvae.compile(optimizer='adam')
        
        # Dummy batch to initialize shapes
        n_features = cvae_config['input_shape'][0]
        dummy_x = tf.zeros((2, n_features, 1))
        num_classes = cvae_config.get('num_classes', 6)
        dummy_y = tf.convert_to_tensor(
            tf.keras.utils.to_categorical([0, 1], num_classes=num_classes),
            dtype=tf.float32
        )
        cvae.train_step((dummy_x, dummy_y))
        
        # Load fold-specific weights
        enc_path = f"{model_dir}/encoder_{fold_name}.weights.h5"
        dec_path = f"{model_dir}/decoder_{fold_name}.weights.h5"
        
        if not os.path.exists(enc_path) or not os.path.exists(dec_path):
            print(f"    CVAE weights not found for {fold_name}. Skipping CbR for this fold.")
            continue
        
        cvae.encoder.load_weights(enc_path)
        cvae.decoder.load_weights(dec_path)
        
        encoder = cvae.encoder
        decoder = cvae.decoder
        
        # PHASE 1: ENCODE
        z_mean, _, _, _ = encoder.predict(X_val_exp, batch_size=256, verbose=0)
        
        # PHASE 2: DECODE (for each class) and compute MSE
        mse_scores = np.zeros((n_samples, num_classes), dtype=np.float32)
        for class_idx in range(num_classes):
            requested_classes = tf.convert_to_tensor(
                tf.keras.utils.to_categorical([class_idx] * n_samples, num_classes=num_classes),
                dtype=tf.float32
            )
            reconstructed = decoder.predict([z_mean, requested_classes], batch_size=256, verbose=0)
            reconstructed = np.squeeze(reconstructed, axis=-1)
            mse_scores[:, class_idx] = np.mean((X_val - reconstructed)**2, axis=1)
        
        # PHASE 3: CLASSIFY by argmin(MSE)
        y_pred_cbr = np.argmin(mse_scores, axis=1)
        
        cbr_preds_all.append(y_pred_cbr)
        if sample_names is not None:
            cbr_sample_names_all.append(sample_names)
        cbr_classes_all.append(y_val)
        
        acc_cbr = accuracy_score(y_val, y_pred_cbr)
        print(f"   Accuracy: {acc_cbr:.4f}")
    
    # Concatenate across folds
    cbr_preds_all = np.concatenate(cbr_preds_all)
    cbr_classes_all = np.concatenate(cbr_classes_all)
    
    if cbr_sample_names_all:
        cbr_sample_names_all = np.concatenate(cbr_sample_names_all)
    else:
        cbr_sample_names_all = None
    
    return cbr_preds_all, cbr_sample_names_all, cbr_classes_all


def stratified_slide_bootstrap_cbr(preds_cbr, y_true, slide_ids, n_boot=1000, seed=42):
    """
    Slide-level stratified bootstrap for CbR (no CVAE re-inference).
    
    Since CVAE is computationally expensive, this function does NOT re-run CVAE
    for each bootstrap iteration. Instead, it:
    1. Pre-computes CbR predictions once (via compute_cbr_predictions_per_fold)
    2. Bootstraps by resampling slides (not spectra) within each class
    3. Recomputes accuracy on the resampled slide set
    
    Args:
        preds_cbr: (n_samples,) array of CbR predictions (pre-computed, not re-inferred)
        y_true: (n_samples,) array of true labels
        slide_ids: (n_samples,) array of slide IDs
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        np.ndarray: (n_boot,) array of bootstrap accuracies
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_true)
    
    # PRE-COMPUTE slide indices (for speed)
    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        if sid not in slide_to_indices:
            slide_to_indices[sid] = []
        slide_to_indices[sid].append(idx)
    
    for sid in slide_to_indices:
        slide_to_indices[sid] = np.array(slide_to_indices[sid])
    
    # Map class -> unique slides in that class
    class_to_slides = {}
    for cls in unique_classes:
        class_to_slides[cls] = np.unique(slide_ids[y_true == cls])
    
    boot_acc_cbr = []
    
    print("\nRunning slide-level bootstrap for CbR (no CVAE re-inference)...")
    for i in range(n_boot):
        boot_acc = []
        
        for cls in unique_classes:
            slides_in_cls = class_to_slides[cls]
            # Resample slides with replacement
            sampled_slides = rng.choice(slides_in_cls, size=len(slides_in_cls), replace=True)
            
            for slide in sampled_slides:
                s_idx = slide_to_indices[slide]
                # Get indices where slide's samples belong to this class
                cls_idx = s_idx[y_true[s_idx] == cls]
                
                # Calculate accuracy for this slide
                slide_acc = accuracy_score(y_true[cls_idx], preds_cbr[cls_idx])
                boot_acc.append(slide_acc)
        
        # Average slide accuracies
        acc = np.mean(boot_acc)
        boot_acc_cbr.append(acc)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_boot} iterations complete")
    
    return np.array(boot_acc_cbr)
