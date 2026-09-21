"""
Stage: Statistical evaluation (baseline vs candidate regimen).

Compares two trained model sets on the held-out CV validation spectra using the
slide-level stratified bootstrap from src/utils/statistical_evaluation.py (no
data leakage: each fold's validation spectra are scored only by that fold's
models).

For the ratio-ablation experiments this compares the reused TRTR baseline
(exp_201) against the experiment's own TSRTR models. Both directories are
parameterised so any pair of regimens can be compared.

Outputs (to ``config['paths']['statistical_dir']``):
  statistical_evaluation_{timestamp}.json   full per-model metrics + bootstrap CIs
  statistical_summary_{timestamp}.csv        compact table
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.statistical_evaluation import (
    find_model_path, load_and_predict, load_and_predict_proba,
    stratified_slide_accuracy, stratified_slide_bootstrap,
    stratified_slide_bootstrap_cbr,
    compute_full_metrics, summarize_bootstrap_diffs, build_summary_rows,
)

# (prefix, file extension, loader type) for the 4 benchmark models
MODEL_SPECS = [
    ('apollo', '.joblib', 'sklearn'),
    ('rf', '.joblib', 'sklearn'),
    ('xgb', '.json', 'xgboost'),
    ('cnn', '.keras', 'keras'),
]
MODEL_LABELS = {'apollo': 'Apollo', 'rf': 'RF_Standard', 'xgb': 'XGBoost', 'cnn': '1D_CNN'}


def _binary_map(y, num_classes):
    if y.size and np.min(y) == 1:
        y = y - 1
    if num_classes == 2:
        y = np.where(y < 3, 0, 1)
    return y


def _build_fold_val_sets(config):
    """
    Reconstruct each fold's validation spectra (X, y, per-spectrum slide_ids) from
    the split file + full (cropped) loader arrays — mirrors GliomaDataLoader's own
    masking but also exposes per-spectrum slide ids (needed for slide-level bootstrap).

    Returns list of dicts: {'fold_idx', 'fold_name', 'X', 'y', 'slide_ids'}.
    """
    from pipeline._common import make_loader, num_classes_of

    loader = make_loader(config)
    num_classes = num_classes_of(config)

    split_file = loader.split_file
    with open(split_file, 'r') as f:
        splits = json.load(f)
    cv_splits = splits['cv_folds'] if 'cv_folds' in splits else splits

    folds = []
    for i, (fold_name, split_data) in enumerate(cv_splits.items()):
        test_mask = np.isin(loader.groups, split_data['test'])
        folds.append({
            'fold_idx': i,
            'fold_name': fold_name,
            'X': loader.X[test_mask].astype(np.float32),
            'y': _binary_map(loader.y_class[test_mask].copy(), num_classes),
            'slide_ids': loader.groups[test_mask],
        })
    return folds


def _gather_predictions(folds, model_dir, prefix, ext, model_type):
    """Concatenate fold-wise validation predictions (fold i scored by fold-i model)."""
    preds, y_true, slide_ids = [], [], []
    for fd in folds:
        path = find_model_path(str(model_dir), prefix, ext, fd['fold_idx'])
        preds.append(load_and_predict(path, model_type, fd['X']))
        y_true.append(fd['y'])
        slide_ids.append(fd['slide_ids'])
    return (np.concatenate(preds), np.concatenate(y_true), np.concatenate(slide_ids))


def _gather_probas(folds, model_dir, prefix, ext, model_type):
    """Concatenate fold-wise validation probabilities (n_samples, n_classes)."""
    probas = []
    for fd in folds:
        path = find_model_path(str(model_dir), prefix, ext, fd['fold_idx'])
        probas.append(load_and_predict_proba(path, model_type, fd['X']))
    return np.concatenate(probas)


def _cbr_predictions(folds, config, n_features, cvae_weights_dir):
    """
    Classification by Reconstruction (CbR) predictions, pooled across folds.

    For each fold, encode the validation spectra with that fold's CVAE encoder,
    decode under each class label, and predict the class whose reconstruction has
    the lowest MSE. No data leakage: each fold's validation spectra use only that
    fold's CVAE (trained without them).

    Returns (preds, y_true, slide_ids) concatenated across folds, or None if any
    fold's CVAE weights are missing. This is the single source of truth for CbR
    predictions, shared by the bootstrap evaluation (``_compute_cbr``) and the
    offline metric-patching script (``patch_cbr_metrics.py``).
    """
    import tensorflow as tf
    from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder

    cvae_config = build_cvae_config(config, n_features)
    num_classes = cvae_config['num_classes']

    preds_all, y_all, slide_all = [], [], []
    for fd in folds:
        try:
            encoder, decoder = load_encoder_decoder(cvae_config, cvae_weights_dir, fd['fold_name'])
        except FileNotFoundError as e:
            print(f"  [CbR skip] {e}")
            return None

        X = fd['X']
        X_exp = X.reshape(-1, n_features, 1)
        z_mean = encoder.predict(X_exp, batch_size=256, verbose=0)[0]  # encoder -> [z_mean, ...]

        mse = np.zeros((len(X), num_classes), dtype=np.float32)
        for k in range(num_classes):
            label = tf.keras.utils.to_categorical([k] * len(X), num_classes=num_classes).astype('float32')
            rec = decoder.predict([z_mean, label], batch_size=256, verbose=0)
            if rec.ndim == 3:
                rec = rec.reshape(rec.shape[0], -1)
            mse[:, k] = np.mean((X - rec) ** 2, axis=1)

        preds_all.append(np.argmin(mse, axis=1))
        y_all.append(fd['y'])
        slide_all.append(fd['slide_ids'])
        tf.keras.backend.clear_session()

    return (np.concatenate(preds_all), np.concatenate(y_all), np.concatenate(slide_all))


def _compute_cbr(folds, config, n_features, cvae_weights_dir, n_boot, seed):
    """
    Classification by Reconstruction (CbR): compute pooled predictions and return
    a result dict with slide-level accuracy + bootstrap CI, or None if weights are
    missing.
    """
    out = _cbr_predictions(folds, config, n_features, cvae_weights_dir)
    if out is None:
        return None
    preds, y_true, slide_ids = out
    acc = stratified_slide_accuracy(preds, y_true, slide_ids)
    boot = stratified_slide_bootstrap_cbr(preds, y_true, slide_ids, n_boot=n_boot, seed=seed)
    full = compute_full_metrics(y_true, preds, slide_ids=slide_ids)
    result = {
        'model': 'Classification_by_Reconstruction',
        'cbr_accuracy': acc,
        'cbr_boot_mean': float(np.mean(boot)),
        'cbr_ci_lo': float(np.percentile(boot, 2.5)),
        'cbr_ci_hi': float(np.percentile(boot, 97.5)),
        'full_metrics': full,
    }
    # Flat metric keys read by paper/gen_appendix_tables.py (CbR row).
    for k in ('f1_macro', 'precision_macro', 'recall_macro', 'balanced_accuracy', 'kappa'):
        if k in full:
            result[k] = full[k]
    return result


def evaluation_stage(config, output_dir=None, baseline_dir=None, candidate_dir=None,
                     baseline_label='TRTR', candidate_label='TSRTR', n_boot=None,
                     cbr=True, cvae_weights_dir=None):
    baseline_dir = (baseline_dir
                    or config['paths'].get('models_baseline_dir')
                    or config['paths'].get('models_trtr_dir'))
    candidate_dir = candidate_dir or config['paths'].get('models_tsrtr_dir')
    cvae_weights_dir = (cvae_weights_dir or config.get('tsrtr', {}).get('cvae_weights_dir')
                        or config['paths'].get('models_cvae_dir'))
    stats_cfg = config.get('statistics', {})
    n_boot = int(n_boot if n_boot is not None else stats_cfg.get('bootstrap_samples', 1000))
    seed = int(stats_cfg.get('random_seed', 42))

    out_dir = Path(config['paths']['statistical_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}\nEVALUATION STAGE: {baseline_label} vs {candidate_label}\n{'='*80}")
    print(f"  baseline  ({baseline_label}):  {baseline_dir}")
    print(f"  candidate ({candidate_label}): {candidate_dir}")
    print(f"  bootstrap iterations: {n_boot}")

    folds = _build_fold_val_sets(config)
    n_val = sum(len(fd['y']) for fd in folds)
    print(f"  Validation spectra across {len(folds)} folds: {n_val}")

    results = []
    for prefix, ext, model_type in MODEL_SPECS:
        label = MODEL_LABELS[prefix]
        try:
            preds_A, y_true, slide_ids = _gather_predictions(folds, baseline_dir, prefix, ext, model_type)
            preds_B, _, _ = _gather_predictions(folds, candidate_dir, prefix, ext, model_type)
        except FileNotFoundError as e:
            print(f"  [skip {label}] {e}")
            continue

        # Probabilities (best-effort; None if unavailable)
        try:
            proba_A = _gather_probas(folds, baseline_dir, prefix, ext, model_type)
            proba_B = _gather_probas(folds, candidate_dir, prefix, ext, model_type)
        except Exception:
            proba_A = proba_B = None

        acc_A = stratified_slide_accuracy(preds_A, y_true, slide_ids)
        acc_B = stratified_slide_accuracy(preds_B, y_true, slide_ids)

        _, _, diffs = stratified_slide_bootstrap(
            preds_A, preds_B, y_true, slide_ids, n_boot=n_boot, seed=seed)

        boot = summarize_bootstrap_diffs(diffs)
        full_A = compute_full_metrics(y_true, preds_A, proba_A, slide_ids=slide_ids)
        full_B = compute_full_metrics(y_true, preds_B, proba_B, slide_ids=slide_ids)

        row = {
            'model': label,
            # Slide-level bootstrap comparison (primary inference)
            f'slide_acc_{baseline_label}': acc_A,
            f'slide_acc_{candidate_label}': acc_B,
            'delta': acc_B - acc_A,
            **boot,
            # Full descriptive metrics for both regimens (macro F1/precision/recall
            # are slide-level; accuracy/kappa/auroc remain spectrum-level)
            f'full_metrics_{baseline_label}': full_A,
            f'full_metrics_{candidate_label}': full_B,
        }
        results.append(row)
        print(f"  {label:>12}: {baseline_label}={acc_A:.4f}  {candidate_label}={acc_B:.4f}  "
              f"delta={row['delta']:+.4f}  95%CI[{boot['ci_diff_lo']:+.4f},{boot['ci_diff_hi']:+.4f}]  "
              f"p={boot['p_bootstrap']:.4f}")
        print(f"  {' ':>12}  F1_macro: {baseline_label}={full_A['f1_macro']:.4f}  "
              f"{candidate_label}={full_B['f1_macro']:.4f}  "
              f"prec: {baseline_label}={full_A['precision_macro']:.4f}  "
              f"rec: {baseline_label}={full_A['recall_macro']:.4f}")

    # Classification by Reconstruction (CVAE encoder/decoder MSE matching)
    cbr_result = None
    if cbr:
        n_features = folds[0]['X'].shape[1]
        print(f"\n  Classification by Reconstruction (CVAE weights: {cvae_weights_dir})")
        cbr_result = _compute_cbr(folds, config, n_features, cvae_weights_dir, n_boot, seed)
        if cbr_result:
            print(f"  {'CbR':>12}: slide_acc={cbr_result['cbr_accuracy']:.4f}  "
                  f"95%CI[{cbr_result['cbr_ci_lo']:.4f},{cbr_result['cbr_ci_hi']:.4f}]  "
                  f"F1={cbr_result.get('f1_macro', float('nan')):.4f}  "
                  f"prec={cbr_result.get('precision_macro', float('nan')):.4f}  "
                  f"rec={cbr_result.get('recall_macro', float('nan')):.4f}")
            results.append(cbr_result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_tag = f"{baseline_label}_vs_{candidate_label}"
    payload = {
        'experiment': config['name'],
        'baseline_label': baseline_label,
        'candidate_label': candidate_label,
        'baseline_dir': str(baseline_dir),
        'candidate_dir': str(candidate_dir),
        'cvae_weights_dir': str(cvae_weights_dir),
        'n_bootstrap': n_boot,
        'n_val_spectra': int(n_val),
        'tsrtr': config.get('tsrtr', {}),
        'results': results,
        'cbr': cbr_result,
    }
    json_path = out_dir / f"statistical_evaluation_{comparison_tag}_{timestamp}.json"
    csv_path  = out_dir / f"statistical_summary_{comparison_tag}_{timestamp}.csv"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    # Flat CSV: one row per model, key metrics for both regimens side-by-side
    csv_rows = build_summary_rows(results, baseline_label, candidate_label)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"\n  Evaluation complete -> {json_path.name}, {csv_path.name}")
    return {'json_path': str(json_path), 'csv_path': str(csv_path), 'results': results}


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Statistical evaluation (baseline vs candidate)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_401")
    parser.add_argument('--baseline-dir', default=None, help="override baseline model dir (TRTR)")
    parser.add_argument('--candidate-dir', default=None, help="override candidate model dir (TSRTR)")
    parser.add_argument('--baseline-label', default='TRTR')
    parser.add_argument('--candidate-label', default='TSRTR')
    parser.add_argument('--n-boot', type=int, default=None)
    parser.add_argument('--no-cbr', action='store_true', help="skip Classification by Reconstruction")
    parser.add_argument('--cvae-weights-dir', default=None, help="override CVAE weights dir for CbR")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    evaluation_stage(
        cfg, out_dir,
        baseline_dir=args.baseline_dir, candidate_dir=args.candidate_dir,
        baseline_label=args.baseline_label, candidate_label=args.candidate_label,
        n_boot=args.n_boot, cbr=not args.no_cbr, cvae_weights_dir=args.cvae_weights_dir,
    )
