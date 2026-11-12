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
    within_bag_instance_recall, stratified_slide_bootstrap,
    stratified_slide_bootstrap_cbr,
    bag_decisions, bag_level_bootstrap, bag_level_bootstrap_cbr,
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


def _cbr_predictions(folds, config, n_features, cvae_weights_dir, return_scores=False):
    """
    Classification by Reconstruction (CbR) predictions, pooled across folds.

    For each fold, encode the validation spectra with that fold's CVAE encoder,
    decode under each class label, and predict the class whose reconstruction has
    the lowest MSE. No data leakage: each fold's validation spectra use only that
    fold's CVAE (trained without them).

    Returns (preds, y_true, slide_ids) concatenated across folds, or None if any
    fold's CVAE weights are missing. This is the single source of truth for CbR
    predictions, shared by the bootstrap evaluation (``_compute_cbr``) and the
    offline AUROC-patching script (``scripts/patch_cbr_auroc.py``).

    ``return_scores=True`` (binary tasks only) appends a fourth array: the per-
    spectrum normalised MSE margin ``(mse_0 - mse_1) / (mse_0 + mse_1)``. It is
    positive exactly when class 1 reconstructs better, i.e. when the argmin
    rule predicts class 1, so it is the continuous score underlying the hard
    decision (see ``_cbr_bag_auroc``).
    """
    import tensorflow as tf
    from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder

    cvae_config = build_cvae_config(config, n_features)
    num_classes = cvae_config['num_classes']

    if return_scores and num_classes != 2:
        raise ValueError("return_scores is defined for binary CbR only")
    preds_all, y_all, slide_all, score_all = [], [], [], []
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
        if return_scores:
            score_all.append((mse[:, 0] - mse[:, 1]) / (mse[:, 0] + mse[:, 1] + 1e-12))
        tf.keras.backend.clear_session()

    out = (np.concatenate(preds_all), np.concatenate(y_all), np.concatenate(slide_all))
    return out + (np.concatenate(score_all),) if return_scores else out


def _cbr_bag_auroc(scores, y_true, slide_ids):
    """
    Bag-level AUROC for CbR. The per-spectrum score (normalised MSE margin, positive
    = class 1 reconstructs better) is mean-pooled within each bag -- the analogue
    of the other classifiers' mean-pooled class-1 probability -- and ranked against
    the bag labels. Duplicated bags (a slide in two test folds) pool all of their
    spectra, exactly as ``bag_decisions`` does.

    Also returns the AUROC of the vote share (fraction of a bag's spectra predicted
    class 1) as a sensitivity check: a majority vote is a threshold of 0.5 on it.
    Returns (auroc_margin, auroc_vote_share).
    """
    from sklearn.metrics import roc_auc_score

    scores, y_true, slide_ids = np.asarray(scores), np.asarray(y_true), np.asarray(slide_ids)
    bag_ids = np.unique(slide_ids)
    bag_margin, bag_vote, bag_y = [], [], []
    for b in bag_ids:
        m = slide_ids == b
        assert np.all(y_true[m] == y_true[m][0]), f'bag {b!r} has mixed labels'
        bag_margin.append(scores[m].mean())
        bag_vote.append((scores[m] > 0).mean())
        bag_y.append(int(y_true[m][0]))
    return (float(roc_auc_score(bag_y, bag_margin)), float(roc_auc_score(bag_y, bag_vote)))


def _compute_cbr(folds, config, n_features, cvae_weights_dir, n_boot, seed):
    """
    Classification by Reconstruction (CbR): compute pooled predictions and return
    a result dict with bag-level accuracy + bootstrap CI (primary), plus the
    within-bag instance diagnostic under its own name, or None if weights are
    missing.
    """
    binary = int(config['models']['cvae']['num_classes']) == 2
    out = _cbr_predictions(folds, config, n_features, cvae_weights_dir, return_scores=binary)
    if out is None:
        return None
    preds, y_true, slide_ids = out[:3]
    full = compute_full_metrics(y_true, preds, slide_ids=slide_ids)

    bag_ids, bag_pred, bag_true = bag_decisions(preds, slide_ids, y_true)
    bag_acc = full['bag_accuracy']
    bag_boot = bag_level_bootstrap_cbr(bag_pred, bag_true, n_boot=n_boot, seed=seed)

    wbir = within_bag_instance_recall(preds, y_true, slide_ids)
    wbir_boot = stratified_slide_bootstrap_cbr(preds, y_true, slide_ids, n_boot=n_boot, seed=seed)

    result = {
        'model': 'Classification_by_Reconstruction',
        'n_bags': len(bag_ids),
        'cbr_bag_accuracy': bag_acc,
        'cbr_bag_boot_mean': float(np.mean(bag_boot)),
        'cbr_bag_ci_lo': float(np.percentile(bag_boot, 2.5)),
        'cbr_bag_ci_hi': float(np.percentile(bag_boot, 97.5)),
        # Diagnostic (demoted): kept under the old key names for continuity,
        # but this is within-bag instance recall, NOT a bag accuracy.
        'cbr_accuracy': wbir,
        'cbr_boot_mean': float(np.mean(wbir_boot)),
        'cbr_ci_lo': float(np.percentile(wbir_boot, 2.5)),
        'cbr_ci_hi': float(np.percentile(wbir_boot, 97.5)),
        'full_metrics': full,
    }
    # Flat metric keys read by paper/gen_appendix_controlled_experiments.py (CbR row). _slide_
    # level_metrics no longer returns bare accuracy/f1_macro/precision_macro/
    # recall_macro/balanced_accuracy/kappa at all (Stage 2 retired the
    # degenerate within-bag-instance versions of these); only the bag_* and
    # within_bag_instance_recall/f1_weighted names exist now.
    for k in ('bag_f1_macro', 'bag_precision_macro', 'bag_recall_macro',
             'bag_balanced_accuracy', 'bag_kappa', 'within_bag_instance_recall',
             'f1_weighted'):
        if k in full:
            result[k] = full[k]
    if binary:
        result['bag_auroc'], result['bag_auroc_vote_share'] = _cbr_bag_auroc(out[3], y_true, slide_ids)
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

        # Full metric sets -- _slide_level_metrics computes BOTH the genuine
        # bag-level metrics (bag_accuracy etc., one majority-vote decision per
        # bag) and the within-bag instance diagnostics in one pass, already
        # correctly merging duplicated bags (docs/MIL_REFRAME_PLAN.md §7.1) via
        # bag_decisions's own pooling.
        full_A = compute_full_metrics(y_true, preds_A, proba_A, slide_ids=slide_ids)
        full_B = compute_full_metrics(y_true, preds_B, proba_B, slide_ids=slide_ids)

        # Primary inference: bag-level accuracy + bootstrap CI over bag decisions.
        bag_ids, bag_pred_A, bag_true = bag_decisions(preds_A, slide_ids, y_true)
        _, bag_pred_B = bag_decisions(preds_B, slide_ids)
        bag_acc_A, bag_acc_B = full_A['bag_accuracy'], full_B['bag_accuracy']
        _, _, bag_diffs = bag_level_bootstrap(
            bag_pred_A, bag_pred_B, bag_true, n_boot=n_boot, seed=seed)
        bag_boot = summarize_bootstrap_diffs(bag_diffs)

        # Diagnostic (demoted): within-bag instance recall + its own CI, kept
        # for continuity with pre-reframe reporting -- not a bag decision.
        wbir_A = within_bag_instance_recall(preds_A, y_true, slide_ids)
        wbir_B = within_bag_instance_recall(preds_B, y_true, slide_ids)
        _, _, wbir_diffs = stratified_slide_bootstrap(
            preds_A, preds_B, y_true, slide_ids, n_boot=n_boot, seed=seed)
        wbir_boot = summarize_bootstrap_diffs(wbir_diffs)

        row = {
            'model': label,
            'n_bags': len(bag_ids),
            # Bag-level bootstrap comparison (primary inference)
            f'bag_acc_{baseline_label}': bag_acc_A,
            f'bag_acc_{candidate_label}': bag_acc_B,
            'bag_delta': bag_acc_B - bag_acc_A,
            **{f'bag_{k}': v for k, v in bag_boot.items()},
            # Within-bag instance recall (diagnostic; NOT a bag accuracy)
            f'within_bag_instance_recall_{baseline_label}': wbir_A,
            f'within_bag_instance_recall_{candidate_label}': wbir_B,
            'within_bag_instance_recall_delta': wbir_B - wbir_A,
            **{f'wbir_{k}': v for k, v in wbir_boot.items()},
            # Full descriptive metrics for both regimens (bag_* keys are
            # bag-level; the rest are within-bag instance diagnostics)
            f'full_metrics_{baseline_label}': full_A,
            f'full_metrics_{candidate_label}': full_B,
        }
        results.append(row)
        print(f"  {label:>12}: BAG  {baseline_label}={bag_acc_A:.4f}  {candidate_label}={bag_acc_B:.4f}  "
              f"delta={row['bag_delta']:+.4f}  "
              f"95%CI[{bag_boot['ci_diff_lo']:+.4f},{bag_boot['ci_diff_hi']:+.4f}]  "
              f"p={bag_boot['p_bootstrap']:.4f}  (n={len(bag_ids)} bags)")
        print(f"  {' ':>12}  [diagnostic] within_bag_instance_recall: "
              f"{baseline_label}={wbir_A:.4f}  {candidate_label}={wbir_B:.4f}")
        print(f"  {' ':>12}  bag_F1_macro: {baseline_label}={full_A['bag_f1_macro']:.4f}  "
              f"{candidate_label}={full_B['bag_f1_macro']:.4f}  "
              f"bag_prec: {baseline_label}={full_A['bag_precision_macro']:.4f}  "
              f"bag_rec: {baseline_label}={full_A['bag_recall_macro']:.4f}")

    # Classification by Reconstruction (CVAE encoder/decoder MSE matching)
    cbr_result = None
    if cbr:
        n_features = folds[0]['X'].shape[1]
        print(f"\n  Classification by Reconstruction (CVAE weights: {cvae_weights_dir})")
        cbr_result = _compute_cbr(folds, config, n_features, cvae_weights_dir, n_boot, seed)
        if cbr_result:
            print(f"  {'CbR':>12}: BAG bag_acc={cbr_result['cbr_bag_accuracy']:.4f}  "
                  f"95%CI[{cbr_result['cbr_bag_ci_lo']:.4f},{cbr_result['cbr_bag_ci_hi']:.4f}]  "
                  f"(n={cbr_result['n_bags']} bags)  "
                  f"bag_F1={cbr_result.get('bag_f1_macro', float('nan')):.4f}  "
                  f"bag_prec={cbr_result.get('bag_precision_macro', float('nan')):.4f}  "
                  f"bag_rec={cbr_result.get('bag_recall_macro', float('nan')):.4f}")
            print(f"  {' ':>12}  [diagnostic] within_bag_instance_recall="
                  f"{cbr_result['cbr_accuracy']:.4f}  "
                  f"95%CI[{cbr_result['cbr_ci_lo']:.4f},{cbr_result['cbr_ci_hi']:.4f}]")
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
