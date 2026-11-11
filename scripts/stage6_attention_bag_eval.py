"""
Stage 6, Tier 3: bag-level accuracy vs Tier 0, from the attention-weighted
bag-level artifact (pipeline/mil_attention_stage.py writes
results/{exp}/latent/attention_bag_level.json).

Why this is a SEPARATE script from scripts/stage6_bag_eval.py rather than
another --method choice there: that script reads the latent-label STORE's
per-instance posterior and pools it (mean, across folds) -- correct for
mi-SVM/Tier 4, whose evaluation doesn't depend on attention. For Tier 3,
the store's per-instance posterior is deliberately attention-FREE (right
for Stage 5's spatial validation, wrong for bag classification -- see
AttentionMIL.bag_posterior's docstring and docs/stage6_generative_mil.md's
backlog). Reusing stage6_bag_eval.py's pooling on Tier 3's store posterior
would reproduce the exact misleading, chance-level result the smoke tests
already exposed and fixed. This script reads the ALREADY attention-weighted,
already bag-level artifact instead -- no further pooling rule needed, since
each fold already produced one decision per held-out bag directly.

Usage:
    python scripts/stage6_attention_bag_eval.py --config exp_501
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage6_bag_eval import _tier0_reference
from src.utils.statistical_evaluation import bag_level_bootstrap, summarize_bootstrap_diffs


def _load_bag_level_across_folds(bag_level_path):
    """
    Merge every fold's {bag_ids, posterior, y_true} into one (bag_ids, K+1
    posterior, y_true) triple. A bag appearing in two folds' test sets
    (docs/stage1_data_layer.md's 4 duplicated slides) gets both folds'
    posteriors averaged -- the same duplicate-bag convention Stage 2/
    src/mil/store.py::merge_out_of_sample_posterior already use elsewhere,
    reapplied here since this JSON artifact is a parallel structure to the
    HDF5 store, not read through it.
    """
    with open(bag_level_path) as f:
        by_fold = json.load(f)

    sums, counts, true_label = {}, {}, {}
    for fold_name, fold_result in by_fold.items():
        for bag_id, post, y in zip(fold_result['bag_ids'], fold_result['posterior'],
                                   fold_result['y_true']):
            post = np.array(post, dtype=np.float64)
            sums[bag_id] = sums.get(bag_id, np.zeros_like(post)) + post
            counts[bag_id] = counts.get(bag_id, 0) + 1
            if bag_id in true_label:
                assert true_label[bag_id] == y, (
                    f"bag {bag_id!r} has inconsistent labels across folds: "
                    f"{true_label[bag_id]} vs {y}")
            true_label[bag_id] = y

    bag_ids = sorted(sums.keys())
    posterior = np.array([sums[b] / counts[b] for b in bag_ids])
    posterior = posterior / posterior.sum(axis=1, keepdims=True)
    y_true = np.array([true_label[b] for b in bag_ids])
    return np.array(bag_ids, dtype=object), posterior, y_true


def run(config_name, n_boot=1000, seed=42, regimen_label='TRTR', reference_model='cnn',
       out_path=None):
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    bag_level_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / 'attention_bag_level.json'

    print(f"\n{'='*80}\nSTAGE 6 TIER 3 BAG-LEVEL EVAL (attention-weighted)  ({config_name})\n{'='*80}")
    bag_ids, posterior, y_true = _load_bag_level_across_folds(bag_level_path)
    n_bags = len(bag_ids)
    bag_pred = posterior[:, 1:].argmax(axis=1)
    classes = sorted(np.unique(y_true).tolist())

    metrics = {
        'n_bags': n_bags,
        'bag_accuracy': float(accuracy_score(y_true, bag_pred)),
        'bag_balanced_accuracy': float(balanced_accuracy_score(y_true, bag_pred)),
        'bag_f1_macro': float(f1_score(y_true, bag_pred, average='macro',
                                       zero_division=0, labels=classes)),
        'bag_precision_macro': float(precision_score(y_true, bag_pred, average='macro',
                                                      zero_division=0, labels=classes)),
        'bag_recall_macro': float(recall_score(y_true, bag_pred, average='macro',
                                               zero_division=0, labels=classes)),
        'bag_kappa': float(cohen_kappa_score(y_true, bag_pred))
                     if len(np.unique(y_true)) > 1 else float('nan'),
    }
    # Same convention as scripts/stage6_bag_eval.py: binary-only sensitivity/
    # specificity as per-class recall, AUC from the diagnostic-only columns
    # RENORMALIZED to sum to 1 after dropping column 0 (non-diagnostic mass) --
    # matching stage6_bag_eval.py's _store_fold_data exactly, so the two
    # scripts' AUC numbers sit on the same scale and are comparable.
    if len(classes) == 2:
        recall_per_class = recall_score(y_true, bag_pred, average=None,
                                        zero_division=0, labels=classes)
        metrics['bag_sensitivity'] = float(recall_per_class[1])
        metrics['bag_specificity'] = float(recall_per_class[0])
        diag = posterior[:, 1:].astype(np.float64)
        diag_renorm = diag / diag.sum(axis=1, keepdims=True)
        try:
            metrics['bag_auroc'] = float(roc_auc_score(y_true, diag_renorm[:, 1]))
            metrics['bag_auprc'] = float(average_precision_score(y_true, diag_renorm[:, 1]))
        except ValueError:
            pass
    print(f"  attention: bag_acc={metrics['bag_accuracy']:.4f}  "
          f"bag_F1={metrics['bag_f1_macro']:.4f}  bag_kappa={metrics['bag_kappa']:.4f}  "
          f"(n={n_bags} bags)")

    ref_bag_ids, ref_pred_all, ref_true_all, ref_label = _tier0_reference(
        config, n_boot, seed, regimen_label=regimen_label, reference_model=reference_model)
    common = sorted(set(bag_ids.tolist()) & set(ref_bag_ids.tolist()))
    if len(common) < n_bags or len(common) < len(ref_bag_ids):
        print(f"  [note] comparing on {len(common)}/{n_bags} attention bags "
              f"({len(ref_bag_ids)} reference bags) -- not identical coverage.")
    idx_att = {b: i for i, b in enumerate(bag_ids)}
    idx_ref = {b: i for i, b in enumerate(ref_bag_ids)}
    common_pred = np.array([bag_pred[idx_att[b]] for b in common])
    common_true = np.array([y_true[idx_att[b]] for b in common])
    ref_true_common = np.array([ref_true_all[idx_ref[b]] for b in common])
    ref_pred_common = np.array([ref_pred_all[idx_ref[b]] for b in common])
    assert np.array_equal(common_true, ref_true_common), (
        "bag labels disagree between the attention artifact and the reference classifier "
        "for the same bag id -- investigate before trusting anything below")

    _, _, diffs = bag_level_bootstrap(ref_pred_common, common_pred, common_true,
                                      n_boot=n_boot, seed=seed)
    vs_tier0 = summarize_bootstrap_diffs(diffs)
    ref_acc = float(accuracy_score(common_true, ref_pred_common))
    print(f"  Tier 0 reference ({ref_label}, {regimen_label}, majority vote): "
          f"bag_acc={ref_acc:.4f}  (n={len(common)} bags)")
    print(f"  delta (attention - Tier0) = {metrics['bag_accuracy'] - ref_acc:+.4f}  "
          f"95%CI[{vs_tier0['ci_diff_lo']:+.4f},{vs_tier0['ci_diff_hi']:+.4f}]  "
          f"p={vs_tier0['p_bootstrap']:.4f}  significant={vs_tier0['significant_95']}")

    report = {
        'config': config_name, 'method': 'attention', 'variant': 'attention',
        'n_bootstrap': n_boot, 'metrics': metrics,
        'tier0_reference': {
            'model': ref_label, 'regimen': regimen_label, 'rule': 'majority',
            'n_bags': len(common), 'bag_accuracy': ref_acc,
        },
        'vs_tier0': vs_tier0,
    }
    if out_path is None:
        out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'mil_eval' / 'stage6_bag_eval_attention_attention.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 3 attention-weighted bag-level eval")
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regimen-label', default='TRTR', choices=['TRTR', 'TSTR', 'TSRTR'])
    parser.add_argument('--reference-model', default='cnn',
                        choices=['apollo', 'rf', 'xgb', 'cnn'])
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()
    run(args.config, n_boot=args.n_boot, seed=args.seed, regimen_label=args.regimen_label,
       reference_model=args.reference_model, out_path=args.out_path)
