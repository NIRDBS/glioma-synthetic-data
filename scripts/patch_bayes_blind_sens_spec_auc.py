"""
Retroactively add bag_sensitivity/bag_specificity/bag_auroc/bag_auprc to an
already-written bayes_blind_bag_classification.json, WITHOUT refitting the
Gibbs chains -- tier5_blind_bag_classification.py's own run() now computes
these (see its extra_metrics block), but re-running it means a full refit
(3 chains x 5 folds, ~hours per experiment, confirmed directly this
session). The per-bag log-likelihood vectors needed are already saved in
each fold's 'bags' entry, so this reconstructs the identical duplicate-bag
merge (average log-likelihood across the folds a bag appears in, same
spirit as merge_out_of_sample_posterior) and the identical metric formulas
the live script now uses, purely as a post-hoc read of existing JSON.

Usage:
    python scripts/patch_bayes_blind_sens_spec_auc.py --config exp_608
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def patch(config_name):
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score,
    )

    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / 'bayes_blind_bag_classification.json'
    with open(path) as f:
        report = json.load(f)

    log_lik_across_folds = defaultdict(list)
    true_label_by_bag = {}
    for fold_result in report['per_fold'].values():
        for bag_id, info in fold_result['bags'].items():
            log_lik_across_folds[bag_id].append(np.asarray(info['log_lik']))
            true_label_by_bag[bag_id] = info['true']

    merged_bags = sorted(log_lik_across_folds)
    merged_ll = np.array([np.mean(log_lik_across_folds[b], axis=0) for b in merged_bags])
    merged_pred = merged_ll.argmax(axis=1)
    merged_true = np.array([true_label_by_bag[b] for b in merged_bags])
    assert float((merged_true == merged_pred).mean()) == report['merged_bag_acc'], (
        f"{config_name}: reconstructed merged_bag_acc disagrees with the saved value -- "
        f"merge logic drifted from tier5_blind_bag_classification.py's own, investigate")

    extra = {}
    if len(set(merged_true.tolist())) == 2:
        shifted = merged_ll - merged_ll.max(axis=1, keepdims=True)
        proba = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        recall_per_class = recall_score(merged_true, merged_pred, average=None, zero_division=0)
        extra['bag_balanced_accuracy'] = float(balanced_accuracy_score(merged_true, merged_pred))
        extra['bag_sensitivity'] = float(recall_per_class[1])
        extra['bag_specificity'] = float(recall_per_class[0])
        try:
            extra['bag_auroc'] = float(roc_auc_score(merged_true, proba[:, 1]))
            extra['bag_auprc'] = float(average_precision_score(merged_true, proba[:, 1]))
        except ValueError:
            pass

    report.update(extra)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"{config_name}: bag_acc={report['merged_bag_acc']:.4f}  "
         f"sensitivity={extra.get('bag_sensitivity', float('nan')):.4f}  "
         f"specificity={extra.get('bag_specificity', float('nan')):.4f}  "
         f"AUC={extra.get('bag_auroc', float('nan')):.4f}  -> patched {path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    patch(args.config)
