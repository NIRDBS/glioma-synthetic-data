"""
Cheap post-processing: compute the Tier 0 bootstrap comparison for an
ALREADY-COMPUTED Tier 5 blind bag classification report
(scripts/tier5_blind_bag_classification.py's output,
results/{exp}/latent/bayes_blind_bag_classification.json), without
re-running the expensive Gibbs refit. Every bag's per-fold log-likelihood
vector is already stored in that file; this just merges duplicated bags
(average log-lik across the folds they appear in, same convention as
merge_out_of_sample_posterior, docs/mil_tiers_methodology.md §2) and runs
the same bootstrap comparison tier5_blind_bag_classification.py now
computes for any NEW run.

Usage:
    python scripts/tier5_blind_vs_tier0.py --config exp_608
    python scripts/tier5_blind_vs_tier0.py --config exp_2.608 --reference-model apollo
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from scripts.stage6_bag_eval import _tier0_reference
from src.utils.statistical_evaluation import bag_level_bootstrap, summarize_bootstrap_diffs


def run(config_name, report_path=None, reference_model='apollo', n_boot=1000, seed=42, out_path=None):
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    if report_path is None:
        report_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'latent' / 'bayes_blind_bag_classification.json'
    with open(report_path) as f:
        existing = json.load(f)

    log_lik_across_folds = {}
    true_label_by_bag = {}
    for fold_name, fold in existing['per_fold'].items():
        for b, rec in fold['bags'].items():
            log_lik_across_folds.setdefault(b, []).append(np.asarray(rec['log_lik']))
            true_label_by_bag[b] = rec['true']

    merged_bags = sorted(log_lik_across_folds)
    merged_pred = np.array([int(np.argmax(np.mean(log_lik_across_folds[b], axis=0)))
                            for b in merged_bags])
    merged_true = np.array([true_label_by_bag[b] for b in merged_bags])
    merged_acc = float((merged_true == merged_pred).mean())
    print(f"Loaded {report_path}")
    print(f"merged (duplicate-averaged, n={len(merged_bags)} unique bags) bag_acc: {merged_acc:.4f}")

    print(f"\n{'='*70}\nTIER 0 COMPARISON (reference_model={reference_model})\n{'='*70}")
    tier0_bag_ids, tier0_pred, tier0_true, tier0_label = _tier0_reference(
        config, n_boot=n_boot, seed=seed, regimen_label='TRTR', reference_model=reference_model)
    tier0_bag_ids = np.asarray(tier0_bag_ids)
    common = [b for b in merged_bags if b in set(tier0_bag_ids.tolist())]
    idx_t5 = {b: i for i, b in enumerate(merged_bags)}
    idx_t0 = {b: i for i, b in enumerate(tier0_bag_ids.tolist())}
    aligned_t5_pred = np.array([merged_pred[idx_t5[b]] for b in common])
    aligned_t5_true = np.array([merged_true[idx_t5[b]] for b in common])
    aligned_t0_pred = np.array([tier0_pred[idx_t0[b]] for b in common])
    aligned_t0_true = np.array([tier0_true[idx_t0[b]] for b in common])
    assert np.array_equal(aligned_t5_true, aligned_t0_true), \
        "bag true-label mismatch between Tier 5 (blind) and Tier 0 reference -- alignment bug"

    tier0_acc = float((aligned_t0_true == aligned_t0_pred).mean())
    _, _, diffs = bag_level_bootstrap(aligned_t0_pred, aligned_t5_pred, aligned_t0_true,
                                      n_boot=n_boot, seed=seed)
    boot = summarize_bootstrap_diffs(diffs)
    print(f"  Tier 0 ({tier0_label}, TRTR, majority vote): bag_acc={tier0_acc:.4f}  (n={len(common)} bags)")
    print(f"  Tier 5 (blind): bag_acc={merged_acc:.4f}")
    print(f"  delta (Tier5_blind - Tier0) = {merged_acc - tier0_acc:+.4f}  "
         f"95%CI[{boot['ci_diff_lo']:+.4f},{boot['ci_diff_hi']:+.4f}]  "
         f"p={boot['p_bootstrap']:.4f}  significant={boot['p_bootstrap'] < 0.05}")

    report = dict(existing)
    report['merged_bag_acc'] = merged_acc
    report['n_merged_bags'] = len(merged_bags)
    report['tier0_reference'] = {'model': tier0_label, 'regimen': 'TRTR', 'rule': 'majority',
                                 'n_bags': len(common), 'bag_accuracy': tier0_acc}
    report['vs_tier0'] = {'boot_mean_diff': boot['boot_mean_diff'],
                          'ci_diff_lo': boot['ci_diff_lo'], 'ci_diff_hi': boot['ci_diff_hi'],
                          'p_bootstrap': boot['p_bootstrap'], 'significant_95': boot['significant_95']}
    if out_path is None:
        out_path = report_path
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport updated -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 0 comparison for an existing Tier 5 blind report")
    parser.add_argument('--config', required=True)
    parser.add_argument('--report-path', default=None)
    parser.add_argument('--reference-model', default='apollo', choices=['apollo', 'rf', 'xgb', 'cnn'])
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()
    run(args.config, report_path=args.report_path, reference_model=args.reference_model,
       n_boot=args.n_boot, seed=args.seed, out_path=args.out_path)
