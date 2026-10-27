"""
Genuine, non-circular Tier 5 bag classification (docs/tier5_bayesian_mil.md
section 4.0): for each fold, refit the exact same Gibbs chains (same seeds,
same data prep as pipeline/mil_fit_stage.py::_fit_one_fold_bayes_mil --
confirmed bit-exact reproduction of the saved chain_log_liks, so this is
reconstructing the already-fitted model, not training a different one; the
raw posterior draws were never persisted to disk, only summary diagnostics
were), then score each fold's own held-out bags with
src/mil/bayes.py::score_bayes_mil_blind, which is never given the true
label -- it evaluates each candidate class's posterior-predictive
likelihood and predicts by argmax, exactly like an ordinary held-out
classifier evaluation.

Also computes the Tier 0 bootstrap comparison every other tier's bag-eval
reports (`scripts/stage6_bag_eval.py`'s convention) -- this was missing
entirely until 2026-10-12: duplicated bags (4 slides forced into two CV
test folds each) are merged by averaging their log-likelihood vectors
across folds before the final argmax, the same spirit as
`merge_out_of_sample_posterior`'s duplicate-averaging convention
(docs/mil_tiers_methodology.md §2), giving one decision per unique bag
before the bootstrap runs.

Usage:
    python scripts/tier5_blind_bag_classification.py --config exp_608
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.decomposition import PCA

from pipeline._common import make_loader, num_classes_of, prepare_fold
from src.mil.bayes import fit_bayes_mil_chains, score_bayes_mil_blind
from src.utils.statistical_evaluation import bag_level_bootstrap, summarize_bootstrap_diffs


def run(config_name, n_folds=5, pca_components=10, n_chains=3, n_iter=600,
       burn_in=200, thin=5, seed=42, out_path=None, reference_model='apollo',
       n_boot=1000):
    from collections import defaultdict

    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    n_classes = num_classes_of(config)
    loader = make_loader(config)

    per_fold = {}
    all_true, all_pred, all_bag_ids = [], [], []
    log_lik_across_folds = defaultdict(list)  # bag -> [per-fold log-lik vectors]
    true_label_by_bag = {}

    for fold_data in loader:
        fold_name = fold_data['fold_name']
        fd = prepare_fold(fold_data, config, cap_train=False)

        pca = PCA(n_components=pca_components, random_state=seed)
        Z_train = pca.fit_transform(fd.X_train)
        Z_val = pca.transform(fd.X_val)

        print(f"\n{'='*70}\n{fold_name}: fitting (reconstructing the saved chains)\n{'='*70}")
        fit = fit_bayes_mil_chains(
            Z_train, fd.train_bag_ids, fd.y_train, n_classes,
            n_chains=n_chains, n_iter=n_iter, burn_in=burn_in, thin=thin,
            seed=seed, verbose=False)
        print(f"  chain_rhat={fit['chain_rhat']:.4f}  chain_log_liks="
              f"{[round(v, 1) for v in fit['chain_log_liks']]}")

        log_lik_by_bag = score_bayes_mil_blind(fit, Z_val, fd.val_bag_ids, n_classes)
        true_label = {b: int(fd.y_val[fd.val_bag_ids == b][0]) for b in np.unique(fd.val_bag_ids)}

        fold_true, fold_pred, fold_bags = [], [], []
        for b, ll in log_lik_by_bag.items():
            pred = int(np.argmax(ll))
            fold_true.append(true_label[b])
            fold_pred.append(pred)
            fold_bags.append(b)
            all_true.append(true_label[b])
            all_pred.append(pred)
            all_bag_ids.append(b)
            log_lik_across_folds[b].append(np.asarray(ll))
            true_label_by_bag[b] = true_label[b]

        fold_true, fold_pred = np.array(fold_true), np.array(fold_pred)
        acc = float((fold_true == fold_pred).mean())
        majority_baseline = float(max((fold_true == 0).mean(), (fold_true == 1).mean()))
        per_fold[fold_name] = {
            'n_bags': len(fold_bags), 'bag_acc': acc,
            'majority_baseline': majority_baseline,
            'bags': {b: {'true': t, 'pred': p, 'log_lik': log_lik_by_bag[b].tolist()}
                    for b, t, p in zip(fold_bags, fold_true.tolist(), fold_pred.tolist())},
        }
        print(f"  BLIND bag_acc={acc:.4f}  (n={len(fold_bags)}, majority baseline={majority_baseline:.4f})")

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    overall_acc = float((all_true == all_pred).mean())
    print(f"\n{'='*70}\nOVERALL (per-fold average, no cross-fold duplicate merging -- see docs)\n{'='*70}")
    print(f"mean per-fold bag_acc: {np.mean([v['bag_acc'] for v in per_fold.values()]):.4f}")
    print(f"pooled (all {len(all_true)} fold-bag rows) bag_acc: {overall_acc:.4f}")

    # Merge duplicated bags (4 slides forced into 2 CV test folds each) by
    # averaging their log-likelihood vectors before the final argmax -- one
    # decision per unique bag, same spirit as merge_out_of_sample_posterior's
    # duplicate-averaging convention (docs/mil_tiers_methodology.md §2).
    merged_bags = sorted(log_lik_across_folds)
    merged_pred = np.array([int(np.argmax(np.mean(log_lik_across_folds[b], axis=0)))
                            for b in merged_bags])
    merged_true = np.array([true_label_by_bag[b] for b in merged_bags])
    merged_acc = float((merged_true == merged_pred).mean())
    print(f"merged (duplicate-averaged, n={len(merged_bags)} unique bags) bag_acc: {merged_acc:.4f}")

    # Binary-only sensitivity/specificity (per-class recall) and AUC, same
    # convention as scripts/stage6_bag_eval.py -- the continuous score is a
    # softmax of the (duplicate-averaged) log-likelihood vector, since
    # score_bayes_mil_blind returns log p(X_bag | Y_b=k), not a normalized
    # probability.
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score,
    )
    extra_metrics = {}
    if len(set(merged_true.tolist())) == 2:
        merged_ll = np.array([np.mean(log_lik_across_folds[b], axis=0) for b in merged_bags])
        merged_ll = merged_ll - merged_ll.max(axis=1, keepdims=True)
        merged_proba = np.exp(merged_ll) / np.exp(merged_ll).sum(axis=1, keepdims=True)
        recall_per_class = recall_score(merged_true, merged_pred, average=None, zero_division=0)
        extra_metrics['bag_balanced_accuracy'] = float(balanced_accuracy_score(merged_true, merged_pred))
        extra_metrics['bag_sensitivity'] = float(recall_per_class[1])
        extra_metrics['bag_specificity'] = float(recall_per_class[0])
        try:
            extra_metrics['bag_auroc'] = float(roc_auc_score(merged_true, merged_proba[:, 1]))
            extra_metrics['bag_auprc'] = float(average_precision_score(merged_true, merged_proba[:, 1]))
        except ValueError:
            pass
        print(f"  sensitivity={extra_metrics.get('bag_sensitivity', float('nan')):.4f}  "
             f"specificity={extra_metrics.get('bag_specificity', float('nan')):.4f}  "
             f"AUC={extra_metrics.get('bag_auroc', float('nan')):.4f}")

    print(f"\n{'='*70}\nTIER 0 COMPARISON (reference_model={reference_model})\n{'='*70}")
    from scripts.stage6_bag_eval import _tier0_reference
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

    report = {'config': config_name, 'per_fold': per_fold,
             'pooled_bag_acc_no_dedup': overall_acc,
             'merged_bag_acc': merged_acc, 'n_merged_bags': len(merged_bags),
             **extra_metrics,
             'tier0_reference': {'model': tier0_label, 'regimen': 'TRTR', 'rule': 'majority',
                                 'n_bags': len(common), 'bag_accuracy': tier0_acc},
             'vs_tier0': {'boot_mean_diff': boot['boot_mean_diff'],
                         'ci_diff_lo': boot['ci_diff_lo'], 'ci_diff_hi': boot['ci_diff_hi'],
                         'p_bootstrap': boot['p_bootstrap'],
                         'significant_95': boot['significant_95']}}
    if out_path is None:
        out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'latent' / 'bayes_blind_bag_classification.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 5 blind (non-circular) bag classification")
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-chains', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=600)
    parser.add_argument('--burn-in', type=int, default=200)
    parser.add_argument('--thin', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--reference-model', default='apollo', choices=['apollo', 'rf', 'xgb', 'cnn'])
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()
    run(args.config, n_chains=args.n_chains, n_iter=args.n_iter, burn_in=args.burn_in,
       thin=args.thin, seed=args.seed, out_path=args.out_path,
       reference_model=args.reference_model, n_boot=args.n_boot)
