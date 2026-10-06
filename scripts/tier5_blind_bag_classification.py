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


def run(config_name, n_folds=5, pca_components=10, n_chains=3, n_iter=600,
       burn_in=200, thin=5, seed=42, out_path=None):
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    n_classes = num_classes_of(config)
    loader = make_loader(config)

    per_fold = {}
    all_true, all_pred, all_bag_ids = [], [], []

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

    report = {'config': config_name, 'per_fold': per_fold,
             'pooled_bag_acc_no_dedup': overall_acc}
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
    args = parser.parse_args()
    run(args.config, n_chains=args.n_chains, n_iter=args.n_iter, burn_in=args.burn_in,
       thin=args.thin, seed=args.seed, out_path=args.out_path)
