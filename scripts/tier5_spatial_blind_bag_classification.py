"""
Real-data test of the spatial (Potts-coupled) Tier 5 extension
(src/mil/bayes_spatial.py), mirroring scripts/tier5_blind_bag_classification.py
exactly (same data prep, same genuine blind -- never given Y_b --
classification protocol) but swept over several fixed J values per fold,
since tests/test_bayes_spatial.py found accuracy is NOT monotonic in J on
synthetic data (J=0.5/1.0 hurt, J=1.5+ helped at that iteration budget) --
there is no a-priori right J for real data, so this reports the whole
response curve rather than a single guessed value.

J=0 is included deliberately as a built-in control: it should reproduce
scripts/tier5_blind_bag_classification.py's already-measured per-fold
numbers (mean 0.710, pooled 0.698, docs/tier5_bayesian_mil.md section 4.3),
confirming this is the same underlying fit machinery, not a different one,
before trusting any J>0 result against it.

Usage:
    python scripts/tier5_spatial_blind_bag_classification.py --config exp_608
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
from src.mil.bayes_spatial import fit_bayes_mil_spatial, score_bayes_mil_spatial_blind


def run(config_name, J_values=(0.0, 1.0, 2.0, 4.0), n_folds=5, pca_components=10,
       n_chains=3, n_iter=600, burn_in=200, thin=5, seed=42, out_path=None):
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    n_classes = num_classes_of(config)
    loader = make_loader(config)

    per_J = {J: {'per_fold': {}} for J in J_values}

    for fold_data in loader:
        fold_name = fold_data['fold_name']
        fd = prepare_fold(fold_data, config, cap_train=False)

        pca = PCA(n_components=pca_components, random_state=seed)
        Z_train = pca.fit_transform(fd.X_train)
        Z_val = pca.transform(fd.X_val)
        true_label = {b: int(fd.y_val[fd.val_bag_ids == b][0]) for b in np.unique(fd.val_bag_ids)}

        for J in J_values:
            print(f"\n{'='*70}\n{fold_name}  J={J}\n{'='*70}")
            best_fit, best_ll = None, -np.inf
            chain_rhats_info = []
            for c in range(n_chains):
                fit_c = fit_bayes_mil_spatial(
                    Z_train, fd.train_bag_ids, fd.y_train, fd.train_coords, n_classes, J,
                    n_iter=n_iter, burn_in=burn_in, thin=thin, seed=seed + 1000 * c,
                    verbose=False)
                # same marginal-log-likelihood ranking convention as fit_bayes_mil_chains,
                # ignoring the spatial term (comparable, model-agnostic chain ranking)
                from src.mil.bayes import marginal_log_likelihood
                ll_c = marginal_log_likelihood(fit_c, Z_train, fd.train_bag_ids, fd.y_train, n_classes)
                chain_rhats_info.append(ll_c)
                if ll_c > best_ll:
                    best_ll, best_fit = ll_c, fit_c
            print(f"  chain log-liks: {[round(v, 1) for v in chain_rhats_info]}")

            log_lik_by_bag = score_bayes_mil_spatial_blind(
                best_fit, Z_val, fd.val_bag_ids, fd.val_coords, n_classes, J,
                n_mini_iter=150, n_mini_burn=50, seed=0)

            fold_true, fold_pred = [], []
            for b, ll in log_lik_by_bag.items():
                pred = int(np.argmax(ll))
                fold_true.append(true_label[b])
                fold_pred.append(pred)
            fold_true, fold_pred = np.array(fold_true), np.array(fold_pred)
            acc = float((fold_true == fold_pred).mean())
            per_J[J]['per_fold'][fold_name] = {
                'n_bags': len(fold_true), 'bag_acc': acc,
                'chain_log_liks': [float(v) for v in chain_rhats_info],
            }
            print(f"  BLIND bag_acc (J={J}) = {acc:.4f}  (n={len(fold_true)})")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for J in J_values:
        accs = [v['bag_acc'] for v in per_J[J]['per_fold'].values()]
        print(f"J={J}: mean per-fold bag_acc = {np.mean(accs):.4f}   per-fold = "
              f"{[round(a, 3) for a in accs]}")

    if out_path is None:
        out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'latent' / 'bayes_spatial_blind_bag_classification.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({str(J): v for J, v in per_J.items()}, f, indent=2)
    print(f"\nReport written -> {out_path}")
    return per_J


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 5 spatial blind bag classification, J sweep")
    parser.add_argument('--config', required=True)
    parser.add_argument('--J-values', type=float, nargs='+', default=[0.0, 1.0, 2.0, 4.0])
    parser.add_argument('--n-chains', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=600)
    parser.add_argument('--burn-in', type=int, default=200)
    parser.add_argument('--thin', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()
    run(args.config, J_values=tuple(args.J_values), n_chains=args.n_chains,
       n_iter=args.n_iter, burn_in=args.burn_in, thin=args.thin, seed=args.seed,
       out_path=args.out_path)
