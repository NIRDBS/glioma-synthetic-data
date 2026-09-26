"""
Stage: MIL formulation fitting (docs/MIL_REFRAME_PLAN.md Stage 4).

Fits Tier 1 (mi-SVM/MI-SVM) and/or Tier 2 (EM) on each CV fold's TRAINING bags
only, scores that same fold's training AND validation instances with the
fold's own model, and writes both to the latent-label store
(results/{exp}/latent/latent_labels.h5) with the leakage-safe in_sample flag
(docs/MIL_REFRAME_PLAN.md §3).

No patient_cap here -- Stage 4 retires it in favour of using every available
instance for latent inference (bag-size-aware pooling is the model's own job,
via pi_b's per-bag n_b-dependent shrinkage, not a preprocessing-time cap).
"""
import json
from pathlib import Path

import numpy as np

from pipeline._common import make_loader, num_classes_of, prepare_fold, set_seeds
from src.mil.em import fit_em, score_em
from src.mil.misvm import fit_misvm, predict_misvm
from src.mil.store import write_posteriors

METHODS = ('misvm', 'em')


def _config_hash(config):
    try:
        return str(hash(json.dumps(config, sort_keys=True, default=str)))
    except Exception:
        return ''


def _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                         max_iters, seed):
    full_bag_ids = loader.groups
    full_coords = loader.coords
    n_full = len(full_bag_ids)
    config_hash = _config_hash(config)

    for fold_data in loader:
        fold_name = fold_data['fold_name']
        fd = prepare_fold(fold_data, config, cap_train=False)

        train_tissue_cluster = fold_data.get('train_tissue_cluster')
        val_tissue_cluster = fold_data.get('val_tissue_cluster')

        print(f"\n  [{method}] fold={fold_name}  "
              f"train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")

        if method == 'misvm':
            fit = fit_misvm(fd.X_train, fd.train_bag_ids, fd.y_train, n_classes,
                            pi_min=pi_min, variant='mi_svm', max_iters=max_iters, seed=seed)
            train_posterior = fit['posterior']
            val_posterior = predict_misvm(fit, fd.X_val)
            variant = 'mi_svm'
        elif method == 'em':
            fit = fit_em(fd.X_train, fd.train_bag_ids, fd.y_train, n_classes,
                        tissue_cluster=train_tissue_cluster, pi_min=pi_min,
                        max_iters=max_iters, seed=seed)
            train_posterior = fit['posterior']
            val_posterior = score_em(fit, fd.X_val, fd.val_bag_ids, fd.y_val, n_classes)
            variant = 'constrained'
            print(f"    shrinkage_ratio={fit['shrinkage']['shrinkage_ratio']:.3f}  "
                  f"(collapsed={fit['shrinkage']['collapsed']})  "
                  f"null_effective_n={fit['null_collapse']['effective_n_components']:.2f}")
        else:
            raise ValueError(f"unknown MIL method {method!r}")

        # Scatter fold-relative posteriors back onto the FULL dataset's row
        # order (§3: "rows aligned 1:1 to the main HDF5 row order"), matched
        # by absolute position via the same train/val bag-id masks the loader
        # itself used to build this fold (fold_data['train_sample_ids'] /
        # ['val_sample_ids'] are exactly those bag id sets).
        train_mask_full = np.isin(full_bag_ids, fold_data['train_sample_ids'])
        val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])

        K1 = train_posterior.shape[1]
        posterior_full = np.full((n_full, K1), np.nan, dtype=np.float32)
        in_sample_full = np.zeros(n_full, dtype=bool)

        posterior_full[train_mask_full] = train_posterior
        in_sample_full[train_mask_full] = True
        posterior_full[val_mask_full] = val_posterior
        in_sample_full[val_mask_full] = False

        write_posteriors(
            store_path, method=method, variant=variant, fold_name=fold_name,
            posterior=posterior_full, in_sample=in_sample_full,
            sample_ids=full_bag_ids, coords=full_coords, config_hash=config_hash,
        )
        print(f"    -> wrote {method}/{variant}/{fold_name} to {store_path}")


def mil_fit_stage(config, output_dir=None, methods=METHODS, pi_min=None, max_iters=10,
                  seed=None):
    seed = seed if seed is not None else set_seeds(config)
    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)

    store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
        / 'latent' / 'latent_labels.h5'

    print(f"\n{'='*80}\nMIL FIT STAGE: {config['name']}\n{'='*80}")
    print(f"  methods: {methods}  pi_min={pi_min}  n_classes={n_classes}")
    print(f"  store -> {store_path}")

    loader = make_loader(config)

    results = {}
    for method in methods:
        _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                             max_iters, seed)
        results[method] = str(store_path)

    print(f"\n  MIL fit stage complete -> {store_path}")
    return {'store_path': str(store_path), 'methods': list(methods)}


if __name__ == '__main__':
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Fit MIL formulations (mi-SVM, EM)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_501")
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    parser.add_argument('--pi-min', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=10)
    args = parser.parse_args()

    from src.notebooks.experiment_utils import setup_experiment
    cfg, out_dir = setup_experiment(args.config)
    mil_fit_stage(cfg, out_dir, methods=args.methods, pi_min=args.pi_min,
                 max_iters=args.max_iters)
