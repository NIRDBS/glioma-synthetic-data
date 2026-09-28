"""
Stage: MIL formulation fitting (docs/MIL_REFRAME_PLAN.md Stage 4; Stage 6 adds
Tier 4, 'cbr', below).

Fits Tier 1 (mi-SVM/MI-SVM) and/or Tier 2 (EM) on each CV fold's TRAINING bags
only, scores that same fold's training AND validation instances with the
fold's own model, and writes both to the latent-label store
(results/{exp}/latent/latent_labels.h5) with the leakage-safe in_sample flag
(docs/MIL_REFRAME_PLAN.md §3).

No patient_cap here -- Stage 4 retires it in favour of using every available
instance for latent inference (bag-size-aware pooling is the model's own job,
via pi_b's per-bag n_b-dependent shrinkage, not a preprocessing-time cap).

Tier 4 ('cbr') is not a *fit* at all -- it scores every fold's instances with
that fold's own already-trained CVAE weights (src/mil/generative.py), so it
slots into this same per-fold/store-writing machinery via
``_score_one_fold_cbr`` rather than ``_fit_one_fold``, but produces TWO
variants ('reconstruction', 'encoder_head') from one encoder forward pass per
fold instead of one. It is NOT in the default ``METHODS`` tuple (deliberately
-- see ``_fit_one_formulation``'s dispatch) so existing callers that rely on
``mil_fit_stage``'s bare default (``pipeline/run_suite.py``) are unaffected;
request it explicitly with ``methods=('cbr',)`` or ``--methods cbr``.
"""
import json
from pathlib import Path

import numpy as np

from pipeline._common import make_loader, num_classes_of, prepare_fold, set_seeds
from src.mil.em import fit_em, score_em
from src.mil.generative import instance_posteriors_from_cvae
from src.mil.misvm import fit_misvm, predict_misvm
from src.mil.store import write_posteriors

METHODS = ('misvm', 'em')
CBR_VARIANTS = ('reconstruction', 'encoder_head')
ALL_METHODS = METHODS + ('cbr',)


def _config_hash(config):
    try:
        return str(hash(json.dumps(config, sort_keys=True, default=str)))
    except Exception:
        return ''


def _fit_one_fold(fold_data, config, method, n_classes, pi_min, max_iters, seed,
                  full_bag_ids, full_coords, svm_subsample_per_bag=500):
    """
    Fit one MIL formulation on one fold and return its posteriors scattered
    onto the full dataset's row order -- everything ``_fit_one_formulation``
    needs per fold, factored out so a single fold's work is independently
    callable (this is exactly what a parallel-fold Modal container calls --
    see pipeline/modal_mil_fit.py -- as well as the plain local loop below).

    Returns (fold_name, variant, posterior_full, in_sample_full) -- no I/O,
    the caller (local loop or Modal driver) decides how/where to write it.
    """
    fold_name = fold_data['fold_name']
    fd = prepare_fold(fold_data, config, cap_train=False)
    train_tissue_cluster = fold_data.get('train_tissue_cluster')

    print(f"\n  [{method}] fold={fold_name}  "
          f"train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")

    if method == 'misvm':
        fit = fit_misvm(fd.X_train, fd.train_bag_ids, fd.y_train, n_classes,
                        pi_min=pi_min, variant='mi_svm', max_iters=max_iters, seed=seed,
                        svm_subsample_per_bag=svm_subsample_per_bag)
        train_posterior = fit['posterior']
        val_posterior = predict_misvm(fit, fd.X_val)
        variant = 'mi_svm'
    elif method == 'em':
        fit = fit_em(fd.X_train, fd.train_bag_ids, fd.y_train, n_classes,
                    tissue_cluster=train_tissue_cluster, pi_min=pi_min,
                    max_iters=max_iters, seed=seed,
                    svm_subsample_per_bag=svm_subsample_per_bag)
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

    n_full = len(full_bag_ids)
    K1 = train_posterior.shape[1]
    posterior_full = np.full((n_full, K1), np.nan, dtype=np.float32)
    in_sample_full = np.zeros(n_full, dtype=bool)

    posterior_full[train_mask_full] = train_posterior
    in_sample_full[train_mask_full] = True
    posterior_full[val_mask_full] = val_posterior
    in_sample_full[val_mask_full] = False

    return fold_name, variant, posterior_full, in_sample_full


def _score_one_fold_cbr(fold_data, config, n_classes, full_bag_ids, full_coords,
                        cvae_weights_dir, temperature):
    """
    Tier 4 (docs/MIL_REFRAME_PLAN.md Stage 6): score one fold's full train+val
    instance set with that fold's own trained CVAE, producing BOTH
    'reconstruction' and 'encoder_head' variants. No training happens here --
    both variants score with the CVAE weights the ``cvae`` stage already
    saved (results/{exp}/models/cvae/{encoder,decoder}_{fold}.weights.h5),
    the same weights Classification by Reconstruction already reuses in
    ``pipeline/evaluation_stage.py``.

    Mirrors ``_fit_one_fold``'s per-fold contract (full-dataset scatter-back,
    same train/val masks from the loader's own fold assignment) but returns a
    LIST of (variant, posterior_full, in_sample_full) -- two entries instead
    of one -- since one encoder pass yields both variants' raw ingredients
    (src/mil/generative.py::instance_posteriors_from_cvae).
    """
    from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder

    fold_name = fold_data['fold_name']
    fd = prepare_fold(fold_data, config, cap_train=False)
    n_features = fd.X_train.shape[1]

    print(f"\n  [cbr] fold={fold_name}  train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")

    cvae_config = build_cvae_config(config, n_features)
    encoder, decoder = load_encoder_decoder(cvae_config, cvae_weights_dir, fold_name)

    train_posteriors = instance_posteriors_from_cvae(
        fd.X_train, encoder, decoder, n_classes, temperature=temperature)
    val_posteriors = instance_posteriors_from_cvae(
        fd.X_val, encoder, decoder, n_classes, temperature=temperature)

    train_mask_full = np.isin(full_bag_ids, fold_data['train_sample_ids'])
    val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
    n_full = len(full_bag_ids)

    in_sample_full = np.zeros(n_full, dtype=bool)
    in_sample_full[train_mask_full] = True
    in_sample_full[val_mask_full] = False

    results = []
    for variant in CBR_VARIANTS:
        K1 = train_posteriors[variant].shape[1]
        posterior_full = np.full((n_full, K1), np.nan, dtype=np.float32)
        posterior_full[train_mask_full] = train_posteriors[variant]
        posterior_full[val_mask_full] = val_posteriors[variant]
        results.append((variant, posterior_full, in_sample_full))

    import tensorflow as tf
    tf.keras.backend.clear_session()
    return fold_name, results


def _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                         max_iters, seed, svm_subsample_per_bag=500,
                         cvae_weights_dir=None, cbr_temperature=1.0):
    full_bag_ids = loader.groups
    full_coords = loader.coords
    config_hash = _config_hash(config)

    for fold_data in loader:
        if method == 'cbr':
            fold_name, variant_results = _score_one_fold_cbr(
                fold_data, config, n_classes, full_bag_ids, full_coords,
                cvae_weights_dir, cbr_temperature)
        else:
            fold_name, variant, posterior_full, in_sample_full = _fit_one_fold(
                fold_data, config, method, n_classes, pi_min, max_iters, seed,
                full_bag_ids, full_coords, svm_subsample_per_bag=svm_subsample_per_bag)
            variant_results = [(variant, posterior_full, in_sample_full)]

        for variant, posterior_full, in_sample_full in variant_results:
            write_posteriors(
                store_path, method=method, variant=variant, fold_name=fold_name,
                posterior=posterior_full, in_sample=in_sample_full,
                sample_ids=full_bag_ids, coords=full_coords, config_hash=config_hash,
            )
            print(f"    -> wrote {method}/{variant}/{fold_name} to {store_path}")


def mil_fit_stage(config, output_dir=None, methods=METHODS, pi_min=None, max_iters=10,
                  seed=None, backend='local'):
    """
    backend='local' (default): fit all folds sequentially in this process.
    backend='modal': fit every (method, fold) pair in its own parallel Modal
                     container (pipeline/modal_mil_fit.py) -- each fold is an
                     independent fit (no shared state), so this is CPU-only
                     parallelism across folds, not a GPU need; see
                     docs/stage4_mil.md for why this was chosen over local
                     multiprocessing (this machine's free RAM does not cover
                     5 folds' worth of independent data copies at once).
    """
    if backend == 'modal':
        if 'cbr' in methods:
            raise ValueError(
                "backend='modal' does not support method='cbr' yet -- it is inference-only "
                "and cheap enough to run with backend='local' (see docs/stage6_generative_mil.md)")
        from pipeline.modal_mil_fit import run_mil_fit_parallel
        return run_mil_fit_parallel(config, methods=methods, pi_min=pi_min,
                                    max_iters=max_iters, seed=seed)

    seed = seed if seed is not None else set_seeds(config)
    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    svm_subsample_per_bag = mil_cfg.get('svm_subsample_per_bag', 500)
    cbr_temperature = mil_cfg.get('cbr_temperature', 1.0)
    cvae_weights_dir = config['paths'].get('models_cvae_dir')

    store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
        / 'latent' / 'latent_labels.h5'

    print(f"\n{'='*80}\nMIL FIT STAGE: {config['name']}\n{'='*80}")
    print(f"  methods: {methods}  pi_min={pi_min}  n_classes={n_classes}  "
          f"svm_subsample_per_bag={svm_subsample_per_bag}")
    if 'cbr' in methods:
        print(f"  cbr_temperature={cbr_temperature}  cvae_weights_dir={cvae_weights_dir}")
    print(f"  store -> {store_path}")

    loader = make_loader(config)

    results = {}
    for method in methods:
        _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                             max_iters, seed, svm_subsample_per_bag=svm_subsample_per_bag,
                             cvae_weights_dir=cvae_weights_dir, cbr_temperature=cbr_temperature)
        results[method] = str(store_path)

    print(f"\n  MIL fit stage complete -> {store_path}")
    return {'store_path': str(store_path), 'methods': list(methods)}


if __name__ == '__main__':
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Fit MIL formulations (mi-SVM, EM, cbr)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_501")
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(ALL_METHODS),
                        help="default is mi-SVM+EM only; 'cbr' (Tier 4, inference-only, no "
                             "training) must be requested explicitly")
    parser.add_argument('--pi-min', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--backend', default='local', choices=['local', 'modal'],
                        help="'modal' fits every (method, fold) pair in its own parallel "
                             "CPU container instead of one sequential local loop")
    args = parser.parse_args()

    from src.notebooks.experiment_utils import setup_experiment
    cfg, out_dir = setup_experiment(args.config)
    mil_fit_stage(cfg, out_dir, methods=args.methods, pi_min=args.pi_min,
                 max_iters=args.max_iters, backend=args.backend)
