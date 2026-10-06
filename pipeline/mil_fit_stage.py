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
from src.mil.generative import instance_posteriors_from_cvae, compute_z_mean, latent_rf_posteriors
from src.mil.misvm import fit_misvm, predict_misvm
from src.mil.store import write_posteriors

METHODS = ('misvm', 'em')
CBR_VARIANTS = ('reconstruction', 'encoder_head')
ALL_METHODS = METHODS + ('cbr', 'latent_rf', 'bayes_mil')


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

    em_diagnostics = None
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
        # pi_b/mu/sigma/shrinkage/null_collapse/history were computed and
        # immediately discarded here in every prior run -- only the collapsed
        # (n, K+1) posterior ever reached the store. Persisted from here on
        # (mil_fit_stage's caller writes this to em_diagnostics.json) so the
        # backlog's pi_b-external-validity and sigma-identifiability checks
        # (docs/MIL_REFRAME_PLAN.md §13) don't need a from-scratch refit again.
        em_diagnostics = {
            'pi_b': {str(b): float(v) for b, v in fit['pi_b'].items()},
            'mu': fit['mu'], 'sigma': fit['sigma'],
            'shrinkage': fit['shrinkage'],
            'null_collapse': {
                'pairwise_cosine': np.asarray(fit['null_collapse']['pairwise_cosine']).tolist(),
                'effective_n_components': fit['null_collapse']['effective_n_components'],
            },
            'history': fit['history'],
        }
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

    return fold_name, variant, posterior_full, in_sample_full, em_diagnostics


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


def _score_one_fold_latent_rf(fold_data, config, n_classes, full_bag_ids, full_coords,
                              cvae_weights_dir, rf_kwargs, seed):
    """
    A third CVAE-derived Tier 4 variant: fit a plain RandomForestClassifier on
    the frozen encoder's z_mean for this fold (src/mil/generative.py::
    latent_rf_posteriors), using Tier 0's own RF hyperparameters
    (``rf_kwargs``) for a same-classifier, different-features comparison
    against Tier 0's raw-spectrum RF. Unlike ``_score_one_fold_cbr``, this
    DOES fit something per fold -- mirrors ``_fit_one_fold``'s contract
    (full-dataset scatter-back) rather than being "free" inference.
    """
    from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder

    fold_name = fold_data['fold_name']
    fd = prepare_fold(fold_data, config, cap_train=False)
    n_features = fd.X_train.shape[1]

    print(f"\n  [latent_rf] fold={fold_name}  train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")

    cvae_config = build_cvae_config(config, n_features)
    encoder, _ = load_encoder_decoder(cvae_config, cvae_weights_dir, fold_name)

    z_mean_train = compute_z_mean(fd.X_train, encoder)
    z_mean_val = compute_z_mean(fd.X_val, encoder)
    posteriors = latent_rf_posteriors(z_mean_train, fd.y_train, z_mean_val, rf_kwargs, seed=seed)

    train_mask_full = np.isin(full_bag_ids, fold_data['train_sample_ids'])
    val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
    n_full = len(full_bag_ids)
    K1 = posteriors['train'].shape[1]

    posterior_full = np.full((n_full, K1), np.nan, dtype=np.float32)
    in_sample_full = np.zeros(n_full, dtype=bool)
    posterior_full[train_mask_full] = posteriors['train']
    in_sample_full[train_mask_full] = True
    posterior_full[val_mask_full] = posteriors['val']
    in_sample_full[val_mask_full] = False

    import tensorflow as tf
    tf.keras.backend.clear_session()
    return fold_name, [('rf', posterior_full, in_sample_full)]


def _fit_one_fold_bayes_mil(fold_data, config, n_classes, full_bag_ids, full_coords,
                            seed, bayes_cfg):
    """
    Tier 5 (src/mil/bayes.py): the constrained-MIL model fit by Gibbs, the only
    tier that returns a posterior rather than a point estimate.

    PCA first, fit on this fold's TRAINING instances only and applied to both
    sides -- a full-covariance Gaussian in 571 bands is neither tractable nor
    identifiable at this sample size, so reduction is a precondition of the
    model, and fitting it on train only keeps the usual leakage discipline
    (section 2). PCA deliberately, NOT the CVAE's latent: Tier 4's `latent_rf`
    showed the CVAE latent is worse than raw spectra through the same
    classifier, and this tier should not inherit that.

    Returns the same per-fold contract as the other dispatches, plus a
    diagnostics dict (R-hat, per-chain log-likelihoods, pi_b credible
    intervals) -- the honest-failure signal this tier exists for.
    """
    from sklearn.decomposition import PCA
    from src.mil.bayes import fit_bayes_mil_chains, score_bayes_mil

    fold_name = fold_data['fold_name']
    fd = prepare_fold(fold_data, config, cap_train=False)

    n_components = int(bayes_cfg.get('pca_components', 10))
    pca = PCA(n_components=n_components, random_state=seed)
    Z_train = pca.fit_transform(fd.X_train)
    Z_val = pca.transform(fd.X_val)
    evr = float(pca.explained_variance_ratio_.sum())

    print(f"\n  [bayes_mil] fold={fold_name}  train={fd.X_train.shape[0]}  "
          f"val={fd.X_val.shape[0]}  PCA {n_components}d (EVR={evr:.3f})")

    fit = fit_bayes_mil_chains(
        Z_train, fd.train_bag_ids, fd.y_train, n_classes,
        n_chains=int(bayes_cfg.get('n_chains', 3)),
        n_iter=int(bayes_cfg.get('n_iter', 600)),
        burn_in=int(bayes_cfg.get('burn_in', 200)),
        thin=int(bayes_cfg.get('thin', 5)),
        seed=seed, verbose=True)

    val_posterior = score_bayes_mil(fit, Z_val, fd.val_bag_ids, fd.y_val, n_classes)

    train_mask_full = np.isin(full_bag_ids, fold_data['train_sample_ids'])
    val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
    n_full = len(full_bag_ids)
    K1 = fit['posterior'].shape[1]

    posterior_full = np.full((n_full, K1), np.nan, dtype=np.float32)
    in_sample_full = np.zeros(n_full, dtype=bool)
    posterior_full[train_mask_full] = fit['posterior']
    in_sample_full[train_mask_full] = True
    posterior_full[val_mask_full] = val_posterior
    in_sample_full[val_mask_full] = False

    pi_ci = {str(b): [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
             for b, v in fit['pi_b_samples'].items()}
    diagnostics = {
        'pca_components': n_components, 'pca_explained_variance_ratio': evr,
        'chain_rhat': float(fit['chain_rhat']),
        'chain_log_liks': [float(v) for v in fit['chain_log_liks']],
        'n_chains': fit['n_chains'], 'n_retained_draws': fit['n_retained'],
        'metropolis_accept_rate': float(fit['accept_rate']),
        'beta_a': float(fit['a']), 'beta_b': float(fit['b']),
        'pi_b_mean': {str(k): float(v) for k, v in fit['pi_b'].items()},
        'pi_b_95ci': pi_ci,
    }
    print(f"    R-hat={fit['chain_rhat']:.3f}  mean pi_b="
          f"{np.mean(list(fit['pi_b'].values())):.4f}  "
          f"median 95%CI width="
          f"{np.median([hi - lo for lo, hi in pi_ci.values()]):.4f}")

    return fold_name, [('gibbs', posterior_full, in_sample_full)], diagnostics


def _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                         max_iters, seed, svm_subsample_per_bag=500,
                         cvae_weights_dir=None, cbr_temperature=1.0,
                         rf_kwargs=None, bayes_cfg=None, em_diagnostics_path=None):
    full_bag_ids = loader.groups
    full_coords = loader.coords
    config_hash = _config_hash(config)

    em_diagnostics_by_fold = {}
    bayes_diagnostics_by_fold = {}
    if em_diagnostics_path is None and method == 'em':
        em_diagnostics_path = Path(store_path).parent / 'em_diagnostics.json'

    for fold_data in loader:
        em_diagnostics = None
        if method == 'cbr':
            fold_name, variant_results = _score_one_fold_cbr(
                fold_data, config, n_classes, full_bag_ids, full_coords,
                cvae_weights_dir, cbr_temperature)
        elif method == 'latent_rf':
            fold_name, variant_results = _score_one_fold_latent_rf(
                fold_data, config, n_classes, full_bag_ids, full_coords,
                cvae_weights_dir, rf_kwargs or {}, seed)
        elif method == 'bayes_mil':
            fold_name, variant_results, bayes_diag = _fit_one_fold_bayes_mil(
                fold_data, config, n_classes, full_bag_ids, full_coords,
                seed, bayes_cfg or {})
            bayes_diagnostics_by_fold[fold_name] = bayes_diag
            bayes_path = Path(store_path).parent / 'bayes_diagnostics.json'
            bayes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bayes_path, 'w') as f:
                json.dump(bayes_diagnostics_by_fold, f, indent=2)
            print(f"    -> wrote bayes_diagnostics/{fold_name} to {bayes_path}")
        else:
            fold_name, variant, posterior_full, in_sample_full, em_diagnostics = _fit_one_fold(
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

        if em_diagnostics is not None:
            em_diagnostics_by_fold[fold_name] = em_diagnostics
            em_diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(em_diagnostics_path, 'w') as f:
                json.dump(em_diagnostics_by_fold, f, indent=2)
            print(f"    -> wrote em_diagnostics/{fold_name} to {em_diagnostics_path}")


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
        if {'cbr', 'latent_rf', 'bayes_mil'} & set(methods):
            raise ValueError(
                "backend='modal' does not support cbr/latent_rf/bayes_mil yet -- run them "
                "with backend='local' (see docs/stage6_generative_mil.md)")
        from pipeline.modal_mil_fit import run_mil_fit_parallel
        return run_mil_fit_parallel(config, methods=methods, pi_min=pi_min,
                                    max_iters=max_iters, seed=seed)

    seed = seed if seed is not None else set_seeds(config)
    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    svm_subsample_per_bag = mil_cfg.get('svm_subsample_per_bag', 500)
    cbr_temperature = mil_cfg.get('cbr_temperature', 1.0)
    rf_kwargs = config['models'].get('random_forest', {})
    bayes_cfg = mil_cfg.get('bayes', {})
    cvae_weights_dir = config['paths'].get('models_cvae_dir')

    store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
        / 'latent' / 'latent_labels.h5'

    print(f"\n{'='*80}\nMIL FIT STAGE: {config['name']}\n{'='*80}")
    print(f"  methods: {methods}  pi_min={pi_min}  n_classes={n_classes}  "
          f"svm_subsample_per_bag={svm_subsample_per_bag}")
    if 'cbr' in methods:
        print(f"  cbr_temperature={cbr_temperature}  cvae_weights_dir={cvae_weights_dir}")
    if 'latent_rf' in methods:
        print(f"  latent_rf rf_kwargs={rf_kwargs}  cvae_weights_dir={cvae_weights_dir}")
    if 'bayes_mil' in methods:
        print(f"  bayes_mil cfg={bayes_cfg or '(defaults)'}")
    print(f"  store -> {store_path}")

    loader = make_loader(config)

    results = {}
    for method in methods:
        _fit_one_formulation(method, loader, config, n_classes, store_path, pi_min,
                             max_iters, seed, svm_subsample_per_bag=svm_subsample_per_bag,
                             cvae_weights_dir=cvae_weights_dir, cbr_temperature=cbr_temperature,
                             rf_kwargs=rf_kwargs, bayes_cfg=bayes_cfg)
        results[method] = str(store_path)

    print(f"\n  MIL fit stage complete -> {store_path}")
    return {'store_path': str(store_path), 'methods': list(methods)}


if __name__ == '__main__':
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Fit MIL formulations (mi-SVM, EM, cbr, latent_rf)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_501")
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(ALL_METHODS),
                        help="default is mi-SVM+EM only; 'cbr' (Tier 4, inference-only, no "
                             "training) and 'latent_rf' (Tier 4, a RandomForest fit on the "
                             "frozen CVAE encoder's z_mean) must be requested explicitly")
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
