"""
Stage 6, Tier 3: gated-attention MIL, trained per fold on a frozen CVAE
encoder trunk (docs/MIL_REFRAME_PLAN.md §4, backlog §13).

Unlike Tier 4 ('cbr', pure inference), this tier genuinely trains a small
instance head + attention branch per fold, on randomly-resampled sub-bags
(fresh draw every epoch, matching the per-epoch resampling pattern already
established in pipeline/cvae_stage.py) drawn only from that fold's own
training bags. Early stopping uses an INNER validation split carved from
those same training bags (pipeline/_common.py::_inner_validation_split,
group-aware -- never the true CV test fold), the same leak-avoidance Stage 3
put in place for the CNN. Scores both training (in-sample) and validation
(out-of-sample) instances with the fold's own trained model and writes both
to the latent-label store, exactly like Tier 1/2/4.

Per the agreed Stage 6 reordering ("Tier 3 attention MIL second with cheap
smoke test"), do not launch a full 5-fold run without a cheap ``--folds``
subset run first (see docs/stage6_generative_mil.md / backlog §13 for the
two real, informative smoke-test failures this caught before a full run).
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from pipeline._common import _inner_validation_split, make_loader, num_classes_of, prepare_fold, set_seeds
from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder
from src.mil.attention import AttentionMIL, required_sub_bag_size, sample_sub_bags, sub_bag_null_rate
from src.mil.store import config_hash, write_posteriors

VARIANT = 'attention'


def _bag_accuracy(model, X, bag_ids, y_true):
    """
    Bag-level accuracy via ``AttentionMIL.bag_posterior`` -- the
    attention-WEIGHTED aggregate, not a naive majority-vote/mean-pool over
    instance_posterior's per-instance values. That naive approach was tried
    first and is a poor fit for this architecture specifically: an
    instance attention learns to downweight during training can develop an
    arbitrary individual prediction (near-zero training gradient regardless
    of what it predicts), so pooling it back in unweighted just reintroduces
    noise the training objective never touched -- confirmed empirically on
    two real smoke tests (fold_0, fold_1: excellent attention-pooled
    training loss, chance-level naive-pooled bag accuracy on the SAME
    model), not assumed. See AttentionMIL.bag_posterior's own docstring.
    """
    bag_ids = np.asarray(bag_ids)
    y_true = np.asarray(y_true)
    bag_id_to_true = {b: int(y_true[bag_ids == b][0]) for b in np.unique(bag_ids)}

    bags_sorted, posterior = model.bag_posterior(X, bag_ids)
    preds = posterior[:, 1:].argmax(axis=1)
    trues = np.array([bag_id_to_true[b] for b in bags_sorted])
    return float((preds == trues).mean())


def fit_one_fold_attention(fold_data, config, n_classes, full_bag_ids, full_coords,
                           cvae_weights_dir, sub_bag_size=None, n_per_bag_per_epoch=50,
                           max_epochs=200, patience=15, pi_min=0.02, epsilon_target=0.01,
                           inner_val_fraction=0.15, seed=42, verbose=True,
                           attn_dim=32, dropout_rate=0.3, l2=1e-3, center_by_bag=True):
    """
    Train Tier 3 on one fold and return its posteriors scattered onto the
    full dataset's row order -- same per-fold contract as
    pipeline/mil_fit_stage.py's _fit_one_fold, so it slots into the same
    store-writing machinery (docs/MIL_REFRAME_PLAN.md §3).

    Returns (fold_name, posterior_full, in_sample_full, diagnostics) --
    diagnostics carries the epoch history, best inner-val bag accuracy, and
    the post-training sub-bag null-rate check (§4.1's named diagnostic).
    """
    fold_name = fold_data['fold_name']
    fd = prepare_fold(fold_data, config, cap_train=False)
    n_features = fd.X_train.shape[1]

    if sub_bag_size is None:
        sub_bag_size = required_sub_bag_size(pi_min, epsilon_target)
    # epsilon actually implied by the size in use (matters if sub_bag_size
    # was overridden rather than derived) -- what the loss's label smoothing
    # and the null-rate diagnostic are both checked against.
    epsilon = (1 - pi_min) ** sub_bag_size

    train_idx, inner_val_idx = _inner_validation_split(
        fd.y_train, bag_ids=fd.train_bag_ids, val_fraction=inner_val_fraction, seed=seed)
    X_fit, y_fit, bag_fit = fd.X_train[train_idx], fd.y_train[train_idx], fd.train_bag_ids[train_idx]
    X_inner_val = fd.X_train[inner_val_idx]
    y_inner_val, bag_inner_val = fd.y_train[inner_val_idx], fd.train_bag_ids[inner_val_idx]

    if verbose:
        print(f"\n  [attention] fold={fold_name}  sub_bag_size={sub_bag_size} "
              f"(epsilon={epsilon:.4f})  fit_bags={len(np.unique(bag_fit))}  "
              f"inner_val_bags={len(np.unique(bag_inner_val))}  "
              f"held_out_val={fd.X_val.shape[0]}")

    cvae_config = build_cvae_config(config, n_features)
    encoder, _ = load_encoder_decoder(cvae_config, cvae_weights_dir, fold_name)
    latent_dim = cvae_config['latent_dim']

    model = AttentionMIL(encoder, latent_dim, n_classes, epsilon=epsilon, attn_dim=attn_dim,
                         dropout_rate=dropout_rate, l2=l2, center_by_bag=center_by_bag)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

    # Early stopping is monitored on INNER-VAL LOSS over sub-bags drawn from
    # the inner-val bags, not raw whole-bag accuracy: with only a handful of
    # inner-val bags (7 on exp_501 fold_0), accuracy moves in coarse jumps
    # (1/7 ~= 0.14 per bag) that are too noisy to reliably tell "got better"
    # from "got worse by chance" -- exactly what made the first real smoke
    # test's signal hard to read. Loss over many sub-bags is continuous and
    # far less noisy. Whole-bag accuracy is still computed and logged every
    # epoch as an interpretable diagnostic, just not what picks the
    # checkpoint.
    best_val_loss, best_weights, patience_left, epoch_log = np.inf, None, patience, []
    for epoch in range(max_epochs):
        X_sub, y_sub, _ = sample_sub_bags(
            X_fit, bag_fit, y_fit, n_per_bag=n_per_bag_per_epoch,
            sub_bag_size=sub_bag_size, seed=seed + epoch)
        y_oh = tf.keras.utils.to_categorical(y_sub, num_classes=n_classes)
        h = model.fit(X_sub, y_oh, epochs=1, batch_size=32, verbose=0)

        X_val_sub, y_val_sub, _ = sample_sub_bags(
            X_inner_val, bag_inner_val, y_inner_val, n_per_bag=n_per_bag_per_epoch,
            sub_bag_size=sub_bag_size, seed=90000 + seed + epoch)  # fixed-offset seed space,
            # deliberately disjoint from the training draws' (seed+epoch) space above
        y_val_oh = tf.keras.utils.to_categorical(y_val_sub, num_classes=n_classes)
        val_metrics = model.evaluate(X_val_sub, y_val_oh, batch_size=32, verbose=0, return_dict=True)

        val_acc = _bag_accuracy(model, X_inner_val, bag_inner_val, y_inner_val)
        epoch_log.append({'epoch': epoch, 'train_loss': float(h.history['loss'][0]),
                          'train_bag_acc_subbags': float(h.history['bag_accuracy'][0]),
                          'inner_val_loss_subbags': float(val_metrics['loss']),
                          'inner_val_bag_acc': val_acc})
        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"    epoch {epoch:3d}: train_loss={h.history['loss'][0]:.4f}  "
                  f"inner_val_loss={val_metrics['loss']:.4f} (best={best_val_loss:.4f})  "
                  f"inner_val_bag_acc={val_acc:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss, best_weights, patience_left = val_metrics['loss'], model.get_weights(), patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"    early stop at epoch {epoch} (patience exhausted)")
                break

    model.set_weights(best_weights)

    train_posterior = model.instance_posterior(fd.X_train, fd.train_bag_ids)
    val_posterior = model.instance_posterior(fd.X_val, fd.val_bag_ids)
    best_val_acc = _bag_accuracy(model, X_inner_val, bag_inner_val, y_inner_val)

    # The attention-weighted bag-level posterior for the TRUE held-out test
    # fold -- what scripts/stage6_bag_eval.py's cross-tier comparison needs,
    # and NOT what instance_posterior/the latent store give it (that naive,
    # attention-free pooling is the exact thing _bag_accuracy's docstring
    # documents as misleading for this architecture). Computed here, while
    # the trained model is still in memory -- it is discarded at the end of
    # this function (tf.keras.backend.clear_session() below) and cannot be
    # reconstructed from the store afterward, so this is the only point
    # this can be captured.
    test_bags_sorted, test_bag_posterior = model.bag_posterior(fd.X_val, fd.val_bag_ids)
    val_bag_ids_arr = np.asarray(fd.val_bag_ids)
    test_bag_true = np.array([
        int(fd.y_val[val_bag_ids_arr == b][0]) for b in test_bags_sorted
    ])
    bag_level_result = {
        'bag_ids': test_bags_sorted.tolist(),
        'posterior': test_bag_posterior.tolist(),
        'y_true': test_bag_true.tolist(),
    }

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

    # §4.1's named diagnostic, using this fold's OWN converged instance
    # posterior on its training bags as the proxy for "which instances are
    # diagnostic" (there is no ground truth -- see
    # [[project_no_tumor_ground_truth]]).
    null_rate = sub_bag_null_rate(
        fd.X_train, fd.train_bag_ids, fd.y_train, train_posterior,
        sub_bag_size=sub_bag_size, n_per_bag=200, seed=seed)

    diagnostics = {
        'sub_bag_size': sub_bag_size, 'epsilon_target': epsilon,
        'best_inner_val_bag_acc': best_val_acc, 'n_epochs_run': len(epoch_log),
        'sub_bag_null_rate_final_model': null_rate, 'epoch_log': epoch_log,
    }
    if verbose:
        print(f"    -> best_epoch inner_val_loss={best_val_loss:.4f}  "
              f"inner_val_bag_acc={best_val_acc:.4f}  "
              f"sub_bag_null_rate={null_rate:.4f} (target epsilon={epsilon:.4f})")

    tf.keras.backend.clear_session()
    return fold_name, posterior_full, in_sample_full, diagnostics, bag_level_result


def mil_attention_stage(config, output_dir=None, sub_bag_size=None, n_per_bag_per_epoch=50,
                        max_epochs=200, patience=15, pi_min=None, seed=None, fold_names=None,
                        attn_dim=32, dropout_rate=0.3, l2=1e-3, center_by_bag=True,
                        loader=None, store_path=None, bag_level_path=None):
    """
    Full run: every fold (or ``fold_names``, for a smoke test on a subset),
    written to the same latent-label store Tier 1/2/4 use, method='attention',
    variant='attention'.

    ``loader``, ``store_path``, ``bag_level_path``: optional overrides, unused
    by the real fit (which builds its own loader and uses the default paths
    below). scripts/stage6_shuffled_label_control.py passes all three
    explicitly -- a loader with its y_class already permuted (built the same
    way _fit_one_formulation's caller does it, before any fold splitting) and
    a separate pair of output paths, so a shuffled run can never collide with
    or corrupt the real store/bag-level artifact. Everything else in this
    function, and all of fit_one_fold_attention, is unchanged either way.
    """
    seed = seed if seed is not None else set_seeds(config)
    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    cvae_weights_dir = config['paths'].get('models_cvae_dir')

    if store_path is None:
        store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
            / 'latent' / 'latent_labels.h5'
    else:
        store_path = Path(store_path)

    print(f"\n{'='*80}\nTIER 3 (ATTENTION MIL) FIT STAGE: {config['name']}\n{'='*80}")
    print(f"  pi_min={pi_min}  n_classes={n_classes}  cvae_weights_dir={cvae_weights_dir}")
    print(f"  attn_dim={attn_dim}  dropout_rate={dropout_rate}  l2={l2}  "
          f"center_by_bag={center_by_bag}")
    print(f"  store -> {store_path}")

    if loader is None:
        loader = make_loader(config)
    full_bag_ids = loader.groups
    full_coords = loader.coords
    cfg_hash = config_hash(config)

    if bag_level_path is None:
        bag_level_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
            / 'latent' / 'attention_bag_level.json'
    else:
        bag_level_path = Path(bag_level_path)

    results = {}
    bag_level_by_fold = {}
    for fold_data in loader:
        if fold_names is not None and fold_data['fold_name'] not in fold_names:
            continue
        fold_name, posterior_full, in_sample_full, diagnostics, bag_level_result = fit_one_fold_attention(
            fold_data, config, n_classes, full_bag_ids, full_coords, cvae_weights_dir,
            sub_bag_size=sub_bag_size, n_per_bag_per_epoch=n_per_bag_per_epoch,
            max_epochs=max_epochs, patience=patience, pi_min=pi_min, seed=seed,
            attn_dim=attn_dim, dropout_rate=dropout_rate, l2=l2, center_by_bag=center_by_bag)

        write_posteriors(
            store_path, method='attention', variant=VARIANT, fold_name=fold_name,
            posterior=posterior_full, in_sample=in_sample_full,
            sample_ids=full_bag_ids, coords=full_coords, config_hash=cfg_hash,
        )
        print(f"    -> wrote attention/{VARIANT}/{fold_name} to {store_path}")
        results[fold_name] = diagnostics
        bag_level_by_fold[fold_name] = bag_level_result

        # Written after EVERY fold, not just at the end: on a 5-fold run
        # spanning hours, this means a fold's attention-weighted bag-level
        # result survives even if a later fold crashes or the run is
        # stopped -- the per-instance store above already has this property
        # (write_posteriors runs per fold); this keeps the same guarantee
        # for the bag-level artifact scripts/stage6_bag_eval.py needs.
        bag_level_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bag_level_path, 'w') as f:
            json.dump(bag_level_by_fold, f, indent=2)

    print(f"\n  Tier 3 fit stage complete -> {store_path}")
    print(f"  Attention-weighted bag-level results -> {bag_level_path}")
    return {'store_path': str(store_path), 'bag_level_path': str(bag_level_path), 'per_fold': results}


if __name__ == '__main__':
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Fit Tier 3 (attention MIL)")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_501")
    parser.add_argument('--folds', nargs='+', default=None,
                        help="subset of fold names to run, e.g. fold_0 (smoke-test use)")
    parser.add_argument('--sub-bag-size', type=int, default=None)
    parser.add_argument('--n-per-bag-per-epoch', type=int, default=50)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--pi-min', type=float, default=None)
    parser.add_argument('--attn-dim', type=int, default=32)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--no-center-by-bag', action='store_true',
                        help="disable per-bag feature centering (on by default)")
    args = parser.parse_args()

    from src.notebooks.experiment_utils import setup_experiment
    cfg, out_dir = setup_experiment(args.config)
    mil_attention_stage(
        cfg, out_dir, sub_bag_size=args.sub_bag_size,
        n_per_bag_per_epoch=args.n_per_bag_per_epoch, max_epochs=args.max_epochs,
        patience=args.patience, pi_min=args.pi_min, fold_names=args.folds,
        attn_dim=args.attn_dim, dropout_rate=args.dropout_rate, l2=args.l2,
        center_by_bag=not args.no_center_by_bag,
    )
