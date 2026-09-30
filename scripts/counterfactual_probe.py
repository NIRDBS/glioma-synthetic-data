"""
CVAE counterfactual probe -- a free-standing analysis, not a Stage-N MIL
tier (see src/mil/counterfactual.py's module docstring and
docs/counterfactual_probe.md for the full write-up and why).

Per fold: load that fold's own trained CVAE encoder/decoder (real or, with
--shuffled, the already-existing shuffled-label-trained weights -- zero new
training either way), fit a fresh z_mean->class logistic-regression probe on
that fold's TRAINING instances, then on that fold's held-out VALIDATION
instances: decode every instance's z_mean under both class conditions,
compute Delta(z) = D(z,mutant) - D(z,wildtype), score the z-probe, compute
the within-class reconstruction spread, and score both a class-swapped
counterfactual and the real spectrum through that fold's already-trained,
independent RF/XGBoost models (the flip-rate check). Merges every fold's
out-of-sample Delta into one leak-free per-instance array (the same
duplicate-bag-averaging convention used everywhere else in this project),
reduces to per-bag and grand-mean Delta, reports cross-fold stability and
the named-band assignment table.

IMPORTANT with --shuffled: the loaded CVAE's decoder condition "0"/"1"
corresponds to a PERMUTED label assignment from training
(permutation_seed=123, matching every other shuffled-label control this
project runs), NOT true mutant/wildtype. Delta here means
D(z, permuted=1) - D(z, permuted=0) -- scored against the TRUE labels for
every check in this script (fd.y_train/fd.y_val are never permuted here;
only the CVAE's OWN internal training was permuted, earlier, on Modal).

Usage:
    python scripts/counterfactual_probe.py --config exp_501
    python scripts/counterfactual_probe.py --config exp_501 --shuffled
    python scripts/counterfactual_probe.py --config exp_501 --folds fold_0   # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import make_loader, num_classes_of, prepare_fold, set_seeds
from pipeline._cvae_gen import build_cvae_config, load_encoder_decoder
from pipeline.modal_cvae import _effective_output_dirs
from src.mil.counterfactual import (
    band_assignment_report, bootstrap_ci_mean, compute_counterfactual_reconstructions,
    compute_delta, cosine_to_reference, delta_effect_size, fit_z_class_probe,
    independent_classifier_flip_rate, mean_delta_per_bag, merge_out_of_sample_arrays,
    normalized_distance_to_reference, pairwise_cosine_stability, random_direction_null,
    score_z_class_probe, within_class_reconstruction_spread,
)
from src.utils.statistical_evaluation import aggregate_bag_scores_across_folds, find_model_path


def _fold_idx(fold_name):
    return int(fold_name.split('_')[1])


def run(config_name, store_suffix=None, fold_names=None, seed=42, n_boot=1000, out_path=None):
    """
    ``store_suffix``: None for the real CVAE; 'shuffled' for the shuffled-
    bag-label-trained CVAE; 'alpha0' for the alpha=0 ablation (§5) -- any
    suffix pipeline.modal_cvae._effective_output_dirs understands.
    `build_cvae_config`'s own `alpha` value only ever affects `train_step`,
    never `.predict()`, so which config's `alpha` gets read here has zero
    effect on inference -- ``config_name`` need not be switched to
    'exp_501_alpha0' for this to score that variant's weights correctly.
    """
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    n_classes = num_classes_of(config)
    if n_classes != 2:
        raise ValueError(
            f"counterfactual probe is K=2-only (Delta has no natural K>2 generalisation, "
            f"docs/mil_tiers_methodology.md §1) -- config {config_name!r} has {n_classes} classes")

    cvae_weights_dir, _ = _effective_output_dirs(config, store_suffix)
    baseline_dir = config['paths']['models_baseline_dir']

    label = store_suffix.upper() if store_suffix else 'REAL'
    print(f"\n{'='*80}\nCOUNTERFACTUAL PROBE ({label}): {config_name}\n{'='*80}")
    print(f"  cvae_weights_dir -> {cvae_weights_dir}")
    if store_suffix == 'shuffled':
        print("  NOTE: this CVAE's decoder condition '0'/'1' is a PERMUTED label from "
              "training (permutation_seed=123), not true mutant/wildtype. Delta here means "
              "D(z, permuted=1) - D(z, permuted=0). Every check below is still scored "
              "against the TRUE labels. See docs/counterfactual_probe.md.")
    elif store_suffix == 'alpha0':
        print("  NOTE: this CVAE was trained with the auxiliary class head's loss weight "
              "(alpha) set to 0 -- the class head received no training signal at all, so "
              "its own accuracy (logged in the training history) is meaningless; only "
              "z_mean/the decoder are of interest here. See docs/counterfactual_probe.md §5.")

    loader = make_loader(config)
    full_bag_ids = loader.groups
    wavenumbers = loader.wavenumbers

    seed = seed if seed is not None else set_seeds(config)

    delta_fold_results = []       # (val_mask_full, delta_val) per fold -> cross-fold merge
    zprobe_fold_data = []         # {'proba','y_true','bag_ids'} per fold -> aggregate_bag_scores_across_folds
    per_fold_mean_delta = []      # this fold's own mean-Delta over its own val instances
    per_fold_spread = []
    flip_rf_rates, flip_xgb_rates = [], []

    for fold_data in loader:
        fold_name = fold_data['fold_name']
        if fold_names is not None and fold_name not in fold_names:
            continue
        fold_idx = _fold_idx(fold_name)

        fd = prepare_fold(fold_data, config, cap_train=False)
        n_features = fd.X_train.shape[1]
        print(f"\n  fold={fold_name}  train={fd.X_train.shape[0]}  val={fd.X_val.shape[0]}")

        cvae_config = build_cvae_config(config, n_features)
        encoder, decoder = load_encoder_decoder(cvae_config, cvae_weights_dir, fold_name)

        # z-probe: fit fresh on this fold's TRAINING z_mean (in-sample -- the
        # same naive-MIL discipline as Tier 0's own instance classifiers),
        # scored later on held-out z_mean.
        X_train_exp = fd.X_train.reshape(-1, n_features, 1)
        z_mean_train, _, _, _ = encoder.predict(X_train_exp, batch_size=256, verbose=0)
        scaler, probe_model = fit_z_class_probe(z_mean_train, fd.y_train, seed=seed)

        # Main probe: full counterfactual reconstruction on VAL (out-of-sample) only.
        cf = compute_counterfactual_reconstructions(fd.X_val, encoder, decoder, n_classes)
        z_mean_val, recon_val = cf['z_mean'], cf['recon']
        delta_val = compute_delta(recon_val)
        zprobe_proba_val = score_z_class_probe(scaler, probe_model, z_mean_val)
        spread = within_class_reconstruction_spread(recon_val, fd.y_val)

        recon_swap = recon_val[np.arange(len(fd.y_val)), 1 - fd.y_val, :]
        rf_path = find_model_path(baseline_dir, 'rf', '.joblib', fold_idx)
        flip_rf = independent_classifier_flip_rate(
            fd.X_val, fd.y_val, recon_swap, rf_path, 'sklearn')
        xgb_path = find_model_path(baseline_dir, 'xgb', '.json', fold_idx)
        flip_xgb = independent_classifier_flip_rate(
            fd.X_val, fd.y_val, recon_swap, xgb_path, 'xgboost')

        z_val_acc = float((zprobe_proba_val.argmax(axis=1) == fd.y_val).mean())
        print(f"    mean|Delta|={np.abs(delta_val).mean():.4f}  "
              f"z_probe_instance_acc={z_val_acc:.4f}  "
              f"flip_to_target(rf)={flip_rf['flip_to_target_rate']:.4f}  "
              f"flip_to_target(xgb)={flip_xgb['flip_to_target_rate']:.4f}")

        val_mask_full = np.isin(full_bag_ids, fold_data['val_sample_ids'])
        delta_fold_results.append((val_mask_full, delta_val))
        zprobe_fold_data.append(
            {'proba': zprobe_proba_val, 'y_true': fd.y_val, 'bag_ids': fd.val_bag_ids})
        per_fold_mean_delta.append(delta_val.mean(axis=0))
        per_fold_spread.append(spread['pooled'])
        flip_rf_rates.append(flip_rf['flip_to_target_rate'])
        flip_xgb_rates.append(flip_xgb['flip_to_target_rate'])

        import tensorflow as tf
        tf.keras.backend.clear_session()

    # ---- Cross-fold merge + reduction ----
    delta_merged, n_folds_covering = merge_out_of_sample_arrays(delta_fold_results, full_bag_ids)
    bag_ids_sorted, per_bag_mean_delta, grand_mean_delta = mean_delta_per_bag(
        delta_merged, full_bag_ids)

    covered = n_folds_covering > 0
    per_instance_cosine = cosine_to_reference(delta_merged[covered], grand_mean_delta)
    per_instance_ndist = normalized_distance_to_reference(delta_merged[covered], grand_mean_delta)
    per_bag_cosine = cosine_to_reference(per_bag_mean_delta, grand_mean_delta)
    per_bag_ndist = normalized_distance_to_reference(per_bag_mean_delta, grand_mean_delta)

    per_fold_mean_delta_arr = np.array(per_fold_mean_delta)
    stability = pairwise_cosine_stability(per_fold_mean_delta_arr)
    null = random_direction_null(
        d=per_fold_mean_delta_arr.shape[1], m=per_fold_mean_delta_arr.shape[0], seed=seed)
    z_stability = ((stability['mean_offdiag_cosine'] - null['null_mean']) / null['null_std']
                  if null['null_std'] > 0 else float('nan'))

    mean_pooled_spread = np.mean(per_fold_spread, axis=0)
    effect_size = delta_effect_size(grand_mean_delta, mean_pooled_spread)

    per_bag_norm = np.linalg.norm(per_bag_mean_delta, axis=1)
    delta_norm_ci = bootstrap_ci_mean(per_bag_norm, n_boot=n_boot, seed=seed)

    band_report = band_assignment_report(
        grand_mean_delta, wavenumbers, per_fold_mean_delta=per_fold_mean_delta_arr,
        spread=mean_pooled_spread)

    # z-probe bag-level accuracy, via the SAME machinery Tier 0/every MIL tier uses.
    zprobe_bag_ids, zprobe_scores, zprobe_bag_true = aggregate_bag_scores_across_folds(
        zprobe_fold_data, rule='mean')
    zprobe_bag_pred = zprobe_scores.argmax(axis=1)
    zprobe_bag_acc = float((zprobe_bag_pred == zprobe_bag_true).mean())

    print(f"\n{'='*70}\nSUMMARY ({label})\n{'='*70}")
    print(f"  ||Delta|| (bag-of-bags): mean={delta_norm_ci['mean']:.4f}  "
          f"95%CI[{delta_norm_ci['ci_lo']:.4f},{delta_norm_ci['ci_hi']:.4f}]")
    print(f"  cross-fold stability: mean_offdiag_cosine={stability['mean_offdiag_cosine']:.4f}  "
          f"z_vs_random_null={z_stability:+.2f}  effective_n={stability['effective_n_components']:.2f} "
          f"(of {per_fold_mean_delta_arr.shape[0]} folds)")
    print(f"  z-class-probe: instance-level acc per fold above; bag-level acc={zprobe_bag_acc:.4f} "
          f"(cf. encoder_head's 0.7966, chance=0.5593)")
    print(f"  flip_to_target_rate: rf={np.mean(flip_rf_rates):.4f}  xgb={np.mean(flip_xgb_rates):.4f}")
    print(f"\n  Band assignments:")
    for name, info in band_report.items():
        print(f"    {name:30s} wn={info['wavenumber_matched']:.1f}  delta={info['delta']:+.4f}  "
              f"({info['direction']})  effect_size={info['effect_size']}  "
              f"fold_sign_agreement={info['fold_sign_agreement_frac']:.2f}")

    report = {
        'config': config_name, 'store_suffix': store_suffix, 'n_bags': len(bag_ids_sorted),
        'n_folds': len(per_fold_mean_delta),
        'delta_norm_bootstrap_ci': delta_norm_ci,
        'cross_fold_stability': {
            'mean_offdiag_cosine': stability['mean_offdiag_cosine'],
            'effective_n_components': stability['effective_n_components'],
            'null_mean': null['null_mean'], 'null_std': null['null_std'],
            'z_vs_random_null': z_stability,
        },
        'z_class_probe': {'bag_accuracy': zprobe_bag_acc,
                          'chance_bag_accuracy': 0.5593,
                          'encoder_head_bag_accuracy_reference': 0.7966},
        'flip_to_target_rate': {'rf': float(np.mean(flip_rf_rates)),
                                'xgb': float(np.mean(flip_xgb_rates))},
        'band_assignments': band_report,
        'per_instance_consistency': {
            'cosine_to_grand_mean': {'mean': float(np.nanmean(per_instance_cosine)),
                                     'std': float(np.nanstd(per_instance_cosine))},
            'normalized_distance_to_grand_mean': {'mean': float(np.nanmean(per_instance_ndist)),
                                                  'std': float(np.nanstd(per_instance_ndist))},
            'note': ('descriptive only -- confounded by within-slide spatial '
                    'autocorrelation, same caveat as spatial-validation lags 1-2; '
                    'the per_bag statistic below is primary'),
        },
        'per_bag_consistency': {
            'cosine_to_grand_mean': {'mean': float(np.nanmean(per_bag_cosine)),
                                     'std': float(np.nanstd(per_bag_cosine))},
            'normalized_distance_to_grand_mean': {'mean': float(np.nanmean(per_bag_ndist)),
                                                  'std': float(np.nanstd(per_bag_ndist))},
        },
    }

    if out_path is None:
        suffix = f'_{store_suffix}' if store_suffix else ''
        out_dir = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'counterfactual_probe'
        out_path = out_dir / f'report{suffix}.json'
        npz_path = out_dir / f'arrays{suffix}.npz'
    else:
        out_path = Path(out_path)
        npz_path = out_path.with_suffix('.npz')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    np.savez(npz_path,
            bag_ids=np.asarray(bag_ids_sorted, dtype=object),
            per_bag_mean_delta=per_bag_mean_delta,
            per_fold_mean_delta=per_fold_mean_delta_arr,
            grand_mean_delta=grand_mean_delta,
            wavenumbers=wavenumbers)
    print(f"\nReport written -> {out_path}\nArrays written -> {npz_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CVAE counterfactual probe")
    parser.add_argument('--config', required=True)
    parser.add_argument('--shuffled', action='store_true',
                        help="shorthand for --store-suffix shuffled")
    parser.add_argument('--store-suffix', default=None,
                        help="score an already-trained CVAE variant instead of the real one, "
                             "e.g. 'shuffled' or 'alpha0' -- any suffix "
                             "pipeline.modal_cvae._effective_output_dirs understands")
    parser.add_argument('--folds', nargs='+', default=None,
                        help="subset of fold names, e.g. fold_0 (smoke-test use)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()
    store_suffix = args.store_suffix or ('shuffled' if args.shuffled else None)
    run(args.config, store_suffix=store_suffix, fold_names=args.folds, seed=args.seed,
       n_boot=args.n_boot, out_path=args.out_path)
