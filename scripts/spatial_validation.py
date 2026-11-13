"""
Real-data spatial validation.

Reads a MIL formulation's fitted posteriors from the latent-label store,
merges them into one leak-free per-instance posterior (out-of-sample only --
src/mil/store.py::merge_out_of_sample_posterior), reprojects each slide's
non-diagnostic mass onto its pixel grid, and reports:

  - Moran's I and the empirical variogram, BY LAG BAND, per slide and pooled
    across slides -- the headline claim is coherence at lag >= 3, which the
    3x3 median filter provably cannot produce (lags 1-2 are shown but
    excluded from the claim).
  - A permutation-null p-value/z-score per lag (shuffling values within each
    slide's own mask).
  - External agreement (ARI/AMI) between the model's hard diagnostic/
    non-diagnostic assignment and `viable_legacy` -- the old manually-curated
    tumour-cluster selection, read directly from the main HDF5 (verified
    row-aligned to the loader's own order; no loader change needed).

Usage:
    python scripts/spatial_validation.py --config exp_608 --method misvm
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.spatial import reproject, slide_shape_lookup
from src.evaluation.spatial import (
    external_agreement, morans_i_by_lag, permutation_null_by_lag,
    pool_lag_reports, variogram_by_lag, variogram_range,
)
from src.mil.store import merge_out_of_sample_posterior

# See hard_diagnostic_threshold()'s docstring for why these two,
# specifically, get a rescaled hard-diagnostic threshold and mi-SVM does not.
JOINT_SOFTMAX_METHODS = {'attention', 'cbr'}


def hard_diagnostic_threshold(method, K):
    """
    The cutoff on non-diagnostic mass used to call an instance "diagnostic"
    for the external-agreement (ARI/AMI) check. A flat 0.5 is only reachable
    at all for a method whose non-diagnostic mass isn't bounded below 0.5 by
    construction. mi-SVM's independent one-vs-rest scores aren't bounded
    that way (already non-degenerate under 0.5 -- 0.96% hard-diagnostic, a
    real if small number) and are deliberately left alone here, so this fix
    doesn't retroactively change an already-reported result for a tier that
    didn't need it.

    For a genuine joint K-way softmax (Tier 3's attention head, Tier 4's cbr
    variants), max_k p_k is
    bounded below at 1/K by construction, so after the "1 - max_p" K+1 lift,
    non-diagnostic mass can never reach 0.5 at all: confirmed directly on
    real data (ARI=AMI=0.0000, every single instance called "diagnostic",
    for both attention and both cbr variants). Provisional fix, documented
    and not swept -- same treatment as this project's other provisional
    constants (cbr_temperature, IARPLS lambda): half of the maximum
    non-diagnostic mass this construction can EVER produce, (K-1)/(2K-1),
    instead of a threshold inherited from a differently-shaped score source.
    """
    if method in JOINT_SOFTMAX_METHODS and K > 1:
        return (K - 1) / (2.0 * (2 * K - 1))
    return 0.5


def _load_raw_side_tables(h5_path):
    with h5py.File(h5_path, 'r') as f:
        sample_ids = f['sample_ids'][:]
        sample_ids = (np.char.decode(sample_ids.astype('S'), 'utf-8')
                     if sample_ids.dtype.kind != 'U' else sample_ids.astype(str))
        coords = f['coords'][:]
        # v2 preprocessing (src/data/preprocessing_v2.py) has no clustering
        # stage at all, so this column does not exist there -- external
        # agreement is skipped for that dataset (configs/exp_2.*.yaml's own
        # notes), not a bug.
        viable_legacy = f['viable_legacy'][:] if 'viable_legacy' in f else None
        slide_names = f['slide_names'][:]
        slide_names = (np.char.decode(slide_names.astype('S'), 'utf-8')
                      if slide_names.dtype.kind != 'U' else slide_names.astype(str))
        slide_shapes = f['slide_shapes'][:]
    return sample_ids, coords, viable_legacy, slide_shape_lookup(slide_names, slide_shapes)


def run(config_name, method, variant, n_folds=5, max_lag=10, n_perm=200, seed=42,
       out_path=None, store_path=None):
    """
    ``store_path``: override the default real-store path -- e.g. to read
    ``latent_labels_shuffled.h5`` (scripts/shuffled_label_control.py)
    for the shuffled-bag-label control's own spatial statistics, compared
    directly against the real fit's, without touching the real store.
    """
    from src.utils.config_manager import ConfigManager
    config = ConfigManager.load_config(config_name)

    if store_path is None:
        store_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'latent' / 'latent_labels.h5'
    else:
        store_path = Path(store_path)
    fold_names = [f"fold_{i}" for i in range(n_folds)]

    print(f"Merging out-of-sample posteriors: {method}/{variant} from {store_path}")
    posterior, n_folds_covering = merge_out_of_sample_posterior(
        store_path, method, variant, fold_names)
    print(f"  {np.sum(n_folds_covering > 0)}/{len(posterior)} instances covered "
          f"({np.sum(n_folds_covering == 2)} from duplicated bags)")

    h5_path = config['paths']['processed_file']
    sample_ids, coords, viable_legacy, shape_lookup = _load_raw_side_tables(h5_path)
    assert len(sample_ids) == len(posterior), (
        f"row count mismatch: store has {len(posterior)}, raw H5 has {len(sample_ids)} -- "
        "alignment_checksum should have caught this; investigate before trusting anything below."
    )

    non_diag_mass = posterior[:, 0]  # column 0: combined non-diagnostic posterior mass

    per_slide_morans = {}
    per_slide_variogram = {}
    per_slide_perm_morans = {}
    slides = sorted(set(sample_ids.tolist()))
    print(f"\nReprojecting {len(slides)} slides...")
    for slide in slides:
        idx = np.where(sample_ids == slide)[0]
        if slide not in shape_lookup:
            print(f"  [skip {slide}] no shape in slide_shapes registry")
            continue
        shape = shape_lookup[slide]
        vals = non_diag_mass[idx]
        valid = ~np.isnan(vals)
        if valid.sum() < 20:
            print(f"  [skip {slide}] only {valid.sum()} covered instances")
            continue
        image = reproject(vals[valid], coords[idx][valid], shape, fill=np.nan)

        per_slide_morans[slide] = morans_i_by_lag(image, max_lag=max_lag)
        per_slide_variogram[slide] = variogram_by_lag(image, max_lag=max_lag)
        per_slide_perm_morans[slide] = permutation_null_by_lag(
            image, max_lag=max_lag, n_perm=n_perm, seed=seed, statistic='morans_i')

    pooled_morans = pool_lag_reports(list(per_slide_morans.values()), key='I')
    pooled_variogram = pool_lag_reports(list(per_slide_variogram.values()), key='gamma')
    pooled_perm_p = pool_lag_reports(list(per_slide_perm_morans.values()), key='p')
    pooled_perm_z = pool_lag_reports(list(per_slide_perm_morans.values()), key='z')

    print(f"\n{'='*70}\nPOOLED MORAN'S I BY LAG ({len(per_slide_morans)} slides)\n{'='*70}")
    for lag in sorted(pooled_morans.keys()):
        tag = "[contaminated]" if lag <= 2 else "[HEADLINE]" if lag <= 5 else ""
        print(f"  lag {lag:2d}: I={pooled_morans[lag]['mean']:+.4f} "
              f"(std={pooled_morans[lag]['std']:.4f}, n={pooled_morans[lag]['n_slides']})  "
              f"mean_p={pooled_perm_p.get(lag, {}).get('mean', float('nan')):.4f}  "
              f"mean_z={pooled_perm_z.get(lag, {}).get('mean', float('nan')):+.2f}  {tag}")

    print(f"\n{'='*70}\nPOOLED VARIOGRAM BY LAG\n{'='*70}")
    for lag in sorted(pooled_variogram.keys()):
        print(f"  lag {lag:2d}: gamma={pooled_variogram[lag]['mean']:.4f} "
              f"(std={pooled_variogram[lag]['std']:.4f}, n={pooled_variogram[lag]['n_slides']})")
    range_lags = [variogram_range(per_slide_variogram[s]) for s in per_slide_variogram]
    range_lags = [r for r in range_lags if r is not None]
    if range_lags:
        print(f"\n  variogram range: median={np.median(range_lags):.1f} px "
              f"(n={len(range_lags)} slides with an estimable range)")

    # External agreement vs viable_legacy: hard assignment = "diagnostic" iff
    # non-diagnostic mass < threshold (see hard_diagnostic_threshold's
    # docstring), compared against the OLD manually-curated tumour-cluster
    # boolean -- the finest-grained comparison available from the
    # (K+1)-collapsed posterior alone (the full (M+K)-way ARI against
    # hard_labels.npy's 3-cluster id is deferred, not done here).
    K = posterior.shape[1] - 1
    threshold = hard_diagnostic_threshold(method, K)
    covered = n_folds_covering > 0
    hard_diagnostic = (non_diag_mass[covered] < threshold).astype(int)
    if viable_legacy is None:
        agreement = None
        print(f"\n{'='*70}\nEXTERNAL AGREEMENT: skipped -- this dataset has no "
              f"'viable_legacy' column (v2 preprocessing has no clustering "
              f"stage)\n{'='*70}")
    else:
        agreement = external_agreement(hard_diagnostic, viable_legacy[covered].astype(int))
        print(f"\n{'='*70}\nEXTERNAL AGREEMENT vs viable_legacy (n={agreement['n']}, "
              f"hard_diagnostic_threshold={threshold:.4f})\n{'='*70}")
        print(f"  ARI={agreement['ari']:.4f}  AMI={agreement['ami']:.4f}")
        print(f"  (model says 'diagnostic': {hard_diagnostic.sum()}/{len(hard_diagnostic)}; "
              f"viable_legacy true: {viable_legacy[covered].sum()}/{covered.sum()})")

    report = {
        'config': config_name, 'method': method, 'variant': variant,
        'n_slides': len(per_slide_morans),
        'pooled_morans_i_by_lag': pooled_morans,
        'pooled_variogram_by_lag': pooled_variogram,
        'pooled_permutation_p_by_lag': pooled_perm_p,
        'pooled_permutation_z_by_lag': pooled_perm_z,
        'variogram_range_median_px': float(np.median(range_lags)) if range_lags else None,
        'external_agreement_vs_viable_legacy': agreement,
        'hard_diagnostic_threshold': threshold,
    }
    if out_path is None:
        out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'spatial_validation' / f'stage5_{method}_{variant}.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Spatial validation")
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True,
                        choices=['misvm', 'cbr', 'attention'])
    parser.add_argument('--variant', default=None,
                        help="defaults to 'mi_svm' for misvm, "
                             "'reconstruction' for cbr (its other variant is 'encoder_head'), "
                             "'attention' for attention")
    parser.add_argument('--max-lag', type=int, default=10)
    parser.add_argument('--n-perm', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--store-path', default=None,
                        help="override the real store, e.g. latent_labels_shuffled.h5 "
                             "for the shuffled-bag-label control")
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()

    _DEFAULT_VARIANT = {'misvm': 'mi_svm', 'cbr': 'reconstruction', 'attention': 'attention'}
    variant = args.variant or _DEFAULT_VARIANT[args.method]
    run(args.config, args.method, variant, max_lag=args.max_lag, n_perm=args.n_perm,
       seed=args.seed, store_path=args.store_path, out_path=args.out_path)
