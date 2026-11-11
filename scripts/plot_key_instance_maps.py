"""
Spatial visualization of the key-instance utility test's own binarization
(docs/mil_validation_test_suite.md SS B, scripts/cvae_weighted_rf_vote.py's
``_key_instance_vote_bag_decision``): for a given MIL tier's per-instance
posterior, "key" = non-diagnostic mass below the GLOBAL median (pooled over
every out-of-sample instance), NOT a per-slide median -- reusing the exact
threshold convention the key-instance utility test itself uses, so this map
shows what that test actually operates on, not a different definition.

Reprojects the resulting key/not-key binary onto each slide's own pixel grid
(src/data/spatial.py::reproject, the project's single verified round-trip)
for a random subsample of slides, same two slides across every requested
tier so the maps are directly comparable.

This is a diagnostic/exploratory visualization -- unlike the key-instance
UTILITY test itself (which trains and compares RF classifiers), this script
makes no classification claim; it only shows which instances a tier's
posterior calls "key".

Also renders a biological false-color RGB panel of each chosen slide's full
raw cube (background included, not just the strong-signal mask), using the
project's own established convention -- ``_bio_rgb`` from
notebooks/01_NMF_GMM_Tissue_Classification.ipynb (R=Phenylalanine band sum
990-1020 cm-1, G=C-H stretch/lipids 2800-3000 cm-1, B=CH2 deformation
1430-1460 cm-1, each channel min-max normalised then gamma 0.8), reproduced
verbatim here rather than redefined, after the SAME 3x3 per-wavelength
median filter that notebook applies before computing it
(src/data/preprocessing.py's ``_MEDIAN_KERNEL``). The lipid band (G) falls
outside the [700,1800] cm-1 fingerprint-region crop this project's MIL
pipeline trains on, so this panel is necessarily built from the untouched
raw per-slide ``.npy`` cube (``data/raw/LGm-*/<slide>.npy``) and the full
wavenumber calibration (``data/RamanFreqTable.csv``), not from the processed/
cropped per-instance spectra the rest of this script reads.

Usage:
    python scripts/plot_key_instance_maps.py --config exp_4.608 \
        --tiers misvm:mi_svm bayes_mil:gibbs --n-slides 10 --seed 42

A tier may carry an optional third field, a store suffix, to read
latent_labels_<suffix>.h5 instead of the real store (e.g. the shuffled-bag-
label control's fits):
    --tiers cbr:reconstruction cbr:reconstruction:shuffled --all-slides
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage5_spatial_validation import _load_raw_side_tables
from src.data.spatial import reproject
from src.mil.store import merge_out_of_sample_posterior

TIER_LABELS = {
    'misvm': 'mi-SVM', 'bayes_mil': 'Bayesian MIL', 'attention': 'Attention',
    'cbr': 'CbR', 'em': 'EM',
}

_MEDIAN_KERNEL = 3  # matches src/data/preprocessing.py and notebook 01


def _bio_rgb(cube, wn):
    """Verbatim from notebooks/01_NMF_GMM_Tissue_Classification.ipynb
    (the "Biological RGB helper", R=Phe, G=C-H stretch, B=CH2) -- do not
    redefine the band edges or normalisation independently elsewhere."""
    def band_sum(lo, hi):
        return cube[:, :, np.argmin(np.abs(wn - lo)):np.argmin(np.abs(wn - hi))].sum(axis=2)

    def norm(c):
        c = c - c.min()
        return c / (c.max() + 1e-8)

    R = norm(band_sum(990, 1020))    # Phenylalanine
    G = norm(band_sum(2800, 3000))   # C-H stretch (lipids)
    B = norm(band_sum(1430, 1460))   # CH2 deformation
    return np.power(np.stack([R, G, B], axis=-1), 0.8)


def _load_bio_rgb(slide_name, raw_data_dir, wavenumbers):
    matches = list(Path(raw_data_dir).glob(f"LGm-*/{slide_name}.npy"))
    if not matches:
        raise FileNotFoundError(f"no raw cube found for slide {slide_name!r} under {raw_data_dir}")
    cube = np.load(matches[0]).astype(np.float32)
    filtered = np.empty_like(cube)
    for i in range(cube.shape[2]):
        filtered[:, :, i] = median_filter(cube[:, :, i], size=_MEDIAN_KERNEL, mode='reflect')
    return _bio_rgb(filtered, wavenumbers)


def _tier_key_map(config, method, variant, suffix='', n_folds=5):
    # suffix='' -> the real store; suffix='shuffled' -> latent_labels_shuffled.h5
    # (scripts/stage6_shuffled_label_control.py's permuted-bag-label fits)
    store_name = f"latent_labels_{suffix}.h5" if suffix else 'latent_labels.h5'
    store_path = Path(config['paths'].get('experiment_root', f"results/{config['name']}")) \
        / 'latent' / store_name
    fold_names = [f"fold_{i}" for i in range(n_folds)]
    posterior, n_folds_covering = merge_out_of_sample_posterior(store_path, method, variant, fold_names)
    non_diag_mass = posterior[:, 0]
    covered = n_folds_covering > 0
    threshold = float(np.median(non_diag_mass[covered]))
    key = non_diag_mass < threshold
    print(f"  {method}/{variant}: {covered.sum()}/{len(covered)} covered instances, "
          f"global key threshold (median non-diag mass)={threshold:.4f}, "
          f"{key[covered].mean():.1%} of covered instances are 'key'")
    return non_diag_mass, covered, key, threshold


def run(config_name, tiers, n_slides=10, seed=42, out_path=None, n_folds=5,
       all_slides_mode=False, cols=10, min_covered=20):
    from src.utils.config_manager import ConfigManager
    config = ConfigManager.load_config(config_name).to_dict()

    h5_path = config['paths']['processed_file']
    sample_ids, coords, _viable_legacy, shape_lookup = _load_raw_side_tables(h5_path)

    per_tier = {}
    for method, variant, suffix in tiers:
        non_diag_mass, covered, key, threshold = _tier_key_map(
            config, method, variant, suffix, n_folds)
        per_tier[(method, variant, suffix)] = {
            'non_diag_mass': non_diag_mass, 'covered': covered, 'key': key, 'threshold': threshold,
        }

    # Eligible slides: in the shape registry AND with >=20 covered instances
    # for EVERY requested tier (stage5_spatial_validation.py's own >=20
    # cutoff, reused here so a slide isn't shown for one tier but silently
    # skipped for another). min_covered=0 keeps every slide in the registry;
    # slides below 20 covered instances are then flagged in their panel titles.
    all_slides = sorted(set(sample_ids.tolist()))
    eligible = []
    for slide in all_slides:
        if slide not in shape_lookup:
            continue
        idx = np.where(sample_ids == slide)[0]
        ok = all(per_tier[t]['covered'][idx].sum() >= min_covered for t in per_tier)
        if ok:
            eligible.append(slide)
    print(f"\n{len(eligible)}/{len(all_slides)} slides eligible (in shape registry, "
          f">={min_covered} covered instances for every requested tier)")

    if all_slides_mode:
        chosen = eligible  # already sorted
        print(f"Showing all {len(chosen)} eligible slides, batched {cols}-per-row")
    else:
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(eligible, size=min(n_slides, len(eligible)), replace=False).tolist())
        print(f"Chosen {len(chosen)} slides (seed={seed}): {chosen}")
        cols = len(chosen)

    raw_data_dir = config['paths']['raw_data_dir']
    wavenumbers = pd.read_csv(config['paths']['calib_file']).iloc[:, 1].values.astype(np.float64)
    print(f"\nLoading raw cubes + computing biological false-color RGB for {len(chosen)} slides...")
    bio_rgb_by_slide = {}
    for slide in chosen:
        bio_rgb_by_slide[slide] = _load_bio_rgb(slide, raw_data_dir, wavenumbers)
        print(f"  {slide}: cube shape {bio_rgb_by_slide[slide].shape}")

    import math

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['#6250d6', '#e8e8e8'])  # key=purple, not-key=light gray
    cmap.set_bad('#ffffff')  # uncovered/out-of-mask = white

    rows_per_batch = 1 + len(tiers)  # false-color RGB row + one row per tier
    n_batches = math.ceil(len(chosen) / cols)
    n_rows = n_batches * rows_per_batch
    fig, axes = plt.subplots(n_rows, cols, figsize=(2.0 * cols, 2.2 * n_rows), squeeze=False)

    for batch in range(n_batches):
        batch_slides = chosen[batch * cols:(batch + 1) * cols]
        row0 = batch * rows_per_batch

        for col in range(cols):
            ax = axes[row0][col]
            if col >= len(batch_slides):
                ax.axis('off')
                continue
            slide = batch_slides[col]
            ax.imshow(bio_rgb_by_slide[slide], aspect='equal', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(slide, fontsize=7, rotation=45, ha='left')
            if col == 0:
                ax.set_ylabel("False color\n(R=Phe,G=Lipid,B=CH$_2$)", fontsize=8)

        for t_idx, (method, variant, suffix) in enumerate(tiers, start=1):
            data = per_tier[(method, variant, suffix)]
            label = TIER_LABELS.get(method, method)
            if method == 'cbr' or len({v for m, v, _ in tiers if m == method}) > 1:
                label = f"{label} {variant}"
            if suffix:
                label = f"{label} [{suffix}]"
            for col in range(cols):
                ax = axes[row0 + t_idx][col]
                if col >= len(batch_slides):
                    ax.axis('off')
                    continue
                slide = batch_slides[col]
                idx = np.where(sample_ids == slide)[0]
                shape = shape_lookup[slide]
                covered_idx = data['covered'][idx]
                key_vals = np.where(data['key'][idx], 1.0, 0.0)
                key_vals = np.where(covered_idx, key_vals, np.nan)
                image = reproject(key_vals, coords[idx], shape, fill=np.nan)
                ax.imshow(image, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                n_cov = int(covered_idx.sum())
                if n_cov < 20:
                    ax.set_title(f"{n_cov} covered (<20)", fontsize=6, color='#b00020')
                if col == 0:
                    ax.set_ylabel(f"{label}\n(thr={data['threshold']:.5f})", fontsize=8)

    fig.suptitle(f"Key-instance map ({config_name}): purple = 'key' (non-diagnostic mass < "
                 f"global median), light gray = 'not key', white = uncovered -- false-color "
                 f"row is each slide's full raw cube in the project's biological RGB "
                 f"convention", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if n_batches > 1 else 0.95))

    if out_path is None:
        tier_tag = '_'.join(f"{m}-{v}" + (f"-{x}" if x else '') for m, v, x in tiers)
        suffix = f"all{len(chosen)}" if all_slides_mode else f"n{len(chosen)}_seed{seed}"
        out_path = Path(f"results/{config_name}/key_instance_maps/"
                        f"key_instance_maps_{tier_tag}_{suffix}.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved -> {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='exp_4.608')
    parser.add_argument('--tiers', nargs='+', default=['misvm:mi_svm', 'bayes_mil:gibbs'],
                        help="method:variant pairs, e.g. misvm:mi_svm bayes_mil:gibbs")
    parser.add_argument('--n-slides', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--all-slides', action='store_true',
                        help="show every eligible slide, batched --cols per row, instead of "
                             "a random --n-slides subsample")
    parser.add_argument('--cols', type=int, default=10, help="slides per row in --all-slides mode")
    parser.add_argument('--min-covered', type=int, default=20,
                        help="minimum covered instances per slide for every tier (0 = every slide)")
    args = parser.parse_args()
    tiers = [tuple((t.split(':') + [''])[:3]) for t in args.tiers]  # method:variant[:suffix]
    run(args.config, tiers, n_slides=args.n_slides, seed=args.seed, out_path=args.out_path,
       all_slides_mode=args.all_slides, cols=args.cols, min_covered=args.min_covered)
