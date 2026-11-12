"""Histogram of per-spectrum max intensity, pooled across all 59 raw slides,
after the same slide-tensor normalisation used by the v2 preprocessing
pipeline (src/data/preprocessing_v2.py) -- BEFORE compute_strong_signal_mask's
threshold filter is applied. Diagnostic for inspecting/tuning that threshold
(0.95 saturated-branch cutoff, 0.8 floor), not part of the pipeline itself.

Reads raw .npy cubes directly (data/raw/**/*.npy), not the processed HDF5.

Per slide:
    cube = np.load(raw_path)                      # (rows, cols, wavelengths)
    filtered = 3x3 median filter per wavelength    # matches process_slide_v2
    normalized = cube_normalize(filtered)          # shift/scale/clip to [0, 1]
    slide_intensities = normalized.max(axis=-1)    # per-pixel max intensity
    max_intensity_list.append(slide_intensities.ravel())

All slides' arrays are then concatenated into one pooled array for the
histogram. Results are cached to .npy after the (slower) load+filter+normalize
pass, so re-running the script to tweak the plot only re-reads the cache --
pass --recompute to force a fresh pass over the raw data.

Usage:
    python scripts/plot_max_intensity_histogram.py
    python scripts/plot_max_intensity_histogram.py --recompute
    python scripts/plot_max_intensity_histogram.py --bins 100 --out my_hist.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.ndimage import median_filter

from src.data.preprocessing_v2 import cube_normalize

_MEDIAN_KERNEL = 3  # matches src/data/preprocessing_v2.py's _MEDIAN_KERNEL


def compute_max_intensities(raw_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Per slide: load -> median filter -> cube_normalize -> per-pixel max
    intensity. Returns (list of 1-D arrays, one per slide; matching slide names)."""
    slide_paths = sorted(raw_dir.rglob("*.npy"))
    print(f"Found {len(slide_paths)} raw slides under {raw_dir}")

    max_intensity_list = []
    slide_names = []
    for i, raw_path in enumerate(slide_paths):
        cube = np.load(raw_path).astype(np.float32)

        filtered = np.empty_like(cube)
        for w in range(cube.shape[2]):
            filtered[:, :, w] = median_filter(cube[:, :, w], size=_MEDIAN_KERNEL, mode="reflect")

        normalized = cube_normalize(filtered)
        slide_intensities = normalized.max(axis=-1)  # (rows, cols) -- per-pixel max intensity

        max_intensity_list.append(slide_intensities.ravel())
        slide_names.append(raw_path.stem)
        print(f"  [{i + 1}/{len(slide_paths)}] {raw_path.stem}: "
             f"{slide_intensities.size} pixels, max_intensity range "
             f"[{slide_intensities.min():.4f}, {slide_intensities.max():.4f}]")

    return max_intensity_list, slide_names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cache", default="results/diagnostics/max_intensity_pooled.npy",
                    help="pooled per-pixel max-intensity array, cached here after computing")
    ap.add_argument("--recompute", action="store_true", help="ignore the cache, redo the raw-data pass")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--out", default="paper/img/max_intensity_histogram.png")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.recompute:
        print(f"Loading cached pooled max intensities from {cache_path}")
        pooled = np.load(cache_path)
    else:
        t0 = time.perf_counter()
        max_intensity_list, slide_names = compute_max_intensities(Path(args.raw_dir))
        pooled = np.concatenate(max_intensity_list)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, pooled)
        print(f"Computed in {time.perf_counter() - t0:.1f}s, "
             f"{pooled.size} pixels pooled across {len(slide_names)} slides -> cached to {cache_path}")

    print(f"\nPooled max intensity: n={pooled.size}  "
         f"min={pooled.min():.4f}  max={pooled.max():.4f}  "
         f"mean={pooled.mean():.4f}  median={np.median(pooled):.4f}")

    # ---- Plot -- tweak freely from here ----
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(pooled, bins=args.bins, color="#3b6fa0", edgecolor="white", linewidth=0.3)

    # Reference lines for compute_strong_signal_mask's thresholds (src/data/preprocessing_v2.py):
    # Rule 1 (saturated branch): max_intensity > 0.95.  Rule 3 (floor, rule2+3 branch): max_intensity > 0.8.
    # ax.axvline(0.95, color="#c0392b", linestyle="--", linewidth=1.2, label="0.95")
    ax.axvline(0.8, color="#e67e22", linestyle="--", linewidth=1.2, label="0.8")
    ax.axvline(0.6, color="#f39c12", linestyle="--", linewidth=1.2, label="0.6")
    ax.axvline(0.4, color="#f1c40f", linestyle="--", linewidth=1.2, label="0.4")

    ax.set_xlabel("Per-spectrum max intensity", fontsize=16)
    ax.set_ylabel("Count (pixels, pooled across all slides)", fontsize=16)
    ax.set_title("Max intensity distribution, all 59 raw slides, pre-threshold (log scale)", fontsize=18)
    ax.legend()
    ax.set_yscale("log")  # log scale to see the long tail of low-intensity pixels
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved histogram -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
