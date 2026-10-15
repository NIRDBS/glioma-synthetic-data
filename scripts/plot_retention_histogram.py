"""Histogram of per-slide retention rate -- rerun of the paper's existing
figure (`paper/img/tumor_retention_histogram.png`), generalised to any
processed HDF5 (v1 `glioma_mil.h5`, v2 `glioma-preproc-2.h5`, v3
`glioma-preproc-3.h5`), 20 bins, same style as the original.

IMPORTANT difference from the original figure: the original
(`_archive/old_notebooks/02_NMF_GMM_Unmixing_Pipeline.ipynb`, cell 21) used
`tumor_maps.npy` -- the old, narrower, manually-curated NMF/GMM tumour-
cluster mask (the source of v1's `viable_legacy` column). v2/v3 have no
such column at all (no clustering stage), so "retention" here is
necessarily **strong-signal-mask retention** (retained instances / original
full pixel grid per slide, from the processed H5's own `slide_shapes` +
`sample_ids` -- no need to touch raw cubes at all, this is a cheap read).
For v1 this is a WIDER retention rate than the original figure's (the
current default mask, not the old narrower tumor-only one) -- pass
--viable-legacy-only to reproduce the original figure's exact v1 definition
for a side-by-side comparison; that flag has no effect on v2/v3 (the
column doesn't exist there and the script will say so).

Usage:
    python scripts/plot_retention_histogram.py --h5 data/processed/glioma_mil.h5
    python scripts/plot_retention_histogram.py --h5 data/processed/glioma-preproc-2.h5 --out paper/img/tumor_retention_histogram_v2.png
    python scripts/plot_retention_histogram.py --h5 data/processed/glioma_mil.h5 --viable-legacy-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def compute_retention_pcts(h5_path: Path, viable_legacy_only: bool = False) -> tuple[np.ndarray, list[str]]:
    with h5py.File(h5_path, "r") as f:
        slide_names = f["slide_names"][:]
        if isinstance(slide_names[0], bytes):
            slide_names = np.char.decode(slide_names.astype("S"), "utf-8")
        slide_shapes = f["slide_shapes"][:]  # (n_slides, 2) -- original full (rows, cols) per slide

        sample_ids = f["sample_ids"][:]
        if isinstance(sample_ids[0], bytes):
            sample_ids = np.char.decode(sample_ids.astype("S"), "utf-8")

        if viable_legacy_only:
            if "viable_legacy" not in f:
                raise ValueError(
                    f"{h5_path} has no 'viable_legacy' column (v2/v3 have no clustering "
                    "stage at all) -- --viable-legacy-only is only valid for v1 (glioma_mil.h5)."
                )
            viable_legacy = f["viable_legacy"][:]
            sample_ids = sample_ids[viable_legacy]

    pcts = []
    for name, (rows, cols) in zip(slide_names, slide_shapes):
        n_retained = int((sample_ids == name).sum())
        n_total = int(rows) * int(cols)
        pcts.append(100.0 * n_retained / n_total)
    return np.array(pcts), slide_names.tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="data/processed/glioma_mil.h5")
    ap.add_argument("--viable-legacy-only", action="store_true",
                    help="v1 only -- reproduce the ORIGINAL paper figure's exact definition "
                         "(old narrower tumor-cluster mask) instead of strong-signal retention")
    ap.add_argument("--out", default=None, help="default: paper/img/tumor_retention_histogram_<h5 stem>.png")
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    h5_path = Path(args.h5)
    pcts, slide_names = compute_retention_pcts(h5_path, viable_legacy_only=args.viable_legacy_only)

    print(f"{h5_path.name}" + (" [viable_legacy only]" if args.viable_legacy_only else " [strong-signal mask]"))
    print(f"  Slides: {len(pcts)}")
    print(f"  Mean retention:   {pcts.mean():.1f}%")
    print(f"  Median retention: {np.median(pcts):.1f}%")
    print(f"  Min / Max:        {pcts.min():.1f}% / {pcts.max():.1f}%")
    print(f"  Slides >80%:      {int(np.sum(pcts > 80))} / {len(pcts)}")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(pcts, bins=args.bins, range=(0, 100), color="#4878CF", alpha=0.82,
           edgecolor="white", linewidth=0.6)

    mean_val, median_val = pcts.mean(), np.median(pcts)
    ax.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=1.6, label=f"Mean: {mean_val:.1f}%")
    ax.axvline(median_val, color="#55A868", linestyle="-.", linewidth=1.6, label=f"Median: {median_val:.1f}%")

    ax.set_title("Distribution of retention rates across slides", fontsize=12, pad=8)
    ax.set_xlabel("Retained spectra (%)", fontsize=11)
    ax.set_ylabel("Number of slides", fontsize=11)
    ax.set_xlim(70, 105)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=10, framealpha=0.85)
    ax.grid(axis="y", alpha=0.35, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = Path(args.out) if args.out else Path(f"paper/img/tumor_retention_histogram_{h5_path.stem}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
