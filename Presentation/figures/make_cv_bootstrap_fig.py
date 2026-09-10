"""
Slides 13-14: the two clustering units.

(a) Cross-validation splits by PATIENT   -> prevents leakage
(b) Bootstrap resamples by SLIDE         -> estimates variability

Run with the glioma env:
    python make_cv_bootstrap_fig.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_CV = Path(__file__).resolve().parent / "cv_design.png"
OUT_BS = Path(__file__).resolve().parent / "bootstrap_design.png"

TRAIN = "#4c8fd1"
TEST = "#e07b39"
GREY = "#b8b8b8"
EDGE = "0.30"


def slide_box(ax, x, y, w, h, color, label=None, fs=8, alpha=1.0, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=color, edgecolor=EDGE, lw=lw, alpha=alpha))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color="white", fontweight="bold")


# ---------------------------------------------------------------- (a) CV ----
def make_cv():
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.set_xlim(0, 11.5); ax.set_ylim(0.30, 4.75); ax.axis("off")

    # patients, each holding a different number of slides
    patients = [("P1", 1), ("P2", 3), ("P3", 1), ("P4", 2), ("P5", 1)]
    # which patients are held out in each fold
    folds = {1: {"P2"}, 2: {"P4"}, 3: {"P1"}, 4: {"P5"}, 5: {"P3"}}

    ax.text(0.1, 4.52, "Cross-validation splits by PATIENT",
            fontsize=14, fontweight="bold", color="0.15")
    ax.text(0.1, 4.22,
            "every slide from one patient goes to the same side of the split",
            fontsize=10, color="0.40")

    x0, ytop, bw, bh, gap = 1.55, 3.30, 0.42, 0.30, 0.07
    # patient group header
    xx = x0
    for pname, nsl in patients:
        width = nsl * bw + (nsl - 1) * gap
        ax.add_patch(FancyBboxPatch((xx - 0.05, ytop - 0.06), width + 0.10, 0.42,
                                    boxstyle="round,pad=0.01,rounding_size=0.03",
                                    facecolor="none", edgecolor="0.55",
                                    lw=1.0, linestyle=(0, (3, 2))))
        ax.text(xx + width / 2, ytop + 0.44, pname, ha="center", fontsize=9.5,
                color="0.3", fontweight="bold")
        for s in range(nsl):
            slide_box(ax, xx + s * (bw + gap), ytop, bw, bh, GREY)
        xx += width + 0.42
    ax.text(0.08, ytop + 0.12, "58 slides\n45 patients", fontsize=9.5,
            color="0.35", va="center")

    # folds
    for i, (f, held) in enumerate(sorted(folds.items())):
        y = 2.58 - i * 0.50
        ax.text(0.08, y + bh / 2, f"Fold {f}", fontsize=10, color="0.25",
                va="center", fontweight="bold")
        xx = x0
        for pname, nsl in patients:
            width = nsl * bw + (nsl - 1) * gap
            col = TEST if pname in held else TRAIN
            for s in range(nsl):
                slide_box(ax, xx + s * (bw + gap), y, bw, bh, col)
            xx += width + 0.42

    # legend
    slide_box(ax, 8.55, 2.58, bw, bh, TRAIN)
    ax.text(9.08, 2.73, "train", fontsize=10, va="center", color="0.2")
    slide_box(ax, 8.55, 2.14, bw, bh, TEST)
    ax.text(9.08, 2.29, "held out", fontsize=10, va="center", color="0.2")

    ax.text(8.55, 1.62,
            "P2 has 3 slides:\nall three move together.\nA model is never tested on a\n"
            "patient it has already seen.",
            fontsize=9.5, color="0.30", va="top")

    fig.savefig(OUT_CV, dpi=200, bbox_inches="tight")
    print("saved:", OUT_CV)


# --------------------------------------------------------- (b) bootstrap ----
def make_bootstrap():
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.0); ax.axis("off")

    ax.text(0.1, 3.70, "The bootstrap resamples by SLIDE",
            fontsize=14, fontweight="bold", color="0.15")
    ax.text(0.1, 3.38,
            "one slide = one Raman acquisition; each draw simulates a fresh batch of acquisitions",
            fontsize=10, color="0.40")

    bw, bh, gap = 0.52, 0.36, 0.10
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    # original cohort
    ax.text(0.1, 2.62, "observed\ncohort", fontsize=10, color="0.25",
            va="center", fontweight="bold")
    for i, lab in enumerate(labels):
        slide_box(ax, 1.35 + i * (bw + gap), 2.44, bw, bh, TRAIN, lab, fs=9)

    # draws
    draws = [["B", "B", "D", "E", "E", "F", "H", "H"],
             ["A", "C", "C", "C", "D", "G", "G", "H"]]
    for r, draw in enumerate(draws):
        y = 1.62 - r * 0.56
        ax.text(0.1, y + bh / 2, f"draw {r+1}", fontsize=10, color="0.35", va="center")
        seen = {}
        for i, lab in enumerate(draw):
            seen[lab] = seen.get(lab, 0) + 1
            col = TEST if seen[lab] > 1 else TRAIN
            slide_box(ax, 1.35 + i * (bw + gap), y, bw, bh, col, lab, fs=9,
                      alpha=0.95)
    ax.text(1.35, 0.62, r"$\cdots$ 1000 draws", fontsize=11, color="0.35")

    arr = FancyArrowPatch((6.55, 2.38), (6.55, 2.02), arrowstyle="-|>",
                          mutation_scale=16, color="0.45", lw=1.4)
    ax.add_patch(arr)
    ax.text(6.72, 2.20, "sample slides with replacement,\nkeeping class proportions",
            fontsize=9.5, color="0.35", va="center")

    slide_box(ax, 6.72, 1.02, bw * 0.6, bh * 0.72, TEST)
    ax.text(7.10, 1.15, "a slide drawn more than once", fontsize=9.5,
            va="center", color="0.3")
    ax.text(6.72, 0.52, "Spectra are never resampled on their own:\n"
                        "thousands from one acquisition are not\n"
                        "thousands of independent observations.",
            fontsize=9.5, color="0.30", va="top")

    fig.savefig(OUT_BS, dpi=200, bbox_inches="tight")
    print("saved:", OUT_BS)


if __name__ == "__main__":
    make_cv()
    make_bootstrap()
