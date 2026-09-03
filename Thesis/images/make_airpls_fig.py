"""
Figure for Section 3.7 (airPLS): one raw spectrum, the fitted airPLS baseline, and the
corrected result. Uses the same slide as Figure 2.2 so the two figures pair up.

Run:  python make_airpls_fig.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[0]
PROJ = Path(r"c:\Users\Andrei\incdsb\glioma-synthetic-data")

SLIDE = "HF-1295_V3AM_2.h5_1"          # same slide as Figure 2.2
CLASS_DIR = "LGm-1"
LAMBDA, PORDER, ITERMAX = 100, 1, 15
PIXEL = 8134                            # a representative tissue spectrum (see caption)


def airpls(x, lambda_=LAMBDA, porder=PORDER, itermax=ITERMAX):
    """Port of GliomaPreprocessor._airPLS (src/data/preprocessing.py)."""
    m = x.shape[0]
    D = eye(m, format="csc")
    for _ in range(porder):
        D = D[1:] - D[:-1]
    DD = (D.T @ D).tocsc()

    w = np.ones(m)
    z = x.copy()
    for i in range(1, itermax + 1):
        W = diags(w)
        z = spsolve((W + lambda_ * DD).tocsc(), w * x)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * np.abs(x).sum():
            break
        w[d >= 0] = 0
        neg = d < 0
        if neg.any():
            w[neg] = np.exp(i * np.abs(d[neg]) / dssn)
            w[0] = np.exp(i * d[d < 0].max() / dssn)
        else:
            w[0] = 0
        w[-1] = w[0]
    return z


def main():
    cube = np.load(PROJ / "data" / "raw" / CLASS_DIR / f"{SLIDE}.npy")
    print("slide cube:", cube.shape)
    flat = cube.reshape(-1, cube.shape[-1])

    wn = pd.read_csv(PROJ / "data" / "RamanFreqTable.csv")["freq"].values
    assert len(wn) == flat.shape[1], (len(wn), flat.shape[1])

    idx = PIXEL if PIXEL is not None else int(np.argmax(flat.max(axis=1)))
    raw = flat[idx].astype(np.float64)
    print(f"pixel {idx}: max={raw.max():.1f}")

    base = airpls(raw)
    corr = np.maximum(raw - base, 0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})

    axes[0].plot(wn, raw, color="#1f77b4", lw=0.9, label="Raw spectrum")
    axes[0].plot(wn, base, color="#d62728", lw=1.4, ls="--",
                 label=rf"airPLS baseline ($\lambda={LAMBDA}$, $p={PORDER}$)")
    axes[0].set_ylabel("Intensity (counts)")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].set_title("(a) Raw spectrum and fitted baseline", fontsize=10, loc="left")

    axes[1].plot(wn, corr, color="#2ca02c", lw=0.9, label="Baseline-corrected")
    axes[1].axhline(0, color="gray", lw=0.6, ls=":")
    axes[1].set_xlabel(r"Raman shift (cm$^{-1}$)")
    axes[1].set_ylabel("Intensity (counts)")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].set_title("(b) After subtracting the baseline", fontsize=10, loc="left")

    # mark the fingerprint region used throughout this work
    for ax in axes:
        ax.axvspan(700, 1800, color="k", alpha=0.05, zorder=0)
    axes[1].text(1250, axes[1].get_ylim()[1] * 0.90,
                 "fingerprint region\n(700–1800 cm$^{-1}$)",
                 ha="center", va="top", fontsize=8, color="0.35")

    fig.subplots_adjust(hspace=0.18)
    out = PROJ / "Thesis" / "images" / "airpls_baseline_correction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
