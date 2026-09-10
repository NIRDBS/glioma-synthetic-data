"""
Slide 3-4 figure: the hyperspectral data cube.

Shows a slide as a (H, W, N_lambda) tensor -- two spatial axes and one spectral axis --
by drawing several wavenumber slices in perspective, and pulling the spectrum out of a
single pixel to make "one pixel = one full spectrum" concrete.

Run with the glioma env:
    python make_tensor_fig.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(r"C:\Users\Andrei\incdsb\glioma-synthetic-data")
OUT = Path(__file__).resolve().parent / "data_cube.png"

SLIDE = "LGm-1/HF-1295_V3AM_2.h5_1.npy"
# (centre wavenumber, half-window in bands, short label)
BANDS = [(2930, 12, "Protein / lipid"),
         (1450, 12, r"CH$_2$ deformation"),
         (1003, 8,  "Phenylalanine")]
PIXEL = (120, 30)          # (row, col) whose spectrum is drawn on the right


def band_image(cube, wn, centre, half):
    """Mean intensity over a small window, which is far less noisy than one band."""
    i = int(np.argmin(np.abs(wn - centre)))
    img = cube[:, :, max(i - half, 0): i + half + 1].mean(axis=2)
    lo, hi = np.percentile(img, [3, 97])
    return np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)


def main():
    cube = np.load(PROJ / "data" / "raw" / SLIDE).astype(np.float32)
    wn = pd.read_csv(PROJ / "data" / "RamanFreqTable.csv")["freq"].values
    H, W, L = cube.shape

    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.10)

    # ---- left: the cube as stacked spectral slices -------------------------
    ax = fig.add_subplot(gs[0], projection="3d")
    ax.set_box_aspect((2.0, 0.62, 1.25))

    # transpose so the long spatial axis runs left-to-right: slices read as wide strips
    xs = np.linspace(0, 1, H)
    ys = np.linspace(0, 1, W)
    Xg, Yg = np.meshgrid(xs, ys)

    for k, (centre, half, label) in enumerate(BANDS):
        img = band_image(cube, wn, centre, half).T        # -> (W, H)
        colors = plt.cm.viridis(img)
        colors[..., 3] = 0.96
        z = k * 1.25
        ax.plot_surface(Xg, Yg, np.full_like(Xg, z), facecolors=colors,
                        rstride=1, cstride=3, shade=False)
        ax.text(1.06, 0.5, z + 0.18, f"{centre} cm" + r"$^{-1}$", fontsize=9.5,
                color="0.15", va="center", ha="left", fontweight="bold")
        ax.text(1.06, 0.5, z - 0.14, label, fontsize=8, color="0.45",
                va="center", ha="left")

    # the spectral axis, drawn through one pixel
    px = PIXEL[0] / (H - 1)
    py = PIXEL[1] / (W - 1)
    ax.plot([px, px], [py, py], [-0.5, 3.05], color="#d62728", lw=2.4, zorder=99)
    for k in range(len(BANDS)):
        ax.scatter([px], [py], [k * 1.25], color="#d62728", s=30, zorder=100,
                   depthshade=False)
    ax.text(px, py, 3.42, "one pixel =\none spectrum", color="#d62728",
            fontsize=10.5, ha="center", fontweight="bold")

    ax.set_xlabel(f"Height  ({H} px)", labelpad=2, fontsize=9.5)
    ax.set_ylabel(f"Width  ({W} px)", labelpad=-8, fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_zlabel("")
    ax.view_init(elev=20, azim=-70)
    ax.set_title(r"(a) One slide $=$ a $(H,\; W,\; N_\lambda)$ tensor",
                 fontsize=12, y=0.99)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
        axis.line.set_alpha(0.25)
    ax.grid(False)

    # ---- right: the spectrum at that pixel --------------------------------
    ax2 = fig.add_subplot(gs[1])
    spec = cube[PIXEL[0], PIXEL[1], :]
    ax2.plot(wn, spec, color="#d62728", lw=0.85)
    top = spec.max() * 1.16
    ax2.set_ylim(spec.min() - 0.05 * abs(spec.min()) - 500, top)
    for centre, _, _ in BANDS:
        ax2.axvline(centre, color="0.65", lw=0.8, ls="--", zorder=0)
        ax2.text(centre, top * 0.985, f"{centre}", rotation=90, fontsize=8,
                 color="0.35", va="top", ha="right")
    ax2.set_xlabel(r"Raman shift (cm$^{-1}$)", fontsize=10)
    ax2.set_ylabel("Intensity (counts)", fontsize=10)
    ax2.set_title(rf"(b) The spectrum at that pixel: $N_\lambda = {L}$ values",
                  fontsize=12)
    ax2.margins(x=0.01)
    ax2.tick_params(labelsize=9)

    fig.text(0.5, 0.005,
             r"58 slides  $\rightarrow$  $\sim$270 000 spectra, each a "
             rf"{L}-dimensional vector",
             ha="center", fontsize=10.5, color="0.25")

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
