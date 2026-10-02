"""
Suite 600 beta-sweep diagnostic: plots the reconstruction (MSE) loss curve,
averaged across the 5 CV folds, for each beta value -- to distinguish whether
lower-beta's synthetic-diversity gain (docs/counterfactual_probe.md's
collapse finding; see the Suite 600 sweep) reflects genuinely better
reconstruction (lower mse_loss) or just a noisier decoder (similar or worse
mse_loss despite higher variance).

beta=2.0's history is PARTIAL (epochs 71-100 only) -- epochs 1-70 were lost
when the stale 70-epoch checkpoint's history_fold_*.csv was overwritten
during the resume-and-promote step rather than concatenated (a real mistake,
noted plainly rather than silently plotting a misleading full-looking
curve). Its segment is drawn dashed and only from epoch 71 to mark this.

Usage:
    python scripts/plot_beta_sweep_recon_loss.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIG_DPI = 150

# Ordinal-safe steps from the dataviz skill's reference sequential blue ramp
# (references/palette.md): light-mode chart, so the only documented
# constraint is "start no lighter than step 250" -- spread across the valid
# range, lightest=lowest beta, darkest=highest beta.
BETA_COLORS = {
    1.0:  "#86b6ef",  # step 250
    1.25: "#5598e7",  # step 350
    1.5:  "#2a78d6",  # step 450
    1.75: "#184f95",  # step 600
    2.0:  "#0d366b",  # step 700
}

CONFIGS = [
    (1.0,  "results/exp_601/models/cvae", None),
    (1.25, "results/exp_602/models/cvae", None),
    (1.5,  "results/exp_603/models/cvae", None),
    (1.75, "results/exp_604/models/cvae", None),
    (2.0,  "results/exp_501/models/cvae", 71),  # partial: epochs 71-100 only
]


def _load_mean_mse_loss(model_dir, min_epoch=None):
    """Average mse_loss across all 5 folds' history CSVs, per epoch."""
    per_fold = []
    for i in range(5):
        df = pd.read_csv(Path(model_dir) / f"history_fold_{i}.csv")
        if min_epoch is not None:
            df = df[df["epoch"] >= min_epoch]
        per_fold.append(df.set_index("epoch")["mse_loss"])
    combined = pd.concat(per_fold, axis=1)
    return combined.mean(axis=1)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))

    for beta, model_dir, min_epoch in CONFIGS:
        series = _load_mean_mse_loss(model_dir, min_epoch=min_epoch)
        label = f"β={beta}" + (" (epochs 71-100 only, see note)" if min_epoch else "")
        linestyle = "--" if min_epoch else "-"
        ax.plot(series.index, series.values, color=BETA_COLORS[beta],
                linewidth=2, linestyle=linestyle, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss (MSE, mean over 5 folds)")
    ax.set_title("Suite 600: CVAE reconstruction loss by β")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    fig.text(0.01, -0.02,
             "Note: β=2.0's epochs 1-70 were lost (history file overwritten, not\n"
             "concatenated, during a resume-and-promote step) -- only its converged\n"
             "tail (71-100) is shown, dashed, for honest comparison.",
             fontsize=7, color="#52514e", ha="left", va="top")

    out_dir = PROJECT_ROOT / "results" / "suite600_beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reconstruction_loss_by_beta.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\nFinal-epoch (converged) mean reconstruction loss by beta:")
    for beta, model_dir, min_epoch in CONFIGS:
        series = _load_mean_mse_loss(model_dir, min_epoch=min_epoch)
        print(f"  beta={beta}: final={series.iloc[-1]:.4f}  "
             f"last-10-epoch mean={series.iloc[-10:].mean():.4f}  "
             f"last-10-epoch std={series.iloc[-10:].std():.4f}")


if __name__ == "__main__":
    main()
