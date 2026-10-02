"""
Suite 600 low-beta extension: reconstruction (MSE) and KL loss curves for
beta=0.25/0.5/0.75 (exp_607/608/609, flat, trained from scratch for 120
epochs) alongside beta=1.0 (exp_601, flat, now with a full 1-200 epoch
history after its continuation + concatenation -- sliced to the same range).

Two separate figures (same split as plot_suite600_loss_curves.py), epochs
20-120 only: the steep epoch 1-20 drop otherwise dominates the y-scale and
hides the plateau region these configs are actually being compared on.

Usage:
    python scripts/plot_beta_sweep_low_range.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIG_DPI = 150
EPOCH_LO, EPOCH_HI = 20, 120

# Same ordinal-safe sequential-blue steps as plot_suite600_loss_curves.py --
# lighter = lower beta, darker = higher beta.
BETA_COLORS = {
    0.25: "#86b6ef",  # step 250
    0.5:  "#5598e7",  # step 350
    0.75: "#2a78d6",  # step 450
    1.0:  "#184f95",  # step 600
}

CONFIGS = [
    (0.25, "results/exp_607/models/cvae"),
    (0.5,  "results/exp_608/models/cvae"),
    (0.75, "results/exp_609/models/cvae"),
    (1.0,  "results/exp_601/models/cvae"),
]


def _load_mean_loss(model_dir, column):
    per_fold = []
    for i in range(5):
        df = pd.read_csv(Path(model_dir) / f"history_fold_{i}.csv")
        df = df[(df["epoch"] >= EPOCH_LO) & (df["epoch"] <= EPOCH_HI)]
        per_fold.append(df.set_index("epoch")[column])
    combined = pd.concat(per_fold, axis=1)
    return combined.mean(axis=1)


def _plot_one(column, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    for beta, model_dir in CONFIGS:
        series = _load_mean_loss(model_dir, column)
        ax.plot(series.index, series.values, color=BETA_COLORS[beta],
               linewidth=2, label=f"β={beta}")
        print(f"  [{column}] beta={beta}: epoch{EPOCH_HI}={series.iloc[-1]:.4f}  "
             f"epoch{EPOCH_LO}={series.iloc[0]:.4f}  "
             f"delta({EPOCH_LO}->{EPOCH_HI})={series.iloc[-1]-series.iloc[0]:+.5f}")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(EPOCH_LO, EPOCH_HI)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", linewidth=0.5, alpha=0.3)

    out_dir = PROJECT_ROOT / "results" / "suite600_beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out_path}\n")


def main():
    print("Reconstruction (MSE) loss:")
    _plot_one("mse_loss", f"Reconstruction loss (MSE, mean over 5 folds, log scale)",
             "Suite 600 low-β extension: reconstruction loss (epochs 20-120)",
             "low_beta_recon_loss.png")

    print("KL loss:")
    _plot_one("kl_loss", "KL loss (mean over 5 folds, log scale)",
             "Suite 600 low-β extension: KL loss (epochs 20-120)",
             "low_beta_kl_loss.png")


if __name__ == "__main__":
    main()
