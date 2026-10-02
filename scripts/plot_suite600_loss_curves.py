"""
Suite 600 beta sweep: full-trajectory loss curves for exp_601-604 (beta=1.0/
1.25/1.5/1.75 -- all four have complete 1-100 epoch histories, unlike the
corrected exp_501/beta=2.0 reference, which is left out of these two plots
since the user's request scoped to "the 600 suite experiments" specifically).

Two separate figures (reconstruction MSE, KL), each averaged across the 5 CV
folds -- to judge whether the loss is still decreasing at epoch 100 (extended
epochs would plausibly help) or has plateaued (they likely wouldn't).

Log-scale y-axis on both: the steep epoch 1-5 drop otherwise compresses the
epoch 20-100 plateau region into an unreadable sliver on a linear axis, which
is exactly the region this plot needs to resolve.

Usage:
    python scripts/plot_suite600_loss_curves.py
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

# Same ordinal-safe sequential-blue steps as plot_beta_sweep_recon_loss.py,
# restricted to the 4 Suite 600 values (beta=2.0's step 700 dropped here).
BETA_COLORS = {
    1.0:  "#86b6ef",  # step 250
    1.25: "#5598e7",  # step 350
    1.5:  "#2a78d6",  # step 450
    1.75: "#184f95",  # step 600
}

CONFIGS = [
    (1.0,  "results/exp_601/models/cvae"),
    (1.25, "results/exp_602/models/cvae"),
    (1.5,  "results/exp_603/models/cvae"),
    (1.75, "results/exp_604/models/cvae"),
]


def _load_mean_loss(model_dir, column):
    per_fold = []
    for i in range(5):
        df = pd.read_csv(Path(model_dir) / f"history_fold_{i}.csv")
        per_fold.append(df.set_index("epoch")[column])
    combined = pd.concat(per_fold, axis=1)
    return combined.mean(axis=1)


def _plot_one(column, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    for beta, model_dir in CONFIGS:
        series = _load_mean_loss(model_dir, column)
        ax.plot(series.index, series.values, color=BETA_COLORS[beta],
               linewidth=2, label=f"β={beta}")
        print(f"  [{column}] beta={beta}: epoch100={series.iloc[-1]:.4f}  "
             f"epoch90={series.iloc[89]:.4f}  "
             f"delta(90->100)={series.iloc[-1]-series.iloc[89]:+.5f}")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    _plot_one("mse_loss", "Reconstruction loss (MSE, mean over 5 folds, log scale)",
             "Suite 600: reconstruction loss by β", "suite600_recon_loss.png")

    print("KL loss:")
    _plot_one("kl_loss", "KL loss (mean over 5 folds, log scale)",
             "Suite 600: KL loss by β", "suite600_kl_loss.png")


if __name__ == "__main__":
    main()
