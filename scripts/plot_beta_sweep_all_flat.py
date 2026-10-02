"""
All flat (constant-beta) Suite 600 variants together: beta=0.25/0.5/0.75
(exp_607/608/609, trained 120 epochs), 1.0 (exp_601, trained to 200 but
sliced here), 1.25/1.5/1.75 (exp_602/603/604, trained 100 epochs only).

Epochs 20-120 where available, 20-100 for the three configs never trained
past 100 -- each series simply ends at its own config's max epoch rather
than being padded or interpolated.

Categorical palette (not the single-hue sequential ramp used for the
narrower beta<1 comparisons): with 7 series on a log-scale plot, a single
light->dark blue ramp compresses into hard-to-distinguish adjacent steps,
so this uses the dataviz skill's fixed-order 8-hue categorical theme
instead, assigned in ascending beta order (references/palette.md's
categorical table, which validates the adjacent-pairlist CVD/contrast
gates for a line chart's full 8 slots).

Usage:
    python scripts/plot_beta_sweep_all_flat.py
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
EPOCH_LO = 20

# Fixed-order categorical palette (references/palette.md), assigned in
# ascending beta order -- never cycled, never re-ordered per chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7"]

# (beta, model_dir, epoch_hi)
CONFIGS = [
    (0.25, "results/exp_607/models/cvae", 120),
    (0.5,  "results/exp_608/models/cvae", 120),
    (0.75, "results/exp_609/models/cvae", 120),
    (1.0,  "results/exp_601/models/cvae", 120),
    (1.25, "results/exp_602/models/cvae", 100),
    (1.5,  "results/exp_603/models/cvae", 100),
    (1.75, "results/exp_604/models/cvae", 100),
]
BETA_COLORS = {beta: color for (beta, _, _), color in zip(CONFIGS, CATEGORICAL)}


def _load_mean_loss(model_dir, column, epoch_hi):
    per_fold = []
    for i in range(5):
        df = pd.read_csv(Path(model_dir) / f"history_fold_{i}.csv")
        df = df[(df["epoch"] >= EPOCH_LO) & (df["epoch"] <= epoch_hi)]
        per_fold.append(df.set_index("epoch")[column])
    combined = pd.concat(per_fold, axis=1)
    return combined.mean(axis=1)


def _plot_one(column, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for beta, model_dir, epoch_hi in CONFIGS:
        series = _load_mean_loss(model_dir, column, epoch_hi)
        ax.plot(series.index, series.values, color=BETA_COLORS[beta],
               linewidth=2, label=f"β={beta}")
        print(f"  [{column}] beta={beta}: epoch{epoch_hi}={series.iloc[-1]:.4f}  "
             f"epoch{EPOCH_LO}={series.iloc[0]:.4f}")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(EPOCH_LO, 120)
    ax.legend(frameon=False, fontsize=9, ncol=2)
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
             "Suite 600 (all flat β): reconstruction loss (epochs 20-120/100)",
             "all_flat_beta_recon_loss.png")

    print("KL loss:")
    _plot_one("kl_loss", "KL loss (mean over 5 folds, log scale)",
             "Suite 600 (all flat β): KL loss (epochs 20-120/100)",
             "all_flat_beta_kl_loss.png")


if __name__ == "__main__":
    main()
