"""
KL loss, same 3-way parallel CVAE continuation as
plot_beta_window_continuation.py (exp_601 constant-beta continuation vs
exp_605/exp_606 windowed beta schedules), epochs 50-200.

A separate figure from the reconstruction-loss one (not a second pane under
it): unlike reconstruction loss, KL loss is NOT beta-invariant -- it's
expected to visibly track each schedule's beta value directly, so plotting
it alongside recon loss on a shared y-scale would be misleading. Kept here
as its own single-pane chart instead.

Usage:
    python scripts/plot_beta_window_continuation_kl.py
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
N_FOLDS = 5

EXPERIMENTS = [
    ("exp_601 (beta=1.0, constant)",
     [("results/exp_601/models/cvae", 50, 100), ("results/exp_601/models/cvae", 101, 200)],
     "#2a78d6"),
    ("exp_605 (beta window: 1.25/1 alt.)",
     [("results/exp_601/models/cvae", 50, 100), ("results/exp_605/models/cvae", 101, 200)],
     "#eb6834"),
    ("exp_606 (beta window: 0.75/1 alt.)",
     [("results/exp_601/models/cvae", 50, 100), ("results/exp_606/models/cvae", 101, 200)],
     "#1baf7a"),
]

BRANCH_EPOCH = 100.5


def _load_mean_series(segments, column):
    per_fold = []
    for fold_idx in range(N_FOLDS):
        parts = []
        for model_dir, lo, hi in segments:
            df = pd.read_csv(Path(model_dir) / f"history_fold_{fold_idx}.csv")
            parts.append(df[(df["epoch"] >= lo) & (df["epoch"] <= hi)])
        fold_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="epoch")
        per_fold.append(fold_df.set_index("epoch")[column].sort_index())
    combined = pd.concat(per_fold, axis=1)
    return combined.mean(axis=1)


def main():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, segments, color in EXPERIMENTS:
        series = _load_mean_series(segments, "kl_loss")
        ax.plot(series.index, series.values, color=color, linewidth=2, label=label)

    ax.axvline(BRANCH_EPOCH, color="#c3c2b7", linewidth=1, linestyle="--", zorder=0)
    ax.text(BRANCH_EPOCH + 1, ax.get_ylim()[1], "branch point",
           fontsize=7, color="#898781", ha="left", va="top")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL loss (mean over 5 folds)")
    ax.set_title("exp_601 continuation: constant β vs windowed β-annealing — KL loss (epochs 50-200)")
    ax.set_xlim(50, 200)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    out_dir = PROJECT_ROOT / "results" / "suite600_beta_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "continuation_kl_loss.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\nFinal-epoch (epoch 200) mean KL loss:")
    for label, segments, _ in EXPERIMENTS:
        series = _load_mean_series(segments, "kl_loss")
        safe_label = label.encode("ascii", "replace").decode("ascii")
        print(f"  {safe_label}: epoch200={series.iloc[-1]:.4f}  "
             f"last-10-epoch mean={series.iloc[-10:].mean():.4f}")


if __name__ == "__main__":
    main()
