"""
The 3-way parallel CVAE continuation (epochs 101-200, all branching from the
same exp_601 beta=1.0, 100-epoch checkpoint): a constant-beta continuation
(exp_601 itself) vs two windowed beta-annealing schedules (exp_605, exp_606).

Two x-synced panes: reconstruction (MSE) loss on top (the metric that's
beta-invariant and therefore comparable across differently-scheduled runs --
same principle as plot_beta_sweep_recon_loss.py), beta trace on the bottom
(narrow -- it only ever takes 3 distinct values: 0.75/1.0/1.25).

Epochs 10-200: epoch 10 (not 1) as the left edge for the same reason as
plot_beta_sweep_all_flat.py -- the steep initial drop is confined to
epochs 1-2 (confirmed directly against the history CSVs, both for a low-β
and a β=1.0 series; epoch-10 values are already the same order of magnitude
as epoch-50 ones), so a uniform epoch-10 floor across every appendix figure
costs nothing here either. Epochs 10-100 are themselves identical across
all three experiments (literally the same training run, read once from
exp_601's history) -- the three lines are drawn coincident up to epoch 100,
then diverge at 101.

Usage:
    python scripts/plot_beta_window_continuation.py
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

# Categorical palette (references/palette.md), fixed order -- these are 3
# distinct experiment identities, not an ordinal continuum, so categorical
# hues (not the sequential blue ramp used for the beta-value sweep) are the
# right tool here.
EXPERIMENTS = [
    ("exp_601 (β=1.0, constant)",
     [("results/exp_601/models/cvae", 10, 100), ("results/exp_601/models/cvae", 101, 200)],
     "#2a78d6"),  # slot 1 blue
    ("exp_605 (β window: 1.25/1 alt.)",
     [("results/exp_601/models/cvae", 10, 100), ("results/exp_605/models/cvae", 101, 200)],
     "#eb6834"),  # slot 2 orange
    ("exp_606 (β window: 0.75/1 alt.)",
     [("results/exp_601/models/cvae", 10, 100), ("results/exp_606/models/cvae", 101, 200)],
     "#1baf7a"),  # slot 3 aqua
]

BRANCH_EPOCH = 100.5  # divergence point between the 3 runs


def _load_mean_series(segments, column):
    """Average `column` over the 5 CV folds, for one or more (dir, lo, hi)
    segments concatenated end-to-end (the shared-trunk + own-branch pattern
    above)."""
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
    fig, (ax_loss, ax_beta) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    for label, segments, color in EXPERIMENTS:
        loss_series = _load_mean_series(segments, "mse_loss")
        ax_loss.plot(loss_series.index, loss_series.values, color=color,
                    linewidth=2, label=label)

        beta_series = _load_mean_series(segments, "beta")
        ax_beta.step(beta_series.index, beta_series.values, color=color,
                     linewidth=2, where="post")

    for ax in (ax_loss, ax_beta):
        ax.axvline(BRANCH_EPOCH, color="#c3c2b7", linewidth=1, linestyle="--", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    ax_loss.set_ylabel("Reconstruction loss (MSE, mean over 5 folds)")
    ax_loss.set_title("exp_601 continuation: constant β vs windowed β-annealing (epochs 10-200)")
    ax_loss.legend(frameon=False, fontsize=9, loc="upper right")
    ax_loss.text(BRANCH_EPOCH + 1, ax_loss.get_ylim()[1], "branch point",
                fontsize=7, color="#898781", ha="left", va="top")

    ax_beta.set_ylabel("β", fontsize=9)
    ax_beta.set_xlabel("Epoch")
    ax_beta.set_yticks([0.75, 1.0, 1.25])
    ax_beta.set_ylim(0.6, 1.4)
    ax_beta.set_xlim(10, 200)

    out_dir = PROJECT_ROOT / "paper" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "beta_window_continuation_recon.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\nFinal-epoch (epoch 200) mean reconstruction loss:")
    for label, segments, _ in EXPERIMENTS:
        series = _load_mean_series(segments, "mse_loss")
        safe_label = label.encode("ascii", "replace").decode("ascii")
        print(f"  {safe_label}: epoch200={series.iloc[-1]:.4f}  "
             f"last-10-epoch mean={series.iloc[-10:].mean():.4f}")


if __name__ == "__main__":
    main()
