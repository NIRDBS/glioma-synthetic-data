r"""
Combined 2x2 figure for the paper's beta-sensitivity appendix: row 1 is the
flat-beta sweep (reconstruction | KL), row 2 is the windowed beta-schedule
continuation (reconstruction+beta-trace | KL), laid out as one matplotlib
figure instead of four separate images.

Why one native figure rather than four images placed side by side in LaTeX:
cramming two full-size images into half-width each via `\includegraphics`
shrinks everything in them -- including their text -- proportionally, and
at 0.48\linewidth the axis labels/legends become illegible. Building one
combined canvas lets matplotlib lay out all four panels with font sizes
set explicitly for this layout, so the legibility of the final page is a
property of the source figure, not of how much LaTeX ends up shrinking it.
Font sizes below are picked so that after this whole combined image is
placed at \linewidth (one figure, not shrunk further), each panel's text
renders at roughly the same effective on-page size the four original
single-panel figures had at their own \linewidth inclusion.

Usage:
    python scripts/plot_beta_sensitivity_combined.py
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
EPOCH_LO = 10
N_FOLDS = 5

# Font sizes compensate for the ~0.4x shrink this combined canvas undergoes
# when placed at \linewidth (vs. ~0.75x for each original standalone
# figure) -- roughly double the originals' sizes, not a stylistic choice.
FS_TITLE = 22
FS_LABEL = 18
FS_TICK = 16
FS_LEGEND_ROW1 = 15
FS_LEGEND_ROW2 = 15
FS_ROWLABEL = 20

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#6250d6", "#e34948"]

FIXED_CONFIGS = [
    (0.25, "results/exp_607/models/cvae", 120),
    (0.5,  "results/exp_608/models/cvae", 120),
    (0.75, "results/exp_609/models/cvae", 120),
    (1.0,  "results/exp_601/models/cvae", 120),
    (1.25, "results/exp_602/models/cvae", 100),
    (1.5,  "results/exp_603/models/cvae", 100),
    (1.75, "results/exp_604/models/cvae", 100),
    (2.0,  "results/exp_501/models/cvae", 100),
]
FIXED_COLORS = {beta: color for (beta, _, _), color in zip(FIXED_CONFIGS, CATEGORICAL)}

WINDOWED_EXPERIMENTS = [
    ("exp_601 (β=1.0, constant)",
     [("results/exp_601/models/cvae", EPOCH_LO, 100), ("results/exp_601/models/cvae", 101, 200)],
     "#2a78d6"),
    ("exp_605 (β window: 1.25/1 alt.)",
     [("results/exp_601/models/cvae", EPOCH_LO, 100), ("results/exp_605/models/cvae", 101, 200)],
     "#eb6834"),
    ("exp_606 (β window: 0.75/1 alt.)",
     [("results/exp_601/models/cvae", EPOCH_LO, 100), ("results/exp_606/models/cvae", 101, 200)],
     "#1baf7a"),
]
BRANCH_EPOCH = 100.5


def _load_mean_loss(model_dir, column, epoch_hi):
    per_fold = []
    for i in range(N_FOLDS):
        df = pd.read_csv(Path(model_dir) / f"history_fold_{i}.csv")
        df = df[(df["epoch"] >= EPOCH_LO) & (df["epoch"] <= epoch_hi)]
        per_fold.append(df.set_index("epoch")[column])
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _load_mean_series(segments, column):
    per_fold = []
    for fold_idx in range(N_FOLDS):
        parts = []
        for model_dir, lo, hi in segments:
            df = pd.read_csv(Path(model_dir) / f"history_fold_{fold_idx}.csv")
            parts.append(df[(df["epoch"] >= lo) & (df["epoch"] <= hi)])
        fold_df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="epoch")
        per_fold.append(fold_df.set_index("epoch")[column].sort_index())
    return pd.concat(per_fold, axis=1).mean(axis=1)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis="y", which="both", linewidth=0.6, alpha=0.3)


def _draw_fixed_beta_panels(ax_recon, ax_kl):
    """Flat-beta sweep: reconstruction (left) and KL (right) loss curves.
    Shared by the 4-pane and the 2-pane figure so they cannot drift apart.
    Returns (lines, labels) for the legend."""
    lines, labels = [], []
    for beta, model_dir, epoch_hi in FIXED_CONFIGS:
        recon = _load_mean_loss(model_dir, "mse_loss", epoch_hi)
        line, = ax_recon.plot(recon.index, recon.values, color=FIXED_COLORS[beta], linewidth=2)
        lines.append(line)
        labels.append(f"β={beta}")
        kl = _load_mean_loss(model_dir, "kl_loss", epoch_hi)
        ax_kl.plot(kl.index, kl.values, color=FIXED_COLORS[beta], linewidth=2)

    for ax, title in ((ax_recon, "Reconstruction loss"), (ax_kl, "KL loss")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=FS_LABEL)
        ax.set_title(title, fontsize=FS_TITLE)
        ax.set_xlim(EPOCH_LO, 120)
        _style_axes(ax)
    ax_recon.set_ylabel("MSE (log scale)", fontsize=FS_LABEL)
    ax_kl.set_ylabel("KL (log scale)", fontsize=FS_LABEL)
    return lines, labels


def _save(fig, name):
    out_dir = PROJECT_ROOT / "paper" / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_two_pane():
    """Just the flat-beta row (reconstruction | KL) of the combined figure,
    same canvas width and font sizes so it drops in at \\linewidth with the
    same on-page text size; only the height differs."""
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.08, right=0.98, top=0.78, bottom=0.17)
    ax_recon = fig.add_subplot(gs[0, 0])
    ax_kl = fig.add_subplot(gs[0, 1])
    lines, labels = _draw_fixed_beta_panels(ax_recon, ax_kl)
    fig.legend(lines, labels, loc="upper center", ncol=8, frameon=False,
              fontsize=FS_LEGEND_ROW1, bbox_to_anchor=(0.53, 0.985), columnspacing=1.2,
              handlelength=1.4)
    _save(fig, "beta_sensitivity_fixed.png")


def make_four_pane():
    fig = plt.figure(figsize=(16, 12))
    outer = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.38, wspace=0.22,
                             left=0.08, right=0.98, top=0.90, bottom=0.07)

    # --- Row 1: fixed-beta sweep (recon | KL) ---
    ax_fixed_recon = fig.add_subplot(outer[0, 0])
    ax_fixed_kl = fig.add_subplot(outer[0, 1])
    fixed_lines, fixed_labels = _draw_fixed_beta_panels(ax_fixed_recon, ax_fixed_kl)

    fig.legend(fixed_lines, fixed_labels, loc="upper center", ncol=8, frameon=False,
              fontsize=FS_LEGEND_ROW1, bbox_to_anchor=(0.53, 0.985), columnspacing=1.2,
              handlelength=1.4)

    # --- Row 2: windowed continuation (recon+beta-trace | KL) ---
    inner = outer[1, 0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
    ax_win_recon = fig.add_subplot(inner[0])
    ax_win_beta = fig.add_subplot(inner[1], sharex=ax_win_recon)
    ax_win_kl = fig.add_subplot(outer[1, 1])

    win_lines, win_labels = [], []
    for label, segments, color in WINDOWED_EXPERIMENTS:
        recon = _load_mean_series(segments, "mse_loss")
        line, = ax_win_recon.plot(recon.index, recon.values, color=color, linewidth=2)
        win_lines.append(line)
        win_labels.append(label)

        beta_trace = _load_mean_series(segments, "beta")
        ax_win_beta.step(beta_trace.index, beta_trace.values, color=color, linewidth=2, where="post")

        kl = _load_mean_series(segments, "kl_loss")
        ax_win_kl.plot(kl.index, kl.values, color=color, linewidth=2)

    for ax in (ax_win_recon, ax_win_beta, ax_win_kl):
        ax.axvline(BRANCH_EPOCH, color="#c3c2b7", linewidth=1, linestyle="--", zorder=0)
        _style_axes(ax)
        ax.set_xlim(EPOCH_LO, 200)
    ax_win_recon.text(BRANCH_EPOCH + 2, ax_win_recon.get_ylim()[1], "branch point",
                      fontsize=FS_TICK - 2, color="#898781", ha="left", va="top")

    ax_win_recon.set_ylabel("MSE", fontsize=FS_LABEL)
    ax_win_recon.set_title("Reconstruction loss (windowed)", fontsize=FS_TITLE)
    plt.setp(ax_win_recon.get_xticklabels(), visible=False)

    ax_win_beta.set_ylabel("β", fontsize=FS_LABEL - 2)
    ax_win_beta.set_xlabel("Epoch", fontsize=FS_LABEL)
    ax_win_beta.set_yticks([0.75, 1.0, 1.25])
    ax_win_beta.set_ylim(0.6, 1.4)
    ax_win_beta.tick_params(axis="y", labelsize=FS_TICK - 2)

    ax_win_kl.set_ylabel("KL loss", fontsize=FS_LABEL)
    ax_win_kl.set_xlabel("Epoch", fontsize=FS_LABEL)
    ax_win_kl.set_title("KL loss (windowed)", fontsize=FS_TITLE)

    # Row-group labels and the row-2 legend are placed from each row's ACTUAL
    # post-layout bounding box (via get_position(), after a draw pass) rather
    # than guessed figure-fraction coordinates -- guessed fractions collided
    # with axis labels/data curves on the first pass.
    fig.canvas.draw()
    pos_fixed_recon = ax_fixed_recon.get_position()
    pos_fixed_kl = ax_fixed_kl.get_position()
    pos_win_recon = ax_win_recon.get_position()
    pos_win_beta = ax_win_beta.get_position()

    row1_top, row1_bot = pos_fixed_recon.y1, pos_fixed_recon.y0
    row2_top, row2_bot = pos_win_recon.y1, pos_win_beta.y0

    fig.text(0.015, (row1_top + row1_bot) / 2, "Fixed β", fontsize=FS_ROWLABEL,
             rotation=90, va="center", ha="center")
    fig.text(0.015, (row2_top + row2_bot) / 2, "Windowed β", fontsize=FS_ROWLABEL,
             rotation=90, va="center", ha="center")

    # Row-2 legend (3 entries) in the gap between the two rows, mirroring how
    # row-1's 8-entry legend sits above row-1 -- inside the axes, it collided
    # with the data curves (the shared pre-branch history occupies most of
    # the panel's otherwise-open space).
    fig.legend(win_lines, win_labels, loc="upper center", ncol=3, frameon=False,
              fontsize=FS_LEGEND_ROW2, bbox_to_anchor=(0.53, (row1_bot + row2_top) / 2 + 0.03),
              columnspacing=1.2, handlelength=1.4)

    _save(fig, "beta_sensitivity_combined.png")


def main(panes="both"):
    if panes in ("2", "both"):
        make_two_pane()
    if panes in ("4", "both"):
        make_four_pane()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panes", choices=["2", "4", "both"], default="both",
                        help="2: flat-beta row only (beta_sensitivity_fixed.png); "
                             "4: full 2x2 (beta_sensitivity_combined.png); both: default")
    main(parser.parse_args().panes)
