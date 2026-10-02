"""
Concatenate a CVAE continuation run's history CSVs onto the checkpoint it
resumed from, instead of overwriting -- built specifically so Experiment 1 of
the exp_601 beta=1.0 extension (another 100 epochs, same constant beta) does
NOT repeat the exp_501 resume-and-promote mistake earlier this session, where
`cp cvae_resumed/history_fold_*.csv cvae/` silently destroyed epochs 1-70
(never backed up, since only .weights.h5 are uploaded to the Modal resume
checkpoint, not .csv) instead of merging with them.

For each fold, reads OLD_DIR/history_fold_N.csv (epochs 1..E_old) and
NEW_DIR/history_fold_N.csv (epochs E_old+1..E_old+E_new, written by a
resumed/epoch_offset=E_old run), verifies the epoch sequence is contiguous
with no gaps or duplicates, and writes the concatenated result. Also copies
the NEW run's (latest) .weights.h5 files, since those represent the
further-trained model.

Promoting onto OLD_DIR itself (the common case: OUT_DIR == OLD_DIR) backs up
the existing history_fold_N.csv to history_fold_N.csv.bak first, belt-and-
suspenders given the prior incident -- the merged content already includes
every original row, but a recoverable backup costs nothing.

Usage:
    python scripts/concat_cvae_history.py \\
        --old results/exp_601/models/cvae \\
        --new results/exp_601/models/cvae_extended \\
        --out results/exp_601/models/cvae \\
        [--dry-run]
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

N_FOLDS = 5


def _concat_one_fold(old_dir, new_dir, fold_idx):
    old_path = Path(old_dir) / f"history_fold_{fold_idx}.csv"
    new_path = Path(new_dir) / f"history_fold_{fold_idx}.csv"
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)

    old_epochs = old_df["epoch"].tolist()
    new_epochs = new_df["epoch"].tolist()

    if sorted(set(old_epochs) & set(new_epochs)):
        raise ValueError(
            f"fold {fold_idx}: overlapping epoch numbers between old and new "
            f"history ({sorted(set(old_epochs) & set(new_epochs))}) -- refusing "
            f"to concatenate, since this would duplicate or silently prefer one "
            f"run's rows over the other's."
        )
    expected_start = max(old_epochs) + 1
    if min(new_epochs) != expected_start:
        raise ValueError(
            f"fold {fold_idx}: new history starts at epoch {min(new_epochs)}, "
            f"expected {expected_start} (old history's last epoch + 1) -- "
            f"epoch_offset was likely wrong at launch time."
        )

    merged = pd.concat([old_df, new_df], ignore_index=True).sort_values("epoch")
    gaps = merged["epoch"].diff().dropna()
    if not (gaps == 1).all():
        raise ValueError(
            f"fold {fold_idx}: merged epoch sequence has gaps: "
            f"{merged['epoch'].tolist()}"
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="dir with the original (e.g. epochs 1-100) history + weights")
    parser.add_argument("--new", required=True, help="dir with the continuation (e.g. epochs 101-200) history + weights")
    parser.add_argument("--out", required=True, help="dir to write the concatenated history + promoted weights into")
    parser.add_argument("--dry-run", action="store_true", help="print what would happen, write nothing")
    args = parser.parse_args()

    old_dir, new_dir, out_dir = Path(args.old), Path(args.new), Path(args.out)
    merged_per_fold = {}
    for fold_idx in range(N_FOLDS):
        merged = _concat_one_fold(old_dir, new_dir, fold_idx)
        merged_per_fold[fold_idx] = merged
        print(f"  fold {fold_idx}: {len(merged)} epochs merged "
             f"({merged['epoch'].min()}-{merged['epoch'].max()})")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, merged in merged_per_fold.items():
        out_csv = out_dir / f"history_fold_{fold_idx}.csv"
        if out_csv.exists() and out_csv.resolve() != (old_dir / f"history_fold_{fold_idx}.csv").resolve():
            pass  # out_dir differs from old_dir -- no pre-existing file of ours to protect
        elif out_csv.exists():
            backup = out_dir / f"history_fold_{fold_idx}.csv.bak"
            shutil.copy(out_csv, backup)
            print(f"  backed up existing {out_csv} -> {backup}")
        merged.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}")

    for weight_file in sorted(new_dir.glob("*.weights.h5")):
        dst = out_dir / weight_file.name
        shutil.copy(weight_file, dst)
        print(f"  promoted weights: {weight_file} -> {dst}")

    print(f"\nDone. {out_dir} now holds the full concatenated history and the "
         f"latest ({args.new}) weights.")


if __name__ == "__main__":
    main()
