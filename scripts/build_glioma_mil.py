"""Build data/processed/glioma_mil.h5 -- the Stage 1 canonical dataset.

Requires the per-slide clustering artifacts in data/processed/<slide>/ to already
exist (produced by notebook 01 sections 2-3: strong_signal_mask.npy, hard_labels.npy,
tumor_mask.npy). This script does not recompute them.

Usage:
    python scripts/build_glioma_mil.py [--out data/processed/glioma_mil.h5] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import RamanPreprocessor  # noqa: E402
from src.preprocessing.baseline import IARPLS_LAMBDA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--out", default="data/processed/glioma_mil.h5")
    ap.add_argument("--calib", default="data/RamanFreqTable.csv")
    ap.add_argument("--lambda", dest="lam", type=float, default=IARPLS_LAMBDA)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc = RamanPreprocessor(
        raw_data_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        output_file=args.out,
        calibration_path=args.calib,
        baseline_lambda=args.lam,
        n_workers=args.workers,
    )
    out = proc.run()
    dt = time.perf_counter() - t0
    print(f"\nWrote {out} in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
