"""Build data/processed/glioma-preproc-3.h5 -- the fixed-threshold v3 dataset.

Same automated, no-NMF/GMM-clustering pipeline as
scripts/build_glioma_preproc_2.py, with one difference: the strong-signal
mask is a fixed max-intensity threshold (default 0.4) instead of v2's
adaptive per-slide percentile rule. See src/data/preprocessing_v3.py for the
full pipeline and why 0.4 specifically (an informal 0.6-vs-0.8 comparison on
the current pipeline found looser cuts performing better; v1's own current
default dataset turns out to already BE the fixed-0.6 case, confirmed
directly against the archived batch-processing script that produced its
strong_signal_mask.npy artifacts).

Usage:
    python scripts/build_glioma_preproc_3.py [--threshold 0.4] [--out data/processed/glioma-preproc-3.h5] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing_v3 import DEFAULT_THRESHOLD, RamanPreprocessorV3  # noqa: E402
from src.preprocessing.baseline import IARPLS_LAMBDA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default=None,
                    help="if given, persists strong_signal_mask_v3.npy per slide here "
                         "for inspectability (optional -- not required for correctness)")
    ap.add_argument("--out", default="data/processed/glioma-preproc-3.h5")
    ap.add_argument("--calib", default="data/RamanFreqTable.csv")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--lambda", dest="lam", type=float, default=IARPLS_LAMBDA)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc = RamanPreprocessorV3(
        raw_data_dir=args.raw_dir,
        output_file=args.out,
        calibration_path=args.calib,
        threshold=args.threshold,
        baseline_lambda=args.lam,
        processed_dir=args.processed_dir,
        n_workers=args.workers,
    )
    out = proc.run()
    dt = time.perf_counter() - t0
    print(f"\nWrote {out} in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
