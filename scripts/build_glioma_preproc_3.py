"""Build a dataset with a fixed maximum-intensity threshold (src/data/preprocessing_v3.py).

    --threshold 0.6  ->  data/processed/glioma_mil.h5   (the paper's main dataset)
    --threshold 0.4  ->  data/processed/glioma-preproc-3.h5   (exp_4.608; the default)

Usage:
    python scripts/build_glioma_preproc_3.py --threshold 0.6 --out data/processed/glioma_mil.h5 [--workers N]
    python scripts/build_glioma_preproc_3.py --threshold 0.4 --out data/processed/glioma-preproc-3.h5
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
