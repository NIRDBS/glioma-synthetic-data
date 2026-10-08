"""Build data/processed/glioma-preproc-2.h5 -- the fully-automated v2 dataset.

No precomputed artifacts required (unlike scripts/build_glioma_mil.py): this
pipeline has no NMF/GMM clustering and no manual tumour-cluster curation step,
so a single run does load -> median filter -> cube-normalise -> adaptive
strong-signal mask -> IARPLS -> per-spectrum normalise -> HDF5, start to finish.
See src/data/preprocessing_v2.py for the full pipeline and what it deliberately
no longer does.

Usage:
    python scripts/build_glioma_preproc_2.py [--out data/processed/glioma-preproc-2.h5] [--workers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing_v2 import RamanPreprocessorV2  # noqa: E402
from src.preprocessing.baseline import IARPLS_LAMBDA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--processed-dir", default=None,
                    help="if given, persists strong_signal_mask_v2.npy per slide here "
                         "for inspectability (optional -- not required for correctness)")
    ap.add_argument("--out", default="data/processed/glioma-preproc-2.h5")
    ap.add_argument("--calib", default="data/RamanFreqTable.csv")
    ap.add_argument("--lambda", dest="lam", type=float, default=IARPLS_LAMBDA)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    proc = RamanPreprocessorV2(
        raw_data_dir=args.raw_dir,
        output_file=args.out,
        calibration_path=args.calib,
        baseline_lambda=args.lam,
        processed_dir=args.processed_dir,
        n_workers=args.workers,
    )
    out = proc.run()
    dt = time.perf_counter() - t0
    print(f"\nWrote {out} in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
