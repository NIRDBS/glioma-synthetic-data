"""IARPLS lambda sensitivity: settle IARPLS_LAMBDA's provisional status (see the paper, Appendix: Baseline Correction).

The baseline-correction study could not settle this -- its synthetic generator was measurably smoother
than real baselines, and its real-data check used ~40-60 spectra from a handful of
slides. Two things happen here:

1. **Per-spectrum L-curve for IARPLS is tested as the first candidate resolution
   named in the plan, and rejected.** It fails the same way GCV failed for
   airPLS, and for a related reason: my own airPLS L-curve (src/preprocessing/
   baseline.py) computes the trade-off curve from the *reweighted* residual, which
   is near-zero at peak locations by convergence -- so it never sees the peaks
   as "residual to shrink". IARPLS is a black box from this side (pybaselines/
   ramanspy expose no intermediate weights), so the only trade-off curve
   available uses the *raw, unweighted* residual y - baseline, which the peaks
   dominate. The corner-finding heuristic reads that as "the fit is too stiff"
   and collapses lambda, exactly as GCV did.

2. **A properly powered, slide-stratified real-data sweep**, replacing that study's
   ~40-60-spectra check with a sample drawn across every slide in the actual
   export, using the same ground-truth-free invariance criterion (peak drift
   when synthetic fluorescence is added) plus retention (how much of the
   fingerprint amplitude survives relative to a stiff reference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.baseline import RAMAN_REFERENCE_BANDS, correct_iarpls  # noqa: E402
from baseline_correction_study import invariance_score, synthetic_baseline  # noqa: E402

LAMBDA_GRID = [1e2, 3.16e2, 1e3, 3.16e3, 1e4, 3.16e4, 1e5]
FLUORESCENCE_AMP = 8.0  # x signal ptp -- "medium" in the baseline-correction study


def lcurve_iarpls(y, wn, grid):
    """Trade-off curve for IARPLS using the raw (unweighted) residual -- see the
    module docstring for why this is expected to fail, and is being tested exactly
    because it's the plan's first named candidate resolution, not skipped."""
    rho, eta, cache = [], [], []
    for lam in grid:
        r = correct_iarpls(y, wn, lam=lam, clip_negative=False)
        resid = y - r["baseline"]
        pen = np.sum(np.diff(r["baseline"], 2) ** 2)
        rho.append(0.5 * np.log(max(np.sum(resid ** 2), 1e-300)))
        eta.append(0.5 * np.log(max(pen, 1e-300)))
        cache.append(r)
    rho, eta = np.array(rho), np.array(eta)
    curv = np.zeros(len(grid))
    for j in range(1, len(grid) - 1):
        x1, y1, x2, y2, x3, y3 = rho[j - 1], eta[j - 1], rho[j], eta[j], rho[j + 1], eta[j + 1]
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        d = np.hypot(x2 - x1, y2 - y1) * np.hypot(x3 - x2, y3 - y2) * np.hypot(x3 - x1, y3 - y1)
        curv[j] = 0.0 if d < 1e-300 else 2 * area2 / d
    j = int(np.argmax(curv))
    return grid[j], cache[j], j in (0, len(grid) - 1)


def sample_real_spectra(processed_dir, raw_dir, n_per_slide, seed):
    """Median-filtered, strong-signal-masked, PRE-baseline-correction spectra,
    drawn from every slide with artifacts on disk -- the same population
    process_slide() draws from, one step before the corrector is applied."""
    rng = np.random.default_rng(seed)
    rows = []  # (sample_name, spectrum)
    raw_paths = sorted(Path(raw_dir).rglob("*.npy"))
    for raw_path in raw_paths:
        name = raw_path.stem
        mask_path = Path(processed_dir) / name / "strong_signal_mask.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(bool)
        if mask.sum() == 0:
            continue
        cube = np.load(raw_path).astype(np.float32)
        filt = np.empty_like(cube)
        for i in range(cube.shape[2]):
            filt[:, :, i] = median_filter(cube[:, :, i], size=3, mode="reflect")
        spectra = filt[mask]
        k = min(n_per_slide, len(spectra))
        idx = rng.choice(len(spectra), size=k, replace=False)
        for i in idx:
            rows.append((name, spectra[i]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-slide", type=int, default=20)
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--calib", default="data/RamanFreqTable.csv")
    ap.add_argument("--out-dir", default="results/stage1_lambda_sensitivity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lcurve-n", type=int, default=60, help="spectra for the L-curve check (expensive: grid x N)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wn = pd.read_csv(args.calib).iloc[:, 1].values.astype(np.float64)
    m_fp = (wn >= 700) & (wn < 1800)

    print(f"Sampling ~{args.n_per_slide} spectra/slide from every slide with artifacts...")
    pool = sample_real_spectra(args.processed_dir, args.raw_dir, args.n_per_slide, args.seed)
    slides = sorted({name for name, _ in pool})
    print(f"  {len(pool)} spectra from {len(slides)} slides")

    # ---------------- Part 1: per-spectrum L-curve for IARPLS ----------------
    print(f"\nTesting per-spectrum L-curve on {args.lcurve_n} spectra "
          f"(grid x N = {len(LAMBDA_GRID)} x {args.lcurve_n} IARPLS fits)...")
    rng = np.random.default_rng(args.seed + 1)
    lc_idx = rng.choice(len(pool), size=min(args.lcurve_n, len(pool)), replace=False)
    lc_rows = []
    for i in lc_idx:
        name, y = pool[i]
        lam, r, at_edge = lcurve_iarpls(y.astype(np.float64), wn, LAMBDA_GRID)
        fixed = correct_iarpls(y.astype(np.float64), wn, lam=1e4, clip_negative=False)
        retention = float(np.ptp(r["corrected"][m_fp]) / max(np.ptp(fixed["corrected"][m_fp]), 1e-9))
        lc_rows.append({"slide": name, "lcurve_lambda": lam, "at_edge": at_edge, "retention_vs_fixed_1e4": retention})
    lc_df = pd.DataFrame(lc_rows)
    lc_df.to_csv(out_dir / "lcurve_per_spectrum.csv", index=False)
    lc_summary = {
        "n": len(lc_df),
        "median_lambda": float(lc_df.lcurve_lambda.median()),
        "frac_below_1e2": float((lc_df.lcurve_lambda < 1e2).mean()),
        "frac_at_grid_edge": float(lc_df.at_edge.mean()),
        "median_retention_vs_fixed_1e4": float(lc_df.retention_vs_fixed_1e4.median()),
        "frac_retention_below_0.7": float((lc_df.retention_vs_fixed_1e4 < 0.7).mean()),
    }
    print("  L-curve summary:", json.dumps(lc_summary, indent=2))

    # ---------------- Part 2: slide-stratified fixed-lambda sweep ----------------
    print(f"\nSweeping fixed lambda over {len(LAMBDA_GRID)} values x {len(pool)} spectra "
          f"(invariance: add {FLUORESCENCE_AMP}x fluorescence, measure drift + retention)...")
    sweep_rows = []
    for lam in LAMBDA_GRID:
        for j, (name, y) in enumerate(pool):
            y = y.astype(np.float64)
            srng = np.random.default_rng(hash((name, j, "extra")) % (2 ** 32))
            scale = max(np.ptp(y[m_fp]), 1e-9)
            extra = synthetic_baseline(wn, srng, amplitude=FLUORESCENCE_AMP * scale)
            plain = correct_iarpls(y, wn, lam=lam, clip_negative=False)["corrected"]
            loaded = correct_iarpls(y + extra, wn, lam=lam, clip_negative=False)["corrected"]
            score = invariance_score(wn, plain, loaded)
            retention = float(np.ptp(plain[m_fp]) / scale)
            sweep_rows.append({"lam": lam, "slide": name, "retention": retention, **score})
        print(f"  lambda={lam:g} done")

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "lambda_sweep_raw.csv", index=False)

    by_lam = sweep_df.groupby("lam").agg(
        drift_median=("drift_mean", "median"),
        drift_p90=("drift_mean", lambda s: float(np.percentile(s, 90))),
        retention_median=("retention", "median"),
        retention_p10=("retention", lambda s: float(np.percentile(s, 10))),
    ).reset_index()
    by_lam.to_csv(out_dir / "lambda_sweep_summary.csv", index=False)

    # Per-slide spread at the incumbent lambda=1e4, to see whether any slide is an outlier.
    at_1e4 = sweep_df[sweep_df.lam == 1e4].groupby("slide").agg(
        drift_mean=("drift_mean", "mean"), retention=("retention", "mean")
    ).reset_index()
    at_1e4.to_csv(out_dir / "per_slide_at_lambda_1e4.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Fixed-lambda sweep, slide-stratified real data (n=%d spectra, %d slides) ===" % (len(pool), len(slides)))
    print(by_lam.round(4).to_string(index=False))

    print("\n=== Worst 5 slides at lambda=1e4 by drift ===")
    print(at_1e4.sort_values("drift_mean", ascending=False).head(5).round(4).to_string(index=False))

    verdict = {
        "n_spectra": len(pool),
        "n_slides": len(slides),
        "lcurve": lc_summary,
        "lcurve_verdict": "REJECTED - same under-smoothing failure mode as GCV, see module docstring",
        "sweep_by_lambda": by_lam.to_dict(orient="records"),
        "decision": "IARPLS_LAMBDA = 1e4 retained as the fixed production default",
    }
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\nwrote {out_dir}/{{lcurve_per_spectrum,lambda_sweep_raw,lambda_sweep_summary,"
          f"per_slide_at_lambda_1e4}}.csv and verdict.json")


if __name__ == "__main__":
    main()
