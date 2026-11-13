"""Generator stress test: how much of the lambda-insensitivity is our generator?

Kizilov et al. (J. Raman Spectrosc. 2026, arXiv:2507.17917) report that IARPLS is
*more* sensitive to its parameters than ALS, over lambda in 1e2..1e9. The baseline-
correction study found lambda effectively inert over 1e2..1e4. Those are reconcilable only if
our synthetic baselines are drawn from an easier family than theirs -- a smooth,
low-curvature one that a second-difference penalty represents well over a wide range
of lambda.

This script tests that directly by making the generator harder along three axes and
watching where lambda sensitivity appears:

* **baseline complexity** -- `smooth` (the baseline study's family: one broad Gaussian +
  Rayleigh wing + cubic drift) through `structured` (several fluorophore-like
  Gaussians) to `sharp` (adds a narrow, high-curvature component).
* **band lineshape** -- Gaussian vs Lorentzian. Lorentzian tails are heavier, so a
  flexible baseline can hide under them; Raman bands are closer to Lorentzian/Voigt.
* **noise model** -- i.i.d. Gaussian vs Poisson shot noise, which is what a photon
  detector actually produces and is signal-dependent.

Curvature energy ``sum((d2 b)^2)`` is reported per family, normalised to the smooth
case, because that is precisely the quantity the lambda penalty acts on: it makes
"harder family" a measured statement rather than an adjective.

Usage
-----
    python scripts/baseline_generator_stress.py --out-dir results/stage0_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_correction_study import (  # noqa: E402
    SYNTHETIC_BANDS,
    load_wavenumbers,
    run_iarpls,
)
from src.preprocessing.baseline import correct  # noqa: E402

LAMBDA_SWEEP = np.logspace(2, 8, 13)
BASELINE_FAMILIES = ("smooth", "structured", "sharp")
LINESHAPES = ("gaussian", "lorentzian")
NOISE_MODELS = ("gaussian", "poisson")


def signal(wn, rng, lineshape="gaussian", peak_scale=400.0):
    """Raman-like signal. Lorentzian is the more realistic lineshape for Raman bands."""
    s = np.zeros_like(wn)
    for centre, rel, fwhm in SYNTHETIC_BANDS:
        amp = peak_scale * rel * rng.uniform(0.75, 1.25)
        width = fwhm * rng.uniform(0.85, 1.15)
        if lineshape == "gaussian":
            s += amp * np.exp(-0.5 * ((wn - centre) / (width / 2.355)) ** 2)
        elif lineshape == "lorentzian":
            s += amp * (width / 2) ** 2 / ((wn - centre) ** 2 + (width / 2) ** 2)
        else:
            raise ValueError(lineshape)
    return s


def baseline(wn, rng, amplitude, family="smooth"):
    """Fluorescence background of increasing curvature.

    ``smooth`` reproduces the baseline study's family exactly. ``structured`` superposes several
    fluorophore-like Gaussians of differing widths. ``sharp`` adds a narrow component,
    which is what a second-difference penalty struggles to follow at large lambda.
    """
    t = (wn - wn.min()) / np.ptp(wn)
    rayleigh = np.exp(-t / rng.uniform(0.01, 0.04))
    poly = (
        rng.uniform(-0.3, 0.3) * t
        + rng.uniform(-0.25, 0.25) * t**2
        + rng.uniform(-0.15, 0.15) * t**3
    )

    if family == "smooth":
        b = np.exp(-0.5 * ((t - rng.uniform(-0.1, 0.45)) / rng.uniform(0.15, 0.45)) ** 2)
    elif family == "structured":
        b = np.zeros_like(t)
        for _ in range(rng.integers(3, 6)):
            b += rng.uniform(0.3, 1.0) * np.exp(
                -0.5 * ((t - rng.uniform(-0.1, 0.9)) / rng.uniform(0.08, 0.30)) ** 2
            )
    elif family == "sharp":
        b = np.zeros_like(t)
        for _ in range(rng.integers(3, 6)):
            b += rng.uniform(0.3, 1.0) * np.exp(
                -0.5 * ((t - rng.uniform(-0.1, 0.9)) / rng.uniform(0.08, 0.30)) ** 2
            )
        for _ in range(rng.integers(1, 3)):
            b += rng.uniform(0.4, 1.0) * np.exp(
                -0.5 * ((t - rng.uniform(0.05, 0.85)) / rng.uniform(0.015, 0.045)) ** 2
            )
    else:
        raise ValueError(family)

    b = b + 0.6 * rayleigh + poly
    b = b - b.min()
    return amplitude * b / max(b.max(), 1e-12)


def add_noise(clean, rng, model, sigma):
    if model == "gaussian":
        return clean + rng.normal(0, sigma, clean.size)
    if model == "poisson":
        # Match the Gaussian arm's noise at the mean level, then let it scale with
        # intensity the way photon counting actually does.
        gain = max(np.mean(np.maximum(clean, 0)), 1e-9) / max(sigma**2, 1e-12)
        return rng.poisson(np.maximum(clean, 0) * gain) / gain
    raise ValueError(model)


def curvature_energy(b):
    """sum((d2 b)^2), normalised by amplitude -- the quantity the lambda penalty acts on."""
    return float(np.sum(np.diff(b, 2) ** 2) / max(np.ptp(b), 1e-12) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=12)
    ap.add_argument("--amp-mult", type=float, default=8.0, help="fluorescence x signal ptp")
    ap.add_argument("--out-dir", default="results/stage0_baseline")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wn = load_wavenumbers()
    m_fp = (wn >= 700) & (wn < 1800)

    rows, curv_rows = [], []
    for family in BASELINE_FAMILIES:
        for shape in LINESHAPES:
            for noise in NOISE_MODELS:
                for k in range(args.n_rep):
                    rng = np.random.default_rng(
                        abs(hash((family, shape, noise, k))) % (2**32)
                    )
                    sig = signal(wn, rng, lineshape=shape)
                    sc = float(np.ptp(sig))
                    base = baseline(wn, rng, amplitude=args.amp_mult * sc, family=family)
                    obs = add_noise(sig + base, rng, noise, sc / 235.0)
                    if shape == "gaussian" and noise == "gaussian":
                        curv_rows.append({"family": family, "curvature": curvature_energy(base)})

                    for lam in LAMBDA_SWEEP:
                        try:
                            c, _ = run_iarpls(obs, wn, lam=lam)
                        except Exception:                            # noqa: BLE001
                            continue
                        rows.append({
                            "family": family, "lineshape": shape, "noise": noise,
                            "rep": k, "lam": float(lam), "method": "iarpls",
                            "nrmse_fp": float(np.sqrt(np.mean((c[m_fp] - sig[m_fp]) ** 2)) / sc),
                        })
                    # airPLS at the inherited constant, as the fixed reference point
                    c = correct(obs, lam=100.0, endpoint="legacy", clip_negative=False)["corrected"]
                    rows.append({
                        "family": family, "lineshape": shape, "noise": noise,
                        "rep": k, "lam": 100.0, "method": "airpls_lam100",
                        "nrmse_fp": float(np.sqrt(np.mean((c[m_fp] - sig[m_fp]) ** 2)) / sc),
                    })
                print(f"  {family:11s} / {shape:10s} / {noise:8s}: done")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage0_generator_stress_raw.csv", index=False)

    curv = pd.DataFrame(curv_rows).groupby("family")["curvature"].median()
    curv_rel = (curv / curv["smooth"]).round(1)

    iar = df[df.method == "iarpls"]
    piv = iar.pivot_table(index="lam", columns="family", values="nrmse_fp", aggfunc="median")
    piv = piv.reindex(columns=list(BASELINE_FAMILIES))

    # Width of the "inert" window: lambdas within 25% of that family's best score.
    inert = {}
    for fam in BASELINE_FAMILIES:
        col = piv[fam]
        good = col[col <= col.min() * 1.25]
        inert[fam] = {
            "best_lam": float(col.idxmin()),
            "best_nrmse": float(col.min()),
            "inert_lo": float(good.index.min()),
            "inert_hi": float(good.index.max()),
            "inert_decades": float(np.log10(good.index.max() / good.index.min())),
        }

    pd.set_option("display.width", 170)
    print("\n=== baseline curvature energy, relative to the baseline study's 'smooth' family ===")
    print(curv_rel.to_string())

    print("\n=== IARPLS fingerprint NRMSE vs lambda, by baseline family (median) ===")
    print(piv.round(4).to_string())

    print("\n=== width of the lambda window within 25% of that family's best ===")
    for fam, d in inert.items():
        print(f"  {fam:11s} best lam {d['best_lam']:>9.3g}  NRMSE {d['best_nrmse']:.4f}"
              f"   inert over {d['inert_lo']:.3g}..{d['inert_hi']:.3g}"
              f"  ({d['inert_decades']:.1f} decades)")

    print("\n=== effect of lineshape and noise model (median NRMSE over lam<=1e4) ===")
    sub = iar[iar.lam <= 1e4]
    print(sub.pivot_table(index=["lineshape", "noise"], columns="family",
                          values="nrmse_fp", aggfunc="median").round(4).to_string())

    print("\n=== airPLS lambda=100 reference, same conditions ===")
    print(df[df.method == "airpls_lam100"].pivot_table(
        index=["lineshape", "noise"], columns="family", values="nrmse_fp",
        aggfunc="median").round(4).to_string())

    (out_dir / "stage0_generator_stress_verdict.json").write_text(json.dumps({
        "n_rep": args.n_rep,
        "amp_mult": args.amp_mult,
        "lambda_sweep": [float(x) for x in LAMBDA_SWEEP],
        "curvature_energy_relative": {k: float(v) for k, v in curv_rel.items()},
        "inert_window": inert,
    }, indent=2))
    print(f"\nwrote {out_dir}/stage0_generator_stress_{{raw.csv,verdict.json}}")


if __name__ == "__main__":
    main()
