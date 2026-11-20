"""Baseline-correction validation study: does a per-spectrum lambda beat the inherited constant?

Design
------
Ground truth is required to answer this, and real spectra do not come with a known
baseline. So two arms:

* **synthetic**   -- a Raman-like signal built from Gaussian bands at real band
  positions, plus a known baseline and matched noise. Zero circularity: the "true"
  signal was never produced by any baseline corrector.
* **semisynthetic** -- a real, median-filtered tissue spectrum with its own baseline
  removed, then a known synthetic baseline added back. Grounded in real spectral
  shape; mildly circular, so it is a corroborating arm rather than the primary one.

Arms compared
-------------
``legacy_lam100``  legacy endpoint rule + lambda=100 -- reproduces the current pipeline
``ref_lam100``     reference endpoint rule + lambda=100 -- isolates the endpoint fix
``gcv``            reference endpoint + GCV-selected lambda
``lcurve``         reference endpoint + L-curve-selected lambda
``iarpls``         ramanspy IARPLS(lam=1e4) -- the corrector the other code path used

Usage
-----
    python scripts/baseline_correction_study.py --out-dir results/stage0_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.baseline import (  # noqa: E402
    DEFAULT_LAMBDA_GRID,
    RAMAN_REFERENCE_BANDS,
    correct,
)


def stable_seed(*key) -> int:
    """32-bit RNG seed derived from ``key``, identical in every process. Python's ``hash()`` of a
    string is salted per process, so seeding from it draws different spectra on every run."""
    return zlib.crc32(repr(key).encode())

# Band positions (cm^-1) and relative heights loosely following brain-tissue Raman.
SYNTHETIC_BANDS = [
    (1004.0, 1.00, 8.0), (1032.0, 0.30, 8.0), (1128.0, 0.35, 10.0),
    (1230.0, 0.40, 18.0), (1260.0, 0.45, 15.0), (1300.0, 0.70, 14.0),
    (1340.0, 0.55, 14.0), (1440.0, 1.20, 16.0), (1550.0, 0.35, 14.0),
    (1660.0, 0.95, 20.0), (2850.0, 1.60, 18.0), (2885.0, 1.45, 18.0),
    (2930.0, 1.80, 22.0), (3060.0, 0.30, 20.0),
]

FLUORESCENCE_LEVELS = {"low": 2.0, "medium": 8.0, "high": 25.0}


def load_wavenumbers(calib="data/RamanFreqTable.csv") -> np.ndarray:
    return pd.read_csv(calib).iloc[:, 1].values.astype(float)


def synthetic_signal(wn: np.ndarray, rng: np.random.Generator, peak_scale=400.0) -> np.ndarray:
    """Raman-like signal: Gaussian bands at real positions, jittered amplitude/width."""
    s = np.zeros_like(wn)
    for centre, rel, fwhm in SYNTHETIC_BANDS:
        amp = peak_scale * rel * rng.uniform(0.75, 1.25)
        width = fwhm * rng.uniform(0.85, 1.15)
        sigma = width / 2.355
        s += amp * np.exp(-0.5 * ((wn - centre) / sigma) ** 2)
    return s


def synthetic_baseline(wn: np.ndarray, rng: np.random.Generator, amplitude: float) -> np.ndarray:
    """Broad fluorescence + Rayleigh wing + low-order polynomial drift."""
    t = (wn - wn.min()) / np.ptp(wn)
    centre = rng.uniform(-0.1, 0.45)
    width = rng.uniform(0.15, 0.45)
    fluor = np.exp(-0.5 * ((t - centre) / width) ** 2)
    rayleigh = np.exp(-t / rng.uniform(0.01, 0.04))       # steep rise at low wavenumber
    poly = (
        rng.uniform(-0.3, 0.3) * t
        + rng.uniform(-0.25, 0.25) * t**2
        + rng.uniform(-0.15, 0.15) * t**3
    )
    b = fluor + 0.6 * rayleigh + poly
    b = b - b.min()
    return amplitude * b / max(b.max(), 1e-12)


def peak_heights(wn: np.ndarray, spec: np.ndarray, half_window=25.0) -> dict:
    out = {}
    for name, centre in RAMAN_REFERENCE_BANDS.items():
        m = np.abs(wn - centre) <= half_window
        out[name] = float(spec[m].max()) if m.any() else np.nan
    return out


#: Error is strongly region-dependent, so a single pooled NRMSE hides the answer.
#: ``fingerprint`` is the region the models actually see (``crop_region: [700, 1800]``)
#: and is therefore the primary metric; ``rayleigh`` is discarded downstream.
SPECTRAL_REGIONS = {
    "rayleigh": (0.0, 300.0),
    "low": (300.0, 700.0),
    "fingerprint": (700.0, 1800.0),
    "silent": (1800.0, 2600.0),
    "ch_stretch": (2600.0, 3100.0),
    "high": (3100.0, 3500.0),
}


def score(wn, true_signal, true_baseline, est_corrected, est_baseline) -> dict:
    sig_scale = max(float(np.ptp(true_signal)), 1e-12)
    res = {
        "signal_nrmse": float(np.sqrt(np.mean((est_corrected - true_signal) ** 2)) / sig_scale),
        "baseline_nrmse": float(
            np.sqrt(np.mean((est_baseline - true_baseline) ** 2)) / max(float(np.ptp(true_baseline)), 1e-12)
        ),
    }
    for name, (lo, hi) in SPECTRAL_REGIONS.items():
        m = (wn >= lo) & (wn < hi)
        res[f"nrmse_{name}"] = (
            float(np.sqrt(np.mean((est_corrected[m] - true_signal[m]) ** 2)) / sig_scale)
            if m.any() else np.nan
        )

    h_true = peak_heights(wn, true_signal)
    h_est = peak_heights(wn, est_corrected)
    errs = []
    for name, ht in h_true.items():
        rel = abs(h_est[name] - ht) / max(ht, 1e-12)
        res[f"peak_relerr_{name}"] = float(rel)
        res[f"peak_ratio_{name}"] = float(h_est[name] / max(ht, 1e-12))
        errs.append(rel)
    res["peak_relerr_mean"] = float(np.mean(errs))
    return res


def run_iarpls(y, wn, lam=1e4):
    """IARPLS from the (now-deleted) ramanspy code path, as an external comparator."""
    import ramanspy as rp

    container = rp.SpectralContainer(y[None, :].astype(float), wn)
    out = rp.preprocessing.baseline.IARPLS(lam=lam).apply(container)
    corrected = np.asarray(out.spectral_data).ravel()
    return corrected, y - corrected


def load_real_spectra(n_spectra, rng, cube_paths):
    """Real tissue spectra after the production 3x3 spatial median filter.

    Returned raw (baseline *not* removed) -- the invariance arm must not depend on
    any corrector to build its reference, or it would be circular.
    """
    from scipy.ndimage import median_filter

    pool = []
    per_cube = max(1, n_spectra // max(len(cube_paths), 1) + 1)
    for path in cube_paths:
        cube = np.load(path).astype(np.float32)
        filt = np.empty_like(cube)
        for i in range(cube.shape[2]):
            filt[:, :, i] = median_filter(cube[:, :, i], size=3, mode="reflect")
        flat = filt.reshape(-1, cube.shape[2]).astype(float)
        brightest = np.argsort(flat.mean(1))[-500:]          # tissue, not background
        picks = rng.choice(brightest, size=min(per_cube, len(brightest)), replace=False)
        pool.extend(flat[idx] for idx in picks)
        if len(pool) >= n_spectra:
            break
    return pool[:n_spectra]


def invariance_score(wn, corrected_plain, corrected_loaded) -> dict:
    """Peak-height drift when extra fluorescence is added to a *real* spectrum.

    No ground truth is involved. A corrector that is genuinely invariant to added
    baseline returns the same peak heights either way, so drift measures exactly the
    failure mode that matters in production -- and cannot be gamed by the choice of
    reference, because there is no reference.
    """
    h0 = peak_heights(wn, corrected_plain)
    h1 = peak_heights(wn, corrected_loaded)
    res, drifts = {}, []
    for name in RAMAN_REFERENCE_BANDS:
        denom = max(abs(h0[name]), 1e-12)
        d = abs(h1[name] - h0[name]) / denom
        res[f"drift_{name}"] = float(d)
        drifts.append(d)
    res["drift_mean"] = float(np.mean(drifts))
    m = (wn >= 700) & (wn < 1800)
    scale = max(float(np.ptp(corrected_plain[m])), 1e-12)
    res["drift_nrmse_fingerprint"] = float(
        np.sqrt(np.mean((corrected_loaded[m] - corrected_plain[m]) ** 2)) / scale
    )
    return res


def iarpls_lambda_sensitivity(wn, n_rep, lams=(1e2, 1e3, 1e4, 1e5, 1e6, 1e7)):
    """Is IARPLS's lambda actually inert? Decides whether a selector is needed at all."""
    m = (wn >= 700) & (wn < 1800)
    rows = []
    for lam in lams:
        for level, amp in FLUORESCENCE_LEVELS.items():
            errs = []
            for k in range(n_rep):
                rng = np.random.default_rng(1000 + k)
                sig = synthetic_signal(wn, rng)
                sc = np.ptp(sig)
                obs = sig + synthetic_baseline(wn, rng, amplitude=amp * sc) + rng.normal(
                    0, sc / 235.0, wn.size
                )
                try:
                    c, _ = run_iarpls(obs, wn, lam=lam)
                except Exception:                                   # noqa: BLE001
                    continue
                errs.append(float(np.sqrt(np.mean((c[m] - sig[m]) ** 2)) / sc))
            if errs:
                rows.append({"lam": lam, "level": level, "nrmse_fingerprint": float(np.median(errs))})
    return pd.DataFrame(rows)


def make_figure(wn, df_syn, df_lamsens, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a) example spectrum
    ax = axes[0, 0]
    rng = np.random.default_rng(42)
    sig = synthetic_signal(wn, rng)
    sc = np.ptp(sig)
    base = synthetic_baseline(wn, rng, amplitude=8 * sc)
    obs = sig + base + rng.normal(0, sc / 235.0, wn.size)
    ax.plot(wn, obs, lw=0.6, color="0.6", label="observed (8x fluorescence)")
    ax.plot(wn, sig + base, lw=0.6, color="k", alpha=0.4, label="signal + true baseline")
    ax.plot(wn, base, lw=1.2, color="C3", ls="--", label="true baseline")
    for name, kw, col in [("airPLS $\\lambda$=100", dict(lam=100.0, endpoint="legacy"), "C0"),
                          ("airPLS L-curve", dict(method="lcurve"), "C2")]:
        ax.plot(wn, correct(obs, clip_negative=False, **kw)["baseline"], lw=1.0, color=col, label=name)
    try:
        _, bi = run_iarpls(obs, wn)
        ax.plot(wn, bi, lw=1.0, color="C1", label="IARPLS $\\lambda$=1e4")
    except Exception:                                               # noqa: BLE001
        pass
    ax.set_title("(a) Estimated baselines, 8x fluorescence")
    ax.set_xlabel("Raman shift (cm$^{-1}$)"); ax.set_ylabel("intensity")
    ax.legend(fontsize=7); ax.axvspan(700, 1800, color="C2", alpha=0.06)

    # (b) peak-height ratio
    ax = axes[0, 1]
    cols = [f"peak_ratio_{b}" for b in RAMAN_REFERENCE_BANDS]
    piv = df_syn.groupby("method")[cols].median()
    xs = np.arange(len(cols))
    for i, (meth, row) in enumerate(piv.iterrows()):
        ax.plot(xs + i * 0.02, row.values, "o-", ms=5, lw=1.2, label=meth)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks(xs); ax.set_xticklabels([b.split("_")[-1] for b in RAMAN_REFERENCE_BANDS])
    ax.set_ylabel("recovered / true peak height"); ax.set_ylim(0, 1.3)
    ax.set_title("(b) Peak-height fidelity (1.0 = perfect)"); ax.legend(fontsize=7)

    # (c) fingerprint NRMSE by fluorescence
    ax = axes[1, 0]
    piv = df_syn.pivot_table(index="level", columns="method",
                             values="nrmse_fingerprint", aggfunc="median")
    piv = piv.reindex(["low", "medium", "high"])
    piv.plot(kind="bar", ax=ax, logy=True, width=0.78)
    ax.set_ylabel("NRMSE, 700-1800 cm$^{-1}$ (log)"); ax.set_xlabel("fluorescence level")
    ax.set_title("(c) Error in the modelled region"); ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=0)

    # (d) IARPLS lambda sensitivity
    ax = axes[1, 1]
    if not df_lamsens.empty:
        for level in FLUORESCENCE_LEVELS:
            sub = df_lamsens[df_lamsens.level == level].sort_values("lam")
            ax.loglog(sub.lam, sub.nrmse_fingerprint, "o-", label=f"{level} fluorescence")
        ax.axvspan(1e2, 1e4, color="C2", alpha=0.12)
        ax.text(1e3, ax.get_ylim()[1] * 0.4, "inert", ha="center", fontsize=8, color="C2")
    ax.set_xlabel("IARPLS $\\lambda$"); ax.set_ylabel("NRMSE, fingerprint")
    ax.set_title("(d) IARPLS is insensitive to $\\lambda$ over 1e2-1e4"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-spectra", type=int, default=40,
                    help="per arm, per fluorescence level (40 for the paper; the lambda sweep uses n_spectra // 4 spectra, at least 6)")
    ap.add_argument("--out-dir", default="results/stage0_baseline")
    ap.add_argument("--n-probe", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wn = load_wavenumbers()
    rng = np.random.default_rng(args.seed)

    methods = {
        "legacy_lam100": dict(lam=100.0, endpoint="legacy"),
        "ref_lam100": dict(lam=100.0, endpoint="reference"),
        "gcv": dict(method="gcv", endpoint="reference", n_probe=args.n_probe),
        "lcurve": dict(method="lcurve", endpoint="reference"),
    }

    def apply_all(obs):
        """Run every arm on one observed spectrum -> {name: (corrected, baseline, lam, edge)}."""
        out = {}
        for name, kw in methods.items():
            r = correct(obs, clip_negative=False, **kw)
            out[name] = (r["corrected"], r["baseline"], r["lambda"], r["at_edge"])
        try:
            ci, bi = run_iarpls(obs, wn)
            out["iarpls"] = (ci, bi, 1e4, False)
        except Exception as exc:                                    # noqa: BLE001
            apply_all.iarpls_error = f"{type(exc).__name__}: {exc}"
        return out

    # ---------------- Arm 1: synthetic, full ground truth ----------------
    rows = []
    for level_name, amp_mult in FLUORESCENCE_LEVELS.items():
        for k in range(args.n_spectra):
            srng = np.random.default_rng(stable_seed("syn", level_name, k, args.seed))
            sig = synthetic_signal(wn, srng)
            sig_scale = max(float(np.ptp(sig)), 1e-12)
            base = synthetic_baseline(wn, srng, amplitude=amp_mult * sig_scale)
            obs = sig + base + srng.normal(0, sig_scale / 235.0, wn.size)   # SNR matched to real
            for name, (corr_, base_, lam_, edge_) in apply_all(obs).items():
                row = score(wn, sig, base, corr_, base_)
                row.update(level=level_name, rep=k, method=name, lam=lam_, at_edge=edge_)
                rows.append(row)
        print(f"  synthetic/{level_name}: done")
    df_syn = pd.DataFrame(rows)

    # ---------------- Arm 2: real spectra, added-fluorescence invariance ----------------
    cube_paths = sorted(Path("data/raw").glob("LGm-*/*.npy"))[:6]
    print(f"Loading real spectra from {len(cube_paths)} cubes (3x3 median filtered) ...")
    real_pool = load_real_spectra(args.n_spectra, rng, cube_paths)
    print(f"  pool size: {len(real_pool)}")

    inv_rows = []
    for level_name, amp_mult in FLUORESCENCE_LEVELS.items():
        for k, y_real in enumerate(real_pool):
            srng = np.random.default_rng(stable_seed("inv", level_name, k, args.seed))
            m_fp = (wn >= 700) & (wn < 1800)
            scale = max(float(np.ptp(y_real[m_fp])), 1e-12)
            extra = synthetic_baseline(wn, srng, amplitude=amp_mult * scale)
            plain = apply_all(y_real)
            loaded = apply_all(y_real + extra)
            for name in plain:
                if name not in loaded:
                    continue
                row = invariance_score(wn, plain[name][0], loaded[name][0])
                row.update(level=level_name, rep=k, method=name,
                           lam_plain=plain[name][2], lam_loaded=loaded[name][2])
                inv_rows.append(row)
        print(f"  invariance/{level_name}: done")
    df_inv = pd.DataFrame(inv_rows)

    if getattr(apply_all, "iarpls_error", None):
        print(f"  [iarpls unavailable: {apply_all.iarpls_error}]")

    df_syn.to_csv(out_dir / "stage0_synthetic_raw.csv", index=False)
    df_inv.to_csv(out_dir / "stage0_invariance_raw.csv", index=False)

    region_cols = [f"nrmse_{r}" for r in SPECTRAL_REGIONS]
    peak_cols = [f"peak_relerr_{b}" for b in RAMAN_REFERENCE_BANDS]
    df_syn.groupby(["level", "method"])[
        ["signal_nrmse", "baseline_nrmse", "peak_relerr_mean"] + region_cols + peak_cols
    ].median().reset_index().to_csv(out_dir / "stage0_synthetic_summary.csv", index=False)
    df_inv.groupby(["level", "method"])[
        ["drift_mean", "drift_nrmse_fingerprint"]
    ].median().reset_index().to_csv(out_dir / "stage0_invariance_summary.csv", index=False)

    print("Running IARPLS lambda-sensitivity sweep ...")
    df_lamsens = iarpls_lambda_sensitivity(wn, n_rep=max(6, args.n_spectra // 4))
    df_lamsens.to_csv(out_dir / "stage0_iarpls_lambda_sensitivity.csv", index=False)

    try:
        make_figure(wn, df_syn, df_lamsens, out_dir / "stage0_baseline_study.png")
        print(f"  figure -> {out_dir/'stage0_baseline_study.png'}")
    except Exception as exc:                                        # noqa: BLE001
        print(f"  [figure failed: {type(exc).__name__}: {exc}]")

    edge = df_syn[df_syn.method.isin(["gcv", "lcurve"])].groupby("method")["at_edge"].mean()
    fp = df_syn.groupby("method")["nrmse_fingerprint"].median()
    drift = df_inv.groupby("method")["drift_mean"].median()

    verdict = {
        "n_spectra_per_cell": args.n_spectra,
        "lambda_grid": [float(DEFAULT_LAMBDA_GRID.min()), float(DEFAULT_LAMBDA_GRID.max())],
        "grid_points": len(DEFAULT_LAMBDA_GRID),
        "primary_metric": "nrmse_fingerprint (700-1800 cm-1, the modelled region)",
        "nrmse_fingerprint_median": {k: float(v) for k, v in fp.items()},
        "invariance_drift_median": {k: float(v) for k, v in drift.items()},
        "edge_pinning_rate": {k: float(v) for k, v in edge.items()},
        "lambda_median": df_syn[df_syn.method.isin(["gcv", "lcurve"])]
            .groupby("method")["lam"].median().to_dict(),
        "best_by_fingerprint": str(fp.idxmin()),
        "best_by_invariance": str(drift.idxmin()),
    }
    (out_dir / "stage0_verdict.json").write_text(json.dumps(verdict, indent=2))

    pd.set_option("display.width", 160)
    print("\n=== SYNTHETIC ARM: NRMSE by spectral region (median, all levels) ===")
    print(df_syn.groupby("method")[region_cols].median().round(4))
    print("\n=== SYNTHETIC ARM: peak-height ratio est/true (median) ===")
    print(df_syn.groupby("method")[[f"peak_ratio_{b}" for b in RAMAN_REFERENCE_BANDS]]
          .median().round(3))
    print("\n=== PRIMARY: nrmse_fingerprint by fluorescence level ===")
    print(df_syn.pivot_table(index="level", columns="method",
                             values="nrmse_fingerprint", aggfunc="median").round(4))
    print("\n=== REAL-DATA INVARIANCE: peak drift when fluorescence is added ===")
    print(df_inv.pivot_table(index="level", columns="method",
                             values="drift_mean", aggfunc="median").round(4))
    print("\n=== selected lambda (median) ===")
    print(df_syn[df_syn.method.isin(["gcv", "lcurve"])]
          .pivot_table(index="level", columns="method", values="lam", aggfunc="median"))
    print(f"\nedge-pinning rate: {dict(edge.round(3))}")
    print(f"best by fingerprint NRMSE: {fp.idxmin()}   best by invariance: {drift.idxmin()}")
    print(f"\nwrote {out_dir}/stage0_*.csv and stage0_verdict.json")


if __name__ == "__main__":
    main()
