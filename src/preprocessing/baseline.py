"""Per-spectrum baseline correction with a data-driven smoothing parameter.

The project rule is that **no parameter is ever fitted across the dataset**: every
quantity a spectrum is transformed by must be computable from that spectrum alone.
The previous pipeline honoured that for the transform itself but not for its
regularisation strength -- ``airpls_lambda = 100`` was arbitrarily chosen
and tuned by eye.

This module removes the constant. ``lambda`` is selected per spectrum by an
established criterion applied to the Whittaker system that airPLS already solves:

* **GCV** -- generalised cross-validation (Craven & Wahba 1979), in the weighted
  form appropriate to airPLS's asymmetric weights. Needs ``tr(H)``, estimated by a
  seeded Hutchinson probe so the result stays deterministic.
* **L-curve** -- the maximum-curvature corner of ``(log residual, log penalty)``
  (Hansen 1992). Trace-free, so materially cheaper than GCV.

Both are computed on the *converged* airPLS weights, which is what makes them
sensible here: at convergence the peak regions carry ``w ~ 0``, so the criterion
asks how smooth the baseline should be given only the baseline-like points, rather
than how well it fits the peaks.

Implementation note: the linear system is symmetric positive definite and banded
(bandwidth = difference order), so it is solved with :func:`scipy.linalg.solveh_banded`
rather than the general sparse LU the old code used. That is roughly two orders of
magnitude faster and is what makes a per-spectrum grid search affordable at all.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.linalg import solveh_banded
from scipy.special import comb

__all__ = [
    "DEFAULT_LAMBDA_GRID",
    "RAMAN_REFERENCE_BANDS",
    "airpls",
    "correct",
    "difference_penalty_banded",
    "gcv_curve",
    "lcurve",
    "select_lambda",
    "whittaker",
]

#: Log-spaced search grid. Spans well below and well above the inherited
#: ``lambda = 100`` so the selector is never pinned at an endpoint by construction.
DEFAULT_LAMBDA_GRID = np.logspace(0.0, 8.0, 25)

#: Diagnostic Raman bands used for peak-fidelity scoring (cm^-1).
RAMAN_REFERENCE_BANDS = {
    "phenylalanine_1004": 1004.0,
    "ch2_deformation_1440": 1440.0,
    "amide_I_1660": 1660.0,
    "ch_stretch_2930": 2930.0,
}

# Relative ridge added to the diagonal. D'D alone is singular (its null space is the
# polynomials of degree < order), so a spectrum whose weights collapse onto too few
# points would otherwise produce a non-positive-definite system.
_RIDGE = 1e-9


@functools.lru_cache(maxsize=8)
def _difference_penalty_banded_cached(m: int, order: int) -> np.ndarray:
    ab = _difference_penalty_banded_impl(m, order)
    ab.flags.writeable = False  # cached array is shared; never mutate in place
    return ab


def difference_penalty_banded(m: int, order: int = 2) -> np.ndarray:
    """Cached wrapper -- the penalty depends only on ``(m, order)``, never on the data."""
    return _difference_penalty_banded_cached(int(m), int(order))


def _difference_penalty_banded_impl(m: int, order: int = 2) -> np.ndarray:
    """Upper-banded representation of ``D.T @ D`` for order-``order`` differences.

    Returns an ``(order + 1, m)`` array in the layout :func:`scipy.linalg.solveh_banded`
    expects with ``lower=False``: ``ab[order + i - j, j] == (D.T @ D)[i, j]`` for ``i <= j``.

    Built from binomial coefficients rather than by materialising an ``m x m`` matrix,
    which for the 1738-point spectra here would be a 24 MB dense intermediate.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if m <= order:
        raise ValueError(f"signal length {m} must exceed difference order {order}")

    # Row stencil of the order-d difference operator, e.g. [1, -2, 1] for d = 2.
    stencil = np.array(
        [(-1) ** k * comb(order, k, exact=True) for k in range(order + 1)],
        dtype=np.float64,
    )[::-1]

    # (D.T @ D)[i, j] = sum_r D[r, i] D[r, j]; D[r, c] = stencil[c - r] for 0 <= c - r <= order.
    # So band k = j - i is the correlation of the stencil with itself at lag k, except
    # near the edges where fewer rows contribute.
    n_rows = m - order
    ab = np.zeros((order + 1, m), dtype=np.float64)
    for k in range(order + 1):
        # Interior value of the k-th band.
        interior = float(np.dot(stencil[: order + 1 - k], stencil[k:]))
        band = np.full(m - k, interior, dtype=np.float64)

        # Edge corrections: enumerate the rows that actually reach each column pair.
        for j in range(m - k):
            i = j
            rows = range(max(0, j + k - order), min(n_rows, i + 1))
            band[j] = sum(stencil[i - r] * stencil[j + k - r] for r in rows)
        ab[order - k, k:] = band
    return ab


def _assemble(ab_dd: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Banded ``W + lam * D'D`` with a small ridge for positive definiteness."""
    order = ab_dd.shape[0] - 1
    ab = lam * ab_dd
    main = ab[order]
    main += w
    ridge = _RIDGE * max(float(np.max(main)), 1.0)
    main += ridge
    return ab


def whittaker(y: np.ndarray, w: np.ndarray, lam: float, ab_dd: np.ndarray) -> np.ndarray:
    """Weighted Whittaker smoother: ``argmin_z ||W^(1/2)(y - z)||^2 + lam ||D z||^2``."""
    ab = _assemble(ab_dd, w, lam)
    return solveh_banded(ab, w * y, lower=False, check_finite=False)


def airpls(
    y: np.ndarray,
    lam: float,
    ab_dd: np.ndarray | None = None,
    order: int = 2,
    itermax: int = 15,
    tol: float = 1e-3,
    endpoint: str = "reference",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Adaptive iteratively reweighted penalised least squares (Zhang et al. 2010).

    Parameters
    ----------
    endpoint
        ``"reference"`` anchors the two end points with a *large* weight
        (``exp(i * max|d_neg| / sum|d_neg|)``), as in the published algorithm.
        ``"legacy"`` reproduces this repository's previous behaviour, which used
        ``max(d_neg)`` -- the *least* negative residual -- and therefore assigned the
        end points a near-zero weight instead. Retained so the study can separate the
        effect of selecting ``lambda`` from the effect of fixing this.

    Returns
    -------
    (baseline, weights, n_iter)
    """
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[0]
    if ab_dd is None:
        ab_dd = difference_penalty_banded(m, order)

    w = np.ones(m, dtype=np.float64)
    z = y.copy()
    abs_y_sum = float(np.abs(y).sum())
    n_iter = 0

    for i in range(1, itermax + 1):
        n_iter = i
        z = whittaker(y, w, lam, ab_dd)
        d = y - z
        neg = d < 0
        if not neg.any():
            break
        dssn = float(np.abs(d[neg]).sum())
        if dssn < tol * abs_y_sum:
            break

        w = np.zeros(m, dtype=np.float64)
        # Guard the exponential: without it a near-converged spectrum overflows.
        expo = np.clip(i * np.abs(d[neg]) / dssn, None, 50.0)
        w[neg] = np.exp(expo)
        if endpoint == "reference":
            edge = np.exp(min(i * float(np.abs(d[neg]).max()) / dssn, 50.0))
        elif endpoint == "legacy":
            edge = np.exp(min(i * float(d[neg].max()) / dssn, 50.0))
        else:
            raise ValueError(f"unknown endpoint mode {endpoint!r}")
        w[0] = edge
        w[-1] = edge

    return z, w, n_iter


def _trace_hat(
    ab_dd: np.ndarray, w: np.ndarray, lam: float, n_probe: int, seed: int
) -> float:
    """Hutchinson estimate of ``tr(H)`` where ``H = (W + lam D'D)^-1 W``.

    Deterministic: the probe vectors come from a locally-seeded generator, so a
    spectrum yields the same estimate regardless of what else is being processed.
    """
    m = w.shape[0]
    ab = _assemble(ab_dd, w, lam)
    rng = np.random.default_rng(seed)
    probes = rng.integers(0, 2, size=(n_probe, m)).astype(np.float64) * 2.0 - 1.0
    # v' A^-1 W v  for each probe; solveh_banded handles the whole block at once.
    rhs = (probes * w).T
    sol = solveh_banded(ab, rhs, lower=False, check_finite=False)
    return float(np.mean(np.einsum("ij,ji->i", probes, sol)))


def gcv_curve(
    y: np.ndarray,
    grid: np.ndarray = DEFAULT_LAMBDA_GRID,
    order: int = 2,
    itermax: int = 15,
    n_probe: int = 48,
    seed: int = 0,
    endpoint: str = "reference",
) -> dict:
    """GCV score across ``grid``, evaluated on each lambda's converged airPLS weights."""
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[0]
    ab_dd = difference_penalty_banded(m, order)

    scores = np.full(len(grid), np.inf)
    traces = np.zeros(len(grid))
    for j, lam in enumerate(grid):
        z, w, _ = airpls(y, lam, ab_dd, order, itermax, endpoint=endpoint)
        resid = y - z
        rss = float(np.sum(w * resid * resid))
        n_eff = float(np.sum(w > 0))
        if n_eff <= order + 1:
            continue
        tr = _trace_hat(ab_dd, w, lam, n_probe, seed)
        traces[j] = tr
        denom = n_eff - tr
        if denom <= 0:
            continue
        scores[j] = (rss / n_eff) / (denom / n_eff) ** 2

    best = int(np.argmin(scores)) if np.isfinite(scores).any() else len(grid) // 2
    return {
        "lambda": float(grid[best]),
        "index": best,
        "scores": scores,
        "traces": traces,
        "grid": np.asarray(grid, dtype=float),
        "at_edge": best in (0, len(grid) - 1),
    }


def lcurve(
    y: np.ndarray,
    grid: np.ndarray = DEFAULT_LAMBDA_GRID,
    order: int = 2,
    itermax: int = 15,
    endpoint: str = "reference",
) -> dict:
    """L-curve corner (Hansen 1992). Trace-free, so much cheaper than :func:`gcv_curve`."""
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[0]
    ab_dd = difference_penalty_banded(m, order)

    rho = np.zeros(len(grid))  # log residual norm
    eta = np.zeros(len(grid))  # log penalty norm
    for j, lam in enumerate(grid):
        z, w, _ = airpls(y, lam, ab_dd, order, itermax, endpoint=endpoint)
        resid = y - z
        rss = float(np.sum(w * resid * resid))
        pen = float(np.sum(np.diff(z, order) ** 2))
        rho[j] = 0.5 * np.log(max(rss, 1e-300))
        eta[j] = 0.5 * np.log(max(pen, 1e-300))

    # Menger curvature of consecutive triples on the (rho, eta) curve.
    curv = np.zeros(len(grid))
    for j in range(1, len(grid) - 1):
        x1, y1 = rho[j - 1], eta[j - 1]
        x2, y2 = rho[j], eta[j]
        x3, y3 = rho[j + 1], eta[j + 1]
        area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        d12 = np.hypot(x2 - x1, y2 - y1)
        d23 = np.hypot(x3 - x2, y3 - y2)
        d13 = np.hypot(x3 - x1, y3 - y1)
        denom = d12 * d23 * d13
        curv[j] = 0.0 if denom < 1e-300 else 2.0 * area2 / denom

    best = int(np.argmax(curv))
    return {
        "lambda": float(grid[best]),
        "index": best,
        "curvature": curv,
        "rho": rho,
        "eta": eta,
        "grid": np.asarray(grid, dtype=float),
        "at_edge": best in (0, len(grid) - 1),
    }


def select_lambda(
    y: np.ndarray,
    method: str = "gcv",
    grid: np.ndarray = DEFAULT_LAMBDA_GRID,
    **kwargs,
) -> dict:
    """Dispatch to a selector. ``method`` is ``"gcv"`` or ``"lcurve"``."""
    if method == "gcv":
        return gcv_curve(y, grid=grid, **kwargs)
    if method == "lcurve":
        kwargs.pop("n_probe", None)
        kwargs.pop("seed", None)
        return lcurve(y, grid=grid, **kwargs)
    raise ValueError(f"unknown selector {method!r}; expected 'gcv' or 'lcurve'")


#: Wavenumber below which IARPLS does not model the Rayleigh wing (see Stage 0 findings).
#: Above this the corrector is accurate; below it the estimate is unusable.
IARPLS_VALID_FROM_CM1 = 300.0

#: PROVISIONAL. Stage 0 initially found lambda inert over 1e2-1e4; the follow-up study
#: (scripts/baseline_generator_stress.py) withdrew that -- the inert window is 2.5 decades
#: for smooth baselines but only 0.5 for high-curvature ones, and real baselines appear
#: *more* curved than any synthetic family tested. This value sits in the low-drift,
#: reasonable-retention region on real spectra and matches the synthetic oracle, but it
#: is a documented default awaiting a Stage 1 sensitivity analysis -- NOT a settled
#: constant, and not evidence of insensitivity. See the paper, Appendix: Baseline Correction (lambda selection).
IARPLS_LAMBDA = 1e4


def correct_iarpls(
    y: np.ndarray,
    wavenumbers: np.ndarray,
    lam: float = IARPLS_LAMBDA,
    clip_negative: bool = True,
) -> dict:
    """Baseline-correct one spectrum with IARPLS -- the Stage 0 recommended path.

    Chosen over a per-spectrum airPLS selector because it is markedly more accurate --
    10.7x lower fingerprint NRMSE than the inherited airPLS ``lambda = 100``, and it
    beats an *oracle* airPLS lambda chosen against ground truth, so no selector for
    airPLS closes the gap. Verified pure: output is bit-identical whether a spectrum is
    corrected alone or inside a batch, in any order.

    ``lam`` is a provisional default, not a settled one -- see :data:`IARPLS_LAMBDA`.

    Accurate only above :data:`IARPLS_VALID_FROM_CM1`; the Rayleigh wing below that
    is not modelled and must be cropped downstream.
    """
    import ramanspy as rp

    y = np.asarray(y, dtype=float)
    container = rp.SpectralContainer(np.atleast_2d(y), np.asarray(wavenumbers, dtype=float))
    out = rp.preprocessing.baseline.IARPLS(lam=lam).apply(container)
    corrected = np.asarray(out.spectral_data).ravel()
    baseline = y - corrected
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)
    return {
        "corrected": corrected,
        "baseline": baseline,
        "lambda": float(lam),
        "method": "iarpls",
    }


def correct(
    y: np.ndarray,
    method: str = "gcv",
    lam: float | None = None,
    grid: np.ndarray = DEFAULT_LAMBDA_GRID,
    order: int = 2,
    itermax: int = 15,
    clip_negative: bool = True,
    endpoint: str = "reference",
    **selector_kwargs,
) -> dict:
    """Baseline-correct one spectrum.

    ``lam`` overrides the selector entirely (used for the fixed-lambda arms of the
    Stage 0 study). Otherwise ``method`` chooses lambda from ``grid``.

    Returns a dict with ``corrected``, ``baseline``, ``lambda`` and selector diagnostics.
    """
    y = np.asarray(y, dtype=np.float64)
    ab_dd = difference_penalty_banded(y.shape[0], order)

    info: dict = {}
    if lam is None:
        info = select_lambda(
            y, method=method, grid=grid, order=order, itermax=itermax,
            endpoint=endpoint, **selector_kwargs
        )
        lam = info["lambda"]

    baseline, weights, n_iter = airpls(
        y, lam, ab_dd, order, itermax, endpoint=endpoint
    )
    corrected = y - baseline
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)

    return {
        "corrected": corrected,
        "baseline": baseline,
        "weights": weights,
        "lambda": float(lam),
        "n_iter": n_iter,
        "selector": None if not info else method,
        "at_edge": bool(info.get("at_edge", False)) if info else False,
    }
