"""Correctness tests for the per-spectrum baseline selector (Stage 0).

The numerical machinery here is new, so each piece is checked against a dense
reference implementation before the Stage 0 study relies on it.
"""

import numpy as np
import pytest

from src.preprocessing.baseline import (
    DEFAULT_LAMBDA_GRID,
    _assemble,
    _trace_hat,
    airpls,
    correct,
    difference_penalty_banded,
    gcv_curve,
    lcurve,
    whittaker,
)


def _dense_dd(m, order):
    D = np.diff(np.eye(m), order, axis=0)
    return D.T @ D


def _banded_to_dense(ab):
    order = ab.shape[0] - 1
    m = ab.shape[1]
    A = np.zeros((m, m))
    for k in range(order + 1):
        band = ab[order - k, k:]
        idx = np.arange(m - k)
        A[idx, idx + k] = band
        A[idx + k, idx] = band
    return A


@pytest.mark.parametrize("m", [12, 50, 200])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_penalty_matches_dense(m, order):
    """Banded D'D must equal the dense construction the old code used."""
    got = _banded_to_dense(np.array(difference_penalty_banded(m, order)))
    assert np.allclose(got, _dense_dd(m, order), atol=1e-10)


@pytest.mark.parametrize("order", [1, 2])
def test_whittaker_matches_dense_solve(order):
    rng = np.random.default_rng(1)
    m = 120
    y = rng.normal(size=m) + np.linspace(0, 5, m)
    w = rng.uniform(0.0, 1.0, size=m)
    lam = 37.0

    ab_dd = difference_penalty_banded(m, order)
    got = whittaker(y, w, lam, ab_dd)

    A = np.diag(w) + lam * _dense_dd(m, order)
    A += np.eye(m) * 1e-9 * max(np.max(np.diag(A)), 1.0)
    want = np.linalg.solve(A, w * y)
    assert np.allclose(got, want, atol=1e-8)


def test_trace_estimator_matches_exact():
    """Hutchinson tr(H) must track the exact trace; this gates the GCV score."""
    rng = np.random.default_rng(2)
    m = 150
    order = 2
    w = rng.uniform(0.2, 1.0, size=m)
    lam = 25.0
    ab_dd = difference_penalty_banded(m, order)

    A = _banded_to_dense(_assemble(ab_dd, w, lam))
    exact = float(np.trace(np.linalg.solve(A, np.diag(w))))
    est = _trace_hat(ab_dd, w, lam, n_probe=512, seed=0)

    assert abs(est - exact) / exact < 0.05, f"est={est:.3f} exact={exact:.3f}"


def test_trace_estimator_is_deterministic():
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 1.0, size=100)
    ab_dd = difference_penalty_banded(100, 2)
    a = _trace_hat(ab_dd, w, 10.0, n_probe=32, seed=0)
    b = _trace_hat(ab_dd, w, 10.0, n_probe=32, seed=0)
    assert a == b


def test_airpls_recovers_smooth_baseline_without_peaks():
    """With no peaks, airPLS should return essentially the signal itself."""
    x = np.linspace(0, 1, 400)
    baseline = 50 + 30 * np.exp(-((x - 0.3) ** 2) / 0.05)
    got, _, _ = airpls(baseline, lam=1e4)
    assert np.max(np.abs(got - baseline)) / np.ptp(baseline) < 0.05


def test_airpls_ignores_peaks_when_estimating_baseline():
    """Peaks sitting on a smooth baseline must not drag the baseline up."""
    n = 600
    x = np.linspace(0, 1, n)
    baseline = 100 + 60 * x + 40 * np.exp(-((x - 0.6) ** 2) / 0.02)
    signal = np.zeros(n)
    for c in (0.25, 0.5, 0.75):
        signal += 80 * np.exp(-((x - c) ** 2) / 2e-5)
    est, _, _ = airpls(signal + baseline, lam=1e5)
    # The estimate should stay close to the true baseline, well below the peak tops.
    assert np.max(np.abs(est - baseline)) < 0.25 * signal.max()


def test_endpoint_modes_differ():
    """The legacy endpoint rule is a real behavioural difference, not a rename."""
    rng = np.random.default_rng(4)
    n = 300
    x = np.linspace(0, 1, n)
    y = 20 + 40 * x + 50 * np.exp(-((x - 0.4) ** 2) / 1e-4) + rng.normal(0, 0.5, n)
    ref, _, _ = airpls(y, 1e3, endpoint="reference")
    leg, _, _ = airpls(y, 1e3, endpoint="legacy")
    assert not np.allclose(ref, leg)


def test_selectors_return_interior_lambda_on_realistic_input():
    """A selector pinned at a grid edge has not actually chosen anything."""
    rng = np.random.default_rng(5)
    n = 800
    x = np.linspace(0, 1, n)
    y = (
        300 * np.exp(-((x - 0.2) ** 2) / 0.08)          # broad fluorescence
        + 60 * x
        + sum(90 * np.exp(-((x - c) ** 2) / 1e-5) for c in (0.3, 0.45, 0.62, 0.8))
        + rng.normal(0, 1.0, n)
    )
    for res in (gcv_curve(y, n_probe=32), lcurve(y)):
        assert not res["at_edge"], f"selector pinned at grid edge: lam={res['lambda']:.3g}"


def test_correct_is_pure_single_vs_batch():
    """Stage 0 invariant: a spectrum's output cannot depend on its batch context."""
    rng = np.random.default_rng(6)
    n = 500
    x = np.linspace(0, 1, n)
    batch = []
    for k in range(4):
        y = (
            (100 + 50 * k) * np.exp(-((x - 0.25) ** 2) / 0.1)
            + sum(70 * np.exp(-((x - c) ** 2) / 1e-5) for c in (0.35, 0.55, 0.7))
            + rng.normal(0, 1.0, n)
        )
        batch.append(y)
    batch = np.array(batch)

    alone = [correct(y, method="gcv", n_probe=16)["corrected"] for y in batch]
    shuffled_order = [correct(batch[i], method="gcv", n_probe=16)["corrected"] for i in (2, 0, 3, 1)]
    remap = {2: 0, 0: 1, 3: 2, 1: 3}
    for i, orig in enumerate((2, 0, 3, 1)):
        assert np.array_equal(alone[orig], shuffled_order[i]), "output depends on processing order"
    assert remap  # keep the mapping explicit for the reader


def test_correct_with_fixed_lambda_skips_selector():
    rng = np.random.default_rng(7)
    y = rng.uniform(10, 100, 300)
    res = correct(y, lam=100.0)
    assert res["lambda"] == 100.0
    assert res["selector"] is None


def test_correct_clips_negative_by_default():
    rng = np.random.default_rng(8)
    y = rng.uniform(10, 100, 300)
    assert correct(y, lam=100.0)["corrected"].min() >= 0.0
    assert correct(y, lam=100.0, clip_negative=False)["corrected"].min() < 0.0


def test_default_grid_brackets_the_inherited_constant():
    assert DEFAULT_LAMBDA_GRID.min() < 100.0 < DEFAULT_LAMBDA_GRID.max()


# --------------------------------------------------------------------------
# IARPLS -- the Stage 0 recommended production path
# --------------------------------------------------------------------------

def _toy_spectrum(rng, n=800):
    wn = np.linspace(400, 3400, n)
    sig = sum(
        amp * np.exp(-0.5 * ((wn - c) / 8.0) ** 2)
        for c, amp in [(1004, 100), (1440, 120), (1660, 95), (2930, 180)]
    )
    base = 900 * np.exp(-((wn - 600) / 900.0) ** 2) + 0.05 * wn
    return wn, sig, base + sig + rng.normal(0, 1.0, n)


def test_iarpls_is_pure_batch_vs_alone():
    """Project invariant: no spectrum's output may depend on its batch context."""
    pytest.importorskip("ramanspy")
    from src.preprocessing.baseline import correct_iarpls

    rng = np.random.default_rng(11)
    specs, wn = [], None
    for _ in range(5):
        wn, _, obs = _toy_spectrum(rng)
        specs.append(obs)

    alone = np.array([correct_iarpls(s, wn)["corrected"] for s in specs])
    reversed_order = np.array([correct_iarpls(s, wn)["corrected"] for s in specs[::-1]])[::-1]
    assert np.array_equal(alone, reversed_order), "IARPLS output depends on processing order"


def test_iarpls_recovers_peak_heights_better_than_airpls_lambda_100():
    """The Stage 0 headline, pinned as a regression test."""
    pytest.importorskip("ramanspy")
    from src.preprocessing.baseline import correct_iarpls

    rng = np.random.default_rng(12)
    wn, sig, obs = _toy_spectrum(rng)

    def peak_ratio(corrected):
        ratios = []
        for c in (1004, 1440, 1660, 2930):
            m = np.abs(wn - c) <= 25
            ratios.append(corrected[m].max() / sig[m].max())
        return np.array(ratios)

    r_iarpls = peak_ratio(correct_iarpls(obs, wn, clip_negative=False)["corrected"])
    r_airpls = peak_ratio(correct(obs, lam=100.0, endpoint="legacy", clip_negative=False)["corrected"])

    assert np.all(np.abs(r_iarpls - 1.0) < 0.10), f"IARPLS distorted peaks: {r_iarpls}"
    # airPLS at lambda=100 eats the broad bands; IARPLS must be closer to truth overall.
    assert np.mean(np.abs(r_iarpls - 1.0)) < np.mean(np.abs(r_airpls - 1.0))


def test_iarpls_lambda_is_stable_on_a_SMOOTH_baseline():
    """Regression guard for the *smooth* baseline case only -- NOT general insensitivity.

    Stage 0 initially reported lambda inert over 1e2-1e4. The follow-up study
    (scripts/stage0_generator_stress.py) withdrew that as a general claim: the inert
    window is ~2.5 decades for smooth baselines but only ~0.5 for high-curvature ones,
    and real baselines appear more curved than any synthetic family tested. See
    docs/stage0_findings.md section 5.

    `_toy_spectrum` uses a single broad Gaussian background, i.e. the smooth regime, so
    this asserts only that behaviour there has not drifted.
    """
    pytest.importorskip("ramanspy")
    from src.preprocessing.baseline import correct_iarpls

    rng = np.random.default_rng(13)
    wn, sig, obs = _toy_spectrum(rng)
    m = (wn >= 700) & (wn < 1800)
    scale = np.ptp(sig)

    errs = [
        np.sqrt(np.mean((correct_iarpls(obs, wn, lam=lam, clip_negative=False)["corrected"][m]
                         - sig[m]) ** 2)) / scale
        for lam in (1e2, 1e3, 1e4)
    ]
    assert max(errs) - min(errs) < 0.01, f"smooth-baseline behaviour drifted: {errs}"


def test_iarpls_lambda_DOES_matter_on_a_curved_baseline():
    """The withdrawal in section 5, pinned: curvature makes lambda matter.

    If this ever fails, the stress-test conclusion no longer holds and the provisional
    IARPLS_LAMBDA default must be revisited.
    """
    pytest.importorskip("ramanspy")
    from src.preprocessing.baseline import correct_iarpls

    rng = np.random.default_rng(14)
    n = 1200
    wn = np.linspace(400, 3400, n)
    sig = sum(
        amp * np.exp(-0.5 * ((wn - c) / 8.0) ** 2)
        for c, amp in [(1004, 100), (1440, 120), (1660, 95), (2930, 180)]
    )
    scale = np.ptp(sig)
    t = (wn - wn.min()) / np.ptp(wn)
    # High-curvature background: two broad lobes plus a narrow one, at 3x signal amplitude.
    base = 3 * scale * (
        np.exp(-0.5 * ((t - 0.15) / 0.12) ** 2)
        + 0.8 * np.exp(-0.5 * ((t - 0.55) / 0.10) ** 2)
        + 0.6 * np.exp(-0.5 * ((t - 0.40) / 0.040) ** 2)
    )
    obs = sig + base + rng.normal(0, 1.0, n)

    m = (wn >= 700) & (wn < 1800)

    def err(lam):
        c = correct_iarpls(obs, wn, lam=lam, clip_negative=False)["corrected"]
        return float(np.sqrt(np.mean((c[m] - sig[m]) ** 2)) / scale)

    e3, e4, e5 = err(1e3), err(1e4), err(1e5)
    # A small lambda still recovers this baseline well ...
    assert e3 < 0.05, f"low lambda should still work here, got {e3:.4f}"
    # ... while the provisional default and a larger value both fail badly.
    assert e5 > 20 * e3, f"lambda should matter: 1e3 -> {e3:.4f}, 1e5 -> {e5:.4f}"
    assert e4 > 10 * e3, (
        f"IARPLS_LAMBDA=1e4 is provisional precisely because of this: "
        f"1e3 -> {e3:.4f}, 1e4 -> {e4:.4f}"
    )
