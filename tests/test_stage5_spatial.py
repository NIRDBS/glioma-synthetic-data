"""Stage 5 regression tests: spatial validation (docs/MIL_REFRAME_PLAN.md §6).

Covers the plan's Stage 5 test bullets:
  1. Lag-band computation verified on synthetic maps of known correlation
     length.
  2. Assert no MIL model in the chain ever receives coords.
  (The third bullet -- "shuffled-label refit collapses lag >= 3 coherence" --
  is an integration-level check against a real, expensive refit, exercised
  separately against real data, not as a synthetic unit test here.)
"""
import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.evaluation.spatial import (
    external_agreement,
    morans_i_by_lag,
    permutation_null_by_lag,
    pool_lag_reports,
    variogram_by_lag,
    variogram_range,
)
from src.mil.misvm import fit_misvm


def _smooth_field(shape=(60, 60), sigma=3.0, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=shape)
    return gaussian_filter(noise, sigma=sigma)


def _white_noise_field(shape=(60, 60), seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape)


# ---------------------------------------------------------------------------
# 1. Lag-band computation on synthetic maps of known correlation length
# ---------------------------------------------------------------------------

def test_morans_i_near_zero_for_white_noise_at_every_lag():
    image = _white_noise_field(seed=1)
    report = morans_i_by_lag(image, max_lag=8)
    for lag, r in report.items():
        assert abs(r['I']) < 0.15, f"lag {lag}: I={r['I']} too far from 0 for pure noise"


def test_morans_i_high_at_short_lag_for_smooth_field():
    image = _smooth_field(sigma=4.0, seed=2)
    report = morans_i_by_lag(image, max_lag=10)
    assert report[1]['I'] > 0.5


def test_morans_i_decays_with_lag_for_smooth_field():
    image = _smooth_field(sigma=4.0, seed=3)
    report = morans_i_by_lag(image, max_lag=10)
    I_by_lag = [report[l]['I'] for l in range(1, 11)]
    # Not required to be perfectly monotone (noisy field), but the long-range
    # value must be clearly below the short-range one.
    assert I_by_lag[0] > I_by_lag[-1]
    assert I_by_lag[0] > 0.4
    assert I_by_lag[-1] < 0.3


def test_morans_i_confined_to_short_lag_for_narrow_smoothing():
    """This is the actual property Stage 5's headline claim depends on: a
    narrowly-correlated field (small sigma, standing in for the 3x3 median
    filter's own reach) must show coherence at short lag but NOT at lag>=3 --
    otherwise the lag>=3 statistic would not be a meaningful headline at all."""
    image = _smooth_field(sigma=0.8, seed=4)
    report = morans_i_by_lag(image, max_lag=8)
    assert report[1]['I'] > 0.3
    assert report[6]['I'] < 0.15
    assert report[8]['I'] < 0.15


def test_variogram_rises_then_is_higher_at_long_lag_than_short_lag():
    image = _smooth_field(sigma=4.0, seed=5)
    report = variogram_by_lag(image, max_lag=10)
    assert report[1]['gamma'] < report[10]['gamma']


def test_variogram_range_scales_with_smoothing_scale():
    """A larger smoothing sigma must yield a larger estimated range -- the
    plan's own diagnostic use of this number (§6: "1-2 px means the filter,
    10-20 px means tissue") only makes sense if range tracks true
    correlation length, which this test checks directly rather than assuming."""
    small = variogram_by_lag(_smooth_field(sigma=1.0, seed=6), max_lag=15)
    large = variogram_by_lag(_smooth_field(sigma=5.0, seed=7), max_lag=15)
    range_small = variogram_range(small)
    range_large = variogram_range(large)
    assert range_small is not None and range_large is not None
    assert range_large > range_small


def test_permutation_null_flags_smooth_field_as_significant_at_short_lag():
    image = _smooth_field(sigma=4.0, seed=8)
    report = permutation_null_by_lag(image, max_lag=5, n_perm=100, seed=8,
                                     statistic='morans_i')
    assert report[1]['p'] < 0.05
    assert report[1]['z'] > 2.0


def test_permutation_null_does_not_flag_white_noise():
    image = _white_noise_field(seed=9)
    report = permutation_null_by_lag(image, max_lag=5, n_perm=200, seed=9,
                                     statistic='morans_i')
    # Pure noise: p should NOT be small at any lag (loose bound -- this is a
    # random draw, not a guarantee, but should hold overwhelmingly often).
    assert report[1]['p'] > 0.05


def test_permutation_null_variogram_direction_is_reversed():
    """Lower variogram = more coherent, the opposite direction from Moran's I
    -- checked explicitly since it's the one place the two statistics'
    p-value conventions differ and a sign error would be easy to miss."""
    image = _smooth_field(sigma=4.0, seed=10)
    report = permutation_null_by_lag(image, max_lag=5, n_perm=100, seed=10,
                                     statistic='variogram')
    assert report[1]['p'] < 0.05  # observed gamma is LOW relative to the shuffled null


def test_pool_lag_reports_averages_across_slides():
    reports = [
        {1: {'observed': 0.8}, 2: {'observed': 0.6}},
        {1: {'observed': 0.6}, 2: {'observed': 0.4}},
    ]
    pooled = pool_lag_reports(reports, key='observed')
    assert pooled[1]['mean'] == pytest.approx(0.7)
    assert pooled[2]['mean'] == pytest.approx(0.5)
    assert pooled[1]['n_slides'] == 2


def test_pool_lag_reports_excludes_nan_slides_from_that_lags_mean():
    reports = [
        {1: {'observed': 0.8}},
        {1: {'observed': float('nan')}},
    ]
    pooled = pool_lag_reports(reports, key='observed')
    assert pooled[1]['mean'] == pytest.approx(0.8)
    assert pooled[1]['n_slides'] == 1


def test_external_agreement_perfect_match():
    labels = np.array([0, 0, 1, 1, 2, 2])
    result = external_agreement(labels, labels)
    assert result['ari'] == pytest.approx(1.0)
    assert result['ami'] == pytest.approx(1.0)


def test_external_agreement_random_labels_near_zero():
    rng = np.random.default_rng(11)
    a = rng.integers(0, 3, size=500)
    b = rng.integers(0, 3, size=500)
    result = external_agreement(a, b)
    assert abs(result['ari']) < 0.1


# ---------------------------------------------------------------------------
# hard_diagnostic_threshold: the K=2-softmax external-agreement fix
# (docs/MIL_REFRAME_PLAN.md backlog, docs/stage6_generative_mil.md Sec 4,
# docs/stage6_attention_mil.md Sec 5)
# ---------------------------------------------------------------------------

def test_hard_diagnostic_threshold_misvm_is_untouched_at_flat_half():
    """mi-SVM's independent one-vs-rest scores aren't bounded below 0.5 the
    way a joint softmax's are -- already non-degenerate under a flat 0.5, so
    this fix must not silently change its already-reported result."""
    from scripts.stage5_spatial_validation import hard_diagnostic_threshold
    assert hard_diagnostic_threshold('misvm', K=2) == pytest.approx(0.5)


def test_hard_diagnostic_threshold_joint_softmax_k2_matches_the_derived_bound():
    """For a genuine joint K-way softmax, non-diagnostic mass after the
    K+1 lift is bounded above by (K-1)/(2K-1) -- 1/3 at K=2 -- so the
    rescaled threshold (half that bound) must be exactly 1/6, not 0.5."""
    from scripts.stage5_spatial_validation import hard_diagnostic_threshold
    assert hard_diagnostic_threshold('attention', K=2) == pytest.approx(1.0 / 6.0)
    assert hard_diagnostic_threshold('cbr', K=2) == pytest.approx(1.0 / 6.0)


def test_hard_diagnostic_threshold_joint_softmax_scales_with_k():
    """The bound loosens toward, but never reaches, 0.5 as K grows -- the
    rescaled threshold should track that, e.g. K=6 gives a looser (larger)
    cutoff than K=2, but always strictly below the flat 0.5 mi-SVM uses."""
    from scripts.stage5_spatial_validation import hard_diagnostic_threshold
    t2 = hard_diagnostic_threshold('cbr', K=2)
    t6 = hard_diagnostic_threshold('cbr', K=6)
    assert t2 < t6 < 0.5


# ---------------------------------------------------------------------------
# 2. No MIL model in the chain ever receives coords
# ---------------------------------------------------------------------------

def test_no_mil_formulation_accepts_coords():
    """A direct, permanent guard: Tier 1 (mi-SVM)'s fit function has no
    coordinate-shaped parameter at all -- geometry cannot leak into the model
    by construction, not merely by convention."""
    for fn in (fit_misvm,):
        params = set(inspect.signature(fn).parameters.keys())
        coord_like = {p for p in params if 'coord' in p.lower()}
        assert not coord_like, f"{fn.__name__} unexpectedly accepts {coord_like}"
