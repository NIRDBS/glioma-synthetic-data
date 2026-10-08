"""Unit tests for src/data/preprocessing_v2.py's two new functions --
cube_normalize and compute_strong_signal_mask -- the pieces that replace
NMF+GMM+manual curation. The mask's 3-rule algorithm was worked out and
confirmed with the user directly against real-slide numbers (the first,
simpler version checked p90>=1.0 as a saturation trigger, which fired on
EVERY real slide tested -- 62-90% of pixels sit exactly at the clip ceiling
on this data, not a rare sliver -- so that version was measuring the wrong
thing; these tests pin the corrected, confirmed algorithm).
"""
import numpy as np
import pytest

from src.data.preprocessing_v2 import compute_strong_signal_mask, cube_normalize


def test_cube_normalize_shifts_scales_and_clips_to_unit_interval():
    rng = np.random.default_rng(0)
    cube = rng.normal(loc=50, scale=10, size=(20, 20, 30)).astype(np.float32)
    cube[0, 0, 0] = cube.max() * 5  # a single extreme outlier, clipped by the 99.9th-pctile scale

    out = cube_normalize(cube)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out[0, 0, 0] == 1.0, "the outlier should be clipped to exactly the ceiling"
    assert 0.0 < np.median(out) < 1.0


def test_rule2_main_case_keeps_exactly_the_top_90_percent_by_rank():
    """saturation_rate <= 0.9 -> Rule 2 (rank-based 90%) with Rule 3's 0.8
    floor a no-op, since this distribution's weak tail sits above 0.8."""
    rng = np.random.default_rng(1)
    max_vals = rng.uniform(0.85, 0.999, size=(50, 50))  # well above the 0.8 floor, < 1.0 (no saturation)
    cube = max_vals[:, :, None] * rng.uniform(0.9, 1.0, size=(50, 50, 10))
    cube[:, :, -1] = max_vals  # guarantee the true per-pixel max IS max_vals exactly

    mask, info = compute_strong_signal_mask(cube)
    assert info["saturation_rate"] <= 0.9
    assert info["branch"].startswith("rule2")
    assert info["n_kept_by_rank_alone"] == info["n_kept"], "Rule 3's floor should be a no-op here"
    assert info["frac_kept"] == pytest.approx(0.90, abs=0.02)
    recomputed_max = cube.max(axis=-1)
    assert np.array_equal(mask, recomputed_max >= info["p10_rank_threshold"])


def test_rule3_floor_trims_a_genuinely_weak_tail_within_the_top_90_percent():
    """Construct a slide where the rank-based top-90% group still contains a
    chunk of pixels below the 0.8 absolute floor -- Rule 3 must remove them,
    confirmed by n_kept < n_kept_by_rank_alone."""
    rng = np.random.default_rng(2)
    max_vals = np.concatenate([
        rng.uniform(0.3, 0.6, 200),    # weak pixels, below 0.8, inside the top-90% by rank
        rng.uniform(0.85, 0.99, 800),  # strong pixels
    ])
    rng.shuffle(max_vals)
    cube = max_vals.reshape(100, 10, 1) * np.ones((100, 10, 5))
    cube[:, :, 0] = max_vals.reshape(100, 10)  # guarantee true max == max_vals

    mask, info = compute_strong_signal_mask(cube)
    assert info["branch"].startswith("rule2")
    assert info["p10_rank_threshold"] < 0.8, "sanity: this slide's rank-90 group should dip below the floor"
    assert info["n_kept"] < info["n_kept_by_rank_alone"], "Rule 3 should trim the weak tail Rule 2 alone kept"
    # every kept pixel must satisfy BOTH conditions
    recomputed_max = cube.max(axis=-1)
    assert (recomputed_max[mask] > 0.8).all()
    assert (recomputed_max[mask] >= info["p10_rank_threshold"]).all()


def test_rule1_fires_only_above_0point9_saturation_and_uses_fixed_0point95_floor():
    """saturation_rate > 0.9 (strict) -> Rule 1: fixed 0.95 floor, Rule 2/3
    never evaluated at all (no rank threshold reported)."""
    rng = np.random.default_rng(3)
    max_vals = np.concatenate([
        np.full(950, 1.0),             # 95% exactly saturated
        rng.uniform(0.0, 0.5, 50),     # 5% genuinely weak
    ])
    rng.shuffle(max_vals)
    cube = max_vals.reshape(10, 100, 1) * np.ones((10, 100, 5))
    cube[:, :, 0] = max_vals.reshape(10, 100)

    mask, info = compute_strong_signal_mask(cube)
    assert info["saturation_rate"] == pytest.approx(0.95, abs=0.01)
    assert info["branch"].startswith("rule1")
    assert info["threshold"] == 0.95
    assert info["p10_rank_threshold"] is None
    # the 5% weak pixels (well below 0.95) must be excluded; the 95% at the
    # ceiling (> 0.95) must be kept
    recomputed_max = cube.max(axis=-1)
    assert mask.sum() == (recomputed_max > 0.95).sum()


def test_rule1_boundary_is_strict_at_exactly_0point9():
    """saturation_rate == exactly 0.9 must NOT trigger Rule 1 (the condition
    is '> 0.9', confirmed strict, not '>= 0.9')."""
    max_vals = np.concatenate([np.full(900, 1.0), np.full(100, 0.5)])
    cube = max_vals.reshape(10, 100, 1) * np.ones((10, 100, 5))
    cube[:, :, 0] = max_vals.reshape(10, 100)

    mask, info = compute_strong_signal_mask(cube)
    assert info["saturation_rate"] == pytest.approx(0.9)
    assert info["branch"].startswith("rule2"), "exactly 0.9 must fall through to rule2+3, not rule1"
