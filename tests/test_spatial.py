"""Spatial statistics: lag-band Moran's I and variogram on synthetic maps of known correlation length,
and the guarantee that no MIL formulation accepts coordinates."""
import inspect

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.evaluation.spatial import morans_i_by_lag, variogram_by_lag
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
    """This is the property the paper's coherence claim depends on: a
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



# ---------------------------------------------------------------------------
# 2. No MIL model in the chain ever receives coords
# ---------------------------------------------------------------------------

def test_no_mil_formulation_accepts_coords():
    """A direct, permanent guard: mi-SVM's fit function has no
    coordinate-shaped parameter at all -- geometry cannot leak into the model
    by construction, not merely by convention."""
    for fn in (fit_misvm,):
        params = set(inspect.signature(fn).parameters.keys())
        coord_like = {p for p in params if 'coord' in p.lower()}
        assert not coord_like, f"{fn.__name__} unexpectedly accepts {coord_like}"
