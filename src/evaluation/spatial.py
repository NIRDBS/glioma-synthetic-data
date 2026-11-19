"""Spatial statistics of a per-pixel map, by Chebyshev lag: Moran's I and the empirical semivariogram.

The paper uses them to ask whether a MIL posterior reprojected onto the slide's pixel grid
(src/data/spatial.py::reproject) is spatially coherent while no model ever saw the coordinates.
The 3x3 spatial median filter couples a pixel only to its 8 neighbours, so two pixels at Chebyshev
distance >= 3 share no input pixel and the filter cannot create correlation between them; the paper
therefore reports lag 3.
"""
import numpy as np

__all__ = ["morans_i_by_lag", "variogram_by_lag"]


def _offsets_at_lag(lag):
    """All (dy, dx) integer offsets at Chebyshev distance exactly `lag` --
    the 8*lag-cell ring of an L-infinity ball, not the filled square."""
    offsets = []
    for dy in range(-lag, lag + 1):
        for dx in range(-lag, lag + 1):
            if max(abs(dy), abs(dx)) == lag:
                offsets.append((dy, dx))
    return offsets


def _paired_values(image, dy, dx):
    """Every (a, b) pair of pixel values at offset (dy, dx), restricted to
    positions where BOTH pixels are in-mask (not NaN). Returns two 1-D arrays
    of equal length, or two empty arrays if the offset doesn't fit the image
    or no pair is fully in-mask."""
    H, W = image.shape
    y0, y1 = max(0, -dy), H - max(0, dy)
    x0, x1 = max(0, -dx), W - max(0, dx)
    if y1 <= y0 or x1 <= x0:
        return np.array([]), np.array([])
    a = image[y0:y1, x0:x1]
    b = image[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    valid = ~np.isnan(a) & ~np.isnan(b)
    return a[valid], b[valid]


def morans_i_by_lag(image, max_lag=10):
    """
    Moran's I computed separately at each Chebyshev-distance lag band
    (1..max_lag), over the image's own NaN mask (NaN = out-of-bag pixel,
    excluded from every pair it appears in). Returns
    {lag: {'I': float, 'n_pairs': int}}.
    """
    image = np.asarray(image, dtype=np.float64)
    mask = ~np.isnan(image)
    values = image[mask]
    if len(values) < 2:
        return {lag: {'I': float('nan'), 'n_pairs': 0} for lag in range(1, max_lag + 1)}

    x_bar = values.mean()
    denom = np.sum((values - x_bar) ** 2)

    results = {}
    for lag in range(1, max_lag + 1):
        num = 0.0
        n_pairs = 0
        for dy, dx in _offsets_at_lag(lag):
            a, b = _paired_values(image, dy, dx)
            if len(a) == 0:
                continue
            num += np.sum((a - x_bar) * (b - x_bar))
            n_pairs += len(a)
        if n_pairs == 0 or denom == 0:
            results[lag] = {'I': float('nan'), 'n_pairs': 0}
        else:
            n = len(values)
            results[lag] = {'I': float((n / n_pairs) * (num / denom)), 'n_pairs': n_pairs}
    return results


def variogram_by_lag(image, max_lag=10):
    """
    Empirical semivariogram gamma(h) = 0.5 * mean((x_i - x_j)^2) over pixel
    pairs at Chebyshev distance h, same masking convention as
    ``morans_i_by_lag``. Low gamma at a given lag means values at that
    distance are similar (coherent); gamma rising and plateauing with lag is
    the classic variogram signature, and the plateau's onset -- the "range"
    -- is itself evidence (§6: "1-2 px means the filter, 10-20 px means
    tissue"). Returns {lag: {'gamma': float, 'n_pairs': int}}.
    """
    image = np.asarray(image, dtype=np.float64)
    results = {}
    for lag in range(1, max_lag + 1):
        sq_diffs = []
        n_pairs = 0
        for dy, dx in _offsets_at_lag(lag):
            a, b = _paired_values(image, dy, dx)
            if len(a) == 0:
                continue
            sq_diffs.append(np.sum((a - b) ** 2))
            n_pairs += len(a)
        if n_pairs == 0:
            results[lag] = {'gamma': float('nan'), 'n_pairs': 0}
        else:
            results[lag] = {'gamma': float(0.5 * sum(sq_diffs) / n_pairs), 'n_pairs': n_pairs}
    return results
