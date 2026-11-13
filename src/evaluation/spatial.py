"""
Stage 5: spatial validation.

The design: withhold coords from every MIL model, infer the latent posterior,
reproject it onto the slide's pixel grid (src/data/spatial.py::reproject),
and ask whether the resulting map is spatially coherent. Coherence is then
*evidence* a model has recovered real tissue structure, rather than an
artefact of the training objective (which never saw geometry at all).

The confound and its fix (§6): the 3x3 spatial median filter (signal
cleaning, kept -- see §6's methods argument) couples a pixel only to its 8
neighbours, so two pixels at Chebyshev distance >= 3 share no input pixel
whatsoever and the filter provably cannot manufacture correlation between
them. Every statistic here is therefore reported BY LAG BAND, not as one
pooled number: lags 1-2 are contaminated and shown but excluded from the
headline claim; lag >= 3 is the actual evidence.
"""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

__all__ = [
    "morans_i_by_lag", "variogram_by_lag", "permutation_null_by_lag",
    "pool_lag_reports", "external_agreement", "variogram_range",
]


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


def variogram_range(gamma_by_lag, plateau_fraction=0.9):
    """
    A simple, quantitative range estimate: the smallest lag at which gamma(h)
    first reaches ``plateau_fraction`` of the maximum gamma observed across
    the reported lags. Not a model fit (no spherical/exponential variogram
    fitted) -- a plain, inspectable heuristic matching §6's own framing
    ("the range itself becomes the reported evidence"), not a black box.
    Returns None if gamma never reaches the threshold (e.g. all-NaN input).
    """
    lags = sorted(gamma_by_lag.keys())
    gammas = np.array([gamma_by_lag[l]['gamma'] for l in lags])
    if np.all(np.isnan(gammas)) or np.nanmax(gammas) <= 0:
        return None
    threshold = plateau_fraction * np.nanmax(gammas)
    for lag, g in zip(lags, gammas):
        if not np.isnan(g) and g >= threshold:
            return lag
    return None


def permutation_null_by_lag(image, max_lag=10, n_perm=200, seed=42, statistic='morans_i'):
    """
    Permutation null for either statistic: shuffle the IN-MASK values among
    themselves (mask geometry -- which pixels are in-bag at all -- held
    fixed), recompute the by-lag statistic, repeat ``n_perm`` times. Returns
    {lag: {'observed': float, 'null_mean': float, 'null_std': float,
    'z': float, 'p': float, 'n_pairs': int}}.

    p is one-sided in the direction that counts as "more coherent than
    chance": for Moran's I, higher is more coherent (p = P(null >= observed));
    for the variogram, LOWER gamma is more coherent (p = P(null <= observed)).
    """
    if statistic not in ('morans_i', 'variogram'):
        raise ValueError(f"unknown statistic {statistic!r}")
    stat_fn = morans_i_by_lag if statistic == 'morans_i' else variogram_by_lag
    key = 'I' if statistic == 'morans_i' else 'gamma'

    image = np.asarray(image, dtype=np.float64)
    mask = ~np.isnan(image)
    values = image[mask]

    observed = stat_fn(image, max_lag=max_lag)

    rng = np.random.default_rng(seed)
    null_stats = {lag: [] for lag in range(1, max_lag + 1)}
    shuffled = image.copy()
    for _ in range(n_perm):
        shuffled[mask] = rng.permutation(values)
        stat = stat_fn(shuffled, max_lag=max_lag)
        for lag in range(1, max_lag + 1):
            null_stats[lag].append(stat[lag][key])

    report = {}
    for lag in range(1, max_lag + 1):
        obs = observed[lag][key]
        null_arr = np.array(null_stats[lag], dtype=np.float64)
        null_mean = float(np.nanmean(null_arr)) if len(null_arr) else float('nan')
        null_std = float(np.nanstd(null_arr)) if len(null_arr) else float('nan')
        z = ((obs - null_mean) / null_std) if (null_std and null_std > 0
                                               and not np.isnan(obs)) else float('nan')
        if np.isnan(obs):
            p = float('nan')
        elif statistic == 'morans_i':
            p = float(np.mean(null_arr >= obs))
        else:
            p = float(np.mean(null_arr <= obs))
        report[lag] = {
            'observed': obs, 'null_mean': null_mean, 'null_std': null_std,
            'z': z, 'p': p, 'n_pairs': observed[lag]['n_pairs'],
        }
    return report


def pool_lag_reports(per_slide_reports, key='observed'):
    """
    Pool several slides' per-lag reports (whatever ``permutation_null_by_lag``
    or ``morans_i_by_lag``/``variogram_by_lag`` returned, one dict per slide)
    into one across-slide summary per lag: mean/std of ``key`` across slides,
    n_slides contributing (slides where that lag had zero valid pairs are
    excluded from that lag's pool, not counted as zero).
    """
    if not per_slide_reports:
        return {}
    all_lags = sorted(per_slide_reports[0].keys())
    pooled = {}
    for lag in all_lags:
        vals = [r[lag][key] for r in per_slide_reports
               if lag in r and not np.isnan(r[lag].get(key, float('nan')))]
        pooled[lag] = {
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std': float(np.std(vals)) if vals else float('nan'),
            'n_slides': len(vals),
        }
    return pooled


def external_agreement(hard_assignment, reference_labels):
    """
    ARI/AMI between a learned hard assignment and an independent reference
    clustering (§6: "the strongest available check", e.g. against
    hard_labels.npy's NMF+GMM tissue clusters, or against viable_legacy as a
    binary reference when only a K+1-collapsed posterior is available).
    Thin wrapper, kept as one call site so the exact
    metrics used are consistent everywhere they're reported.
    """
    hard_assignment = np.asarray(hard_assignment)
    reference_labels = np.asarray(reference_labels)
    return {
        'ari': float(adjusted_rand_score(reference_labels, hard_assignment)),
        'ami': float(adjusted_mutual_info_score(reference_labels, hard_assignment)),
        'n': int(len(hard_assignment)),
    }
