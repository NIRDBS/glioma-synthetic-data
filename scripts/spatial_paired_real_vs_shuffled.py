"""
Paired per-slide comparison of spatial coherence, real fit vs shuffled-label control.

For every MIL fit stored in an experiment's latent-label stores, compute per slide, on
the out-of-sample non-diagnostic-mass map,
    I_s      Moran's I at Chebyshev lag 3
    V_s      1 - gamma_3 / sigma^2   (variance-standardized variogram; same scale as I)
for the real fit and for the shuffled-label refit. The paired difference
    d_s = coherence_real(s) - coherence_shuffled(s)
is taken on the slides where BOTH fits give a finite value (so the real and shuffled
sets are matched; constant maps are undefined for the ratio and dropped on both sides).
The mean of d_s gets a 95% percentile CI from a bootstrap over slides (resampling
slides with replacement), plus a sign-flip permutation p-value on the mean and the
Wilcoxon signed-rank p-value. "Real more coherent" is supported when the CI of the
mean excludes zero on the positive side.

Writes results/<exp>/spatial_validation/spatial_paired_real_vs_shuffled.json with the
per-slide values, so the test can be re-run without re-reprojecting.

Usage:
    python scripts/spatial_paired_real_vs_shuffled.py --config exp_608
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stage5_spatial_validation import _load_raw_side_tables  # noqa: E402
from src.data.spatial import reproject  # noqa: E402
from src.evaluation.spatial import morans_i_by_lag, variogram_by_lag  # noqa: E402
from src.mil.store import merge_out_of_sample_posterior  # noqa: E402

LAG = 3
N_BOOT = 10000
SEED = 42
MIN_COVERED = 20   # same eligibility as scripts/stage5_spatial_validation.py


def _groups(store_path):
    with h5py.File(store_path, 'r') as f:
        return [(m, v) for m in f for v in f[m]]


def per_slide_coherence(nd, sample_ids, coords, shapes):
    """{slide: (I, 1 - gamma3/sigma^2)}; NaN where undefined (constant map)."""
    out = {}
    for s in sorted(set(sample_ids.tolist())):
        if s not in shapes:
            continue
        idx = np.where(sample_ids == s)[0]
        ok = ~np.isnan(nd[idx])
        if ok.sum() < MIN_COVERED:
            continue
        img = reproject(nd[idx][ok], coords[idx][ok], shapes[s], fill=np.nan)
        var = float(np.nanvar(img))
        g = variogram_by_lag(img, max_lag=LAG)[LAG]['gamma']
        i3 = morans_i_by_lag(img, max_lag=LAG)[LAG]['I']
        v3 = np.nan if (var == 0 or np.isnan(g)) else 1.0 - g / var
        out[s] = (float(i3), float(v3))
    return out


def paired_test(real, shuf, rng):
    """real, shuf: {slide: value}. Matched on slides finite in both."""
    slides = [s for s in real if s in shuf and np.isfinite(real[s]) and np.isfinite(shuf[s])]
    d = np.array([real[s] - shuf[s] for s in slides])
    n = len(d)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, n))
    perm = np.abs((signs * d).mean(axis=1))
    try:
        w_p = float(wilcoxon(d).pvalue)
    except ValueError:   # all differences zero
        w_p = 1.0
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'n_matched': n,
        'mean_real': float(np.mean([real[s] for s in slides])),
        'mean_shuffled': float(np.mean([shuf[s] for s in slides])),
        'delta': float(d.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
        'p_signflip': float((np.sum(perm >= abs(d.mean()) - 1e-15) + 1) / (N_BOOT + 1)),
        'p_wilcoxon': w_p,
        'frac_slides_real_higher': float(np.mean(d > 0)),
        'significant_95_real_higher': bool(lo > 0),
        'slides': slides, 'per_slide_delta': d.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    from src.utils.config_manager import ConfigManager
    config = ConfigManager.load_config(args.config).to_dict()
    root = Path(config['paths'].get('experiment_root', f'results/{args.config}'))
    latent = root / 'latent'
    sample_ids, coords, _viable, shapes = _load_raw_side_tables(config['paths']['processed_file'])
    folds = [f'fold_{i}' for i in range(5)]

    coh = {}   # '<method>_<variant>' -> {'real'|'shuffled' -> {slide: (I, V)}}
    for store in sorted(latent.glob('latent_labels*.h5')):   # later stores overwrite, as in standardized_variogram.py
        suffix = store.stem[len('latent_labels'):].lstrip('_')
        fit = 'real' if suffix == '' else 'shuffled' if suffix.startswith('shuffled') else None
        if fit is None:
            continue
        for method, variant in _groups(store):
            try:
                post, _ = merge_out_of_sample_posterior(store, method, variant, folds)
            except (KeyError, OSError):
                continue
            coh.setdefault(f'{method}_{variant}', {})[fit] = per_slide_coherence(
                post[:, 0], sample_ids, coords, shapes)
            coh[f'{method}_{variant}'][fit + '_store'] = store.name

    rng = np.random.default_rng(SEED)
    out = {'lag': LAG, 'n_boot': N_BOOT, 'seed': SEED, 'fits': {}}
    for key, d in sorted(coh.items()):
        if 'real' not in d or 'shuffled' not in d:
            continue
        res = {'stores': {'real': d['real_store'], 'shuffled': d['shuffled_store']}}
        for j, name in enumerate(('morans_i', 'one_minus_gamma_over_var')):
            res[name] = paired_test({s: v[j] for s, v in d['real'].items()},
                                    {s: v[j] for s, v in d['shuffled'].items()}, rng)
            r = res[name]
            print(f'{key:28s} {name:24s} n={r["n_matched"]:2d} real={r["mean_real"]:+.4f} '
                  f'shuf={r["mean_shuffled"]:+.4f} d={r["delta"]:+.4f} '
                  f'[{r["ci_lo"]:+.4f},{r["ci_hi"]:+.4f}] p_flip={r["p_signflip"]:.3f} '
                  f'p_wsr={r["p_wilcoxon"]:.3f}', flush=True)
        out['fits'][key] = res

    path = root / 'spatial_validation' / 'spatial_paired_real_vs_shuffled.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
