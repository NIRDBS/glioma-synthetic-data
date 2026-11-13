"""
Variance-standardized empirical variogram at Chebyshev lag 3 for every fit stored in
an experiment's latent-label stores (real and shuffled-label controls), per MIL tier.

Why: src/evaluation/spatial.py's variogram gamma(h) = 0.5 * mean((x_i - x_j)^2) is in
absolute units, so it scales with the variance of the posterior map. A shuffled-label
refit can have a very different marginal spread from the real fit, which makes raw
gamma comparisons between the two meaningless as a coherence comparison (Moran's I
divides the variance out; the raw variogram does not). For a stationary field
gamma(h) = sigma^2 (1 - rho(h)), so gamma(h) / sigma^2 is on the same footing as
1 - I(h) and is what should be compared between fits. Per slide, sigma^2 is the
population variance of the in-mask map; slides with sigma^2 = 0 (constant maps) have
an undefined ratio and are excluded and counted. The reported value is the mean over
slides, like the pooled statistics in scripts/spatial_validation.py (same
eligibility: >= 20 covered instances, slide in the shape registry).

Writes results/<exp>/spatial_validation/standardized_variogram_lag3.json:
    {"<method>_<variant>": {"real": {...}, "shuffled": {...}}}
where each fit has gamma3_over_var, n_slides, n_constant_excluded, store.

Usage:
    python scripts/standardized_variogram.py --config exp_608
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spatial_validation import _load_raw_side_tables  # noqa: E402
from src.data.spatial import reproject  # noqa: E402
from src.evaluation.spatial import variogram_by_lag  # noqa: E402
from src.mil.store import merge_out_of_sample_posterior  # noqa: E402

LAG = 3


def _groups(store_path):
    with h5py.File(store_path, 'r') as f:
        return [(m, v) for m in f for v in f[m]]


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
    slides = sorted(set(sample_ids.tolist()))

    out = {}
    for store in sorted(latent.glob('latent_labels*.h5')):
        suffix = store.stem[len('latent_labels'):].lstrip('_')
        fit = 'real' if suffix == '' else 'shuffled' if suffix.startswith('shuffled') else None
        if fit is None:
            continue
        for method, variant in _groups(store):
            try:
                post, _cov = merge_out_of_sample_posterior(store, method, variant, folds)
            except (KeyError, OSError):
                continue
            nd = post[:, 0]
            ratios, n_const = [], 0
            for s in slides:
                if s not in shapes:
                    continue
                idx = np.where(sample_ids == s)[0]
                ok = ~np.isnan(nd[idx])
                if ok.sum() < 20:
                    continue
                img = reproject(nd[idx][ok], coords[idx][ok], shapes[s], fill=np.nan)
                var = float(np.nanvar(img))
                g = variogram_by_lag(img, max_lag=LAG)[LAG]['gamma']
                if var == 0 or np.isnan(g):
                    n_const += 1
                    continue
                ratios.append(g / var)
            out.setdefault(f'{method}_{variant}', {})[fit] = {
                'gamma3_over_var': float(np.mean(ratios)), 'n_slides': len(ratios),
                'n_constant_excluded': n_const, 'store': store.name}
            print(f'{method}/{variant:15s} {fit:8s} gamma3/var={np.mean(ratios):.4f} '
                  f'(n={len(ratios)}, constant maps excluded={n_const})  [{store.name}]', flush=True)

    path = root / 'spatial_validation' / 'standardized_variogram_lag3.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
