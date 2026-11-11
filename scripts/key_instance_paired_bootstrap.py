"""
Paired per-slide bootstrap test of key-instance utility.

For every tier with per-slide predictions in
results/<exp>/mil_eval/key_instance_per_slide/<method>_<variant>.json
(written by scripts/cvae_weighted_rf_vote.py), take for each slide s the correctness
c_s in {0,1} of three arms scored on the same held-out slides:
    selected  RF trained on the tier's key instances, voting with the slide's key instances
    naive     the TR/TR RF, voting with all of the slide's instances
    random    RF trained on a size-matched random subset, voting with the key instances
and the paired differences
    d_s = c_s(selected) - c_s(naive),    d_s = c_s(selected) - c_s(random).
The mean of d_s (the accuracy difference) gets a 95% percentile CI from a bootstrap over
slides (resampling slides with replacement), as in scripts/spatial_paired_real_vs_shuffled.py.
"Selection helps" is supported when the CI lies above zero.

Writes results/<exp>/mil_eval/key_instance_paired.json.

Usage:
    python scripts/key_instance_paired_bootstrap.py --config exp_608
"""
import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEED = 42


def paired_test(sel, other):
    """sel, other: per-slide correctness (0/1), aligned. Each test draws from its own
    generator, so a tier's CI does not depend on which other tiers are present."""
    d = np.asarray(sel, dtype=float) - np.asarray(other, dtype=float)
    n = len(d)
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'n_slides': n,
        'acc_selected': float(np.mean(sel)), 'acc_other': float(np.mean(other)),
        'delta': float(d.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
        'significant_95_selected_higher': bool(lo > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    root = Path('results') / args.config / 'mil_eval'
    out = {'n_boot': N_BOOT, 'seed': SEED, 'tiers': {}}
    for path in sorted((root / 'key_instance_per_slide').glob('*.json')):
        p = json.loads(path.read_text(encoding='utf-8'))
        y = np.array(p['y_true'])
        correct = {arm: (np.array(p[arm]) == y).astype(int) for arm in ('selected', 'naive', 'random')}
        res = {'source': path.name,
               'vs_naive': paired_test(correct['selected'], correct['naive']),
               'vs_random': paired_test(correct['selected'], correct['random'])}
        for name in ('vs_naive', 'vs_random'):
            r = res[name]
            print(f'{path.stem:28s} {name:9s} n={r["n_slides"]} sel={r["acc_selected"]:.4f} '
                  f'other={r["acc_other"]:.4f} d={r["delta"]:+.4f} [{r["ci_lo"]:+.4f},{r["ci_hi"]:+.4f}]',
                  flush=True)
        out['tiers'][path.stem] = res

    path = root / 'key_instance_paired.json'
    path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
