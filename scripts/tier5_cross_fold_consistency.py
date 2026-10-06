"""
Tier 5 cross-fold pi_b consistency check (docs/tier5_bayesian_mil.md section 2).

R-hat (``src/mil/bayes.py::gelman_rubin``) only checks whether a fold's own
chains agree with each other -- it cannot see whether the mode they agree on
is the same one another fold's independent fit would find. Since most bags
are TRAINING bags in several of the 5 folds (a bag is held out in only 1, or
2 for the duplicated bags), the same real bag gets an independent pi_b
estimate from each fold's own model. This script correlates those estimates
pairwise, purely as a post-hoc read of each fold's already-saved, already-
independently-fit ``bayes_diagnostics.json`` -- no retraining, no shared
parameters, no leakage (every fold's pi_b was fit on that fold's own
training bags only, exactly as before; this only compares the RESULTS).

This does not decide, on its own, which fold is "right" -- it reports
pairwise agreement; reading a consensus out of that (e.g. "3 folds agree
with each other, 2 don't") is an interpretive step documented alongside the
numbers, not asserted by the script.

Usage:
    python scripts/tier5_cross_fold_consistency.py --config exp_608
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mil.bayes import cross_fold_pi_b_consistency


def run(config_name, diagnostics_path=None, out_path=None):
    from src.utils.config_manager import ConfigManager
    config = ConfigManager.load_config(config_name)
    root = Path(config['paths'].get('experiment_root', f"results/{config_name}"))

    if diagnostics_path is None:
        diagnostics_path = root / 'latent' / 'bayes_diagnostics.json'
    diagnostics_path = Path(diagnostics_path)
    with open(diagnostics_path) as f:
        diagnostics_by_fold = json.load(f)

    pi_b_by_fold = {fold: info['pi_b_mean'] for fold, info in diagnostics_by_fold.items()}
    rhat_by_fold = {fold: info['chain_rhat'] for fold, info in diagnostics_by_fold.items()}

    report = cross_fold_pi_b_consistency(pi_b_by_fold)
    report['chain_rhat_by_fold'] = rhat_by_fold

    print(f"\n{'='*70}\nTIER 5 CROSS-FOLD pi_b CONSISTENCY: {config_name}\n{'='*70}")
    print(f"{'fold A':10s} {'fold B':10s} {'n_shared':9s} {'corr':8s} {'mean|diff|':10s}")
    for p in sorted(report['pairwise'], key=lambda p: p['corr']):
        print(f"{p['fold_a']:10s} {p['fold_b']:10s} {p['n_shared']:9d} "
              f"{p['corr']:+.4f}  {p['mean_abs_diff']:.4f}")

    print(f"\n{'fold':10s} {'R-hat':8s} {'mean corr vs others':20s}")
    for fold in sorted(report['fold_mean_corr']):
        print(f"{fold:10s} {rhat_by_fold.get(fold, float('nan')):8.4f} "
              f"{report['fold_mean_corr'][fold]:+.4f}")

    if out_path is None:
        out_path = root / 'latent' / 'bayes_cross_fold_consistency.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 5 cross-fold pi_b consistency check")
    parser.add_argument('--config', required=True)
    parser.add_argument('--diagnostics-path', default=None,
                        help="defaults to results/{config}/latent/bayes_diagnostics.json")
    parser.add_argument('--out-path', default=None)
    args = parser.parse_args()
    run(args.config, diagnostics_path=args.diagnostics_path, out_path=args.out_path)
