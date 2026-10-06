"""
Tier 5 (Bayesian MIL, Gibbs) bag-level accuracy PER FOLD, not pooled --
parallels the Tier 0 per-fold table in docs/tier5_bayesian_mil.md section 2.1.

Reuses scripts/stage6_bag_eval.py's own data-loading and circularity-check
functions unchanged (same store read, same binary-label mapping, same
diagnostic_column_competition_frac check) -- only the aggregation step
differs: this calls aggregate_bag_scores_across_folds on ONE fold's data at
a time instead of pooling all 5.

**This means a bag that is a test bag in >1 fold (docs/mil_tiers_methodology.md
section 2's "cross-fold duplicate bags" -- 4 of them here, forced by
stratification, not averaged into one combined decision the way the
properly pooled n=59 evaluation does -- that number is
`results/{exp}/mil_eval/stage6_bag_eval_{method}_{variant}.json`, NOT this
script.** This script prints which bags those are and whether the folds
that share one actually disagree on it (confirmed on exp_608: they don't --
all 4 score bit-identically in fold_0 and fold_4, so merging them would not
change anything reported here), so the gap is visible rather than silent,
but it is a real, deliberate difference from the project's standard
protocol, not an oversight to read past.

**Read this next to docs/tier5_bayesian_mil.md section 4 before trusting any
number here.** Model (C) is enforced at scoring (src/mil/bayes.py::score_bayes_mil),
so every instance's posterior can only ever place diagnostic mass on its own
bag's already-known class -- confirmed per fold below via the same
competition-fraction check stage6_bag_eval.py runs pooled. Expect it to stay
near 0.0000 in every fold, which makes that fold's bag_accuracy trivially
~1.0 and uninformative, not a measurement of anything learned.

Usage:
    python scripts/tier5_per_fold_accuracy.py --config exp_608
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

import stage6_bag_eval as s6
from src.utils.statistical_evaluation import aggregate_bag_scores_across_folds


def run(config_name, method='bayes_mil', variant='gibbs', n_folds=5):
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    fold_names = [f"fold_{i}" for i in range(n_folds)]
    store_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / 'latent_labels.h5'

    sample_ids, y_bag = s6._load_y_bag_and_sample_ids(config['paths']['processed_file'], config)
    fold_data = s6._store_fold_data(store_path, method, variant, fold_names, sample_ids, y_bag)

    print(f"\n{'='*80}\nTIER 5 ({method}/{variant}) BAG ACCURACY PER FOLD: {config_name}\n{'='*80}")
    print(f"{'fold':10s} {'n_bags':8s} {'n_oos_inst':11s} {'competition_frac':17s} {'bag_acc':8s}")

    rows = {}
    for fold_name, fd in zip(fold_names, fold_data):
        if len(fd['y_true']) == 0:
            print(f"{fold_name:10s}  (no out-of-sample instances)")
            continue
        comp_frac = s6.diagnostic_column_competition_frac([fd])
        bag_ids, scores, bag_true = aggregate_bag_scores_across_folds([fd], rule='mean')
        bag_pred = scores.argmax(axis=1)
        bag_acc = float((bag_pred == bag_true).mean())
        n_bags = len(bag_ids)
        rows[fold_name] = {
            'n_bags': n_bags, 'n_oos_instances': len(fd['y_true']),
            'competition_frac': comp_frac, 'bag_acc': bag_acc,
        }
        print(f"{fold_name:10s} {n_bags:8d} {len(fd['y_true']):11d} "
              f"{comp_frac:17.4f} {bag_acc:8.4f}")

    accs = [r['bag_acc'] for r in rows.values()]
    comp = [r['competition_frac'] for r in rows.values()]
    print(f"\nmean bag_acc across folds: {np.mean(accs):.4f}  "
          f"(mean competition_frac: {np.mean(comp):.4f} -- near 0 means every "
          f"fold's number above is circular, see this script's module docstring)")

    # Cross-fold duplicate bags: not merged above (see module docstring) --
    # reported here so a reader can see whether that omission could matter.
    from collections import defaultdict
    bag_to_folds = defaultdict(list)
    bag_to_score = {}
    for fold_name, fd in zip(fold_names, fold_data):
        if len(fd['y_true']) == 0:
            continue
        bag_ids, scores, _ = aggregate_bag_scores_across_folds([fd], rule='mean')
        for bid, sc in zip(bag_ids, scores):
            bag_to_folds[bid].append(fold_name)
            bag_to_score.setdefault(bid, {})[fold_name] = sc

    dupes = {b: f for b, f in bag_to_folds.items() if len(f) > 1}
    if dupes:
        print(f"\n{len(dupes)} bag(s) are test bags in >1 fold (not merged above):")
        for bag, folds in dupes.items():
            scores_by_fold = bag_to_score[bag]
            agree = len({tuple(np.round(s, 6)) for s in scores_by_fold.values()}) == 1
            print(f"  {bag:28s} folds={folds}  "
                  f"{'AGREE' if agree else 'DISAGREE -- per-fold table above double-counts this inconsistently'}: "
                  + ", ".join(f"{f}={scores_by_fold[f]}" for f in folds))
    else:
        print("\nNo cross-fold duplicate bags in this dataset/config.")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tier 5 per-fold bag accuracy")
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', default='bayes_mil')
    parser.add_argument('--variant', default='gibbs')
    args = parser.parse_args()
    run(args.config, method=args.method, variant=args.variant)
