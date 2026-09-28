"""
Stage 6 backlog: the shuffled-bag-label control (docs/MIL_REFRAME_PLAN.md
§6, backlog §13). Permutes which label goes with which bag (exact
permutation of the true label multiset -- src/mil/shuffle_control.py),
holds everything else fixed, refits a MIL formulation under the permuted
labels, and reports the same spatial statistics Stage 5 computes for the
real fit -- so the two can be compared directly. If lag>=3 coherence
survives this, the real result cannot be attributed to the true label.

Reuses pipeline/mil_fit_stage.py's per-fold fitting code UNCHANGED (the
permutation happens once, in the loader's own y_class array, before any
fold splitting -- everything downstream of that is identical to a real
fit). Writes to a SEPARATE store path so a shuffled run can never collide
with or corrupt the real latent-label store.

Usage:
    python scripts/stage6_shuffled_label_control.py --config exp_501 --method misvm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import make_loader, num_classes_of, set_seeds
from pipeline.mil_fit_stage import _fit_one_formulation
from src.mil.shuffle_control import apply_permuted_labels, permuted_bag_label_map


def run(config_name, method, pi_min=None, max_iters=10, svm_subsample_per_bag=500,
       seed=42, permutation_seed=123, store_suffix='shuffled'):
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    n_classes = num_classes_of(config)
    mil_cfg = config.get('mil', {})
    pi_min = pi_min if pi_min is not None else mil_cfg.get('pi_floor', 0.02)
    svm_subsample_per_bag = svm_subsample_per_bag or mil_cfg.get('svm_subsample_per_bag', 500)

    store_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / f'latent_labels_{store_suffix}.h5'

    print(f"\n{'='*80}\nSHUFFLED-LABEL CONTROL: {method}  ({config_name})\n{'='*80}")
    print(f"  permutation_seed={permutation_seed}  store -> {store_path}")

    loader = make_loader(config)

    label_map = permuted_bag_label_map(loader.groups, loader.y_class, seed=permutation_seed)
    n_moved = sum(1 for b in label_map if label_map[b] != loader.y_class[loader.groups == b][0])
    print(f"  {n_moved}/{len(label_map)} bags received a label different from their true one")

    # The one, surgical intervention: substitute the loader's own y_class in
    # place, BEFORE any fold splitting (__iter__ reads self.y_class fresh
    # per fold). Everything else -- features, bag membership, the CV split
    # itself -- is untouched.
    loader.y_class = apply_permuted_labels(loader.groups, label_map)

    seed = seed if seed is not None else set_seeds(config)
    _fit_one_formulation(
        method, loader, config, n_classes, store_path, pi_min, max_iters, seed,
        svm_subsample_per_bag=svm_subsample_per_bag,
    )
    print(f"\n  Shuffled-label control fit complete -> {store_path}")
    return {'store_path': str(store_path), 'label_map': label_map}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shuffled-bag-label control")
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True, choices=['misvm', 'em'])
    parser.add_argument('--pi-min', type=float, default=None)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--svm-subsample-per-bag', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--permutation-seed', type=int, default=123)
    args = parser.parse_args()
    run(args.config, args.method, pi_min=args.pi_min, max_iters=args.max_iters,
       svm_subsample_per_bag=args.svm_subsample_per_bag, seed=args.seed,
       permutation_seed=args.permutation_seed)
