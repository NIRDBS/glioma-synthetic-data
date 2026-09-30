"""
Stage 4 backlog: pi_b's external validity against NMF cluster proportions
(docs/MIL_REFRAME_PLAN.md §1: "regress inferred pi_b against independent
proxies... If inferred pi_b tracks an independent proxy, that is stronger
evidence than spatial coherence"). Never computed until now -- EM's pi_b was
previously discarded every run (pipeline/mil_fit_stage.py, fixed 2026-09-29
to persist results/{exp}/latent/em_diagnostics.json).

Per-bag pi_b (averaged across the ~4/5 folds where a bag is in TRAINING --
pi_b is a training-time quantity, not something EM produces for held-out
bags) vs. per-bag proportion of instances in each of the 3 NMF/GMM tissue
clusters (`tissue_cluster`, 0/1/2 -- src/data schema, no single cluster is
"the" diagnostic one by construction, so all 3 are correlated and reported,
not just whichever looks best). Also includes pi_b vs. viable_legacy
fraction as a second, complementary check to Stage 5's existing ARI/AMI-
based agreement (a direct correlation on the continuous quantities, rather
than a hard-threshold clustering-agreement metric).

Spearman, not Pearson: pi_b's distribution (mostly near pi_min after
shrinkage, docs/stage4_mil.md) is not remotely normal, and the question is
"does higher pi_b track higher cluster-proportion", a monotonic relationship
question, not a linear one.

Usage:
    python scripts/stage4_pi_b_external_validity.py --config exp_501
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


def _load_pi_b_per_bag(em_diagnostics_path):
    """
    Average a bag's pi_b across every fold where it appears (a bag is in
    TRAINING for ~4/5 folds under this project's CV schema -- pi_b is a
    training-time EM quantity, so a bag typically contributes 4 values, not
    5). Simple mean, not the store's leakage-aware out-of-sample averaging
    (src/mil/store.py) -- there is no "held-out" pi_b to prefer here, every
    contributing fold saw this bag in training by construction.
    """
    with open(em_diagnostics_path) as f:
        by_fold = json.load(f)
    sums, counts = {}, {}
    for fold_name, diag in by_fold.items():
        for bag_id, pi in diag['pi_b'].items():
            sums[bag_id] = sums.get(bag_id, 0.0) + pi
            counts[bag_id] = counts.get(bag_id, 0) + 1
    return {b: sums[b] / counts[b] for b in sums}, {b: counts[b] for b in sums}


def _load_raw_side_tables(h5_path):
    with h5py.File(h5_path, 'r') as f:
        sample_ids = f['sample_ids'][:]
        sample_ids = (np.char.decode(sample_ids.astype('S'), 'utf-8')
                     if sample_ids.dtype.kind != 'U' else sample_ids.astype(str))
        tissue_cluster = f['tissue_cluster'][:]
        viable_legacy = f['viable_legacy'][:]
    return sample_ids, tissue_cluster, viable_legacy


def run(config_name):
    from scipy.stats import spearmanr

    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    em_diagnostics_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / 'em_diagnostics.json'
    h5_path = config['paths']['processed_file']

    print(f"\n{'='*80}\nSTAGE 4: pi_b EXTERNAL VALIDITY vs NMF cluster proportions "
          f"({config_name})\n{'='*80}")
    pi_b, fold_counts = _load_pi_b_per_bag(em_diagnostics_path)
    print(f"  {len(pi_b)} bags with a pi_b estimate "
          f"(avg over {np.mean(list(fold_counts.values())):.2f} folds/bag)")

    sample_ids, tissue_cluster, viable_legacy = _load_raw_side_tables(h5_path)
    bags = sorted(pi_b.keys())
    n_clusters = int(tissue_cluster.max()) + 1

    cluster_props = {b: np.zeros(n_clusters) for b in bags}
    viable_frac = {}
    for b in bags:
        idx = sample_ids == b
        n_b = idx.sum()
        if n_b == 0:
            continue
        for c in range(n_clusters):
            cluster_props[b][c] = (tissue_cluster[idx] == c).mean()
        viable_frac[b] = float(viable_legacy[idx].mean())

    pi_b_arr = np.array([pi_b[b] for b in bags])
    report = {'config': config_name, 'n_bags': len(bags), 'correlations': {}}

    print(f"\n  pi_b range: [{pi_b_arr.min():.4f}, {pi_b_arr.max():.4f}]  "
          f"median={np.median(pi_b_arr):.4f}  std={pi_b_arr.std():.4f}")

    for c in range(n_clusters):
        prop_arr = np.array([cluster_props[b][c] for b in bags])
        rho, p = spearmanr(pi_b_arr, prop_arr)
        print(f"  pi_b vs tissue_cluster={c} proportion: "
              f"spearman_rho={rho:+.4f}  p={p:.4f}  "
              f"(cluster proportion range [{prop_arr.min():.3f},{prop_arr.max():.3f}])")
        report['correlations'][f'tissue_cluster_{c}'] = {'spearman_rho': float(rho), 'p': float(p)}

    viable_arr = np.array([viable_frac[b] for b in bags])
    rho, p = spearmanr(pi_b_arr, viable_arr)
    print(f"  pi_b vs viable_legacy fraction: spearman_rho={rho:+.4f}  p={p:.4f}")
    report['correlations']['viable_legacy_fraction'] = {'spearman_rho': float(rho), 'p': float(p)}

    any_significant = any(v['p'] < 0.05 for v in report['correlations'].values())
    print(f"\n  any correlation significant at p<0.05: {any_significant}")
    if not any_significant:
        print(f"  [note] no external proxy tracks pi_b at conventional significance -- "
              f"consistent with Stage 5's existing ARI/AMI-based null result "
              f"(docs/stage5_spatial_validation.md), via a different (continuous, "
              f"not clustering-agreement) statistic.")

    out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'mil_eval' / 'stage4_pi_b_external_validity.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pi_b vs NMF cluster-proportion external validity")
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    run(args.config)
