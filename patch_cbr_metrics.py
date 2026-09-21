"""
Patch CbR (Classification by Reconstruction) full metrics into the existing
statistical-evaluation JSONs.

Motivation
----------
The full stats_eval stage in run_suite recomputes the 4-model slide-level
bootstraps (1000 iterations each) and is slow. CbR, however, is just another
classifier: its F1 / precision / recall are cheap to add once we have its
per-spectrum predictions. The stored JSONs only kept ``cbr_accuracy`` + the
bootstrap CI, so the appendix tables show "--" for CbR's F1/Precision/Recall.

This script recomputes *only* the CbR predictions (one CVAE encode + C decodes
per fold — no 4-model bootstrap), derives the full metric set with the same
``compute_full_metrics`` helper the pipeline uses, and patches the metrics into
each experiment's latest stats-eval JSON **in place**. Re-run
``python paper/gen_appendix_tables.py`` afterwards to refresh appendix.tex.

Predictions come from ``evaluation_stage._cbr_predictions`` — the identical code
path the pipeline uses — so the numbers match what a full re-run would produce.
As a guard, the recomputed slide-level accuracy is checked against the
``cbr_accuracy`` already stored in the JSON.

Usage
-----
    python patch_cbr_metrics.py                      # patch all experiments
    python patch_cbr_metrics.py --experiments exp_201 exp_204
    python patch_cbr_metrics.py --dry-run            # compute + report, write nothing
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep this list in sync with paper/gen_appendix_tables.py:EXPERIMENTS
EXPERIMENTS = ['exp_201', 'exp_202', 'exp_203', 'exp_204', 'exp_205', 'exp_206',
               'exp_302', 'exp_303', 'exp_401', 'exp_402', 'exp_403']

RESULTS = PROJECT_ROOT / 'results'
# Metric keys copied from the CbR full_metrics dict into the top-level cbr dict
# (flat keys are what the appendix generator reads).
FLAT_METRIC_KEYS = ['f1_macro', 'precision_macro', 'recall_macro',
                    'balanced_accuracy', 'kappa']
ACC_TOL = 0.02  # slide-acc sanity-check tolerance (resampling-free, should be tiny)


def _latest_json(exp, candidate):
    """Latest statistical_evaluation_TRTR_vs_<candidate>_*.json path for an experiment."""
    pat = str(RESULTS / exp / 'statistical_eval'
              / f'statistical_evaluation_TRTR_vs_{candidate}_*.json')
    files = glob.glob(pat)
    if not files:
        return None
    files.sort(key=lambda f: re.search(r'(\d{8}_\d{6})', f).group(1))
    return files[-1]


def _patch_cbr_dict(cbr, full_metrics):
    """Write flat metric keys + a nested full_metrics block into a cbr dict."""
    for k in FLAT_METRIC_KEYS:
        if k in full_metrics:
            cbr[k] = full_metrics[k]
    cbr['full_metrics'] = full_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS,
                        help='Subset of experiments to patch (default: all).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report metrics but do not write JSONs.')
    args = parser.parse_args()

    # Heavy imports (TF) deferred until after arg parsing.
    from src.notebooks.experiment_utils import setup_experiment
    from pipeline.evaluation_stage import _build_fold_val_sets, _cbr_predictions
    from src.utils.statistical_evaluation import (
        stratified_slide_accuracy, compute_full_metrics,
    )

    # Cache: CVAE weights dir -> (full_metrics, recomputed_slide_acc). Shared
    # CVAEs (e.g. exp_401/402/403 reuse exp_201's) are computed only once.
    cache = {}
    summary = []

    for exp in args.experiments:
        json_paths = {c: _latest_json(exp, c) for c in ('TSTR', 'TSRTR')}
        if not any(json_paths.values()):
            summary.append(f'  {exp}: SKIP (no stats-eval JSON)')
            continue

        folds = None  # built lazily (TF/data load) only if this exp needs work
        n_features = None
        cfg = None

        for candidate, jpath in json_paths.items():
            if not jpath:
                continue
            with open(jpath, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            cbr = payload.get('cbr')
            if not cbr:
                summary.append(f'  {exp}/{candidate}: no cbr block')
                continue
            if 'f1_macro' in cbr and not args.dry_run:
                summary.append(f'  {exp}/{candidate}: already patched (f1={cbr["f1_macro"]:.3f})')
                continue

            weights_dir = payload.get('cvae_weights_dir')
            if not weights_dir:
                summary.append(f'  {exp}/{candidate}: no cvae_weights_dir in payload')
                continue

            if weights_dir in cache:
                full_metrics, recomputed_acc = cache[weights_dir]
            else:
                if folds is None:
                    cfg, _ = setup_experiment(exp)
                    folds = _build_fold_val_sets(cfg)
                    n_features = folds[0]['X'].shape[1]
                print(f"\n[{exp}/{candidate}] recomputing CbR predictions "
                      f"(weights: {weights_dir})")
                out = _cbr_predictions(folds, cfg, n_features, weights_dir)
                if out is None:
                    summary.append(f'  {exp}/{candidate}: CbR weights missing -> skipped')
                    continue
                preds, y_true, slide_ids = out
                full_metrics = compute_full_metrics(y_true, preds, slide_ids=slide_ids)
                recomputed_acc = stratified_slide_accuracy(preds, y_true, slide_ids)
                cache[weights_dir] = (full_metrics, recomputed_acc)

            stored_acc = cbr.get('cbr_accuracy')
            flag = ''
            if stored_acc is not None and abs(recomputed_acc - stored_acc) > ACC_TOL:
                flag = (f'  !! slide-acc mismatch: stored={stored_acc:.4f} '
                        f'recomputed={recomputed_acc:.4f}')

            summary.append(
                f'  {exp}/{candidate}: f1={full_metrics["f1_macro"]:.3f} '
                f'prec={full_metrics["precision_macro"]:.3f} '
                f'rec={full_metrics["recall_macro"]:.3f}'
                f'{" [DRY-RUN]" if args.dry_run else ""}{flag}')

            if args.dry_run:
                continue

            # Patch both the top-level cbr dict (read by the appendix generator)
            # and the matching entry in the results list (kept consistent).
            _patch_cbr_dict(cbr, full_metrics)
            for r in payload.get('results', []):
                if r.get('model') == 'Classification_by_Reconstruction':
                    _patch_cbr_dict(r, full_metrics)
            with open(jpath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

    print('\n=== CbR metric patch summary ===')
    print('\n'.join(summary))
    if not args.dry_run:
        print('\nNow refresh the tables:  python paper/gen_appendix_tables.py')


if __name__ == '__main__':
    main()
