"""
Stage: MIL Tier 0 -- naive-MIL aggregation baselines (docs/MIL_REFRAME_PLAN.md §4).

Pools the per-instance posteriors the existing TRTR/TSTR/TSRTR classifiers
already emit (no new training) under four aggregation rules -- mean-posterior,
majority vote, log-sum-exp, top-q -- and reports genuine bag-level metrics for
each (model, rule) pair, for one regimen at a time.

This is the reference table every later MIL tier (mi-SVM, attention,
generative) must beat. Majority vote is also what
``pipeline/evaluation_stage.py``'s bag-level headline already uses (via
``bag_decisions``); this stage's distinct value is showing how sensitive that
naive reference is to the *choice* of pooling rule, across all four.

Cross-fold duplicate bags (4 bags appear in two CV folds each -- see
docs/MIL_REFRAME_PLAN.md §7.1) are handled by
``aggregate_bag_scores_across_folds``, never by pooling instances first and
losing the fold boundary.

Outputs (to ``config['paths']['statistical_dir']``):
  mil_tier0_{regimen}_{timestamp}.json   full per-(model, rule) metrics + CIs
  mil_tier0_{regimen}_{timestamp}.csv    flat comparison table
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Must precede the pipeline.*/src.* imports below: this file is documented to
# run standalone (`python pipeline/mil_eval_stage.py --config ...`), in which
# case Python puts only pipeline/ on sys.path, not the repo root, and the
# `pipeline.evaluation_stage` import fails before __main__ ever runs.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation_stage import MODEL_LABELS, MODEL_SPECS, _build_fold_val_sets
from src.utils.statistical_evaluation import (
    find_model_path, load_and_predict_proba,
    aggregate_bag_scores_across_folds, bag_level_bootstrap,
    summarize_bootstrap_diffs,
)

POOLING_RULES = ['mean', 'majority', 'lse', 'topq']
REFERENCE_RULE = 'majority'  # what the other three are compared against


def _fold_proba(folds, model_dir, prefix, ext, model_type):
    """Per-fold (proba, y_true, bag_ids) triples -- the structure
    aggregate_bag_scores_across_folds needs. Does NOT concatenate across folds
    (unlike evaluation_stage._gather_probas), so the fold boundary survives for
    the duplicate-bag averaging rule."""
    out = []
    for fd in folds:
        path = find_model_path(str(model_dir), prefix, ext, fd['fold_idx'])
        proba = load_and_predict_proba(path, model_type, fd['X'])
        out.append({'proba': proba, 'y_true': fd['y'], 'bag_ids': fd['slide_ids']})
    return out


def _metrics_for_rule(bag_true, scores, bag_pred_ref, n_boot, seed):
    """Bag-level point metrics for one pooling rule's decisions, plus a
    bootstrap comparison against the reference rule's decisions (same model,
    different aggregation -- so this is "does the pooling rule matter", not a
    baseline-vs-candidate model comparison)."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score, cohen_kappa_score,
    )
    bag_pred = scores.argmax(axis=1)
    classes = sorted(np.unique(bag_true).tolist())

    m = {
        'bag_accuracy': float(accuracy_score(bag_true, bag_pred)),
        'bag_balanced_accuracy': float(balanced_accuracy_score(bag_true, bag_pred)),
        'bag_f1_macro': float(f1_score(bag_true, bag_pred, average='macro',
                                       zero_division=0, labels=classes)),
        'bag_precision_macro': float(precision_score(bag_true, bag_pred, average='macro',
                                                      zero_division=0, labels=classes)),
        'bag_recall_macro': float(recall_score(bag_true, bag_pred, average='macro',
                                               zero_division=0, labels=classes)),
        'bag_kappa': float(cohen_kappa_score(bag_true, bag_pred))
                     if len(np.unique(bag_true)) > 1 else float('nan'),
    }
    _, _, diffs = bag_level_bootstrap(bag_pred_ref, bag_pred, bag_true, n_boot=n_boot, seed=seed)
    m.update({f'vs_{REFERENCE_RULE}_{k}': v for k, v in summarize_bootstrap_diffs(diffs).items()})
    return m, bag_pred


def mil_eval_stage(config, output_dir=None, regimen_dir=None, regimen_label='TRTR', n_boot=None):
    stats_cfg = config.get('statistics', {})
    n_boot = int(n_boot if n_boot is not None else stats_cfg.get('bootstrap_samples', 1000))
    seed = int(stats_cfg.get('random_seed', 42))

    regimen_dir = regimen_dir or {
        'TRTR': config['paths'].get('models_baseline_dir') or config['paths'].get('models_trtr_dir'),
        'TSTR': config['paths'].get('models_tstr_dir'),
        'TSRTR': config['paths'].get('models_tsrtr_dir'),
    }.get(regimen_label)
    if not regimen_dir:
        raise ValueError(f"no model dir resolved for regimen {regimen_label!r}; pass regimen_dir=")

    out_dir = Path(config['paths']['statistical_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}\nMIL TIER 0 STAGE: {regimen_label}\n{'='*80}")
    print(f"  models: {regimen_dir}")
    print(f"  rules:  {POOLING_RULES}  (reference: {REFERENCE_RULE})")

    folds = _build_fold_val_sets(config)
    n_val = sum(len(fd['y']) for fd in folds)
    print(f"  Validation spectra across {len(folds)} folds: {n_val}")

    results = []
    for prefix, ext, model_type in MODEL_SPECS:
        label = MODEL_LABELS[prefix]
        try:
            fold_proba = _fold_proba(folds, regimen_dir, prefix, ext, model_type)
        except FileNotFoundError as e:
            print(f"  [skip {label}] {e}")
            continue

        rule_scores = {}
        for rule in POOLING_RULES:
            bag_ids, scores, bag_true = aggregate_bag_scores_across_folds(fold_proba, rule=rule)
            rule_scores[rule] = scores
        n_bags = len(bag_ids)

        bag_pred_ref = rule_scores[REFERENCE_RULE].argmax(axis=1)
        row = {'model': label, 'n_bags': n_bags}
        print(f"  {label}:")
        for rule in POOLING_RULES:
            m, bag_pred = _metrics_for_rule(bag_true, rule_scores[rule], bag_pred_ref, n_boot, seed)
            for k, v in m.items():
                row[f'{rule}_{k}'] = v
            tag = ' (reference)' if rule == REFERENCE_RULE else ''
            print(f"    {rule:>9}: bag_acc={m['bag_accuracy']:.4f}  "
                  f"bag_F1={m['bag_f1_macro']:.4f}{tag}")
        results.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        'experiment': config['name'],
        'regimen_label': regimen_label,
        'regimen_dir': str(regimen_dir),
        'pooling_rules': POOLING_RULES,
        'reference_rule': REFERENCE_RULE,
        'n_bootstrap': n_boot,
        'n_val_spectra': int(n_val),
        'results': results,
    }
    json_path = out_dir / f"mil_tier0_{regimen_label}_{timestamp}.json"
    csv_path = out_dir / f"mil_tier0_{regimen_label}_{timestamp}.csv"
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"\n  MIL Tier 0 complete -> {json_path.name}, {csv_path.name}")
    return {'json_path': str(json_path), 'csv_path': str(csv_path), 'results': results}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Tier 0 naive-MIL pooling baselines")
    parser.add_argument('--config', required=True, help="experiment config name, e.g. exp_401")
    parser.add_argument('--regimen-label', default='TRTR', choices=['TRTR', 'TSTR', 'TSRTR'])
    parser.add_argument('--regimen-dir', default=None, help="override the model dir")
    parser.add_argument('--n-boot', type=int, default=None)
    args = parser.parse_args()

    from src.notebooks.experiment_utils import setup_experiment

    cfg, out_dir = setup_experiment(args.config)
    mil_eval_stage(cfg, out_dir, regimen_dir=args.regimen_dir,
                   regimen_label=args.regimen_label, n_boot=args.n_boot)
