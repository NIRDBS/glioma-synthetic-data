"""
Stage 6: bag-level classification accuracy for a stored MIL posterior, vs the
Tier 0 reference (docs/MIL_REFRAME_PLAN.md Stage 6 exit criterion: "Beats
Tier 0 at bag level").

Generic over (method, variant) -- works for Tier 4's 'cbr' variants and
Tier 1 ('misvm') alike, since it reads directly from the latent-label store
rather than any tier-specific code. This closes a real gap: docs/stage4_mil.md
§6 never computed a bag-accuracy number for mi-SVM -- its story stayed at
degeneracy diagnostics and Stage 5's spatial checks. This script is what
"beats Tier 0" actually needs.

The store's posterior is (n, K+1): column 0 is non-diagnostic mass (Stage 5's
own concern, mapped spatially), columns 1..K are the K diagnostic classes.
For BAG CLASSIFICATION specifically -- "which of the K classes does this bag
belong to" -- only the K diagnostic columns are used here, renormalised;
column 0 answers a different question (is this instance diagnostic at all)
and is deliberately not part of this comparison.

**This metric is CIRCULAR for a method that implements Model (C)'s hard
support restriction throughout scoring, not just fitting** (docs/
MIL_REFRAME_PLAN.md §1: each instance's state is restricted to
{null_1..null_M, Y_b} -- never another diagnostic class -- so it cannot,
architecturally, ever place mass on the "wrong" diagnostic column). Such a
posterior has *zero* instances with both diagnostic columns simultaneously
nonzero, giving a trivial, always-correct bag_accuracy of 1.0000 that
reflects the model's known-by-construction bag label, not anything it
learned (docs/stage6_generative_mil.md §5). mi-SVM (independent one-vs-rest scores,
no hard mask) and Tier 4's cbr variants (plain softmax, no notion of
"allowed states" at all) do NOT have this property (16.5% and ~100% of
instances respectively show real competition between the two diagnostic
columns) -- their numbers from this script are genuine. `run()` computes and
reports this fraction (`k_diagnostic_columns_competition_frac`) on every
call and prints a loud warning when it is nearly zero, rather than trusting
the caller to remember which methods are safe to evaluate this way.

Usage:
    python scripts/stage6_bag_eval.py --config exp_501 --method cbr --variant reconstruction
    python scripts/stage6_bag_eval.py --config exp_501 --method cbr --variant encoder_head
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline._common import num_classes_of
from pipeline.evaluation_stage import MODEL_LABELS, MODEL_SPECS, _binary_map, _build_fold_val_sets
from pipeline.mil_eval_stage import _fold_proba
from src.mil.store import read_posteriors
from src.utils.statistical_evaluation import (
    aggregate_bag_scores_across_folds, bag_level_bootstrap, summarize_bootstrap_diffs,
)


def diagnostic_column_competition_frac(fold_data):
    """
    See module docstring's circularity warning. Fraction of instances (across
    ALL folds' out-of-sample rows in ``fold_data``, each a dict with a
    'proba' (n_i, K) array -- the exact structure ``_store_fold_data``
    returns) where MORE THAN ONE diagnostic column is nonzero. Pure numpy,
    independent of the store/H5/reference-model machinery, so it's directly
    unit-testable against hand-built posteriors.

    Near 0.0: this method hard-restricts every instance to its own bag's
    known class throughout scoring (Model (C)) -- the
    bag_accuracy this script computes is then trivially 1.0 and
    UNINFORMATIVE, not a genuine result. Near 1.0 (mi-SVM, Tier 4's cbr
    variants): genuine per-instance competition between classes; the
    resulting bag_accuracy is a real measurement.
    """
    all_diag = np.concatenate([fd['proba'] for fd in fold_data], axis=0) if fold_data else np.empty((0, 0))
    if all_diag.size == 0:
        return float('nan')
    return float((all_diag > 1e-6).all(axis=1).mean())


def _load_y_bag_and_sample_ids(h5_path, config):
    """
    y_bag as stored is the raw 0-indexed 6-class label (docs/stage1_data_layer.md
    -- confirmed 0..5, not 1..6, in glioma_mil.h5), but every MIL formulation
    was fit under prepare_fold's binary mapping when num_classes==2 (y<3 -> 0,
    y>=3 -> 1) -- the SAME mapping evaluation_stage.py's own reference
    pipeline applies (_binary_map). Reusing that function here, rather than
    reimplementing it, is what keeps this script's labels comparable to the
    Tier 0 reference at all.
    """
    with h5py.File(h5_path, 'r') as f:
        sample_ids = f['sample_ids'][:]
        sample_ids = (np.char.decode(sample_ids.astype('S'), 'utf-8')
                     if sample_ids.dtype.kind != 'U' else sample_ids.astype(str))
        y_bag = f['y_bag'][:] if 'y_bag' in f else f['y_class'][:]
    y_bag = _binary_map(y_bag.copy(), num_classes_of(config))
    return sample_ids, y_bag


def _store_fold_data(store_path, method, variant, fold_names, sample_ids, y_bag):
    """
    Per-fold {'proba': (n_i, K), 'y_true': (n_i,), 'bag_ids': (n_i,)} for this
    fold's OUT-OF-SAMPLE rows only (in_sample=False -- scored by a model that
    never trained on them), in the exact structure
    aggregate_bag_scores_across_folds needs. The K diagnostic columns
    (posterior[:, 1:]) are renormalised after dropping column 0 (non-
    diagnostic mass -- a different question, see module docstring).
    """
    fold_data = []
    for fold_name in fold_names:
        data = read_posteriors(store_path, method, variant, fold_name)
        posterior, in_sample = data['posterior'], data['in_sample']
        oos = (~in_sample) & ~np.isnan(posterior).any(axis=1)
        diag = posterior[oos][:, 1:].astype(np.float64)
        row_sums = diag.sum(axis=1, keepdims=True)
        proba = diag / np.where(row_sums > 0, row_sums, 1.0)
        fold_data.append({
            'proba': proba.astype(np.float32),
            'y_true': y_bag[oos],
            'bag_ids': sample_ids[oos],
        })
    return fold_data


def _tier0_reference(config, n_boot, seed, regimen_label='TRTR', reference_model='cnn'):
    """Majority-vote Tier 0 pooling of one classifier's per-instance
    predictions (default: the 1D-CNN, the model this MIL programme is meant
    to improve on -- docs/MIL_REFRAME_PLAN.md §7.2) -- the same reference
    pipeline/mil_eval_stage.py itself compares its own pooling rules against."""
    folds = _build_fold_val_sets(config)
    regimen_dir = {
        'TRTR': config['paths'].get('models_baseline_dir') or config['paths'].get('models_trtr_dir'),
        'TSTR': config['paths'].get('models_tstr_dir'),
        'TSRTR': config['paths'].get('models_tsrtr_dir'),
    }[regimen_label]
    spec = next(s for s in MODEL_SPECS if s[0] == reference_model)
    fold_proba = _fold_proba(folds, regimen_dir, *spec)
    bag_ids, scores, bag_true = aggregate_bag_scores_across_folds(fold_proba, rule='majority')
    return bag_ids, scores.argmax(axis=1), bag_true, MODEL_LABELS[reference_model]


def run(config_name, method, variant, n_folds=5, n_boot=1000, seed=42,
       regimen_label='TRTR', reference_model='cnn', out_path=None):
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    from src.utils.config_manager import ConfigManager

    config = ConfigManager.load_config(config_name)
    fold_names = [f"fold_{i}" for i in range(n_folds)]
    store_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
        / 'latent' / 'latent_labels.h5'

    print(f"\n{'='*80}\nSTAGE 6 BAG-LEVEL EVAL: {method}/{variant}  ({config_name})\n{'='*80}")
    sample_ids, y_bag = _load_y_bag_and_sample_ids(config['paths']['processed_file'], config)
    fold_data = _store_fold_data(store_path, method, variant, fold_names, sample_ids, y_bag)
    n_oos = sum(len(fd['y_true']) for fd in fold_data)
    print(f"  out-of-sample instances across {n_folds} folds: {n_oos}")

    # Circularity guard (see module docstring): a method that hard-restricts
    # each instance to {nulls, own-bag-class} throughout scoring (Model (C))
    # can never place mass on the "wrong" diagnostic column, making
    # the argmax below trivially, always correct -- not a finding. Checked
    # directly on real out-of-sample data, not assumed from the method name.
    competition_frac = diagnostic_column_competition_frac(fold_data)
    print(f"  k_diagnostic_columns_competition_frac={competition_frac:.4f} "
          f"(fraction of instances where more than one diagnostic column is nonzero)")
    if competition_frac < 0.01:
        print(f"  [WARNING] {method}/{variant}: <1% of instances show any competition between "
              f"diagnostic columns -- this method likely hard-restricts each instance to its own "
              f"bag's class throughout scoring, making bag_accuracy below TRIVIALLY 1.0 and "
              f"UNINFORMATIVE, not a genuine result. See this script's module docstring.")

    bag_ids, scores, bag_true = aggregate_bag_scores_across_folds(fold_data, rule='mean')
    bag_pred = scores.argmax(axis=1)
    n_bags = len(bag_ids)
    classes = sorted(np.unique(bag_true).tolist())

    metrics = {
        'n_bags': n_bags,
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
    # Binary-only (K=2 throughout this project): sensitivity/specificity as
    # per-class recall (class 1 / class 0), same convention as
    # src/utils/statistical_evaluation.py's _slide_level_metrics, so these
    # are directly comparable to the Tier 0 classifiers' own numbers.
    # bag_auroc/bag_auprc from the CONTINUOUS pooled score (`scores`, already
    # computed above for the argmax) against class 1 -- never available from
    # bag_accuracy/kappa alone, hence computed here rather than derived.
    if len(classes) == 2:
        recall_per_class = recall_score(bag_true, bag_pred, average=None,
                                        zero_division=0, labels=classes)
        metrics['bag_sensitivity'] = float(recall_per_class[1])
        metrics['bag_specificity'] = float(recall_per_class[0])
        try:
            metrics['bag_auroc'] = float(roc_auc_score(bag_true, scores[:, 1]))
            metrics['bag_auprc'] = float(average_precision_score(bag_true, scores[:, 1]))
        except ValueError:
            pass
    print(f"  {method}/{variant}: bag_acc={metrics['bag_accuracy']:.4f}  "
          f"bag_F1={metrics['bag_f1_macro']:.4f}  bag_kappa={metrics['bag_kappa']:.4f}  "
          f"(n={n_bags} bags)")

    ref_bag_ids, ref_pred_all, ref_true_all, ref_label = _tier0_reference(
        config, n_boot, seed, regimen_label=regimen_label, reference_model=reference_model)
    # Align to the SAME bag set (store's out-of-sample coverage need not be
    # identical to the reference classifier's -- e.g. a slide with zero
    # covered instances, see docs/stage5_spatial_validation.md's HF-1887
    # history). Restrict the comparison to the intersection, not assumed.
    common = sorted(set(bag_ids.tolist()) & set(ref_bag_ids.tolist()))
    if len(common) < n_bags or len(common) < len(ref_bag_ids):
        print(f"  [note] comparing on {len(common)}/{n_bags} store bags "
              f"({len(ref_bag_ids)} reference bags) -- not identical coverage.")
    idx_store = {b: i for i, b in enumerate(bag_ids)}
    idx_ref = {b: i for i, b in enumerate(ref_bag_ids)}
    common_pred = np.array([bag_pred[idx_store[b]] for b in common])
    common_true = np.array([bag_true[idx_store[b]] for b in common])
    ref_true_common = np.array([ref_true_all[idx_ref[b]] for b in common])
    ref_pred_common = np.array([ref_pred_all[idx_ref[b]] for b in common])
    assert np.array_equal(common_true, ref_true_common), (
        "bag labels disagree between the store and the reference classifier for "
        "the same bag id -- investigate before trusting anything below")

    _, _, diffs = bag_level_bootstrap(ref_pred_common, common_pred, common_true,
                                      n_boot=n_boot, seed=seed)
    vs_tier0 = summarize_bootstrap_diffs(diffs)
    ref_acc = float(accuracy_score(common_true, ref_pred_common))
    print(f"  Tier 0 reference ({ref_label}, {regimen_label}, majority vote): "
          f"bag_acc={ref_acc:.4f}  (n={len(common)} bags)")
    print(f"  delta ({method}/{variant} - Tier0) = {metrics['bag_accuracy'] - ref_acc:+.4f}  "
          f"95%CI[{vs_tier0['ci_diff_lo']:+.4f},{vs_tier0['ci_diff_hi']:+.4f}]  "
          f"p={vs_tier0['p_bootstrap']:.4f}  significant={vs_tier0['significant_95']}")

    report = {
        'config': config_name, 'method': method, 'variant': variant,
        'n_bootstrap': n_boot, 'n_oos_instances': int(n_oos),
        'k_diagnostic_columns_competition_frac': competition_frac,
        'metrics': metrics,
        'tier0_reference': {
            'model': ref_label, 'regimen': regimen_label, 'rule': 'majority',
            'n_bags': len(common), 'bag_accuracy': ref_acc,
        },
        'vs_tier0': vs_tier0,
    }
    if out_path is None:
        out_path = Path(config['paths'].get('experiment_root', f"results/{config_name}")) \
            / 'mil_eval' / f'stage6_bag_eval_{method}_{variant}.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written -> {out_path}")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stage 6 bag-level eval of a stored MIL posterior")
    parser.add_argument('--config', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--regimen-label', default='TRTR', choices=['TRTR', 'TSTR', 'TSRTR'])
    parser.add_argument('--reference-model', default='cnn',
                        choices=[s[0] for s in MODEL_SPECS])
    parser.add_argument('--out-path', default=None,
                        help="defaults to results/{exp}/mil_eval/stage6_bag_eval_{method}_{variant}.json")
    args = parser.parse_args()
    run(args.config, args.method, args.variant, n_boot=args.n_boot, seed=args.seed,
       regimen_label=args.regimen_label, reference_model=args.reference_model,
       out_path=args.out_path)
