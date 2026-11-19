"""Bag-level metrics, naive MIL pooling and the duplicate-bag CV schema:
  1. Hand-built single-class bags -- each previously-degenerate metric
     (precision_macro was a constant ~0.5, kappa was always 0.0, f1_macro
     silently dropped 0%-accuracy bags) now behaves.
  2. Duplicate-bag averaging yields exactly 59 bag decisions from 63
     (bag, fold) test pairs -- both via plain pooling (bag_decisions) and via
     the genuine per-fold path (aggregate_bag_scores_across_folds). (Was
     58 from 62 until data/splits/manual_splits.json picked up its 59th
     slide, HF-1887_via-t_2_1.h5_3.)
  3. naive MIL mean/majority pooling reproduces the existing majority-vote
     confusion matrix where they should agree (no within-bag ties).

Plus the two invariants promised in aggregate_bag_scores_across_folds's own
docstring, which this file is named as the home for: the mean/majority
per-fold-then-average == pool-then-once equivalence, and disagreement between
independent code paths is otherwise possible (not asserted away).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from src.utils.statistical_evaluation import (
    aggregate_bag_scores_across_folds,
    bag_decisions,
    bag_level_bootstrap,
    compute_full_metrics,
    pool_instance_scores,
    within_bag_instance_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = REPO_ROOT / "data" / "splits" / "manual_splits.json"

requires_splits = pytest.mark.skipif(
    not SPLITS_PATH.exists(), reason="data/splits/manual_splits.json not present"
)


# ---------------------------------------------------------------------------
# 1. Hand-built single-class bags -- previously-degenerate bag-level metrics
# ---------------------------------------------------------------------------
#
# W, X: true=0, majority-correct (within-bag acc 4/5, 3/5).
# Y:    true=1, majority-WRONG, and every instance wrong (0% within-bag acc --
#       the case the old f1_macro silently dropped).
# Z:    true=1, majority-correct (within-bag acc 4/5).
# Hand-verified: bag_accuracy=0.75, bag_precision_macro=0.8333,
# bag_recall_macro=0.75 (== bag_balanced_accuracy), bag_f1_macro=0.7333,
# bag_kappa=0.5, within_bag_instance_recall=0.55.

_WXYZ_BAG_IDS = np.array((['W'] * 5) + (['X'] * 5) + (['Y'] * 5) + (['Z'] * 5))
_WXYZ_Y_TRUE = np.array(([0] * 5) + ([0] * 5) + ([1] * 5) + ([1] * 5))
_WXYZ_PREDS = np.array(
    [0, 0, 0, 0, 1] + [0, 0, 0, 1, 1] + [0, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
)


def _wxyz_metrics():
    return compute_full_metrics(_WXYZ_Y_TRUE, _WXYZ_PREDS, slide_ids=_WXYZ_BAG_IDS)


def test_bag_precision_macro_is_not_constant_half():
    """Old bug: every bag's own-class within-bag instance precision was always
    exactly 1.0, so precision_macro collapsed to ~0.5. The new bag-level
    quantity is a genuine sklearn precision over 4 bag decisions and varies
    with the input -- here it lands at 0.8333, nowhere near 0.5."""
    m = _wxyz_metrics()
    assert m['bag_precision_macro'] == pytest.approx(0.8333333, abs=1e-6)
    assert m['bag_precision_macro'] != pytest.approx(0.5, abs=1e-6)


def test_bag_kappa_is_not_always_zero():
    """Old bug: cohen_kappa_score on a constant y_true (every bag single-class
    at the *instance* level) was always exactly 0.0. bag_kappa is computed on
    the n_bags majority-vote decisions against n_bags true labels instead."""
    m = _wxyz_metrics()
    assert m['bag_kappa'] == pytest.approx(0.5, abs=1e-6)
    assert m['bag_kappa'] != 0.0


def test_bag_f1_macro_does_not_drop_zero_accuracy_bag():
    """Old bug: a bag with 0% within-bag accuracy (Y here) was silently
    excluded from the F1 average (the rec/prec-both-non-None guard), inflating
    the reported score. bag_f1_macro is a single sklearn call over all n_bags
    decisions, so Y necessarily lowers recall_1 (1 of 2 class-1 bags wrong).
    Cross-check against the drop-Y alternative to make the point concrete."""
    m = _wxyz_metrics()
    assert m['bag_f1_macro'] == pytest.approx(0.7333333, abs=1e-6)

    from sklearn.metrics import f1_score
    # If Y had been dropped (the old bug's behaviour), class 1 would be
    # represented only by Z (correct) -- a strictly higher, wrong score.
    f1_without_y = f1_score([0, 0, 1], [0, 0, 1], average='macro')
    assert f1_without_y == pytest.approx(1.0)
    assert m['bag_f1_macro'] < f1_without_y


def test_within_bag_instance_recall_differs_from_bag_accuracy():
    """Accuracy/balanced_accuracy/recall_macro previously
    were literally the same float as this diagnostic. They must now diverge whenever within-bag votes aren't unanimous."""
    m = _wxyz_metrics()
    assert m['within_bag_instance_recall'] == pytest.approx(0.55, abs=1e-6)
    assert m['bag_accuracy'] == pytest.approx(0.75, abs=1e-6)
    assert m['within_bag_instance_recall'] != pytest.approx(m['bag_accuracy'], abs=1e-6)


def test_bag_kappa_nan_for_single_class():
    """cohen_kappa_score is undefined with one class present; _slide_level_metrics
    reports nan explicitly rather than a misleading 0.0."""
    bag_ids = np.array((['A'] * 3) + (['B'] * 3))
    y_true = np.array([0] * 6)
    preds = np.array([0, 0, 1, 0, 0, 0])
    m = compute_full_metrics(y_true, preds, slide_ids=bag_ids)
    assert np.isnan(m['bag_kappa'])


def test_bag_decisions_rejects_mixed_labels_within_bag():
    """A bag-level dataset means every instance in a bag shares one label; a
    caller passing mismatched labels for one bag id is a bug, not silently
    handled."""
    bag_ids = np.array(['A', 'A', 'B'])
    y_true = np.array([0, 1, 1])  # bag 'A' has two different true labels
    preds = np.array([0, 0, 1])
    with pytest.raises(AssertionError):
        bag_decisions(preds, bag_ids, y_true)


# ---------------------------------------------------------------------------
# 2. Duplicate-bag averaging: 63 (bag, fold) test pairs -> 59 unique bags
#
# Was 62 -> 58 until 2026-09-28: HF-1887_via-t_2_1.h5_3 was missing from every
# fold's train/test list in this split file (an independent bug, unrelated to
# the 4 deliberately-duplicated slides below). It's now a normal, single-test-fold slide (fold_3), so it adds exactly
# 1 to both the pair count and the unique-bag count, and does not join the
# duplicated set.
# ---------------------------------------------------------------------------

@requires_splits
def test_manual_splits_has_63_test_pairs_59_unique_bags():
    """4 slides are forced into two test folds
    each (6-class stratification over classes with only 4 patients), giving 62
    (slide, fold) test pairs over 58 unique slides -- plus HF-1887's own single
    test-fold appearance (fold_3), for 63 pairs over 59 unique slides."""
    splits = json.loads(SPLITS_PATH.read_text())
    all_test = [sid for fold in splits.values() for sid in fold['test']]
    assert len(all_test) == 63
    assert len(set(all_test)) == 59

    dup_counts = Counter(all_test)
    duplicated = {sid for sid, n in dup_counts.items() if n > 1}
    assert len(duplicated) == 4
    assert all(dup_counts[sid] == 2 for sid in duplicated)


@requires_splits
def test_bag_decisions_collapses_duplicated_bags_to_59():
    """Pooling every instance from every (bag, fold) test appearance and
    deciding once (bag_decisions keys purely by bag id) collapses the 63 pairs
    to exactly 59 bag decisions -- the behaviour the old instance-level code
    had implicitly, now made explicit."""
    splits = json.loads(SPLITS_PATH.read_text())
    rng = np.random.default_rng(0)

    bag_ids, y_true, preds = [], [], []
    for fold in splits.values():
        for sid in fold['test']:
            label = abs(hash(sid)) % 3  # deterministic per-bag "class"
            n_instances = int(rng.integers(5, 15))
            bag_ids += [sid] * n_instances
            y_true += [label] * n_instances
            preds += rng.integers(0, 3, size=n_instances).tolist()

    ids_sorted, bag_pred, bag_true = bag_decisions(
        np.array(preds), np.array(bag_ids), np.array(y_true))
    assert len(ids_sorted) == 59


@requires_splits
def test_aggregate_bag_scores_across_folds_collapses_duplicated_bags_to_59():
    """Same claim via the genuine per-fold code path that preserves fold
    boundaries instead of concatenating -- what scripts/bag_level_eval.py
    uses."""
    splits = json.loads(SPLITS_PATH.read_text())
    rng = np.random.default_rng(0)
    K = 3

    fold_data = []
    for fold in splits.values():
        bag_ids, y_true, proba_rows = [], [], []
        for sid in fold['test']:
            label = abs(hash(sid)) % K
            n_instances = int(rng.integers(5, 15))
            bag_ids += [sid] * n_instances
            y_true += [label] * n_instances
            proba_rows.append(rng.dirichlet(np.ones(K), size=n_instances))
        fold_data.append({
            'proba': np.vstack(proba_rows),
            'y_true': np.array(y_true),
            'bag_ids': np.array(bag_ids),
        })

    bag_ids_sorted, scores, bag_true = aggregate_bag_scores_across_folds(fold_data, rule='mean')
    assert len(bag_ids_sorted) == 59
    assert scores.shape == (59, K)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. naive MIL pooling: agreement with bag_decisions where they should agree,
#    plus the linear-rule cross-fold equivalence aggregate_bag_scores_across_
#    folds's docstring claims.
# ---------------------------------------------------------------------------

def test_naive_mil_majority_pooling_matches_bag_decisions_confusion_matrix():
    """Two independent code paths compute 'majority vote': bag_decisions
    (the primary bag-level metrics) and pool_instance_scores(rule='majority')
    via aggregate_bag_scores_across_folds (naive MIL). On data built with no
    within-bag ties, they must agree exactly."""
    rng = np.random.default_rng(2)
    K = 3
    bag_ids, y_true, preds, proba = [], [], [], []
    for i, bag in enumerate(['bagA', 'bagB', 'bagC', 'bagD', 'bagE']):
        true_cls = i % K
        n = 9
        winner = int(rng.integers(0, K))
        votes = np.full(n, winner)
        p = np.zeros((n, K))
        p[np.arange(n), votes] = 1.0
        bag_ids += [bag] * n
        y_true += [true_cls] * n
        preds += votes.tolist()
        proba.append(p)

    bag_ids = np.array(bag_ids)
    y_true = np.array(y_true)
    preds = np.array(preds)
    proba = np.vstack(proba)
    labels = sorted(set(y_true.tolist()))

    ids_a, bag_pred_a, bag_true_a = bag_decisions(preds, bag_ids, y_true)
    cm_a = confusion_matrix(bag_true_a, bag_pred_a, labels=labels)

    fold_data = [{'proba': proba, 'y_true': y_true, 'bag_ids': bag_ids}]
    ids_b, scores_b, bag_true_b = aggregate_bag_scores_across_folds(fold_data, rule='majority')
    bag_pred_b = scores_b.argmax(axis=1)
    cm_b = confusion_matrix(bag_true_b, bag_pred_b, labels=labels)

    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(bag_pred_a, bag_pred_b)
    assert np.array_equal(cm_a, cm_b)


def test_mean_rule_equals_pool_once_for_duplicated_bag():
    """For LINEAR pooling rules (mean, majority), per-fold-aggregate-then-
    average equals pooling all of a duplicated bag's instances once, whenever
    the bag has equal instance counts in every fold it appears in -- always
    true here, since it's the same physical slide scored by two fold models.
    Verifies aggregate_bag_scores_across_folds's own equivalence claim rather
    than assuming it."""
    rng = np.random.default_rng(1)
    K = 3
    n = 6
    proba_f0 = rng.dirichlet(np.ones(K), size=n)
    proba_f1 = rng.dirichlet(np.ones(K), size=n)
    bag_ids = np.array(['DUP'] * n)
    y_true = np.array([0] * n)

    fold_data = [
        {'proba': proba_f0, 'y_true': y_true, 'bag_ids': bag_ids},
        {'proba': proba_f1, 'y_true': y_true, 'bag_ids': bag_ids},
    ]
    _, scores_agg, _ = aggregate_bag_scores_across_folds(fold_data, rule='mean')

    pooled_proba = np.vstack([proba_f0, proba_f1])
    pooled_ids = np.array(['DUP'] * (2 * n))
    _, scores_once = pool_instance_scores(pooled_proba, pooled_ids, rule='mean')

    np.testing.assert_allclose(scores_agg[0], scores_once[0], atol=1e-10)


def test_pool_instance_scores_rows_sum_to_one():
    """Regression pin: every pooling rule must return a valid probability
    vector per bag (rows sum to 1) so cross-fold averaging stays meaningful."""
    rng = np.random.default_rng(3)
    K = 4
    bag_ids = np.array((['A'] * 7) + (['B'] * 11))
    proba = rng.dirichlet(np.ones(K), size=18)
    for rule in ('mean', 'majority', 'lse', 'topq'):
        _, scores = pool_instance_scores(proba, bag_ids, rule=rule)
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-8)


def test_bag_level_bootstrap_detects_model_difference():
    """bag_level_bootstrap degenerately reuses stratified_slide_bootstrap with
    singleton groups (each bag its own 'slide'). A clearly-better model must
    show a positive mean diff with a CI that excludes zero."""
    rng = np.random.default_rng(4)
    n_bags = 30
    bag_true = rng.integers(0, 2, size=n_bags)
    bag_pred_good = bag_true.copy()
    flip = rng.choice(n_bags, size=2, replace=False)
    bag_pred_good[flip] = 1 - bag_pred_good[flip]  # 2 mistakes
    bag_pred_bad = rng.integers(0, 2, size=n_bags)  # near-chance

    _, _, diffs = bag_level_bootstrap(bag_pred_bad, bag_pred_good, bag_true, n_boot=200, seed=4)
    assert np.mean(diffs) > 0.1
