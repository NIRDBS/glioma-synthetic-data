"""
Statistical Evaluation Utilities for TRTR vs TSTR Benchmark

This module provides functions for:
1. Model loading and prediction extraction (sklearn, XGBoost, Keras)
2. Slide-level stratified bootstrap for statistical comparison
3. Classification by Reconstruction (CbR) via CVAE encoder/decoder
4. Confidence interval computation with no data leakage

Key Design Principles:
- Slide-level (not spectrum-level) resampling
- No data leakage: each slide predicted only by fold where they were in validation set
- Fast bootstrap: pre-computes predictions once, resamples slides across iterations
- CVAE integration: supports both standard bootstrap and CbR-specific bootstrap
"""

import os
import numpy as np
import joblib
import xgboost as xgb
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, cohen_kappa_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

from src.models.cvae import CVAE

# Modal's CNN training image (pipeline/modal_classifiers.py) pins tensorflow but
# not keras, so a CNN trained there can be saved by a newer Keras than this
# local environment's -- newer Keras serializes GlorotUniform with extra
# input_axes/output_axes kwargs this version's __init__ doesn't accept, which
# otherwise breaks loading ANY such .keras file here (confirmed on
# Modal-trained CNNs; a per-call custom_objects override does NOT
# intercept this, since GlorotUniform is a built-in, not a custom, name --
# only patching the class itself works). Harmless for files saved without
# those kwargs (old-format files keep working identically).
_orig_glorot_init = keras.initializers.GlorotUniform.__init__
def _tolerant_glorot_init(self, seed=None, **kwargs):
    _orig_glorot_init(self, seed=seed)
keras.initializers.GlorotUniform.__init__ = _tolerant_glorot_init


# ============================================================================
# SECTION 1: Helper Functions (Model Loading & Prediction Extraction)
# ============================================================================

def find_model_path(base_dir: str, prefix: str, ext: str, fold_idx: int) -> str:
    """
    Find model file for a specific fold by checking multiple naming conventions.
    
    Args:
        base_dir: Directory containing models
        prefix: Model filename prefix (e.g., 'apollo', 'rf', 'xgb', 'cnn')
        ext: File extension (e.g., '.joblib', '.json', '.keras')
        fold_idx: Cross-validation fold index
    
    Returns:
        str: Full path to model file
    
    Raises:
        FileNotFoundError: If model not found under any naming convention
    """
    candidates = [
        f"{base_dir}/{prefix}_fold_{fold_idx}{ext}",
        f"{base_dir}/{prefix}_fold{fold_idx}{ext}",
        f"{base_dir}/{prefix}_{fold_idx}{ext}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model '{prefix}' fold {fold_idx} not found in {base_dir}")


def load_and_predict(path: str, model_type: str, X: np.ndarray) -> np.ndarray:
    """
    Load a trained model and return class label predictions.
    
    Supports sklearn, XGBoost, and Keras model types with appropriate
    serialization formats (.joblib, .json, .keras respectively).
    
    Args:
        path: Full path to model file
        model_type: One of 'sklearn', 'xgboost', 'keras'
        X: Input features array (n_samples, n_features)
    
    Returns:
        np.ndarray: (n_samples,) array of integer class labels
    
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "sklearn":
        model = joblib.load(path)
        return model.predict(X).astype(int)
    
    elif model_type == "xgboost":
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model.predict(X).astype(int)
    
    elif model_type == "keras":
        X_3d = X.reshape(-1, X.shape[1], 1)
        model = tf.keras.models.load_model(path, compile=False)
        preds = np.argmax(model.predict(X_3d, verbose=0), axis=1)
        tf.keras.backend.clear_session()
        return preds.astype(int)
    
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


def load_and_predict_proba(path: str, model_type: str, X: np.ndarray) -> np.ndarray:
    """
    Load a trained model and return class probability predictions.

    Returns:
        np.ndarray: (n_samples, n_classes) probability array, or None if the
                    model type does not support predict_proba (shouldn't happen
                    for our four model types).
    """
    if model_type == "sklearn":
        model = joblib.load(path)
        return model.predict_proba(X).astype(np.float32)

    elif model_type == "xgboost":
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model.predict_proba(X).astype(np.float32)

    elif model_type == "keras":
        X_3d = X.reshape(-1, X.shape[1], 1)
        try:
            model = tf.keras.models.load_model(path, compile=False)
            proba = model.predict(X_3d, verbose=0).astype(np.float32)
            tf.keras.backend.clear_session()
            return proba
        except (TypeError, ValueError) as e:
            if "renorm" in str(e) or "BatchNormalization" in str(e):
                raise FileNotFoundError(f"Keras model incompatible: {str(e)[:100]}")
            raise

    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


def get_fold_specific_predictions(preds_by_fold, patient_ids, patient_to_fold):
    """
    Extract fold-specific predictions with no data leakage.
    
    For each patient, uses ONLY the predictions from the fold where that patient
    was in the validation set (i.e., the fold whose model was trained WITHOUT that patient).
    
    This ensures honest evaluation: no patient is evaluated by a model that trained on them.
    
    Args:
        preds_by_fold: (n_folds, n_samples) array of predictions from each fold's model
        patient_ids: (n_samples,) array of patient IDs
        patient_to_fold: dict mapping patient_id -> fold_idx (patient's validation fold)
    
    Returns:
        np.ndarray: (n_samples,) array where preds[i] comes from the model trained on
                    patient_ids[i]'s validation fold (not their training folds)
    """
    preds = np.zeros(len(patient_ids), dtype=int)
    
    for i, patient_id in enumerate(patient_ids):
        fold_j = patient_to_fold[patient_id]
        preds[i] = preds_by_fold[fold_j][i]
    
    return preds


def within_bag_instance_recall(preds, y_true, bag_ids):
    """
    Mean of per-(bag, class) instance accuracies — E_bag[fraction of a bag's
    instances predicted as the bag's own label]. Exactly the statistic
    ``within_bag_instance_recall_bootstrap`` accumulates each iteration.

    **This is a diagnostic, not a bag-level accuracy.** It answers "how much of
    a bag's spectra get the bag's own label under the naive assumption that
    every instance in a bag carries that bag's label" — which is precisely the
    assumption the MIL reframing exists to retire (docs/MIL_REFRAME_PLAN.md §7).
    No instance-level ground truth exists in this dataset, so this number is not
    validated against anything; it is a property of the naive pooling, kept for
    continuity with pre-reframe results. For the actual clinical claim (one
    decision per bag), use ``bag_decisions`` / ``_slide_level_metrics``'s
    ``bag_*`` keys.

    Bags contribute equally (not weighted by spectrum count); within that, class
    contributes equally (flattened over (class, bag) pairs — see
    ``_slide_level_metrics`` for why this is a bag-macro, not a class-macro).
    """
    unique_classes = np.unique(y_true)

    bag_to_indices = {}
    for idx, bid in enumerate(bag_ids):
        bag_to_indices.setdefault(bid, []).append(idx)
    for bid in bag_to_indices:
        bag_to_indices[bid] = np.array(bag_to_indices[bid])

    class_to_bags = {
        cls: np.unique(bag_ids[y_true == cls]) for cls in unique_classes
    }

    accs = []
    for cls in unique_classes:
        for bag in class_to_bags[cls]:
            idx = bag_to_indices[bag]
            cls_idx = idx[y_true[idx] == cls]
            if len(cls_idx) > 0:
                accs.append(accuracy_score(y_true[cls_idx], preds[cls_idx]))
    return float(np.mean(accs)) if accs else 0.0


# Deprecated name -- kept only because notebooks/09_Statistical_evaluation.ipynb
# (exploratory, not a build dependency) still imports it. New code must use
# within_bag_instance_recall, which is the same computation under an honest name.
stratified_slide_accuracy = within_bag_instance_recall


def bag_decisions(preds, bag_ids, y_true=None, rule='majority'):
    """
    Collapse per-instance predictions to one hard decision per bag.

    ``rule='majority'``: each bag's decision is the mode of its instances'
    predicted labels (ties broken by ``np.bincount().argmax()``, i.e. lowest
    label index). This is Tier 0's majority-vote pooling
    (docs/MIL_REFRAME_PLAN.md §4); it was previously computed inline in
    ``_slide_level_metrics`` only to build the confusion matrix and discarded
    otherwise -- this is that same computation, factored out and reused for the
    genuine bag-level metrics below.

    Bags spanning more than one CV fold (duplicated in ``manual_splits.json`` --
    docs/stage1_data_layer.md, or MIL_REFRAME_PLAN.md §7.1) need no special
    handling under majority vote specifically: pooling every instance from every
    fold a bag appears in and voting once is mathematically identical to voting
    per fold and averaging the fold-level vote-count vectors, whenever the bag
    has the same instance count in each fold it appears in -- which duplicated
    bags always do here, since it is the same physical slide scored twice by
    different fold models. (This equivalence does NOT hold for non-linear rules
    like top-q; see ``aggregate_bag_scores_across_folds`` for those.)

    Returns ``(bag_ids_sorted, bag_pred)``, or ``(bag_ids_sorted, bag_pred, bag_true)``
    when ``y_true`` is given -- ``bag_true`` is each bag's single label (every
    instance in a bag is asserted to share one label; this is a bag-level
    dataset, so a mismatch means the caller passed something wrong).
    Bags are sorted by id (Python's default string/number ordering) so two calls
    on data covering the same bags always align positionally without a join.
    """
    if rule != 'majority':
        raise ValueError(f"bag_decisions only implements 'majority'; got {rule!r}")
    preds = np.asarray(preds)
    bag_ids = np.asarray(bag_ids)
    y_true = np.asarray(y_true) if y_true is not None else None

    bag_to_indices = {}
    for idx, bid in enumerate(bag_ids):
        bag_to_indices.setdefault(bid, []).append(idx)

    bags_sorted = sorted(bag_to_indices)
    bag_pred, bag_true = [], []
    for bid in bags_sorted:
        ix = np.array(bag_to_indices[bid])
        bag_pred.append(int(np.bincount(preds[ix]).argmax()))
        if y_true is not None:
            yt = y_true[ix]
            assert np.all(yt == yt[0]), (
                f"bag {bid!r} has mixed instance labels {np.unique(yt)} -- every "
                "instance in a bag must share the bag's label"
            )
            bag_true.append(int(yt[0]))

    if y_true is not None:
        return np.array(bags_sorted), np.array(bag_pred), np.array(bag_true)
    return np.array(bags_sorted), np.array(bag_pred)


def _slide_level_metrics(y_true, preds, slide_ids, proba=None):
    """
    Two separate metric sets in one dict, at two different units of analysis --
    do not mix them when reading this (docs/MIL_REFRAME_PLAN.md §7):

    **Bag-level** (``bag_*`` keys, plus ``confusion_matrix``) -- the actual
    clinical claim. One majority-vote decision per bag (``bag_decisions``), then
    standard sklearn metrics (accuracy, balanced accuracy, macro F1/precision/
    recall, kappa) computed on those ``n_bags`` decisions against the ``n_bags``
    true labels. These are genuine, non-degenerate multiclass metrics -- no
    instance-level ground truth is assumed or required. ``auroc``/``auprc`` use
    per-bag mean-pooled probabilities (Tier 0's mean-posterior rule) ranked
    against the bag labels.

    **Within-bag instance diagnostics** (``within_bag_instance_recall``,
    ``f1_weighted``) -- NOT accuracy against a bag decision; the fraction of a
    bag's own instances that carry the bag's own label under the naive
    assumption every instance does. Kept for continuity with pre-reframe
    reporting; do not treat as evidence about per-instance predictions, since no
    instance-level ground truth exists to validate them against.

    Superseded, no longer computed the old way: ``precision_macro`` (was a
    constant ~0.5 by construction -- every bag is single-class, so a bag's own-
    class instance precision is always exactly 1.0), ``kappa`` (was always 0.0,
    ``cohen_kappa_score`` on a constant ``y_true``), and the old ``f1_macro``
    (was upward-biased and non-monotone in accuracy -- it silently dropped bags
    with 0% within-bag accuracy, e.g. ``results/exp_401``'s
    ``f1_macro=0.926`` beside ``accuracy=0.431``). All three now name genuine
    bag-level quantities instead (see docs/stage0_findings.md-style writeup at
    docs/MIL_REFRAME_PLAN.md §7 for the derivation of each bug).
    """
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    slide_ids = np.asarray(slide_ids)
    proba = np.asarray(proba) if proba is not None else None
    classes = sorted(np.unique(y_true).tolist())
    is_binary = len(classes) == 2

    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        slide_to_indices.setdefault(sid, []).append(idx)
    slide_to_indices = {s: np.array(ix) for s, ix in slide_to_indices.items()}

    per_class_recall = {c: [] for c in classes}
    per_bag_f1w = []
    bag_mean_proba_by_id = {}

    for sid, ix in slide_to_indices.items():
        yt, yp = y_true[ix], preds[ix]
        assert np.all(yt == yt[0]), (
            f"bag {sid!r} has mixed instance labels {np.unique(yt)} -- every instance "
            "in a bag must share the bag's label (this is a bag-level dataset)"
        )

        per_bag_f1w.append(f1_score(yt, yp, average='weighted', zero_division=0))

        for c in classes:
            true_c = yt == c
            n_true = int(true_c.sum())
            if n_true > 0:
                tp = int(np.sum(true_c & (yp == c)))
                per_class_recall[c].append(tp / n_true)

        if proba is not None:
            bag_mean_proba_by_id[sid] = proba[ix].mean(axis=0)

    # ---- within-bag instance diagnostics (NOT bag-level; see docstring) ----
    all_recalls = [r for c in classes for r in per_class_recall[c]]
    within_bag_recall = float(np.mean(all_recalls)) if all_recalls else 0.0

    # ---- genuine bag-level metrics: one decision per bag, standard sklearn ----
    bag_ids_sorted, bag_pred, bag_true = bag_decisions(preds, slide_ids, y_true, rule='majority')

    m = {
        'within_bag_instance_recall': within_bag_recall,
        'f1_weighted':                float(np.mean(per_bag_f1w)) if per_bag_f1w else 0.0,
        'n_bags':                     int(len(bag_ids_sorted)),

        'bag_accuracy':          float(accuracy_score(bag_true, bag_pred)),
        'bag_balanced_accuracy': float(balanced_accuracy_score(bag_true, bag_pred)),
        'bag_f1_macro':          float(f1_score(bag_true, bag_pred, average='macro',
                                                zero_division=0, labels=classes)),
        'bag_precision_macro':   float(precision_score(bag_true, bag_pred, average='macro',
                                                        zero_division=0, labels=classes)),
        'bag_recall_macro':      float(recall_score(bag_true, bag_pred, average='macro',
                                                     zero_division=0, labels=classes)),
        'bag_kappa':             float(cohen_kappa_score(bag_true, bag_pred))
                                  if len(np.unique(bag_true)) > 1 else float('nan'),
    }
    if is_binary:
        pos, neg = classes[1], classes[0]
        # Bag-level sensitivity/specificity: per-class recall of the bag decisions.
        bag_recall_per_class = recall_score(bag_true, bag_pred, average=None,
                                            zero_division=0, labels=classes)
        m['bag_sensitivity'] = float(bag_recall_per_class[1])
        m['bag_specificity'] = float(bag_recall_per_class[0])
        # Diagnostic-level sensitivity/specificity (within-bag instance recall,
        # split by class) -- kept for continuity, same caveat as within_bag_instance_recall.
        m['sensitivity'] = float(np.mean(per_class_recall[pos])) if per_class_recall[pos] else 0.0
        m['specificity'] = float(np.mean(per_class_recall[neg])) if per_class_recall[neg] else 0.0

    # Bag-level confusion matrix (n_bags total, not n_spectra) -- majority-vote decisions.
    m['confusion_matrix'] = confusion_matrix(bag_true, bag_pred, labels=classes).tolist()

    if proba is not None and bag_mean_proba_by_id:
        # Look up by bag_ids_sorted (bag_decisions' own order) rather than
        # re-deriving a sort order -- avoids relying on a second sort agreeing
        # with the first for whatever type bag ids happen to be.
        bl_y = bag_true
        bl_p = np.vstack([bag_mean_proba_by_id[bid] for bid in bag_ids_sorted])
        try:
            if is_binary:
                m['bag_auroc'] = float(roc_auc_score(bl_y, bl_p[:, 1]))
                m['bag_auprc'] = float(average_precision_score(bl_y, bl_p[:, 1]))
            else:
                m['bag_auroc'] = float(roc_auc_score(
                    bl_y, bl_p, multi_class='ovr', average='macro', labels=classes))
        except Exception:
            pass

    return m


def stratified_slide_metrics(preds, y_true, slide_ids):
    """Deprecated: use ``_slide_level_metrics(...)``'s ``bag_f1_macro`` /
    ``bag_precision_macro`` / ``bag_recall_macro`` keys directly. Kept only so
    nothing importing this name at module load time breaks; not used by the
    live pipeline (grepped: no callers outside this module)."""
    full = _slide_level_metrics(y_true, preds, slide_ids)
    return {
        'f1_macro': full['bag_f1_macro'],
        'precision_macro': full['bag_precision_macro'],
        'recall_macro': full['bag_recall_macro'],
    }


# ============================================================================
# SECTION 1c: Tier 0 -- naive-MIL aggregation baselines
# (docs/MIL_REFRAME_PLAN.md §4). Pools per-instance probabilities the existing
# TRTR/TSTR/TSRTR models already emit -- no new training. Everything else in
# the MIL programme must beat these.
# ============================================================================

def pool_instance_scores(proba, bag_ids, rule='mean', topq_fraction=0.1, lse_temperature=1.0):
    """
    Pool per-instance class probabilities into one (K,) score vector per bag,
    WITHOUT taking a final argmax -- returning scores (not decisions) lets
    ``aggregate_bag_scores_across_folds`` average a duplicated bag's per-fold
    scores before deciding once, per docs/MIL_REFRAME_PLAN.md §7.1.

    rule:
      'mean'     -- mean-posterior: bag score = mean of instance probabilities.
      'majority' -- vote-fraction: score[k] = fraction of instances whose own
                    argmax is k. argmax of this equals plain majority vote
                    (``bag_decisions``); expressing it as a score vector is
                    what makes the cross-fold averaging well-defined for it too.
      'lse'      -- log-sum-exp (smooth max) over each class's log-probability,
                    temperature-scaled, then softmax-renormalised across
                    classes so the result is a valid probability vector.
      'topq'     -- mean of the ``topq_fraction`` highest-scoring instances per
                    class (the "witness fraction" baseline). A **fraction** of
                    the bag's own instance count, not a fixed count -- a fixed
                    count would favour large bags (more chances at one
                    high-scoring instance); MIL_REFRAME_PLAN.md §4.

    Returns ``(bag_ids_sorted, scores)``, ``scores`` shape ``(n_bags, K)``, each
    row summing to 1 so different bags'/rules'/folds' scores are comparable and
    can be averaged.
    """
    proba = np.asarray(proba, dtype=np.float64)
    bag_ids = np.asarray(bag_ids)
    n_classes = proba.shape[1]

    bag_to_indices = {}
    for idx, bid in enumerate(bag_ids):
        bag_to_indices.setdefault(bid, []).append(idx)
    bags_sorted = sorted(bag_to_indices)

    scores = np.zeros((len(bags_sorted), n_classes), dtype=np.float64)
    for row, bid in enumerate(bags_sorted):
        ix = np.array(bag_to_indices[bid])
        p = proba[ix]

        if rule == 'mean':
            s = p.mean(axis=0)
        elif rule == 'majority':
            votes = np.bincount(p.argmax(axis=1), minlength=n_classes).astype(np.float64)
            s = votes / max(votes.sum(), 1.0)
        elif rule == 'lse':
            from scipy.special import logsumexp
            log_p = np.log(np.clip(p, 1e-12, 1.0))
            lse = logsumexp(log_p / lse_temperature, axis=0)
            s = np.exp(lse - logsumexp(lse))  # softmax over classes -> sums to 1
        elif rule == 'topq':
            q = max(1, int(np.ceil(topq_fraction * len(ix))))
            s = np.array([np.sort(p[:, k])[-q:].mean() for k in range(n_classes)])
            total = s.sum()
            s = s / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
        else:
            raise ValueError(f"unknown pooling rule {rule!r}")

        scores[row] = s

    return np.array(bags_sorted), scores


def aggregate_bag_scores_across_folds(fold_data, rule='mean', **rule_kwargs):
    """
    Tier 0 pooling with the duplicate-bag cross-fold averaging rule
    (docs/MIL_REFRAME_PLAN.md §7.1): 4 bags in ``manual_splits.json`` appear in
    two CV folds' validation sets each (forced by 6-class stratification over
    classes with only 4 patients). Each gets one score per fold it appears in;
    those per-fold scores are averaged before the final decision, so it is not
    double-counted and its decision reflects both folds' models.

    For the 'mean' and 'majority' rules this is provably identical to simply
    pooling every instance from every fold a bag appears in and aggregating
    once -- a duplicated bag has the same instance count in every fold it
    appears in (same physical slide, scored by a different fold's model each
    time), so per-fold averaging of a *linear* statistic equals pooling then
    computing it once. This function still does genuine per-fold aggregation
    for all four rules, both because 'lse' and 'topq' are not linear in this
    sense (top-q over pooled instances != averaging two folds' top-q means) and
    because that equivalence is meant as a validation check on this function
    (tests/test_stage2_evaluation.py), not as the implementation strategy.

    Args:
        fold_data: list of dicts, each with 'proba' (n_i, K), 'y_true' (n_i,),
            'bag_ids' (n_i,) -- one entry per CV fold's validation set.

    Returns ``(bag_ids_sorted, scores, bag_true)``: ``scores`` is
    ``(n_bags, K)``, each row a valid probability vector (renormalised after
    cross-fold averaging); ``bag_true`` is each bag's label, asserted
    consistent across every fold/instance it appears in.
    """
    per_fold_scores = {}   # bag_id -> list of (K,) score vectors, one per fold
    bag_true = {}
    for fd in fold_data:
        ids, scores = pool_instance_scores(fd['proba'], fd['bag_ids'], rule=rule, **rule_kwargs)
        _, _, fold_bag_true = bag_decisions(
            np.argmax(fd['proba'], axis=1), fd['bag_ids'], fd['y_true'], rule='majority')
        for bid, sc, yt in zip(ids, scores, fold_bag_true):
            per_fold_scores.setdefault(bid, []).append(sc)
            if bid in bag_true:
                assert bag_true[bid] == yt, (
                    f"bag {bid!r} has inconsistent labels across folds: "
                    f"{bag_true[bid]} vs {yt} -- a bag's label must not change "
                    "between the folds it appears in"
                )
            else:
                bag_true[bid] = yt

    bags_sorted = sorted(per_fold_scores)
    scores = np.array([np.mean(per_fold_scores[b], axis=0) for b in bags_sorted])
    row_sums = scores.sum(axis=1, keepdims=True)
    scores = scores / np.where(row_sums > 0, row_sums, 1.0)
    y_true_arr = np.array([bag_true[b] for b in bags_sorted])
    return np.array(bags_sorted), scores, y_true_arr


def bag_level_bootstrap(bag_pred_A, bag_pred_B, bag_true, n_boot=1000, seed=42):
    """
    Class-stratified bootstrap over BAG decisions (not instances, not the
    within-bag-instance-recall statistic) -- the primary-inference CI for the
    reframed metrics (docs/MIL_REFRAME_PLAN.md §7.1's "CI from the existing
    stratified_slide_bootstrap, now resampling bag decisions").

    Implemented by reusing ``stratified_slide_bootstrap`` with each bag as its
    own singleton "slide" (``bag_ids = np.arange(len(bag_true))``): that
    function resamples groups with replacement within class and averages
    per-group accuracy, which for singleton groups is exactly a class-
    stratified bootstrap over the bag decisions themselves -- no new resampling
    logic needed, just fed genuine bag-level (not instance-level) arrays.
    """
    dummy_ids = np.arange(len(bag_true))
    return stratified_slide_bootstrap(bag_pred_A, bag_pred_B, bag_true, dummy_ids,
                                      n_boot=n_boot, seed=seed)


def bag_level_bootstrap_cbr(bag_pred_cbr, bag_true, n_boot=1000, seed=42):
    """Bag-level bootstrap for a single model (CbR) -- see ``bag_level_bootstrap``."""
    dummy_ids = np.arange(len(bag_true))
    return stratified_slide_bootstrap_cbr(bag_pred_cbr, bag_true, dummy_ids,
                                          n_boot=n_boot, seed=seed)


def stratified_slide_bootstrap(preds_A, preds_B, y_true, slide_ids, n_boot=1000, seed=42):
    """
    Fast slide-level stratified bootstrap for comparing two models.
    
    Compares two models (e.g., TRTR vs TSTR) by resampling slides (not spectra)
    within each class and computing bootstrap accuracy distributions.
    
    Speed optimization: Pre-computes slide-to-indices mapping to avoid
    repeated lookups in the bootstrap loop.
    
    Args:
        preds_A: (n_samples,) array of Model A predictions
        preds_B: (n_samples,) array of Model B predictions
        y_true: (n_samples,) array of true labels
        slide_ids: (n_samples,) array of slide IDs
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: Three arrays of shape (n_boot,)
            - acc_A_boot: Bootstrap accuracies for Model A
            - acc_B_boot: Bootstrap accuracies for Model B
            - boot_diffs: Bootstrap differences (Model B - Model A)
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_true)
    
    # 1. PRE-COMPUTE SLIDE INDICES (Speed optimization)
    # Maps 'Slide_ID' -> array of row indices for that slide
    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        if sid not in slide_to_indices:
            slide_to_indices[sid] = []
        slide_to_indices[sid].append(idx)
    
    for sid in slide_to_indices:
        slide_to_indices[sid] = np.array(slide_to_indices[sid])
    
    # 2. Map class -> unique slides in that class
    class_to_slides = {}
    for cls in unique_classes:
        class_to_slides[cls] = np.unique(slide_ids[y_true == cls])
    
    acc_A_boot, acc_B_boot, boot_diffs = [], [], []
    
    # 3. FAST BOOTSTRAP LOOP
    print("Running optimized slide-level bootstrap...")
    for i in range(n_boot):
        boot_acc_A = []
        boot_acc_B = []
        
        for cls in unique_classes:
            slides_in_cls = class_to_slides[cls]
            # Resample slides with replacement
            sampled_slides = rng.choice(slides_in_cls, size=len(slides_in_cls), replace=True)
            
            for slide in sampled_slides:
                s_idx = slide_to_indices[slide]
                # Get indices where slide's samples belong to this class
                cls_idx = s_idx[y_true[s_idx] == cls]
                
                # Calculate accuracy for this slide in this class
                slide_acc_A = accuracy_score(y_true[cls_idx], preds_A[cls_idx])
                slide_acc_B = accuracy_score(y_true[cls_idx], preds_B[cls_idx])
                
                boot_acc_A.append(slide_acc_A)
                boot_acc_B.append(slide_acc_B)
        
        # Average slide accuracies (cluster bootstrap at slide level)
        acc_A = np.mean(boot_acc_A)
        acc_B = np.mean(boot_acc_B)
        
        acc_A_boot.append(acc_A)
        acc_B_boot.append(acc_B)
        boot_diffs.append(acc_B - acc_A)
    
    return np.array(acc_A_boot), np.array(acc_B_boot), np.array(boot_diffs)


# ============================================================================
# SECTION 1a: Patient-level (cluster) stratified bootstrap
#
# Alternative to the slide-level bootstrap above: resamples PATIENTS with
# replacement within each class and takes every slide of each drawn patient,
# so slides from one patient are never split across a resample. The statistic
# stays the slide-weighted mean (ratio estimator, random denominator), i.e. the
# same estimand as the slide-level bootstrap. Additive only -- the slide-level
# functions above are untouched, so published results stay reproducible.
#
# RNG contract: per replicate, classes in np.unique(strata) order, exactly one
# rng.choice(cands, size=n_c, replace=True) per class, cands = cluster indices
# in order of first appearance in the unit array. With singleton clusters in
# positional order this reproduces bag_level_bootstrap / _cbr bit-for-bit for
# 0/1 values (tests/test_patient_bootstrap.py relies on it).
# ============================================================================

def patient_id_from_slide(slide_id):
    """Patient id of a slide/bag id: the prefix before the first '_'
    ('HF-1002_V2AT_1.h5_1' -> 'HF-1002'); matches the H5 ``patient_ids``
    dataset for all 59 slides."""
    return str(slide_id).split('_')[0]


def patient_ids_from_slides(slide_ids):
    return np.array([patient_id_from_slide(s) for s in slide_ids])


def _cluster_setup(strata, clusters):
    """Cluster bookkeeping shared by the cluster bootstraps.

    Returns (unit_cluster, classes, cands): ``unit_cluster[i]`` is unit i's
    cluster index (clusters numbered by first appearance), ``cands[cls]`` the
    cluster indices of class ``cls`` in that same order. Raises if a cluster
    spans two strata (a patient with two labels cannot be class-stratified)."""
    strata = np.asarray(strata)
    clusters = np.asarray(clusters)
    if len(strata) != len(clusters):
        raise ValueError("strata and clusters must have the same length")
    index, cluster_stratum = {}, []
    unit_cluster = np.empty(len(clusters), dtype=np.int64)
    for i, (c, s) in enumerate(zip(clusters.tolist(), strata.tolist())):
        if c not in index:
            index[c] = len(index)
            cluster_stratum.append(s)
        elif cluster_stratum[index[c]] != s:
            raise ValueError(f"cluster {c!r} spans strata {cluster_stratum[index[c]]!r} and {s!r}")
        unit_cluster[i] = index[c]
    classes = np.unique(strata)
    cands = {cls: np.array([k for k, s in enumerate(cluster_stratum) if s == cls], dtype=np.int64)
             for cls in classes}
    return unit_cluster, classes, cands


def _draw_clusters(rng, classes, cands):
    """One stratified replicate: the drawn cluster indices (with multiplicity)."""
    return np.concatenate([rng.choice(cands[cls], size=len(cands[cls]), replace=True)
                           for cls in classes])


def stratified_cluster_bootstrap(values_A, values_B, strata, clusters, n_boot=1000, seed=42):
    """
    Stratified cluster bootstrap of a slide-weighted mean.

    Per replicate, draw n_c clusters with replacement within each stratum c,
    take every unit of each drawn cluster, and compute
    theta = sum(values of drawn units) / (number of drawn units).

    Args:
        values_A: (n_units,) per-unit values (e.g. bag correctness 0/1).
        values_B: (n_units,) paired values on the same units, or None for a
            single-arm bootstrap.
        strata: (n_units,) stratum (class) per unit; constant within a cluster.
        clusters: (n_units,) cluster id (e.g. patient) per unit.

    Returns ``(theta_A_boot, theta_B_boot, diffs)`` with ``diffs = B - A`` on
    the same draw (the last two are None when ``values_B`` is None).
    """
    values_A = np.asarray(values_A, dtype=np.float64)
    single = values_B is None
    values_B = values_A if single else np.asarray(values_B, dtype=np.float64)
    if not (len(values_A) == len(values_B) == len(strata) == len(clusters)):
        raise ValueError("values, strata and clusters must have the same length")

    unit_cluster, classes, cands = _cluster_setup(strata, clusters)
    n_clusters = int(unit_cluster.max()) + 1
    sum_A = np.bincount(unit_cluster, weights=values_A, minlength=n_clusters)
    sum_B = np.bincount(unit_cluster, weights=values_B, minlength=n_clusters)
    size = np.bincount(unit_cluster, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    theta_A = np.empty(n_boot)
    theta_B = np.empty(n_boot)
    for t in range(n_boot):
        drawn = _draw_clusters(rng, classes, cands)
        total = size[drawn].sum()
        theta_A[t] = sum_A[drawn].sum() / total
        theta_B[t] = sum_B[drawn].sum() / total

    if single:
        return theta_A, None, None
    return theta_A, theta_B, theta_B - theta_A


def _bag_clusters(bag_ids, cluster_unit):
    if cluster_unit == 'patient':
        return patient_ids_from_slides(bag_ids)
    if cluster_unit == 'slide':
        return np.asarray(bag_ids)
    raise ValueError(f"cluster_unit must be 'patient' or 'slide', got {cluster_unit!r}")


def bag_level_patient_bootstrap(bag_pred_A, bag_pred_B, bag_true, bag_ids, n_boot=1000, seed=42,
                                cluster_unit='patient'):
    """Patient-clustered, class-stratified counterpart of ``bag_level_bootstrap``:
    same inputs plus ``bag_ids`` (slide ids, to derive patients), same return
    ``(acc_A_boot, acc_B_boot, diffs)``. ``cluster_unit='slide'`` gives the
    class-stratified slide bootstrap through the same code path."""
    bag_true = np.asarray(bag_true)
    return stratified_cluster_bootstrap(
        np.asarray(bag_pred_A) == bag_true, np.asarray(bag_pred_B) == bag_true,
        bag_true, _bag_clusters(bag_ids, cluster_unit), n_boot=n_boot, seed=seed)


def bag_level_patient_bootstrap_cbr(bag_pred, bag_true, bag_ids, n_boot=1000, seed=42,
                                    cluster_unit='patient'):
    """Single-arm counterpart of ``bag_level_bootstrap_cbr``: returns the
    (n_boot,) array of bootstrap bag accuracies."""
    bag_true = np.asarray(bag_true)
    theta, _, _ = stratified_cluster_bootstrap(
        np.asarray(bag_pred) == bag_true, None, bag_true,
        _bag_clusters(bag_ids, cluster_unit), n_boot=n_boot, seed=seed)
    return theta


def paired_cluster_test(d, slide_ids, y_slide, n_boot=10000, seed=42, cluster_unit='patient',
                        stratify=True):
    """
    Paired per-slide test on differences ``d`` (e.g. real minus shuffled
    spatial statistic, or selected minus naive accuracy), clustered by patient.

    - Bootstrap: stratified cluster bootstrap of the slide-weighted mean of d;
      95% percentile CI.
    - Sign-flip permutation: one sign per cluster, T = |sum_p eps_p D_p| / N
      with D_p the cluster's sum of d and N the number of slides (the same
      slide-weighted estimand as delta), +1-corrected.
    - Wilcoxon signed-rank on cluster-mean d (cluster-weighted; diagnostic).

    Strata and clusters are built from the slides passed in, so callers pass
    only the slides actually used (e.g. those finite in both fits).
    ``stratify=False`` puts every slide in one stratum.

    Returns generic fields; callers map them onto their own schema.
    """
    from scipy.stats import wilcoxon

    d = np.asarray(d, dtype=np.float64)
    y_slide = np.asarray(y_slide)
    strata = y_slide if stratify else np.zeros(len(d), dtype=np.int64)
    clusters = _bag_clusters(slide_ids, cluster_unit)
    unit_cluster, classes, cands = _cluster_setup(strata, clusters)
    n_clusters = int(unit_cluster.max()) + 1
    D = np.bincount(unit_cluster, weights=d, minlength=n_clusters)
    size = np.bincount(unit_cluster, minlength=n_clusters).astype(np.float64)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for t in range(n_boot):
        drawn = _draw_clusters(rng, classes, cands)
        boot[t] = D[drawn].sum() / size[drawn].sum()
    signs = rng.choice([-1.0, 1.0], size=(n_boot, n_clusters))
    perm = np.abs(signs @ D) / len(d)
    delta = float(d.mean())
    try:
        w_p = float(wilcoxon(D / size).pvalue)
    except ValueError:   # all differences zero (older scipy raises)
        w_p = 1.0
    if not np.isfinite(w_p):   # all differences zero (newer scipy returns nan)
        w_p = 1.0
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # patients per class: count clusters by the class of their first unit
    first_unit = {}
    for i, c in enumerate(unit_cluster.tolist()):
        first_unit.setdefault(c, i)
    per_class = {str(cls): int(sum(1 for c, i in first_unit.items() if y_slide[i] == cls))
                 for cls in np.unique(y_slide)}
    return {
        'delta': delta, 'ci_lo': float(lo), 'ci_hi': float(hi),
        'p_signflip': float((np.sum(perm >= abs(delta) - 1e-15) + 1) / (n_boot + 1)),
        'p_wilcoxon': w_p,
        'n_units': int(len(d)), 'n_clusters': n_clusters, 'n_clusters_per_class': per_class,
        'cluster_unit': cluster_unit, 'stratified': bool(stratify),
        'n_boot': int(n_boot), 'seed': int(seed),
    }


# ============================================================================
# SECTION 1b: Canonical reporting helpers (single source of truth)
#
# Both the batch pipeline (pipeline/evaluation_stage.py) and the exploratory
# notebook (09_Statistical_evaluation.ipynb) import these so the two paths
# produce identical statistics and an identical output schema.
# ============================================================================

# Scalar metrics emitted (in order) to the flat CSV, per regimen. Bag-level
# (bag_*) is the headline; the rest are within-bag instance diagnostics, kept
# for continuity with pre-reframe reporting (docs/MIL_REFRAME_PLAN.md §7).
# 'accuracy'/'balanced_accuracy' no longer appear here: pre-Stage-2 they were
# aliases of the diagnostic 'within_bag_instance_recall' (the three were
# literally the same float); that ambiguity is exactly what the rename fixes,
# so the bare names are retired rather than quietly repointed to a bag-level
# value under the reader's old assumption.
SCALAR_METRIC_KEYS = [
    'bag_accuracy', 'bag_balanced_accuracy', 'bag_f1_macro', 'bag_precision_macro',
    'bag_recall_macro', 'bag_kappa', 'bag_sensitivity', 'bag_specificity',
    'bag_auroc', 'bag_auprc',
    'within_bag_instance_recall', 'f1_weighted', 'sensitivity', 'specificity',
]


def compute_full_metrics(y_true, preds, proba=None, slide_ids=None):
    """
    Full metric set for one model/regimen.

    When ``slide_ids`` is supplied (the default in both the pipeline and notebook
    09) metrics are computed at **two separate levels** via ``_slide_level_metrics``
    (docs/MIL_REFRAME_PLAN.md §7) -- see that function's docstring for the full
    split. In short: ``bag_*`` keys (``bag_accuracy``, ``bag_balanced_accuracy``,
    ``bag_f1_macro``, ``bag_precision_macro``, ``bag_recall_macro``, ``bag_kappa``,
    ``bag_sensitivity``/``bag_specificity`` for binary tasks,
    ``bag_auroc``/``bag_auprc`` with probabilities) plus ``confusion_matrix`` are
    the genuine bag-level clinical claim -- one majority-vote decision per bag,
    standard sklearn metrics on those decisions, non-degenerate. Everything else
    (``within_bag_instance_recall``, ``f1_weighted``, and for binary tasks the bare
    ``sensitivity``/``specificity``) is a within-bag instance diagnostic: it is NOT
    validated against any instance-level ground truth (none exists in this
    dataset) and must not be read as an accuracy.

    Without ``slide_ids`` the legacy spectrum-level (pooled across folds) metrics
    are returned, for backward compatibility only -- dead code in the live
    pipeline, which always supplies ``slide_ids``.

    Returns a flat, JSON-serialisable dict; see ``SCALAR_METRIC_KEYS`` for the
    canonical key list emitted to the summary CSV.
    """
    if slide_ids is not None:
        return _slide_level_metrics(y_true, preds, slide_ids, proba)

    m = {}
    m['accuracy']          = float(accuracy_score(y_true, preds))
    m['balanced_accuracy'] = float(balanced_accuracy_score(y_true, preds))
    m['f1_macro']          = float(f1_score(y_true, preds, average='macro', zero_division=0))
    m['precision_macro']   = float(precision_score(y_true, preds, average='macro', zero_division=0))
    m['recall_macro']      = float(recall_score(y_true, preds, average='macro', zero_division=0))
    m['f1_weighted']       = float(f1_score(y_true, preds, average='weighted', zero_division=0))
    m['kappa']             = float(cohen_kappa_score(y_true, preds))

    classes = sorted(np.unique(y_true))
    if len(classes) == 2:
        per_class_recall = recall_score(y_true, preds, average=None, labels=classes, zero_division=0)
        m['sensitivity'] = float(per_class_recall[1])  # class 1
        m['specificity'] = float(per_class_recall[0])  # class 0

    cm = confusion_matrix(y_true, preds, labels=classes)
    m['confusion_matrix'] = cm.tolist()

    if proba is not None:
        try:
            n_cls = proba.shape[1]
            if n_cls == 2:
                m['auroc'] = float(roc_auc_score(y_true, proba[:, 1]))
                m['auprc'] = float(average_precision_score(y_true, proba[:, 1]))
            else:
                m['auroc'] = float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
        except Exception:
            pass

    return m


def summarize_bootstrap_diffs(diffs, alpha=0.05):
    """
    Summarise a slide-level bootstrap distribution of paired accuracy
    differences (candidate - baseline) into the canonical inference fields.

    Primary inference is the percentile CI; the bootstrap p-value is reported
    as a secondary descriptor. No multiplicity correction is applied here.

    Args:
        diffs: 1-D array of bootstrap differences (candidate - baseline).
        alpha: two-sided level for the percentile CI (default 0.05 -> 95% CI).

    Returns dict with: boot_mean_diff, ci_diff_lo, ci_diff_hi, p_bootstrap,
    significant_95 (CI excludes 0).
    """
    diffs = np.asarray(diffs)
    lo_q, hi_q = 100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)
    ci_lo = float(np.percentile(diffs, lo_q))
    ci_hi = float(np.percentile(diffs, hi_q))
    p_boot = float(min(2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1.0))
    return {
        'boot_mean_diff': float(np.mean(diffs)),
        'ci_diff_lo': ci_lo,
        'ci_diff_hi': ci_hi,
        'p_bootstrap': p_boot,
        'significant_95': bool(ci_lo > 0 or ci_hi < 0),
    }


def build_summary_rows(results, baseline_label, candidate_label):
    """
    Flatten a list of canonical result rows into flat dicts for a paper-ready
    CSV: one row per model, all scalar metrics for both regimens side-by-side.

    Bag-level columns (``bag_acc_*``, ``bag_delta``, ``bag_boot_mean_diff``,
    ``bag_ci_diff_lo/hi``, ``bag_p_bootstrap``, ``bag_significant_95``) are the
    primary inference. ``within_bag_instance_recall_*`` columns are the demoted
    diagnostic, kept for continuity with pre-reframe reporting -- not a bag
    accuracy (docs/MIL_REFRAME_PLAN.md §7). Each model row is expected to carry
    both sets plus ``full_metrics_{label}`` dicts (see
    ``pipeline/evaluation_stage.py``). A Classification by Reconstruction row
    (model == 'Classification_by_Reconstruction') is handled separately with
    blanks for the comparison-only columns, since it is a single, model-
    independent regimen rather than a baseline-vs-candidate pair.
    """
    rows = []
    for r in results:
        if r.get('model') == 'Classification_by_Reconstruction':
            cbr_flat = {
                'model': r['model'], 'n_bags': r.get('n_bags', ''),
                f'bag_acc_{baseline_label}': '',
                f'bag_acc_{candidate_label}': r.get('cbr_bag_accuracy', ''),
                'bag_delta': '', 'bag_boot_mean_diff': '',
                'bag_ci_diff_lo': r.get('cbr_bag_ci_lo', ''),
                'bag_ci_diff_hi': r.get('cbr_bag_ci_hi', ''),
                'bag_p_bootstrap': '', 'bag_significant_95': '',
                f'within_bag_instance_recall_{baseline_label}': '',
                f'within_bag_instance_recall_{candidate_label}': r.get('cbr_accuracy', ''),
                'within_bag_instance_recall_delta': '', 'wbir_boot_mean_diff': '',
                'wbir_ci_diff_lo': r.get('cbr_ci_lo', ''),
                'wbir_ci_diff_hi': r.get('cbr_ci_hi', ''),
                'wbir_p_bootstrap': '', 'wbir_significant_95': '',
            }
            # CbR is model-independent (single regimen); surface its scalar metrics
            # under the candidate columns, leaving the baseline columns blank.
            cbr_full = r.get('full_metrics', {})
            for key in SCALAR_METRIC_KEYS:
                cbr_flat[f'{key}_{baseline_label}'] = ''
                cbr_flat[f'{key}_{candidate_label}'] = cbr_full.get(key, r.get(key, ''))
            rows.append(cbr_flat)
            continue
        flat = {
            'model': r['model'], 'n_bags': r.get('n_bags', ''),
            f'bag_acc_{baseline_label}': r.get(f'bag_acc_{baseline_label}', ''),
            f'bag_acc_{candidate_label}': r.get(f'bag_acc_{candidate_label}', ''),
            'bag_delta': r.get('bag_delta', ''),
            'bag_boot_mean_diff': r.get('bag_boot_mean_diff', ''),
            'bag_ci_diff_lo': r.get('bag_ci_diff_lo', ''),
            'bag_ci_diff_hi': r.get('bag_ci_diff_hi', ''),
            'bag_p_bootstrap': r.get('bag_p_bootstrap', ''),
            'bag_significant_95': r.get('bag_significant_95', ''),
            f'within_bag_instance_recall_{baseline_label}':
                r.get(f'within_bag_instance_recall_{baseline_label}', ''),
            f'within_bag_instance_recall_{candidate_label}':
                r.get(f'within_bag_instance_recall_{candidate_label}', ''),
            'within_bag_instance_recall_delta': r.get('within_bag_instance_recall_delta', ''),
            'wbir_boot_mean_diff': r.get('wbir_boot_mean_diff', ''),
            'wbir_ci_diff_lo': r.get('wbir_ci_diff_lo', ''),
            'wbir_ci_diff_hi': r.get('wbir_ci_diff_hi', ''),
            'wbir_p_bootstrap': r.get('wbir_p_bootstrap', ''),
            'wbir_significant_95': r.get('wbir_significant_95', ''),
        }
        for key in SCALAR_METRIC_KEYS:
            flat[f'{key}_{baseline_label}'] = r.get(f'full_metrics_{baseline_label}', {}).get(key, '')
            flat[f'{key}_{candidate_label}'] = r.get(f'full_metrics_{candidate_label}', {}).get(key, '')
        rows.append(flat)
    return rows


# ============================================================================
# SECTION 2: CVAE Classification by Reconstruction (CbR) Functions
# ============================================================================

def compute_cbr_predictions_per_fold(loader, cvae_config, model_dir, n_folds):
    """
    Extract per-fold Classification by Reconstruction predictions.
    
    For each fold:
    1. Load validation data (X_val, y_val, sample_names)
    2. Load CVAE encoder/decoder for that fold
    3. Encode X_val -> z_mean
    4. For each class k: decode(z_mean, class_k) -> reconstruct X_k
    5. Calculate MSE for each class -> argmin(MSE) = predicted class
    6. Return predictions and metadata
    
    Args:
        loader: GliomaDataLoader iterator yielding fold_data dicts
        cvae_config: Configuration dict for CVAE architecture
        model_dir: Directory containing CVAE weights (encoder_fold_X.weights.h5, decoder_fold_X.weights.h5)
        n_folds: Number of cross-validation folds
    
    Returns:
        tuple:
            - cbr_preds_all: (n_samples,) array of CbR predictions across all folds
            - cbr_sample_names: (n_samples,) array of sample names (for patient ID extraction)
            - cbr_classes_all: (n_samples,) array of true class labels
    """
    cbr_preds_all = []
    cbr_sample_names_all = []
    cbr_classes_all = []
    
    print("\n" + "="*70)
    print("PHASE 1: Computing Classification by Reconstruction per-fold")
    print("="*70)
    
    for fold_data in loader:
        fold_name = fold_data['fold_name']
        print(f"\n{fold_name}...")
        
        # Unpack validation data
        X_val = fold_data['val'][0].astype(np.float32)
        y_val = fold_data['val'][1]
        sample_names = fold_data['val'][2] if len(fold_data['val']) > 2 else None
        
        # Fix 1-indexed to 0-indexed if necessary
        if np.min(y_val) == 1:
            y_val = y_val - 1
        
        X_val_exp = np.expand_dims(X_val, axis=-1)  # (N, 1738, 1)
        n_samples = X_val.shape[0]
        
        # Initialize CVAE
        tf.keras.backend.clear_session()
        cvae = CVAE(cvae_config)
        cvae.compile(optimizer='adam')
        
        # Dummy batch to initialize shapes
        n_features = cvae_config['input_shape'][0]
        dummy_x = tf.zeros((2, n_features, 1))
        num_classes = cvae_config.get('num_classes', 6)
        dummy_y = tf.convert_to_tensor(
            tf.keras.utils.to_categorical([0, 1], num_classes=num_classes),
            dtype=tf.float32
        )
        cvae.train_step((dummy_x, dummy_y))
        
        # Load fold-specific weights
        enc_path = f"{model_dir}/encoder_{fold_name}.weights.h5"
        dec_path = f"{model_dir}/decoder_{fold_name}.weights.h5"
        
        if not os.path.exists(enc_path) or not os.path.exists(dec_path):
            print(f"    CVAE weights not found for {fold_name}. Skipping CbR for this fold.")
            continue
        
        cvae.encoder.load_weights(enc_path)
        cvae.decoder.load_weights(dec_path)
        
        encoder = cvae.encoder
        decoder = cvae.decoder
        
        # PHASE 1: ENCODE
        z_mean, _, _, _ = encoder.predict(X_val_exp, batch_size=256, verbose=0)
        
        # PHASE 2: DECODE (for each class) and compute MSE
        mse_scores = np.zeros((n_samples, num_classes), dtype=np.float32)
        for class_idx in range(num_classes):
            requested_classes = tf.convert_to_tensor(
                tf.keras.utils.to_categorical([class_idx] * n_samples, num_classes=num_classes),
                dtype=tf.float32
            )
            reconstructed = decoder.predict([z_mean, requested_classes], batch_size=256, verbose=0)
            reconstructed = np.squeeze(reconstructed, axis=-1)
            mse_scores[:, class_idx] = np.mean((X_val - reconstructed)**2, axis=1)
        
        # PHASE 3: CLASSIFY by argmin(MSE)
        y_pred_cbr = np.argmin(mse_scores, axis=1)
        
        cbr_preds_all.append(y_pred_cbr)
        if sample_names is not None:
            cbr_sample_names_all.append(sample_names)
        cbr_classes_all.append(y_val)
        
        acc_cbr = accuracy_score(y_val, y_pred_cbr)
        print(f"   Accuracy: {acc_cbr:.4f}")
    
    # Concatenate across folds
    cbr_preds_all = np.concatenate(cbr_preds_all)
    cbr_classes_all = np.concatenate(cbr_classes_all)
    
    if cbr_sample_names_all:
        cbr_sample_names_all = np.concatenate(cbr_sample_names_all)
    else:
        cbr_sample_names_all = None
    
    return cbr_preds_all, cbr_sample_names_all, cbr_classes_all


def stratified_slide_bootstrap_cbr(preds_cbr, y_true, slide_ids, n_boot=1000, seed=42):
    """
    Slide-level stratified bootstrap for CbR (no CVAE re-inference).
    
    Since CVAE is computationally expensive, this function does NOT re-run CVAE
    for each bootstrap iteration. Instead, it:
    1. Pre-computes CbR predictions once (via compute_cbr_predictions_per_fold)
    2. Bootstraps by resampling slides (not spectra) within each class
    3. Recomputes accuracy on the resampled slide set
    
    Args:
        preds_cbr: (n_samples,) array of CbR predictions (pre-computed, not re-inferred)
        y_true: (n_samples,) array of true labels
        slide_ids: (n_samples,) array of slide IDs
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        np.ndarray: (n_boot,) array of bootstrap accuracies
    """
    rng = np.random.default_rng(seed)
    unique_classes = np.unique(y_true)
    
    # PRE-COMPUTE slide indices (for speed)
    slide_to_indices = {}
    for idx, sid in enumerate(slide_ids):
        if sid not in slide_to_indices:
            slide_to_indices[sid] = []
        slide_to_indices[sid].append(idx)
    
    for sid in slide_to_indices:
        slide_to_indices[sid] = np.array(slide_to_indices[sid])
    
    # Map class -> unique slides in that class
    class_to_slides = {}
    for cls in unique_classes:
        class_to_slides[cls] = np.unique(slide_ids[y_true == cls])
    
    boot_acc_cbr = []
    
    print("\nRunning slide-level bootstrap for CbR (no CVAE re-inference)...")
    for i in range(n_boot):
        boot_acc = []
        
        for cls in unique_classes:
            slides_in_cls = class_to_slides[cls]
            # Resample slides with replacement
            sampled_slides = rng.choice(slides_in_cls, size=len(slides_in_cls), replace=True)
            
            for slide in sampled_slides:
                s_idx = slide_to_indices[slide]
                # Get indices where slide's samples belong to this class
                cls_idx = s_idx[y_true[s_idx] == cls]
                
                # Calculate accuracy for this slide
                slide_acc = accuracy_score(y_true[cls_idx], preds_cbr[cls_idx])
                boot_acc.append(slide_acc)
        
        # Average slide accuracies
        acc = np.mean(boot_acc)
        boot_acc_cbr.append(acc)
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_boot} iterations complete")
    
    return np.array(boot_acc_cbr)
