"""
Tier 1: mi-SVM / MI-SVM (Andrews et al. 2002), one-vs-rest per diagnostic
class, over the constrained model (C) -- docs/MIL_REFRAME_PLAN.md §4.

Reuses Apollo (src/models/baselines.py: RF top-20 feature selection + RBF SVM)
as the base learner per one-vs-rest sub-problem, so the feature pipeline is
unchanged and comparable to the existing TRTR/TSTR/TSRTR benchmark.

Only positive-bag instance labels are ever revised; negative-bag instances are
fixed at label 0 for the whole run (a bag that is NOT class k contributes no
evidence for class k in ANY of its instances -- that is the defining MIL
asymmetry mi-SVM relies on, not an approximation of it).
"""
import numpy as np

from src.models.baselines import ApolloPipeline


def _fit_one_vs_rest(X, bag_ids, y_bag_per_instance, target_class, pi_min, variant,
                     max_iters, seed):
    """
    One binary mi-SVM/MI-SVM sub-problem.

    variant='mi_svm': relabel the top ceil(pi_min * n_b) scoring instances per
        positive bag as positive each round (docs/MIL_REFRAME_PLAN.md §4:
        "relabel instances subject to >= pi_min positives per positive bag").
    variant='MI_SVM': keep only the single top-scoring instance (the witness)
        per positive bag -- sparser, yields an explicit, interpretable
        witness set.

    Returns (fitted ApolloPipeline, final_labels, n_iters_run, churn_history).
    """
    y_bag_per_instance = np.asarray(y_bag_per_instance)
    bag_ids = np.asarray(bag_ids)
    is_positive_bag = (y_bag_per_instance == target_class)

    # Naive init: every instance inherits its bag's label (Tier 0's own
    # assumption) -- exactly the starting point mi-SVM iterates away from.
    labels = is_positive_bag.astype(int)

    bag_to_idx = {b: np.where(bag_ids == b)[0] for b in np.unique(bag_ids[is_positive_bag])}

    model = None
    churn_history = []
    n_iters = 0
    for n_iters in range(1, max_iters + 1):
        model = ApolloPipeline(random_state=seed)
        model.fit(X, labels)
        scores = model.predict_proba(X)[:, 1]

        new_labels = np.zeros_like(labels)
        for b, idx in bag_to_idx.items():
            n_b = len(idx)
            bag_scores = scores[idx]
            k = 1 if variant == 'MI_SVM' else max(1, int(np.ceil(pi_min * n_b)))
            top_k = idx[np.argsort(bag_scores)[-k:]]
            new_labels[top_k] = 1

        churn = int(np.sum(new_labels[is_positive_bag] != labels[is_positive_bag]))
        churn_history.append(churn)
        labels = new_labels
        if churn == 0:
            break

    return model, labels, n_iters, churn_history


def fit_misvm(X, bag_ids, y_bag_per_instance, n_classes, pi_min=0.02, variant='mi_svm',
             max_iters=10, seed=42, verbose=True):
    """
    Fit one one-vs-rest mi-SVM/MI-SVM per diagnostic class and combine into a
    single (n, K+1) posterior (docs/MIL_REFRAME_PLAN.md §4): Platt-scaled
    margins across the K one-vs-rest problems, non-diagnostic mass as
    1 - max_k p_k, renormalised to a valid probability vector.

    Returns dict: 'posterior' (n, K+1), 'models' ({class: ApolloPipeline}),
    'churn_history' ({class: [churn per iter]}), 'witness_idx' ({class: array
    of the converged positive-labelled instance indices} -- MI_SVM's
    interpretable witness set; for mi_svm this is simply every instance
    labelled positive at convergence, which is a larger, softer set).
    """
    n = X.shape[0]
    class_probs = np.zeros((n, n_classes), dtype=np.float64)
    models, churn_histories, witness_sets = {}, {}, {}

    for k in range(n_classes):
        model, labels, n_iters, churn = _fit_one_vs_rest(
            X, bag_ids, y_bag_per_instance, target_class=k,
            pi_min=pi_min, variant=variant, max_iters=max_iters, seed=seed)
        class_probs[:, k] = model.predict_proba(X)[:, 1]
        models[k] = model
        churn_histories[k] = churn
        witness_sets[k] = np.where(labels == 1)[0]
        if verbose:
            print(f"    [mi-SVM class {k}] converged in {n_iters} iters, churn history: {churn}")

    non_diag = np.clip(1.0 - class_probs.max(axis=1, keepdims=True), 0.0, None)
    raw = np.concatenate([non_diag, class_probs], axis=1)
    row_sums = raw.sum(axis=1, keepdims=True)
    posterior = raw / np.where(row_sums > 0, row_sums, 1.0)

    return {
        'posterior': posterior,
        'models': models,
        'churn_history': churn_histories,
        'witness_idx': witness_sets,
    }


def predict_misvm(fit_result, X):
    """Score new instances with an already-fitted mi-SVM/MI-SVM result (e.g.
    a fold's held-out validation instances) -- same combination rule as
    fit_misvm's own training-set posterior."""
    n_classes = len(fit_result['models'])
    class_probs = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for k, model in fit_result['models'].items():
        class_probs[:, k] = model.predict_proba(X)[:, 1]
    non_diag = np.clip(1.0 - class_probs.max(axis=1, keepdims=True), 0.0, None)
    raw = np.concatenate([non_diag, class_probs], axis=1)
    row_sums = raw.sum(axis=1, keepdims=True)
    return raw / np.where(row_sums > 0, row_sums, 1.0)
