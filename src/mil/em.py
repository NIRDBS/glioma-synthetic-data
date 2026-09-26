"""
Tier 2: EM / label-proportion model with hierarchical pi_b
(docs/MIL_REFRAME_PLAN.md §1, §4) -- constrained model (C).

This is the formulation meant to yield *calibrated* soft labels for T3's CVAE
conditioning (mi-SVM's Platt scores are calibrated per one-vs-rest binary
problem, not jointly across bags).

Deliberate simplification, documented rather than silently assumed: this is a
Classification-EM (hard pseudo-labels + confidence sample_weight at the
M-step), not a fit with a genuinely soft-label discriminative model --
sklearn's classifiers don't natively fit soft multiclass targets, and CEM is a
standard, tractable approximation for exactly this situation. The hierarchical
pi_b posterior is likewise approximated by empirical-Bayes shrinkage toward the
population logit-mean, sized by each bag's own instance count, rather than a
full posterior (that would need a Laplace approximation or MCMC) -- Stage 4's
own text already expects sigma to be "poorly determined" at 58 bags, so a full
Bayesian treatment is not proportionate to what 58 bags can actually support.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.mil.common import (
    N_NULL_COMPONENTS, n_states, diagnostic_state, posterior_to_K1,
    pi_b_per_bag, shrinkage_diagnostic, null_component_collapse_diagnostic,
)


def _cold_start_hard_labels(X, bag_ids, y_bag_per_instance, n_classes, pi_init, pi_min, seed):
    """
    Cold start via a quick mi-SVM pre-fit (src/mil/misvm.py) rather than the
    naive "every instance is diagnostic" assumption alone.

    A pure 100%-diagnostic cold start was tried first and failed: it gives the
    RF's very first fit zero exposure to any null-labelled example at all
    (``model.classes_`` only ever contains the K diagnostic states after that
    first fit), which floors every null-state probability at a fixed constant
    for the rest of the run and reliably drifts pi_b -> ~1 regardless of later
    relabeling -- confirmed empirically (tests/test_stage4_mil.py), the
    "collapse to the naive solution" mode docs/MIL_REFRAME_PLAN.md §5 names.

    mi-SVM's own per-bag top-k relabeling is already validated to recover a
    sensible diagnostic/null split on this kind of data
    (tests/test_stage4_mil.py::test_misvm_recovers_class_separation), so
    bootstrapping EM's RF from mi-SVM's split -- rather than inventing a new
    heuristic -- reuses an already-working, already-tested mechanism. Null
    components (which mi-SVM does not subdivide) are seeded via a plain
    unsupervised KMeans(M) over the instances mi-SVM assigned to "null"; EM's
    own iterations refine this subdivision afterward.

    The bootstrap mi-SVM is floored at ``pi_init`` (the caller's actual prior
    belief about the marginal diagnostic rate), NOT ``pi_min`` (EM's own tiny
    numerical identifiability floor, ~2-5%): passing ``pi_min`` here was tried
    first and confirmed empirically to make the bootstrap itself only recover
    ~19% of true diagnostic instances on data with a true rate of 30% (mi-SVM
    has no other pressure to include more than its own floor demands), which
    then starves the very first E-step of real signal regardless of any later
    fix. ``pi_init`` is normally much larger than ``pi_min`` (§1: pi_min is a
    floor "1-5%", not a belief), so this bootstrap needs its own, separate,
    more realistic floor to have something worth bootstrapping from.
    """
    from sklearn.cluster import KMeans

    from src.mil.misvm import fit_misvm

    misvm_fit = fit_misvm(X, bag_ids, y_bag_per_instance, n_classes, pi_min=pi_init,
                         variant='mi_svm', max_iters=5, seed=seed, verbose=False)
    is_diagnostic = misvm_fit['posterior'][:, 1:].max(axis=1) > misvm_fit['posterior'][:, 0]

    n = len(y_bag_per_instance)
    hard_labels = np.zeros(n, dtype=int)
    hard_labels[is_diagnostic] = diagnostic_state(y_bag_per_instance[is_diagnostic])

    null_idx = np.where(~is_diagnostic)[0]
    if len(null_idx) >= N_NULL_COMPONENTS:
        km = KMeans(n_clusters=N_NULL_COMPONENTS, random_state=seed, n_init=10)
        hard_labels[null_idx] = km.fit_predict(X[null_idx])
    # else: leave at the default null component 0 -- too few null instances
    # for a meaningful 3-way split, not worth a special code path for.

    # Per-bag empirical diagnostic fraction from THIS SAME bootstrap split --
    # used to seed pi_b consistently with hard_labels, rather than a flat
    # pi_init that would disagree with what the M-step's labels already show
    # (a real, confirmed source of instability when the two disagreed: the
    # E-step's prior and the M-step's labels pulling pi_b in different
    # directions across iterations).
    pi_b_init = {}
    for b in set(bag_ids.tolist()):
        idx = np.where(bag_ids == b)[0]
        frac = float(is_diagnostic[idx].mean())
        pi_b_init[b] = float(np.clip(frac, pi_min, 1 - pi_min))

    return hard_labels, pi_b_init


def _hard_labels_from_pi_b(gamma, y_bag_per_instance, bag_ids, pi_b):
    """
    Per-bag top-round(pi_b[bag] * n_b) selection by diagnostic-state
    responsibility becomes the hard "diagnostic" label for the NEXT M-step;
    every other instance in the bag is hard-labelled to its own best-scoring
    null component. This mirrors mi-SVM's own successful per-bag top-k
    relabeling (src/mil/misvm.py's ``_fit_one_vs_rest``) instead of an
    unconstrained global argmax over gamma.

    An earlier version used plain ``gamma.argmax(axis=1)`` here and it
    reliably drifted to pi_b -> 0.98 on well-separated synthetic data within a
    few iterations (confirmed empirically -- see
    tests/test_stage4_mil.py): once the RF confidently predicts "diagnostic"
    for most instances, hard-EM's next round reinforces that same labelling
    with nothing pulling it back -- exactly the "collapse to the naive
    solution" mode docs/MIL_REFRAME_PLAN.md §5 names. Tying the discrete
    M-step label count directly to the fitted pi_b removes that fixed point:
    the label count can only be as large as the CURRENT pi_b estimate allows,
    so runaway self-reinforcement has no channel to act through.
    """
    n = len(y_bag_per_instance)
    diag_state = diagnostic_state(y_bag_per_instance)
    hard_labels = np.zeros(n, dtype=int)
    confidence = np.zeros(n, dtype=np.float64)
    bag_ids = np.asarray(bag_ids)

    for b in set(bag_ids.tolist()):
        idx = np.where(bag_ids == b)[0]
        n_b = len(idx)
        this_diag_state = int(diag_state[idx[0]])
        k = int(np.clip(round(pi_b[b] * n_b), 0, n_b))

        diag_scores = gamma[idx, this_diag_state]
        order = np.argsort(diag_scores)
        diag_idx = idx[order[n_b - k:]] if k > 0 else np.array([], dtype=int)
        null_idx = idx[order[:n_b - k]]

        hard_labels[diag_idx] = this_diag_state
        confidence[diag_idx] = gamma[diag_idx, this_diag_state]

        if len(null_idx) > 0:
            null_state_choice = gamma[null_idx, :N_NULL_COMPONENTS].argmax(axis=1)
            hard_labels[null_idx] = null_state_choice
            confidence[null_idx] = gamma[null_idx, :N_NULL_COMPONENTS].max(axis=1)

    return hard_labels, confidence


def _e_step(class_scores, y_bag_per_instance, pi_b_per_instance, n_classes):
    """
    E-step: responsibility gamma_i(z) over the 4 states model (C) allows for
    instance i's bag (M null components + its own diagnostic class), mixing
    the bag's current pi_b (prior weight on "diagnostic") with the
    discriminative model's own class_scores, renormalised.
    """
    n = class_scores.shape[0]
    gamma = np.zeros((n, n_states(n_classes)), dtype=np.float64)
    diag_state = diagnostic_state(y_bag_per_instance)

    null_prior = ((1.0 - pi_b_per_instance) / N_NULL_COMPONENTS)[:, None]
    gamma[:, :N_NULL_COMPONENTS] = null_prior * class_scores[:, :N_NULL_COMPONENTS]
    gamma[np.arange(n), diag_state] = pi_b_per_instance * class_scores[np.arange(n), diag_state]

    row_sums = gamma.sum(axis=1, keepdims=True)
    return gamma / np.where(row_sums > 0, row_sums, 1.0)


def _update_pi_b(gamma, y_bag_per_instance, bag_ids, mu, sigma, pi_min, pi_max=0.98):
    """
    Per-bag pi_b update: raw ML estimate (mean responsibility on the bag's own
    diagnostic state) shrunk toward the population logit-mean mu, weighted by
    bag size n_b / (n_b + kappa) -- larger bags trust their own data more
    (empirical-Bayes partial pooling; see module docstring). Floored/ceilinged
    at [pi_min, pi_max]: pi_min is §1's identifiability floor; pi_max guards
    the mirror-image collapse (pi_b -> 1, §5).
    """
    diag_state = diagnostic_state(y_bag_per_instance)
    own_mass = gamma[np.arange(len(gamma)), diag_state]
    bag_ids = np.asarray(bag_ids)

    kappa = max(sigma, 1e-3) ** -2 * 4.0  # larger sigma (more real bag-to-bag
                                          # spread already) -> less shrinkage
    pi_b_new = {}
    for b in sorted(set(bag_ids.tolist())):
        idx = np.where(bag_ids == b)[0]
        n_b = len(idx)
        raw = float(np.clip(own_mass[idx].mean(), 1e-6, 1 - 1e-6))
        w_b = n_b / (n_b + kappa)
        logit_shrunk = w_b * np.log(raw / (1 - raw)) + (1 - w_b) * mu
        pi_b_new[b] = float(np.clip(1.0 / (1.0 + np.exp(-logit_shrunk)), pi_min, pi_max))
    return pi_b_new


def fit_em(X, bag_ids, y_bag_per_instance, n_classes, tissue_cluster=None, pi_min=0.02,
          pi_init=0.15, max_iters=20, tol=1e-3, seed=42, rf_kwargs=None, verbose=True):
    """
    Fit the constrained-model EM (Tier 2).

    Returns dict:
      'posterior'  (n, K+1), collapsed from the (M+K) state space
      'pi_b'       {bag_id: pi_b}, final per-bag diagnostic fraction
      'mu', 'sigma' population logit(pi_b) parameters, final iteration
      'shrinkage'  §1's shrinkage-check dict (src.mil.common.shrinkage_diagnostic)
      'null_collapse' §1/§5's null-component-collapse dict
                   (src.mil.common.null_component_collapse_diagnostic)
      'model'      final RandomForestClassifier (M+K-way)
      'history'    per-iteration convergence log
    """
    n = X.shape[0]
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    bag_ids = np.asarray(bag_ids)

    # Cold start: see _cold_start_hard_labels's docstring for why a plain
    # 100%-diagnostic naive init or a soft prior-blended init both failed in
    # practice. tissue_cluster is accepted but not used for the cold start
    # under this design (mi-SVM's own split is used instead); using
    # tissue_cluster as an RF feature or a later-round tie-break is a
    # reasonable refinement, not implemented here.
    hard_labels, pi_b = _cold_start_hard_labels(X, bag_ids, y_bag_per_instance, n_classes,
                                                pi_init, pi_min, seed)
    confidence = np.ones(n, dtype=np.float64)

    logit_pi_b_init = np.array([np.log(p / (1 - p)) for p in pi_b.values()])
    mu = float(logit_pi_b_init.mean())
    sigma = float(logit_pi_b_init.std()) if len(logit_pi_b_init) > 1 else 1.0

    rf_kwargs = rf_kwargs or {}
    model = None
    history = []
    gamma = None

    for iteration in range(max_iters):
        # --- M-step: hard pseudo-labels + confidence sample_weight ---
        model = RandomForestClassifier(
            n_estimators=rf_kwargs.get('n_estimators', 200),
            max_depth=rf_kwargs.get('max_depth', None),
            class_weight='balanced_subsample',
            random_state=seed, n_jobs=-1,
        )
        model.fit(X, hard_labels, sample_weight=confidence)

        # RF may not observe every one of the n_states classes (small null
        # components especially) -- align its output columns back onto the
        # full state axis so downstream indexing stays state-index-safe.
        raw_proba = model.predict_proba(X)
        class_scores = np.full((n, n_states(n_classes)), 1e-6, dtype=np.float64)
        for col, cls in enumerate(model.classes_):
            class_scores[:, cls] = np.clip(raw_proba[:, col], 1e-6, None)

        # --- E-step ---
        pi_b_per_instance = np.array([pi_b[b] for b in bag_ids])
        new_gamma = _e_step(class_scores, y_bag_per_instance, pi_b_per_instance, n_classes)

        pi_b = _update_pi_b(new_gamma, y_bag_per_instance, bag_ids, mu, sigma, pi_min)
        logit_pi_b = np.array([np.log(p / (1 - p)) for p in pi_b.values()])
        mu = float(logit_pi_b.mean())
        sigma = float(logit_pi_b.std()) if len(logit_pi_b) > 1 else sigma

        delta = float(np.abs(new_gamma - gamma).mean()) if gamma is not None else float('inf')
        history.append({'iteration': iteration, 'mean_gamma_delta': delta, 'mu': mu,
                        'sigma': sigma, 'mean_pi_b': float(np.mean(list(pi_b.values())))})
        gamma = new_gamma
        hard_labels, confidence = _hard_labels_from_pi_b(gamma, y_bag_per_instance, bag_ids, pi_b)
        if verbose:
            print(f"    [EM] iter {iteration}: mean|delta gamma|={delta:.5f}  "
                  f"mu={mu:.3f} sigma={sigma:.3f} mean(pi_b)={history[-1]['mean_pi_b']:.3f}")
        if delta < tol:
            break

    posterior = posterior_to_K1(gamma, n_classes)
    shrinkage = shrinkage_diagnostic(pi_b, mu, sigma)
    null_collapse = null_component_collapse_diagnostic(X, gamma)

    return {
        'posterior': posterior, 'pi_b': pi_b, 'mu': mu, 'sigma': sigma,
        'shrinkage': shrinkage, 'null_collapse': null_collapse,
        'model': model, 'history': history,
    }


def score_em(fit_result, X, bag_ids, y_bag_per_instance, n_classes):
    """Score new instances with an already-fitted EM result -- one E-step at
    the converged model, using each instance's own bag's fitted pi_b when
    known (in-sample bags), else the population mean pi_b (out-of-sample bags,
    e.g. a fold's held-out validation slides -- the leakage-safe case
    docs/MIL_REFRAME_PLAN.md §3 requires: a bag never used to fit this pi_b
    still gets a defensible prior instead of an in-sample value it has no
    right to)."""
    model = fit_result['model']
    raw_proba = model.predict_proba(X)
    class_scores = np.full((X.shape[0], n_states(n_classes)), 1e-6, dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        class_scores[:, cls] = np.clip(raw_proba[:, col], 1e-6, None)

    mean_pi_b = float(np.mean(list(fit_result['pi_b'].values())))
    bag_ids = np.asarray(bag_ids)
    pi_b_per_instance = np.array([fit_result['pi_b'].get(b, mean_pi_b) for b in bag_ids])

    gamma = _e_step(class_scores, np.asarray(y_bag_per_instance).astype(int),
                    pi_b_per_instance, n_classes)
    return posterior_to_K1(gamma, n_classes)
