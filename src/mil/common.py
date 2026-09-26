"""
Shared MIL formulation utilities (docs/MIL_REFRAME_PLAN.md §1, §4).

Every formulation here implements Model (C), the constrained model: each
instance's latent tissue type z_i is restricted to {null_1..null_M, Y_b} for
its own bag's label Y_b -- never another diagnostic class. M=3 non-diagnostic
components, shared across all bags of every class (sharing is what identifies
them -- §5's degeneracy table).

Model (M), the unconstrained mixture where z_i ranges over every diagnostic
class (admits intratumoral heterogeneity), is Stage 8 and not implemented
here. Both share one (M+K)-way state-index convention (see ``state_labels``),
so extending (C) to (M) later means relaxing a mask, not rewriting the state
representation.
"""
import numpy as np

N_NULL_COMPONENTS = 3  # M in the plan -- stroma/necrosis/normal-brain, shared across bags


def n_states(n_classes):
    """Total (M+K) discriminative state count for a K-class diagnostic task."""
    return N_NULL_COMPONENTS + n_classes


def diagnostic_state(y_bag):
    """The state index (>= N_NULL_COMPONENTS) a bag labelled y_bag maps to."""
    return N_NULL_COMPONENTS + np.asarray(y_bag).astype(int)


def allowed_states_mask(y_bag_per_instance, n_classes):
    """
    (n_instances, M+K) boolean mask: which states each instance may take under
    model (C) -- always all M null components, plus only its own bag's
    diagnostic state (docs/MIL_REFRAME_PLAN.md §1: "a slide graded k contains
    only non-diagnostic tissue plus grade-k tumour").
    """
    y_bag_per_instance = np.asarray(y_bag_per_instance)
    n = len(y_bag_per_instance)
    mask = np.zeros((n, n_states(n_classes)), dtype=bool)
    mask[:, :N_NULL_COMPONENTS] = True
    mask[np.arange(n), diagnostic_state(y_bag_per_instance)] = True
    return mask


def posterior_to_K1(state_scores, n_classes):
    """
    Collapse an (n, M+K) state-score/posterior array to the (n, K+1) shape the
    latent store and downstream consumers (T2/T3, CbR, Tier 0's own pooling)
    expect: column 0 is the combined non-diagnostic mass (summed over the M
    null components), columns 1..K are the K diagnostic classes. Renormalised
    to a valid probability vector per row.
    """
    state_scores = np.asarray(state_scores, dtype=np.float64)
    non_diag = state_scores[:, :N_NULL_COMPONENTS].sum(axis=1, keepdims=True)
    diag = state_scores[:, N_NULL_COMPONENTS:]
    out = np.concatenate([non_diag, diag], axis=1)
    row_sums = out.sum(axis=1, keepdims=True)
    return out / np.where(row_sums > 0, row_sums, 1.0)


def pi_b_per_bag(posterior_K1, y_bag_per_instance, bag_ids):
    """
    Per-bag diagnostic fraction pi_b (§1: pi_b = (1/n_b) sum_i [z_bi = Y_b]),
    the soft/expected version -- mean posterior mass on the bag's own
    diagnostic class, averaged over that bag's instances.

    Returns {bag_id: pi_b}.
    """
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    own_class_mass = posterior_K1[np.arange(len(bag_ids)), 1 + y_bag_per_instance]
    pi_b = {}
    for b in sorted(set(bag_ids.tolist())):
        idx = np.where(bag_ids == b)[0]
        pi_b[b] = float(own_class_mass[idx].mean())
    return pi_b


def shrinkage_diagnostic(pi_b_dict, mu, sigma):
    """
    §1's shrinkage check: compare the observed spread of fitted pi_b against
    the spread the prior alone would imply at (mu, sigma). Near-zero means the
    model is returning mu for every bag -- no per-bag signal -- and must be
    reported honestly rather than presented as a finding.

    Delta-method approximation: on the probability scale, sd(pi) ~= sigma *
    pi_bar * (1 - pi_bar), since d(sigmoid)/d(logit) = pi(1-pi) at the mean.
    """
    values = np.array(list(pi_b_dict.values()))
    observed_spread = float(values.std())
    pi_bar = float(1.0 / (1.0 + np.exp(-mu)))
    prior_implied_spread = float(sigma * pi_bar * (1 - pi_bar))
    ratio = observed_spread / max(prior_implied_spread, 1e-6)
    return {
        'observed_spread': observed_spread,
        'prior_implied_spread': prior_implied_spread,
        'shrinkage_ratio': ratio,
        'collapsed': ratio < 0.1,  # near-total shrinkage -> no per-bag signal
    }


def null_component_collapse_diagnostic(X, responsibilities, top_frac=0.1, min_n=10):
    """
    §1/§5's collapse diagnostic: pairwise cosine similarity between the M null
    components' mean feature vectors (estimated from each component's own
    top-responsibility instances), plus an effective-component count. Values
    near 1.0 mean two null components have collapsed into near-duplicates --
    a benign failure mode, but one that must be visible, not silent.
    """
    n = X.shape[0]
    k = max(int(top_frac * n), min_n)
    means = []
    for m in range(N_NULL_COMPONENTS):
        top_idx = np.argsort(responsibilities[:, m])[-k:]
        means.append(X[top_idx].mean(axis=0))
    means = np.array(means)
    norm = np.linalg.norm(means, axis=1, keepdims=True)
    means_normed = means / np.where(norm > 0, norm, 1.0)
    cos_sim = means_normed @ means_normed.T

    # Effective component count via the eigenvalue participation ratio of the
    # (M, M) cosine-similarity (Gram-like) matrix -- 1.0 if all M collapsed to
    # one direction, M if fully distinct/orthogonal.
    eigvals = np.linalg.eigvalsh(cos_sim)
    eigvals = np.clip(eigvals, 0, None)
    effective_n = float((eigvals.sum() ** 2) / max((eigvals ** 2).sum(), 1e-12))

    return {'pairwise_cosine': cos_sim, 'effective_n_components': effective_n}
