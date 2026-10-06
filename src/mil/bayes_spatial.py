"""
Tier 5, spatial extension: couples each instance's latent state to its
8-connected spatial neighbours (Potts-style pairwise term), on top of the
exact same Model (C)-constrained generative model as src/mil/bayes.py.

MOTIVATION (docs/tier5_bayesian_mil.md, "the one modeling decision left"):
every tier in this project (0 through 5) treats instances within a bag as
exchangeable -- z_bi is drawn i.i.d. given pi_b, with (row, col) used only
for POST-HOC spatial validation, never inside the generative model. If
real diagnostic tissue is spatially contiguous, an exchangeable mixture is
structurally incapable of finding it, however strong the underlying
spectral signal. This module is the one exchangeable-instance assumption
actually relaxed, by adding a single global coupling scalar J to the prior
over z: neighbouring instances in the SAME bag (never across bags -- each
bag is a physically separate slide) prefer to share a state.

J IS NOT INFERRED. Learning a Potts coupling strength needs its own
intractable-normalising-constant machinery (pseudo-likelihood,
contrastive divergence) -- out of scope here. J is a provisional, swept
constant, the same treatment this project already gives other provisional
constants (cbr_temperature, IARPLS_LAMBDA): fixed per run, reported, not
hidden. J=0 must exactly reduce to src/mil/bayes.py::fit_bayes_mil -- this
is tested directly (tests/test_bayes_spatial.py), not assumed.

EVERYTHING ELSE is reused unchanged from src/mil/bayes.py: the NIW
component prior/conjugate update, the Beta(a,b) hyperprior and its
Metropolis step, Model (C)'s hard support restriction, the
structure-aware initialisation. Only the z-step changes.

THE Z-STEP: single-site-exact checkerboard (4-colour) Gibbs
------------------------------------------------------------
z_bi are no longer conditionally independent given pi_b once neighbours
are coupled, so the vectorised one-shot categorical draw
src/mil/bayes.py uses no longer applies. Instead: 4-colour the 8-connected
grid (`four_color_grid` -- any actual king's-move neighbour changes
(row%2, col%2), so same-coloured sites are never adjacent, a pure graph
property checked directly, not assumed), then update each colour class as
one vectorised block: every site in a colour class is conditionally
independent of every other site in that SAME class given everything else
(including the other 3 colours' current values, whichever mix of
already-updated-this-sweep and not-yet-updated they are) -- a valid Gibbs
block update. Sweeping all 4 colours in sequence composes 4 valid kernels
into one valid full-sweep transition. This is the standard, provably
correct construction for Gibbs sampling on a graph coupled by a local
(nearest-neighbour) potential.

See docs/tier5_bayesian_mil.md for the parameter-count/collapse discussion
that motivated the J-as-fixed-constant and planted-data-collapse-test
design decisions here, and the test file for the planted contiguous-blob
recovery test and the collapse-at-high-J test this module's whole
justification rests on.
"""
import numpy as np
from scipy.special import logsumexp

from src.mil.common import (
    N_NULL_COMPONENTS, allowed_states_mask, diagnostic_state, n_states,
    posterior_to_K1,
)
from src.mil.bayes import (
    _default_niw_prior, _log_gaussian_all_components, _niw_posterior_sample,
    _sample_beta_hyper, _sample_pi_b, _structure_aware_init,
)


def four_color_grid(coords):
    """
    (n,) int color in {0,1,2,3}, color = 2*(row%2) + (col%2).

    Pure graph-theory fact, independent of any model: an 8-connected
    (king's-move) grid neighbour offset (dy, dx) in {-1,0,1}^2 \\ {(0,0)}
    always changes row%2 or col%2 (both can stay fixed only for the
    trivial (0,0) offset, since the only even value in {-1,0,1} is 0), so
    two sites sharing this colour index are never 8-adjacent. Tested
    directly against the adjacency matrix in tests/test_bayes_spatial.py,
    not assumed.

    ``coords``: (n, 2) int array, (col, row) per instance -- the same
    convention src/data/spatial.py::reproject uses.
    """
    coords = np.asarray(coords)
    cols, rows = coords[:, 0].astype(int), coords[:, 1].astype(int)
    return 2 * (rows % 2) + (cols % 2)


def build_spatial_adjacency(coords, bag_ids):
    """
    Sparse (n, n) boolean adjacency: instance i and j are connected iff
    they are in the SAME bag and at Chebyshev distance exactly 1 (one of
    the up-to-8 grid neighbours). Never an edge between two different
    bags -- each bag is a physically separate slide, so cross-bag
    "adjacency" is meaningless and would silently leak bag-to-bag
    structure that isn't there. A missing grid neighbour (filtered out
    upstream, e.g. by the 3x3 median-filter/intensity-threshold pipeline)
    simply isn't an edge -- not an error, not treated as a null neighbour.

    ``coords``: (n, 2) int array, (col, row) -- src/data/spatial.py's own
    convention. ``bag_ids``: (n,) array-like, one id per instance.

    Returns a ``scipy.sparse.csr_matrix`` of dtype float64 (1.0 = edge),
    ready to left-multiply against a (n, S) one-hot state matrix to get
    per-instance, per-state neighbour-agreement counts.
    """
    from scipy.sparse import csr_matrix

    coords = np.asarray(coords)
    bag_ids = np.asarray(bag_ids)
    n = len(coords)
    rows_idx, cols_idx = [], []

    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]

    for bag in np.unique(bag_ids):
        idx = np.where(bag_ids == bag)[0]
        bag_coords = coords[idx]
        lookup = {(int(c), int(r)): local for local, (c, r) in enumerate(bag_coords)}
        for local, (c, r) in enumerate(bag_coords):
            global_i = idx[local]
            for dy, dx in offsets:
                neighbor = lookup.get((c + dx, r + dy))
                if neighbor is not None:
                    rows_idx.append(global_i)
                    cols_idx.append(idx[neighbor])

    data = np.ones(len(rows_idx), dtype=np.float64)
    return csr_matrix((data, (rows_idx, cols_idx)), shape=(n, n))


def _sample_z_spatial(z, log_lik, pi_b_per_instance, y_bag_per_instance, n_classes,
                      adjacency, color, J, rng):
    """
    One full checkerboard sweep over all 4 colour classes, in place.
    Exact single-site Gibbs within each colour (see module docstring).
    Returns the SAME array object, mutated, for convenience.
    """
    n = log_lik.shape[0]
    S = n_states(n_classes)
    diag_state = diagnostic_state(y_bag_per_instance)
    mask = allowed_states_mask(y_bag_per_instance, n_classes)

    log_prior = np.full((n, S), -np.inf)
    log_prior[:, :N_NULL_COMPONENTS] = np.log(
        np.clip((1.0 - pi_b_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
    log_prior[np.arange(n), diag_state] = np.log(np.clip(pi_b_per_instance, 1e-300, None))

    for c in range(4):
        idx = np.where(color == c)[0]
        if len(idx) == 0:
            continue
        z_onehot = np.zeros((n, S), dtype=np.float64)
        z_onehot[np.arange(n), z] = 1.0
        neighbor_counts = adjacency[idx] @ z_onehot  # (len(idx), S)

        log_w = log_prior[idx] + log_lik[idx] + J * neighbor_counts
        log_w = np.where(mask[idx], log_w, -np.inf)
        gumbel = rng.gumbel(size=log_w.shape)
        z[idx] = np.argmax(log_w + gumbel, axis=1)
    return z


def _full_conditional_weights(z, log_lik, pi_b_per_instance, y_bag_per_instance,
                              n_classes, adjacency, J):
    """
    (n, S) normalised full-conditional p(z_i=k | everything else), using
    the CURRENT (fully-resolved-this-sweep) neighbour states for every
    instance at once -- the Rao-Blackwellised summary used by
    ``fit_bayes_mil_spatial``'s posterior accumulation. Matches exactly
    what the z-step conditions each site on, snapshotted after the sweep
    completes (a standard "mean-field snapshot" RB estimate for MRFs).
    """
    n = log_lik.shape[0]
    S = n_states(n_classes)
    diag_state = diagnostic_state(y_bag_per_instance)
    mask = allowed_states_mask(y_bag_per_instance, n_classes)

    log_prior = np.full((n, S), -np.inf)
    log_prior[:, :N_NULL_COMPONENTS] = np.log(
        np.clip((1.0 - pi_b_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
    log_prior[np.arange(n), diag_state] = np.log(np.clip(pi_b_per_instance, 1e-300, None))

    z_onehot = np.zeros((n, S), dtype=np.float64)
    z_onehot[np.arange(n), z] = 1.0
    neighbor_counts = adjacency @ z_onehot

    lw = np.where(mask, log_prior + log_lik + J * neighbor_counts, -np.inf)
    lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def _unnormalized_log_weight_at_z(z, log_lik, pi_b_per_instance, y_bag_per_instance,
                                  n_classes, adjacency, J):
    """
    (n,) UNNORMALISED log[prior(z_i) * N(x_i;.) * exp(J*neighbour
    agreement)] evaluated AT the realised ``z`` -- the pseudo-likelihood
    numerator, deliberately NOT divided by the per-instance normaliser the
    way ``_full_conditional_weights`` is.

    This distinction is the fix for a real bug found by testing before
    trusting this (tests/test_bayes_spatial.py): scoring candidate
    classes by the NORMALISED conditional probability of whichever state
    got chosen scored a confidently well-separated planted blob at 0%
    accuracy, because under the WRONG hypothesis the sampler can "explain
    away" a bag's diagnostic-looking instances with a mediocre-but-best-
    available null state and still report high confidence in that local
    choice -- confidence in a forced choice says nothing about whether
    that choice fits the data well in absolute terms. This unnormalised
    version preserves the actual fit-quality gap between hypotheses that
    the normalised version discards.
    """
    n = log_lik.shape[0]
    S = n_states(n_classes)
    diag_state = diagnostic_state(y_bag_per_instance)

    log_prior = np.full((n, S), -np.inf)
    log_prior[:, :N_NULL_COMPONENTS] = np.log(
        np.clip((1.0 - pi_b_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
    log_prior[np.arange(n), diag_state] = np.log(np.clip(pi_b_per_instance, 1e-300, None))

    z_onehot = np.zeros((n, S), dtype=np.float64)
    z_onehot[np.arange(n), z] = 1.0
    neighbor_counts = adjacency @ z_onehot

    log_w = log_prior + log_lik + J * neighbor_counts
    return log_w[np.arange(n), z]


def fit_bayes_mil_spatial(X, bag_ids, y_bag_per_instance, coords, n_classes, J,
                          n_iter=600, burn_in=200, thin=5, seed=42, prior=None,
                          init_z=None, verbose=True):
    """
    Same model/return contract as ``src/mil/bayes.py::fit_bayes_mil``, plus
    a fixed Potts coupling ``J`` on the 8-connected within-bag spatial
    adjacency. ``J=0`` must reproduce ``fit_bayes_mil`` exactly (tested
    directly). ``coords``: (n, 2) int array, (col, row) per instance,
    aligned 1:1 with ``X``/``bag_ids``/``y_bag_per_instance``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    S = n_states(n_classes)
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)

    if prior is None:
        prior = _default_niw_prior(d, data_scale=float(np.var(X, axis=0).mean()))

    adjacency = build_spatial_adjacency(coords, bag_ids)
    color = four_color_grid(coords)

    if init_z is None:
        z = _structure_aware_init(X, y_bag_per_instance, n_classes, seed, rng)
    else:
        z = np.asarray(init_z).astype(int).copy()

    a, b = 1.0, 4.0
    pi_b = {bag: a / (a + b) for bag in np.unique(bag_ids)}

    post_accum = np.zeros((n, S), dtype=np.float64)
    n_accum = 0
    pi_samples = {bag: [] for bag in np.unique(bag_ids)}
    mu_samples, Sigma_samples, a_samples, b_samples = [], [], [], []
    history, n_accept = [], 0

    for it in range(n_iter):
        mus, Sigmas = [], []
        for k in range(S):
            mu_k, Sigma_k = _niw_posterior_sample(X[z == k], prior, rng)
            mus.append(mu_k)
            Sigmas.append(Sigma_k)

        log_lik = _log_gaussian_all_components(X, mus, Sigmas)
        pi_per_instance = np.array([pi_b[bag] for bag in bag_ids])
        z = _sample_z_spatial(z, log_lik, pi_per_instance, y_bag_per_instance,
                             n_classes, adjacency, color, J, rng)

        pi_b = _sample_pi_b(z, y_bag_per_instance, bag_ids, a, b, rng)
        a, b, acc = _sample_beta_hyper(a, b, list(pi_b.values()), rng)
        n_accept += int(acc)

        if it >= burn_in and (it - burn_in) % thin == 0:
            w = _full_conditional_weights(z, log_lik, pi_per_instance,
                                         y_bag_per_instance, n_classes, adjacency, J)
            post_accum += w
            n_accum += 1
            for bag in pi_b:
                pi_samples[bag].append(pi_b[bag])
            mu_samples.append(np.array(mus))
            Sigma_samples.append(np.array(Sigmas))
            a_samples.append(a)
            b_samples.append(b)

        if verbose and (it % 100 == 0 or it == n_iter - 1):
            mean_pi = float(np.mean(list(pi_b.values())))
            diag_state = diagnostic_state(y_bag_per_instance)
            print(f"    [bayes_mil_spatial J={J}] sweep {it:4d}/{n_iter}  "
                 f"mean_pi_b={mean_pi:.4f}  a={a:.3f} b={b:.3f}  "
                 f"n_diag={int((z == diag_state).sum())}/{n}")
        history.append({'iter': it, 'mean_pi_b': float(np.mean(list(pi_b.values()))),
                       'a': float(a), 'b': float(b)})

    state_posterior = post_accum / max(n_accum, 1)
    return {
        'posterior': posterior_to_K1(state_posterior, n_classes),
        'state_posterior': state_posterior,
        'pi_b': {bag: float(np.mean(v)) for bag, v in pi_samples.items()},
        'pi_b_samples': {bag: np.array(v) for bag, v in pi_samples.items()},
        'mus': mus, 'Sigmas': Sigmas,
        'mu_samples': mu_samples, 'Sigma_samples': Sigma_samples,
        'a': float(np.mean(a_samples)) if a_samples else a,
        'b': float(np.mean(b_samples)) if b_samples else b,
        'a_samples': np.array(a_samples), 'b_samples': np.array(b_samples),
        'accept_rate': n_accept / max(n_iter, 1),
        'n_retained': n_accum,
        'history': history,
        'J': J,
        'z_final': z,
    }


def score_bayes_mil_spatial_blind(fit, X, bag_ids, coords, n_classes, J,
                                  n_mini_iter=150, n_mini_burn=50, seed=0):
    """
    Genuine K-way bag classification under the spatial model -- never
    given the true label, exactly like
    ``src/mil/bayes.py::score_bayes_mil_blind``, but the spatial coupling
    removes that function's closed-form per-instance marginalisation (no
    exact way to integrate out a coupled z analytically). This is
    therefore an APPROXIMATION, not an exact posterior-predictive
    computation: for each held-out bag and each candidate class k, run a
    short checkerboard-Gibbs chain restricted to {nulls, class k} on that
    bag's own (isolated -- no edges leave one bag) subgraph, using the
    fit's own posterior-mean (mu, Sigma) and a pi_b drawn fresh from
    Beta(a, b), then score the hypothesis by the mean PSEUDO-log-likelihood
    of the final retained sweeps: sum_i log p(z_i=hat_z_i | neighbours,
    x_i) at the sampled configuration. Predict by argmax_k. This is
    Besag's pseudo-likelihood, the standard tractable substitute for an
    MRF's intractable normalising constant -- documented here as an
    approximation precisely so it is never read as the exact marginal
    likelihood ``score_bayes_mil_blind`` computes for the non-spatial
    model.

    Returns ``{bag_id: (n_classes,) array of approximate log p(X_bag | Y_b=k)}``.
    """
    X = np.asarray(X, dtype=np.float64)
    bag_ids = np.asarray(bag_ids)
    coords = np.asarray(coords)
    rng = np.random.default_rng(seed)

    mu_bar = np.mean(fit['mu_samples'], axis=0)
    Sigma_bar = np.mean(fit['Sigma_samples'], axis=0)
    a_bar, b_bar = fit['a'], fit['b']

    out = {}
    for bag in np.unique(bag_ids):
        m = bag_ids == bag
        X_b = X[m]
        coords_b = coords[m]
        n_b = X_b.shape[0]
        adjacency_b = build_spatial_adjacency(coords_b, np.zeros(n_b, dtype=int))
        color_b = four_color_grid(coords_b)
        log_lik_b = _log_gaussian_all_components(X_b, mu_bar, Sigma_bar)

        ll_per_class = np.full(n_classes, np.nan)
        for k in range(n_classes):
            y_hyp = np.full(n_b, k, dtype=int)
            pi_draw = float(rng.beta(a_bar, b_bar))
            pi_per_instance = np.full(n_b, pi_draw)

            # Warm start at the per-instance MAP (likelihood + pi_b prior,
            # ignoring the spatial term) -- NOT a random/all-null start.
            # With J>0, a site can only ever reach the diagnostic state by
            # copying a neighbour already there; starting every site at a
            # null state gives the diagnostic state zero neighbour support
            # anywhere, for BOTH candidate hypotheses equally, which turns
            # the whole comparison into noise (confirmed: this exact bug
            # produced worse-than-chance accuracy on an otherwise
            # well-separated planted blob before this fix -- see
            # tests/test_bayes_spatial.py's test history/docs/tier5_bayesian_mil.md).
            diag_state_k = N_NULL_COMPONENTS + k
            mask_k = allowed_states_mask(y_hyp, n_classes)
            log_prior_k = np.full((n_b, n_states(n_classes)), -np.inf)
            log_prior_k[:, :N_NULL_COMPONENTS] = np.log(
                np.clip((1.0 - pi_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
            log_prior_k[np.arange(n_b), diag_state_k] = np.log(np.clip(pi_per_instance, 1e-300, None))
            log_w0 = np.where(mask_k, log_prior_k + log_lik_b, -np.inf)
            z_b = np.argmax(log_w0, axis=1)

            pseudo_lls = []
            for it in range(n_mini_iter):
                z_b = _sample_z_spatial(z_b, log_lik_b, pi_per_instance, y_hyp,
                                        n_classes, adjacency_b, color_b, J, rng)
                if it >= n_mini_burn:
                    # UNNORMALISED log-weight at the realised z (see that
                    # function's docstring for why -- not the normalised
                    # conditional probability of whichever state got chosen).
                    log_w_at_z = _unnormalized_log_weight_at_z(
                        z_b, log_lik_b, pi_per_instance, y_hyp, n_classes, adjacency_b, J)
                    pseudo_lls.append(float(log_w_at_z.sum()))
            ll_per_class[k] = float(np.mean(pseudo_lls)) if pseudo_lls else float('nan')

        out[bag] = ll_per_class
    return out
