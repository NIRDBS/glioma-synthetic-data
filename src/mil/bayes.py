"""
Tier 5: a genuinely Bayesian constrained-MIL model, fit by MCMC (Gibbs).

Every other tier in this project is a point-estimate heuristic. Tier 2 in
particular *looks* like a latent-variable model but is not one: its E-step
multiplies a prior by a RandomForest's discriminative ``predict_proba`` and
renormalises (not Bayes' rule -- there is no joint density and no likelihood
that provably increases), its M-step takes hard pseudo-labels from a per-bag
top-k projection (a constraint projection, not inference), and its pi_b is an
empirical-Bayes point estimate with no uncertainty propagated
(docs/mil_tiers_methodology.md section 6, src/mil/em.py's own module docstring,
which names "a Laplace approximation or MCMC" as what it is approximating).

This module implements that missing model directly, so that "MIL rejected on
this data" can be a statement about the *formulation* rather than about four
heuristic fits of it.

THE MODEL (constrained model (C), same state space as Tiers 1/2)
----------------------------------------------------------------
For a bag b with known label Y_b, instance i, in a d-dimensional reduced
feature space (PCA -- see ``pipeline/mil_fit_stage.py``; NOT the CVAE latent,
deliberately, so this tier does not inherit the CVAE's own demonstrated
weaknesses):

    mu_k, Sigma_k ~ NIW(m0, kappa0, nu0, Psi0)      k in {0_1..0_M, 1..K}
    pi_b          ~ Beta(a, b)                       one per bag
    z_bi          ~ Cat(pi_b on state Y_b;  (1 - pi_b)/M on each null)
    x_bi | z_bi=k ~ N(mu_k, Sigma_k)

M = 3 null components, SHARED across every bag regardless of class (which is
what identifies them at all, section 1); the K diagnostic components are shared
across bags of that class. Model (C)'s hard support restriction is enforced in
the z-step: an instance may only occupy a null or its OWN bag's class.

THE POSTERIOR IS MULTIMODAL -- RUN MULTIPLE CHAINS (``fit_bayes_mil_chains``)
-----------------------------------------------------------------------------
This was diagnosed twice and got it wrong once, so the evidence is recorded
here rather than the conclusion alone.

On planted data with known instance labels, a single chain started from
k-means recovers only ~0.62 of the instances, because two components swap
roles: a "shared" null settles on class 1's diagnostic cluster while class
1's own component settles on a true null cluster. The first reading of that
was non-identifiability -- the argument being that a cluster occupied by only
one class's bags looks the same labelled either way.

**That argument is wrong**, and the decisive test was to start a chain AT the
planted truth. It stays there (recovery 1.000, pi_b 0.302 vs planted 0.300,
all five components on their planted means), and the marginal log-likelihood
of the two configurations differs by **241 nats** in favour of the truth. The
swapped configuration is not likelihood-equivalent; it is a distinctly worse
mode. The penalty is real but indirect: in the swapped mode one diagnostic
component has to straddle two well-separated clusters, and a single Gaussian
fits bimodal data badly.

So the model IS identified. What fails is the *sampler*: single-site Gibbs
cannot move an entire component across the feature space in one sweep, so it
is trapped in whichever basin its initialisation lands in. This is ordinary
MCMC multimodality, and the remedy is ordinary MCMC practice -- multiple
chains from dispersed starts, select by marginal likelihood, and report R-hat
so a trapped fit is visible rather than silent. ``fit_bayes_mil_chains`` does
exactly this; ``fit_bayes_mil`` is the single-chain primitive and should not
be used directly on real data.

This also means Tier 2's three collapse hacks (section 6.1's random tie-break,
per-bag top-k projection, cold-start floor) are better read as *mode control*
than as patches over an identifiability gap -- EM has the same multimodal
landscape and no way to detect being stuck in the wrong basin, which is
precisely the diagnostic this tier adds.

WHAT IT BUYS THAT NO OTHER TIER HAS
------------------------------------
Posterior *uncertainty*. pi_b comes back as a sample, not a number, so a bag's
diagnostic fraction has a credible interval; and standard MCMC diagnostics
(R-hat, ESS across chains) say whether the fit converged at all -- something no
other tier in this project can report.
"""
import numpy as np
from scipy.stats import invwishart

from src.mil.common import (
    N_NULL_COMPONENTS, allowed_states_mask, diagnostic_state, n_states,
    posterior_to_K1,
)


def _default_niw_prior(d, data_scale=1.0):
    """Weakly informative NIW: prior mean at the origin (features are expected
    PCA-centred), kappa0 small so the data dominates the mean, nu0 = d + 2 (the
    smallest value giving a finite InvWishart mean), Psi0 scaled to the data's
    own variance so the prior is not accidentally informative in absolute
    units."""
    return {
        'm0': np.zeros(d),
        'kappa0': 0.01,
        'nu0': d + 2.0,
        'Psi0': np.eye(d) * data_scale,
    }


def _niw_posterior_sample(X_k, prior, rng):
    """Conjugate NIW update from the instances currently assigned to one
    component, then one draw of (mu, Sigma). An empty component falls back to a
    prior draw -- it is not dropped, so the state space stays fixed across
    sweeps (a component may legitimately empty and refill)."""
    d = prior['m0'].shape[0]
    n_k = len(X_k)
    if n_k == 0:
        Sigma = invwishart.rvs(df=prior['nu0'], scale=prior['Psi0'], random_state=rng)
        Sigma = np.atleast_2d(Sigma)
        mu = rng.multivariate_normal(prior['m0'], Sigma / prior['kappa0'])
        return mu, Sigma

    xbar = X_k.mean(axis=0)
    dev = X_k - xbar
    S = dev.T @ dev

    kappa_n = prior['kappa0'] + n_k
    nu_n = prior['nu0'] + n_k
    m_n = (prior['kappa0'] * prior['m0'] + n_k * xbar) / kappa_n
    diff = (xbar - prior['m0'])[:, None]
    Psi_n = prior['Psi0'] + S + (prior['kappa0'] * n_k / kappa_n) * (diff @ diff.T)
    Psi_n = 0.5 * (Psi_n + Psi_n.T)  # enforce symmetry against float drift

    Sigma = np.atleast_2d(invwishart.rvs(df=nu_n, scale=Psi_n, random_state=rng))
    mu = rng.multivariate_normal(m_n, Sigma / kappa_n)
    return mu, Sigma


def _log_gaussian_all_components(X, mus, Sigmas):
    """(n, n_states) log N(x_i; mu_k, Sigma_k), via one Cholesky per component.
    Kept in log space throughout -- the z-step's weights span many orders of
    magnitude at 571-band-derived features."""
    from scipy.linalg import solve_triangular

    n, d = X.shape
    out = np.empty((n, len(mus)), dtype=np.float64)
    for k, (mu, Sigma) in enumerate(zip(mus, Sigmas)):
        L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(d))
        sol = solve_triangular(L, (X - mu).T, lower=True)
        maha = (sol ** 2).sum(axis=0)
        log_det = 2.0 * np.log(np.diag(L)).sum()
        out[:, k] = -0.5 * (maha + log_det + d * np.log(2.0 * np.pi))
    return out


def _sample_z(log_lik, pi_b_per_instance, y_bag_per_instance, n_classes, rng):
    """Categorical draw of z for every instance at once, restricted to model
    (C)'s allowed states, via Gumbel-max (one vectorised pass, no per-instance
    Python loop)."""
    n = log_lik.shape[0]
    S = n_states(n_classes)
    diag_state = diagnostic_state(y_bag_per_instance)

    log_prior = np.full((n, S), -np.inf)
    log_prior[:, :N_NULL_COMPONENTS] = np.log(
        np.clip((1.0 - pi_b_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
    log_prior[np.arange(n), diag_state] = np.log(np.clip(pi_b_per_instance, 1e-300, None))

    mask = allowed_states_mask(y_bag_per_instance, n_classes)
    log_w = log_prior + log_lik
    log_w = np.where(mask, log_w, -np.inf)

    gumbel = rng.gumbel(size=log_w.shape)
    return np.argmax(log_w + gumbel, axis=1)


def _sample_pi_b(z, y_bag_per_instance, bag_ids, a, b, rng):
    """pi_b | z ~ Beta(a + n_diag, b + n_null), conjugate, per bag."""
    diag_state = diagnostic_state(y_bag_per_instance)
    is_diag = (z == diag_state)
    bag_ids = np.asarray(bag_ids)
    pi_b = {}
    for bag in np.unique(bag_ids):
        m = bag_ids == bag
        n_diag = int(is_diag[m].sum())
        n_null = int(m.sum()) - n_diag
        pi_b[bag] = float(rng.beta(a + n_diag, b + n_null))
    return pi_b


def _log_beta_hyper_posterior(a, b, pi_values):
    """Unnormalised log posterior of the Beta hyperparameters given the current
    pi_b draws, with weakly informative Gamma(1, 1) priors on a and b."""
    from scipy.special import gammaln
    if a <= 0 or b <= 0:
        return -np.inf
    p = np.clip(np.asarray(pi_values), 1e-12, 1 - 1e-12)
    n = len(p)
    ll = n * (gammaln(a + b) - gammaln(a) - gammaln(b)) \
        + (a - 1.0) * np.log(p).sum() + (b - 1.0) * np.log1p(-p).sum()
    return ll - a - b  # Gamma(1,1) log-priors, up to a constant


def _sample_beta_hyper(a, b, pi_values, rng, step=0.2):
    """The one non-conjugate step: random-walk Metropolis on (log a, log b).
    Returns (a, b, accepted)."""
    cur = _log_beta_hyper_posterior(a, b, pi_values)
    a_new = float(np.exp(np.log(a) + rng.normal(0, step)))
    b_new = float(np.exp(np.log(b) + rng.normal(0, step)))
    prop = _log_beta_hyper_posterior(a_new, b_new, pi_values)
    # log-normal proposal is asymmetric in a/b; the Jacobian is the log-ratio.
    log_accept = prop - cur + (np.log(a_new) - np.log(a)) + (np.log(b_new) - np.log(b))
    if np.log(rng.uniform()) < log_accept:
        return a_new, b_new, True
    return a, b, False


def _structure_aware_init(X, y_bag_per_instance, n_classes, seed, rng):
    """
    Initialise z by clustering into M+K groups and assigning each group to a
    state by WHO USES IT: a cluster whose instances come overwhelmingly from
    one class's bags becomes that class's diagnostic component; the rest
    become nulls.

    This is the identifying structure of model (C) (nulls shared across all
    classes, diagnostics class-specific) applied as a starting point, not as a
    prior -- the sampler may leave it freely, and the module docstring's
    241-nat likelihood gap says the truth basin is where it should want to
    stay once it is in the right one.
    """
    from sklearn.cluster import KMeans

    S = n_states(n_classes)
    y = np.asarray(y_bag_per_instance).astype(int)
    km = KMeans(n_clusters=S, n_init=10, random_state=seed)
    cluster = km.fit_predict(X).astype(int)

    # class-usage profile of each cluster, normalised per class so that a
    # class with more instances overall does not dominate every cluster
    class_tot = np.array([(y == c).sum() for c in range(n_classes)], dtype=float)
    usage = np.zeros((S, n_classes))
    for k in range(S):
        for c in range(n_classes):
            usage[k, c] = ((cluster == k) & (y == c)).sum() / max(class_tot[c], 1)
    share = usage / np.clip(usage.sum(axis=1, keepdims=True), 1e-12, None)

    # greedily claim, for each class, the still-unclaimed cluster most
    # concentrated on it; everything left over is a null
    assigned, claimed = {}, set()
    for c in np.argsort(-share.max(axis=0)):           # most distinctive class first
        order = np.argsort(-share[:, c])
        for k in order:
            if k not in claimed:
                assigned[int(k)] = N_NULL_COMPONENTS + int(c)
                claimed.add(int(k))
                break
    null_slot = 0
    for k in range(S):
        if k not in claimed:
            assigned[k] = null_slot % N_NULL_COMPONENTS
            null_slot += 1

    z = np.array([assigned[c] for c in cluster], dtype=int)

    # an instance can only hold its own bag's diagnostic state; anything the
    # clustering put in another class's component falls back to a null
    diag_state = diagnostic_state(y)
    bad = (z >= N_NULL_COMPONENTS) & (z != diag_state)
    z[bad] = rng.integers(0, N_NULL_COMPONENTS, size=int(bad.sum()))
    return z


def fit_bayes_mil(X, bag_ids, y_bag_per_instance, n_classes, n_iter=1500, burn_in=500,
                 thin=5, seed=42, prior=None, init_z=None, verbose=True):
    """
    Gibbs sampler for the constrained-MIL model above.

    ``X`` must already be reduced (PCA) and reasonably centred -- a full-rank
    NIW in 571 dimensions is neither tractable nor identifiable at this sample
    size, so dimension reduction is a precondition, not an option.

    One sweep: z | mu,Sigma,pi  ->  mu_k,Sigma_k | z  ->  pi_b | z  ->  a,b | pi.
    The first three are conjugate and exact; only (a, b) uses Metropolis.

    Returns a dict with:
      'posterior'        (n, K+1) posterior-mean state probabilities, in the
                         shared store convention (column 0 = pooled null mass)
      'pi_b'             {bag: posterior mean}
      'pi_b_samples'     {bag: array of retained draws}  -- the credible
                         interval no other tier can produce
      'mus', 'Sigmas'    last retained draw (for scoring held-out bags)
      'mu_samples', 'Sigma_samples'  retained draws
      'a', 'b'           Beta hyperparameter posterior means
      'a_samples', 'b_samples'
      'accept_rate'      Metropolis acceptance for (a, b)
      'history'          per-sweep log for convergence inspection
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    S = n_states(n_classes)
    bag_ids = np.asarray(bag_ids)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    diag_state = diagnostic_state(y_bag_per_instance)

    if prior is None:
        prior = _default_niw_prior(d, data_scale=float(np.var(X, axis=0).mean()))

    # --- init -------------------------------------------------------------
    # Structure-aware init (see module docstring on multimodality). Plain
    # k-means into M nulls is reliably trapped: it has no way to tell a
    # diagnostic cluster from a null one, and single-site Gibbs cannot move a
    # whole component across the feature space afterwards.
    #
    # Instead, cluster into M+K groups and assign each group to a state using
    # the ONE thing that distinguishes nulls from diagnostics -- who uses it.
    # A cluster whose instances come overwhelmingly from one class's bags is
    # that class's diagnostic component; a cluster drawn from every class is a
    # null. That is the identifying assumption section 1 states but the
    # likelihood alone cannot enforce, used here where it is legitimate (as a
    # starting point the sampler is then free to leave) rather than as a prior.
    if init_z is None:
        z = _structure_aware_init(X, y_bag_per_instance, n_classes, seed, rng)
    else:
        z = np.asarray(init_z).astype(int)

    a, b = 1.0, 4.0  # prior mean pi_b = 0.2, deliberately close to EM's pi_init
    pi_b = {bag: a / (a + b) for bag in np.unique(bag_ids)}

    post_accum = np.zeros((n, S), dtype=np.float64)
    n_accum = 0
    pi_samples = {bag: [] for bag in np.unique(bag_ids)}
    mu_samples, Sigma_samples, a_samples, b_samples = [], [], [], []
    history, n_accept = [], 0

    for it in range(n_iter):
        # 1. mu_k, Sigma_k | z  (conjugate NIW)
        mus, Sigmas = [], []
        for k in range(S):
            mu_k, Sigma_k = _niw_posterior_sample(X[z == k], prior, rng)
            mus.append(mu_k)
            Sigmas.append(Sigma_k)

        # 2. z | mu, Sigma, pi  (exact categorical, model (C)-restricted)
        log_lik = _log_gaussian_all_components(X, mus, Sigmas)
        pi_per_instance = np.array([pi_b[bag] for bag in bag_ids])
        z = _sample_z(log_lik, pi_per_instance, y_bag_per_instance, n_classes, rng)

        # 3. pi_b | z  (conjugate Beta)
        pi_b = _sample_pi_b(z, y_bag_per_instance, bag_ids, a, b, rng)

        # 4. a, b | {pi_b}  (Metropolis -- the only non-conjugate step)
        a, b, acc = _sample_beta_hyper(a, b, list(pi_b.values()), rng)
        n_accept += int(acc)

        if it >= burn_in and (it - burn_in) % thin == 0:
            # Rao-Blackwellised: accumulate the z-step's own probabilities
            # rather than the hard draw -- lower variance for the same sweeps.
            log_prior = np.full((n, S), -np.inf)
            log_prior[:, :N_NULL_COMPONENTS] = np.log(
                np.clip((1.0 - pi_per_instance) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
            log_prior[np.arange(n), diag_state] = np.log(np.clip(pi_per_instance, 1e-300, None))
            lw = np.where(allowed_states_mask(y_bag_per_instance, n_classes),
                         log_prior + log_lik, -np.inf)
            lw -= lw.max(axis=1, keepdims=True)
            w = np.exp(lw)
            post_accum += w / w.sum(axis=1, keepdims=True)
            n_accum += 1
            for bag in pi_b:
                pi_samples[bag].append(pi_b[bag])
            mu_samples.append(np.array(mus))
            Sigma_samples.append(np.array(Sigmas))
            a_samples.append(a)
            b_samples.append(b)

        if verbose and (it % 100 == 0 or it == n_iter - 1):
            mean_pi = float(np.mean(list(pi_b.values())))
            print(f"    [bayes_mil] sweep {it:4d}/{n_iter}  mean_pi_b={mean_pi:.4f}  "
                 f"a={a:.3f} b={b:.3f}  n_diag={int((z == diag_state).sum())}/{n}")
        history.append({'iter': it, 'mean_pi_b': float(np.mean(list(pi_b.values()))),
                       'a': float(a), 'b': float(b),
                       'n_diagnostic': int((z == diag_state).sum())})

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
    }


def score_bayes_mil(fit, X, bag_ids, y_bag_per_instance, n_classes):
    """
    Score held-out instances by posterior predictive, averaging over the
    retained MCMC draws.

    A held-out bag has no fitted pi_b -- it was never in the training set --
    so pi_b is drawn from the Beta(a, b) hyperparameter posterior for each
    retained sweep, exactly as src/mil/em.py::score_em falls back to a prior
    rather than reusing an in-sample value it has no right to. This is the
    posterior predictive, not a point plug-in.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    S = n_states(n_classes)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    diag_state = diagnostic_state(y_bag_per_instance)
    mask = allowed_states_mask(y_bag_per_instance, n_classes)
    rng = np.random.default_rng(0)

    accum = np.zeros((n, S), dtype=np.float64)
    n_draws = len(fit['mu_samples'])
    for t in range(n_draws):
        mus, Sigmas = fit['mu_samples'][t], fit['Sigma_samples'][t]
        pi_draw = float(rng.beta(fit['a_samples'][t], fit['b_samples'][t]))
        log_lik = _log_gaussian_all_components(X, mus, Sigmas)
        log_prior = np.full((n, S), -np.inf)
        log_prior[:, :N_NULL_COMPONENTS] = np.log((1.0 - pi_draw) / N_NULL_COMPONENTS)
        log_prior[np.arange(n), diag_state] = np.log(pi_draw)
        lw = np.where(mask, log_prior + log_lik, -np.inf)
        lw -= lw.max(axis=1, keepdims=True)
        w = np.exp(lw)
        accum += w / w.sum(axis=1, keepdims=True)

    return posterior_to_K1(accum / max(n_draws, 1), n_classes)


def score_bayes_mil_blind(fit, X, bag_ids, n_classes):
    """
    Genuine K-way bag classification -- unlike ``score_bayes_mil``, Y_b is
    NEVER passed in. Every other scoring path in this project's
    Model (C)-constrained tiers (this function's own sibling above, and
    ``src/mil/em.py::score_em``) is handed the bag's already-known label and
    asked "given that, what fraction of these instances match it" -- never
    "which class is this." That is correct for estimating pi_b (the task
    this tier exists for, docs/tier5_bayesian_mil.md section 0.3) but means
    a bag-classification accuracy computed from it is circular (section
    4.0) -- this function is the non-circular alternative the project did
    not previously have: for each held-out bag and each CANDIDATE class k,
    it evaluates the posterior-predictive log-likelihood of that bag's own
    instances under "this bag's class is k" (the M nulls plus k's own
    diagnostic state -- still Model (C)'s state space, just evaluated once
    per candidate instead of being told which one is true) and predicts by
    argmax_k. A bag's true label is used only afterward, by the caller, to
    check the prediction -- never inside this function.

    Returns ``{bag_id: (n_classes,) log p(X_bag | Y_b=k)}`` for
    k=0..n_classes-1, each a posterior-predictive log-mean-exp over the
    fit's retained draws (same fresh-Beta(a,b)-redraw convention as
    ``score_bayes_mil``, since a held-out bag has no fitted pi_b either
    way -- and the SAME pi_draw is reused across both candidate classes
    within one retained draw, since pi_b is a property of the bag, not of
    which label is being hypothesised, keeping the two hypotheses' noise
    paired rather than independent).
    """
    from scipy.special import logsumexp

    X = np.asarray(X, dtype=np.float64)
    bag_ids = np.asarray(bag_ids)
    unique_bags, bag_codes = np.unique(bag_ids, return_inverse=True)
    n_bags = len(unique_bags)
    n_draws = len(fit['mu_samples'])
    rng = np.random.default_rng(0)

    # (n_draws, n_bags, n_classes): per-draw, per-bag, per-candidate-class
    # summed instance log-likelihood.
    draw_bag_class = np.zeros((n_draws, n_bags, n_classes), dtype=np.float64)

    for t in range(n_draws):
        mus, Sigmas = fit['mu_samples'][t], fit['Sigma_samples'][t]
        pi_draw = float(rng.beta(fit['a_samples'][t], fit['b_samples'][t]))
        log_lik = _log_gaussian_all_components(X, mus, Sigmas)  # (n, S)

        log_null_prior = np.log((1.0 - pi_draw) / N_NULL_COMPONENTS)
        log_diag_prior = np.log(pi_draw)
        # Null-state mass doesn't depend on the candidate class -- computed
        # once per draw, reused for every k below.
        null_lse = logsumexp(log_lik[:, :N_NULL_COMPONENTS] + log_null_prior, axis=1)

        for k in range(n_classes):
            diag_col = log_lik[:, N_NULL_COMPONENTS + k] + log_diag_prior
            # log p(x_i | Y_b=k, draw t) over exactly the M+1 states
            # {nulls, class k} -- logsumexp of 2 terms.
            log_px_k = np.logaddexp(null_lse, diag_col)
            draw_bag_class[t, :, k] = np.bincount(bag_codes, weights=log_px_k, minlength=n_bags)

    # Posterior predictive over retained draws: log-mean-exp along the draw axis.
    out_log = logsumexp(draw_bag_class, axis=0) - np.log(max(n_draws, 1))  # (n_bags, n_classes)
    return {b: out_log[i] for i, b in enumerate(unique_bags)}


def marginal_log_likelihood(fit, X, bag_ids, y_bag_per_instance, n_classes):
    """Observed-data log-likelihood at a fit's final parameter draw, with z
    marginalised out. Used to choose among chains that converged to different
    modes -- see ``fit_bayes_mil_chains``."""
    from scipy.special import logsumexp

    X = np.asarray(X, dtype=np.float64)
    y_bag_per_instance = np.asarray(y_bag_per_instance).astype(int)
    pis = np.array([fit['pi_b'][b] for b in np.asarray(bag_ids)])
    log_lik = _log_gaussian_all_components(X, fit['mus'], fit['Sigmas'])
    n, S = log_lik.shape

    log_prior = np.full((n, S), -np.inf)
    log_prior[:, :N_NULL_COMPONENTS] = np.log(
        np.clip((1.0 - pis) / N_NULL_COMPONENTS, 1e-300, None))[:, None]
    log_prior[np.arange(n), diagnostic_state(y_bag_per_instance)] = np.log(
        np.clip(pis, 1e-300, None))

    w = np.where(allowed_states_mask(y_bag_per_instance, n_classes),
                log_prior + log_lik, -np.inf)
    return float(logsumexp(w, axis=1).sum())


def fit_bayes_mil_chains(X, bag_ids, y_bag_per_instance, n_classes, n_chains=4,
                        seed=42, verbose=True, **kwargs):
    """
    Run several dispersed chains, keep the best by marginal log-likelihood, and
    report R-hat. **This is the entry point for real data** -- see the module
    docstring: a single chain is reliably trapped by this posterior's
    multimodality, and on planted data the trapped mode costs ~241 nats and
    drops instance recovery from 1.00 to 0.62.

    Returns the winning chain's fit dict, with three fields added:
      'chain_log_liks'  marginal log-likelihood of every chain
      'chain_rhat'      R-hat on mean pi_b across chains (>1.1 = chains
                        disagree, i.e. at least one is in a different mode --
                        the honest signal that the fit is not trustworthy)
      'n_chains'
    """
    fits, lls = [], []
    for c in range(n_chains):
        if verbose:
            print(f"  [bayes_mil] chain {c + 1}/{n_chains}")
        fit = fit_bayes_mil(X, bag_ids, y_bag_per_instance, n_classes,
                           seed=seed + 1000 * c, verbose=False, **kwargs)
        fits.append(fit)
        lls.append(marginal_log_likelihood(fit, X, bag_ids, y_bag_per_instance, n_classes))

    pi_traces = np.array([
        [np.mean([h['mean_pi_b'] for h in f['history'][i:i + 1]]) for i in range(len(f['history']))]
        for f in fits
    ])
    rhat = gelman_rubin(pi_traces[:, len(pi_traces[0]) // 2:])

    best = int(np.argmax(lls))
    if verbose:
        print(f"  [bayes_mil] chain log-liks: {[round(v, 1) for v in lls]}  "
             f"-> chain {best} selected;  R-hat(mean pi_b) = {rhat:.3f}"
             + ("  [WARNING: chains disagree, >1.1]" if rhat > 1.1 else ""))

    out = dict(fits[best])
    out['chain_log_liks'] = lls
    out['chain_rhat'] = rhat
    out['n_chains'] = n_chains
    return out


def gelman_rubin(chains):
    """R-hat for a scalar across chains, (n_chains, n_draws). >1.1 is the
    conventional non-convergence flag. Reported per fold so this tier can say
    whether its own fit converged -- no other tier in this project can."""
    chains = np.asarray(chains, dtype=np.float64)
    m, n = chains.shape
    if m < 2 or n < 2:
        return float('nan')
    chain_means = chains.mean(axis=1)
    W = chains.var(axis=1, ddof=1).mean()
    B = n * chain_means.var(ddof=1)
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else float('nan')


def cross_fold_pi_b_consistency(pi_b_by_fold):
    """
    Pairwise consistency, across folds, of each fold's own independently-fit
    pi_b estimate -- the check R-hat cannot do, since R-hat only compares a
    fold's chains against each other, never against another fold's
    independent fit (docs/tier5_bayesian_mil.md section 2).

    This checks agreement on the per-bag DIAGNOSTIC FRACTION pi_b (how much
    of a bag's own instances each fold's model thinks are in state Y_b), not
    a bag-level label -- every bag's class Y_b is already known and is never
    in dispute (docs/mil_tiers_methodology.md section 1). Two folds "agree"
    here if they independently arrive at a similar pi_b for a bag that is a
    TRAINING bag in both -- which, under 5-fold CV, is most bags, since a bag
    is held out in only 1 (or 2, for the duplicated bags) of 5 folds and is
    training data in the rest.

    ``pi_b_by_fold``: {fold_name: {bag_id: pi_b_mean}}, e.g. loaded directly
    from a real fit's ``bayes_diagnostics.json`` (one entry per fold, each
    fold's own ``pi_b_mean``).

    Returns a dict with:
      'pairwise'        list of {fold_a, fold_b, n_shared, corr,
                        mean_abs_diff}, one entry per unordered fold pair
                        with >=2 shared bags (a correlation needs variance in
                        both arms; pairs below 2 shared bags or with a
                        constant arm are skipped, not reported as a
                        misleading nan-adjacent number)
      'fold_mean_corr'  {fold: mean of its own corr entries above, across
                        every other fold it shares >=2 bags with} -- a
                        simple, no-threshold summary; this function does not
                        itself decide what counts as "an outlier" fold
    """
    import itertools

    folds = sorted(pi_b_by_fold.keys())
    pairwise = []
    per_fold_corrs = {f: [] for f in folds}
    for fa, fb in itertools.combinations(folds, 2):
        shared = sorted(set(pi_b_by_fold[fa]) & set(pi_b_by_fold[fb]))
        if len(shared) < 2:
            continue
        a = np.array([pi_b_by_fold[fa][bag] for bag in shared])
        b = np.array([pi_b_by_fold[fb][bag] for bag in shared])
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        mad = float(np.mean(np.abs(a - b)))
        pairwise.append({'fold_a': fa, 'fold_b': fb, 'n_shared': len(shared),
                        'corr': corr, 'mean_abs_diff': mad})
        per_fold_corrs[fa].append(corr)
        per_fold_corrs[fb].append(corr)

    fold_mean_corr = {f: (float(np.mean(v)) if v else float('nan'))
                     for f, v in per_fold_corrs.items()}
    return {'pairwise': pairwise, 'fold_mean_corr': fold_mean_corr}
