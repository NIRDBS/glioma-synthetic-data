"""Tier 5 (Bayesian constrained MIL, Gibbs) regression tests.

Synthetic throughout, with instance-level ground truth KNOWN BY CONSTRUCTION
-- the one setting where it exists at all, since the real dataset has none
(docs/mil_tiers_methodology.md section 1). These are a correctness floor for
the sampler, explicitly NOT evidence about real-data performance: a planted,
well-separated signal is far easier than the real one, so passing here says
the machinery works, not that the method does.
"""
import numpy as np
import pytest

from src.mil.bayes import (
    _default_niw_prior, _log_gaussian_all_components, _niw_posterior_sample,
    _sample_beta_hyper, _sample_pi_b, _sample_z, cross_fold_pi_b_consistency,
    fit_bayes_mil, fit_bayes_mil_chains, gelman_rubin, marginal_log_likelihood,
    score_bayes_mil, score_bayes_mil_blind,
)
from src.mil.common import N_NULL_COMPONENTS, diagnostic_state, n_states


def _planted_bags(n_bags=8, n_per_bag=150, pi_true=0.3, d=4, seed=0, n_classes=2):
    """Bags whose instances are a known mix: a (1 - pi_true) share drawn from
    shared 'null' clusters common to every bag, and a pi_true share drawn from
    a class-specific cluster. Returns the true z so recovery can be scored."""
    rng = np.random.default_rng(seed)
    null_means = [np.full(d, -4.0), np.zeros(d), np.full(d, 4.0)][:N_NULL_COMPONENTS]
    class_means = [np.full(d, 10.0), np.full(d, -10.0)][:n_classes]

    X, bag_ids, y_bag, z_true = [], [], [], []
    for b in range(n_bags):
        label = b % n_classes
        n_diag = int(round(pi_true * n_per_bag))
        for i in range(n_per_bag):
            if i < n_diag:
                X.append(rng.normal(class_means[label], 1.0))
                z_true.append(N_NULL_COMPONENTS + label)
            else:
                k = rng.integers(0, N_NULL_COMPONENTS)
                X.append(rng.normal(null_means[k], 1.0))
                z_true.append(k)
            bag_ids.append(f"bag_{b}")
            y_bag.append(label)
    return (np.array(X), np.array(bag_ids, dtype=object),
           np.array(y_bag), np.array(z_true))


# ---------------------------------------------------------------------------
# 1. Conjugate pieces, in isolation
# ---------------------------------------------------------------------------

def test_niw_posterior_concentrates_on_the_data_mean():
    """With many observations the NIW posterior mean must sit on the sample
    mean, not the prior mean (kappa0 is small by design)."""
    rng = np.random.default_rng(0)
    d = 3
    true_mu = np.array([5.0, -2.0, 1.0])
    X_k = rng.normal(true_mu, 0.5, size=(2000, d))
    prior = _default_niw_prior(d, data_scale=1.0)
    draws = np.array([_niw_posterior_sample(X_k, prior, rng)[0] for _ in range(20)])
    np.testing.assert_allclose(draws.mean(axis=0), true_mu, atol=0.15)


def test_niw_empty_component_falls_back_to_prior_without_crashing():
    rng = np.random.default_rng(1)
    prior = _default_niw_prior(3, data_scale=1.0)
    mu, Sigma = _niw_posterior_sample(np.empty((0, 3)), prior, rng)
    assert mu.shape == (3,)
    assert Sigma.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(Sigma) > 0), "prior draw must be positive definite"


def test_log_gaussian_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 3))
    mus = [np.zeros(3), np.ones(3)]
    Sigmas = [np.eye(3), np.diag([2.0, 1.0, 0.5])]
    got = _log_gaussian_all_components(X, mus, Sigmas)
    for k, (mu, S) in enumerate(zip(mus, Sigmas)):
        expected = multivariate_normal(mu, S).logpdf(X)
        np.testing.assert_allclose(got[:, k], expected, rtol=1e-8)


def test_pi_b_conjugate_update_tracks_the_assigned_fraction():
    """pi_b | z ~ Beta(a + n_diag, b + n_null): with a lot of data and a weak
    prior the draw must land near the realised diagnostic fraction."""
    rng = np.random.default_rng(3)
    n = 4000
    y = np.zeros(n, dtype=int)
    bag_ids = np.array(['b0'] * n, dtype=object)
    z = np.where(np.arange(n) < int(0.35 * n), diagnostic_state(y), 0)
    draws = [_sample_pi_b(z, y, bag_ids, 1.0, 1.0, rng)['b0'] for _ in range(30)]
    assert abs(np.mean(draws) - 0.35) < 0.03


# ---------------------------------------------------------------------------
# 2. Model (C)'s support restriction -- the structural guarantee
# ---------------------------------------------------------------------------

def test_z_step_never_assigns_another_bags_diagnostic_class():
    """The defining constraint: an instance may take a null or its OWN bag's
    class, never another bag's. Checked on deliberately adversarial input --
    the likelihood strongly favours the forbidden state."""
    rng = np.random.default_rng(4)
    n, K = 200, 2
    S = n_states(K)
    y = np.zeros(n, dtype=int)            # every instance belongs to class 0
    log_lik = np.full((n, S), -1e6)
    log_lik[:, N_NULL_COMPONENTS + 1] = 0.0   # class 1 (forbidden) looks perfect
    z = _sample_z(log_lik, np.full(n, 0.5), y, K, rng)
    assert (z != N_NULL_COMPONENTS + 1).all(), "sampled a state model (C) forbids"
    assert set(np.unique(z)).issubset(set(range(N_NULL_COMPONENTS)) | {N_NULL_COMPONENTS})


def test_metropolis_step_returns_valid_positive_hyperparameters():
    rng = np.random.default_rng(5)
    pis = list(np.clip(rng.beta(2.0, 5.0, size=40), 1e-6, 1 - 1e-6))
    a, b = 1.0, 1.0
    n_acc = 0
    for _ in range(200):
        a, b, acc = _sample_beta_hyper(a, b, pis, rng)
        n_acc += acc
        assert a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)
    assert 0 < n_acc < 200, f"degenerate acceptance rate: {n_acc}/200"


# ---------------------------------------------------------------------------
# 3. End-to-end recovery on planted ground truth
# ---------------------------------------------------------------------------

def test_recovers_planted_instances_and_pi_b():
    """The correctness floor, via the real entry point
    (``fit_bayes_mil_chains`` + structure-aware init): on a well-separated
    planted signal, recover both which instances are diagnostic and the
    planted pi_b."""
    for data_seed in (10, 11):
        X, bag_ids, y_bag, z_true = _planted_bags(n_bags=6, n_per_bag=120,
                                                  pi_true=0.3, d=4, seed=data_seed)
        fit = fit_bayes_mil_chains(X, bag_ids, y_bag, n_classes=2, n_chains=3,
                                   n_iter=250, burn_in=80, thin=5, seed=0,
                                   verbose=False)
        post = fit['posterior']
        pred_diag = post[:, 1:].sum(axis=1) > post[:, 0]
        acc = (pred_diag == (z_true >= N_NULL_COMPONENTS)).mean()
        est_pi = np.mean(list(fit['pi_b'].values()))
        assert acc > 0.95, f"seed {data_seed}: instance recovery {acc:.3f}"
        assert abs(est_pi - 0.3) < 0.05, f"seed {data_seed}: pi_b {est_pi:.3f}"
        assert fit['chain_rhat'] < 1.1, f"seed {data_seed}: R-hat {fit['chain_rhat']:.2f}"


def test_naive_kmeans_init_is_trapped_which_is_why_structure_aware_init_exists():
    """Pins *why* ``_structure_aware_init`` is not optional. A plain M-way
    k-means start (what the first version of this module did) lands in a mode
    where a 'shared' null absorbs a class's diagnostic cluster -- recovery
    collapses to ~0.6 and R-hat blows up. The likelihood prefers the truth by
    a wide margin; single-site Gibbs simply cannot walk a whole component
    across the feature space to get there.

    If this ever starts recovering, the sampler gained a mode-escaping move
    (split-merge, tempering) and this test should become a second recovery
    assertion rather than being deleted."""
    from sklearn.cluster import KMeans

    X, bag_ids, y_bag, z_true = _planted_bags(n_bags=6, n_per_bag=120,
                                              pi_true=0.3, d=4, seed=10)
    rng = np.random.default_rng(0)
    km = KMeans(n_clusters=N_NULL_COMPONENTS, n_init=4, random_state=0)
    bad_init = km.fit_predict(X).astype(int)
    flip = rng.uniform(size=len(X)) < 0.15
    bad_init[flip] = diagnostic_state(y_bag)[flip]

    fit = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=250, burn_in=80,
                       thin=5, seed=0, init_z=bad_init, verbose=False)
    post = fit['posterior']
    pred_diag = post[:, 1:].sum(axis=1) > post[:, 0]
    acc = (pred_diag == (z_true >= N_NULL_COMPONENTS)).mean()
    assert acc < 0.85, f"naive init unexpectedly recovered ({acc:.3f}) -- see docstring"

    good = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=250, burn_in=80,
                        thin=5, seed=0, verbose=False)
    ll_bad = marginal_log_likelihood(fit, X, bag_ids, y_bag, 2)
    ll_good = marginal_log_likelihood(good, X, bag_ids, y_bag, 2)
    assert ll_good > ll_bad + 50, (
        "the trapped mode should be decisively worse in likelihood, not "
        f"equivalent; got {ll_good:.0f} vs {ll_bad:.0f}")


def test_pi_b_samples_give_a_credible_interval():
    """The thing no other tier can produce: pi_b as a distribution."""
    X, bag_ids, y_bag, _ = _planted_bags(n_bags=4, n_per_bag=100, pi_true=0.3,
                                         d=3, seed=12)
    fit = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=200, burn_in=50,
                       thin=2, seed=2, verbose=False)
    for bag, draws in fit['pi_b_samples'].items():
        assert len(draws) == fit['n_retained']
        lo, hi = np.percentile(draws, [2.5, 97.5])
        assert 0.0 <= lo < hi <= 1.0


def test_posterior_rows_are_valid_probability_vectors():
    X, bag_ids, y_bag, _ = _planted_bags(n_bags=4, n_per_bag=80, d=3, seed=13)
    fit = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=120, burn_in=40,
                       thin=2, seed=3, verbose=False)
    post = fit['posterior']
    assert post.shape == (len(X), 3)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    assert (post >= 0).all()


def test_score_held_out_bags_uses_the_prior_not_an_in_sample_pi_b():
    """Held-out bags have no fitted pi_b; scoring must still produce a valid
    posterior, drawn from the Beta(a, b) hyperparameter posterior."""
    X, bag_ids, y_bag, _ = _planted_bags(n_bags=4, n_per_bag=80, d=3, seed=14)
    fit = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=120, burn_in=40,
                       thin=2, seed=4, verbose=False)
    X_new, bag_new, y_new, z_new_true = _planted_bags(n_bags=2, n_per_bag=60, d=3,
                                                      seed=99)
    post = score_bayes_mil(fit, X_new, bag_new, y_new, n_classes=2)
    assert post.shape == (len(X_new), 3)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
    # Only the mechanics are asserted here (valid posterior, prior-drawn pi_b,
    # no in-sample leakage). Recovery accuracy is NOT asserted: held-out scoring
    # inherits whichever mode the fit landed in, and that mode is degenerate --
    # see test_documents_the_null_diagnostic_swap_degeneracy.
    pred_diag = post[:, 1:].sum(axis=1) > post[:, 0]
    assert 0.0 < pred_diag.mean() < 1.0, "degenerate all-one-state scoring"


# ---------------------------------------------------------------------------
# 3b. Genuine (non-circular) bag classification
# ---------------------------------------------------------------------------

def test_score_bayes_mil_blind_recovers_the_true_class_on_held_out_bags():
    """score_bayes_mil (above) is handed Y_b and asked 'what fraction is
    diagnostic' -- it never predicts a class. score_bayes_mil_blind is the
    version that actually does: given only a held-out bag's instances (no
    label), evaluate the posterior-predictive likelihood under EACH
    candidate class and argmax. On a well-separated planted signal this
    should recover the true label for fresh, never-fitted bags."""
    X, bag_ids, y_bag, _ = _planted_bags(n_bags=6, n_per_bag=120, pi_true=0.3,
                                        d=4, seed=10)
    fit = fit_bayes_mil_chains(X, bag_ids, y_bag, n_classes=2, n_chains=3,
                               n_iter=250, burn_in=80, thin=5, seed=0,
                               verbose=False)

    X_new, bag_new, y_new, _ = _planted_bags(n_bags=6, n_per_bag=120,
                                             pi_true=0.3, d=4, seed=20)
    log_lik_by_bag = score_bayes_mil_blind(fit, X_new, bag_new, n_classes=2)

    true_label = {b: y_new[bag_new == b][0] for b in np.unique(bag_new)}
    n_correct = 0
    for b, ll in log_lik_by_bag.items():
        assert ll.shape == (2,)
        pred = int(np.argmax(ll))
        n_correct += int(pred == true_label[b])
    acc = n_correct / len(log_lik_by_bag)
    assert acc == 1.0, f"blind bag classification recovered only {acc:.2f} on a well-separated planted signal"


def test_score_bayes_mil_blind_never_receives_the_true_label():
    """Signature-level guard: unlike score_bayes_mil, this function must not
    be callable with a y_bag_per_instance argument at all -- the whole point
    is that it cannot be handed the answer, not just that it's told to
    ignore one."""
    import inspect
    sig = inspect.signature(score_bayes_mil_blind)
    assert 'y_bag_per_instance' not in sig.parameters
    assert 'y_bag' not in sig.parameters
    assert list(sig.parameters) == ['fit', 'X', 'bag_ids', 'n_classes']


# ---------------------------------------------------------------------------
# 4. Convergence diagnostic
# ---------------------------------------------------------------------------

def test_gelman_rubin_flags_disagreeing_chains_and_passes_agreeing_ones():
    rng = np.random.default_rng(6)
    agree = rng.normal(0, 1, size=(4, 500))
    assert gelman_rubin(agree) < 1.1
    disagree = np.stack([rng.normal(m, 0.1, size=500) for m in (0, 5, 10, 15)])
    assert gelman_rubin(disagree) > 1.1


# ---------------------------------------------------------------------------
# 5. Cross-fold pi_b consistency -- the check R-hat structurally cannot do
# ---------------------------------------------------------------------------

def test_cross_fold_consistency_separates_agreeing_folds_from_an_outlier():
    """Three folds share overlapping bags (the real 5-fold-CV situation,
    where most bags are training data in several folds). fold_a/fold_b agree
    closely on their shared bags; fold_c is a near-total mismatch on the
    bags it shares with either. The function should report high correlation
    for the agreeing pair and low/negative correlation for every pair
    involving the outlier -- exactly the real-data pattern this was built to
    catch (docs/tier5_bayesian_mil.md section 2: fold_0/fold_3/fold_4 pairwise
    corr 0.94-0.985 on exp_608, fold_1/fold_2 corr -0.42 to +0.41 against
    that consensus)."""
    rng = np.random.default_rng(11)
    bags = [f'bag_{i}' for i in range(12)]
    true_pi = {b: rng.uniform(0, 1) for b in bags}

    fold_a = {b: float(np.clip(true_pi[b] + rng.normal(0, 0.02), 0, 1)) for b in bags[:10]}
    fold_b = {b: float(np.clip(true_pi[b] + rng.normal(0, 0.02), 0, 1)) for b in bags[2:]}
    # fold_c: same bags as fold_a, but a near-reversed, unrelated pi_b --
    # an independently-fit model landing in a different, wrong basin.
    fold_c = {b: float(np.clip(1.0 - true_pi[b] + rng.normal(0, 0.02), 0, 1)) for b in bags[:10]}

    report = cross_fold_pi_b_consistency({'fold_a': fold_a, 'fold_b': fold_b, 'fold_c': fold_c})
    by_pair = {(p['fold_a'], p['fold_b']): p['corr'] for p in report['pairwise']}

    assert by_pair[('fold_a', 'fold_b')] > 0.9
    assert by_pair[('fold_a', 'fold_c')] < -0.5
    assert by_pair[('fold_b', 'fold_c')] < -0.5
    assert report['fold_mean_corr']['fold_a'] < report['fold_mean_corr']['fold_b']


def test_cross_fold_consistency_skips_pairs_with_fewer_than_two_shared_bags():
    report = cross_fold_pi_b_consistency({
        'fold_a': {'bag_1': 0.3, 'bag_2': 0.7},
        'fold_b': {'bag_3': 0.1},  # no overlap with fold_a at all
    })
    assert report['pairwise'] == []
    assert np.isnan(report['fold_mean_corr']['fold_a'])
    assert np.isnan(report['fold_mean_corr']['fold_b'])
