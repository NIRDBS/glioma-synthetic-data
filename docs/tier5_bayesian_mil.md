# Tier 5 — Bayesian MIL (Gibbs sampling) — Findings

Results companion to docs/mil_tiers_methodology.md §8 (methodology only, no
numbers there). Real fit: `exp_608` (β=0.5 CVAE, PCA-10 instance features),
5 CV folds, 3 dispersed chains per fold, 600 iterations/chain, 200 burn-in,
thin 5 (→ 80 retained draws, the best chain by marginal log-likelihood kept
per fold — `pipeline/mil_fit_stage.py`'s actual `bayes_cfg` defaults;
`src/mil/bayes.py`'s own module-docstring example uses illustrative, larger
numbers (1500/500) that were never the real-run setting). Code:
`src/mil/bayes.py`; diagnostics persisted at
`results/exp_608/latent/bayes_diagnostics.json`.

> **Reproducing this.** `pytest tests/test_bayes_mil.py` (14 tests, synthetic
> planted-ground-truth fakes — no real CVAE weights needed). Real fit:
> `python -m pipeline.mil_fit_stage --config exp_608 --methods bayes_mil`.
> Bag-level eval: `python scripts/stage6_bag_eval.py --config exp_608
> --method bayes_mil --variant gibbs --reference-model cnn` (also `apollo`,
> `rf`, `xgb`). Spatial validation:
> `python scripts/stage5_spatial_validation.py --config exp_608 --method
> bayes_mil`. Cross-fold consistency (§2):
> `python scripts/tier5_cross_fold_consistency.py --config exp_608`.
> Per-fold bag accuracy (§4.1):
> `python scripts/tier5_per_fold_accuracy.py --config exp_608`.
> Genuine, non-circular bag classification (§4.3 — refits each fold's
> chains, ~minutes, not instant):
> `python scripts/tier5_blind_bag_classification.py --config exp_608`.

## 0. The model, precisely

Prose description for a general reader: docs/mil_tiers_methodology.md §8.
This section is the full generative model, prior, and update equations,
for transcription into a paper's Methods section — every symbol matches
`src/mil/bayes.py`'s own variable names.

### 0.1 Preprocessing (outside this model)

Each instance's raw spectrum (571 bands, fingerprint region 700–1800 cm⁻¹)
is reduced by PCA to $d=10$ dimensions, fit on that fold's TRAINING
instances only and applied to both train and validation
(`pipeline/mil_fit_stage.py::_fit_one_fold_bayes_mil`, not part of
`src/mil/bayes.py` itself). Write $\mathbf{x}_{bi} \in \mathbb{R}^d$ for
instance $i$ of bag $b$ **after** this reduction — every equation below
starts from $\mathbf{x}_{bi}$, not the raw spectrum.

### 0.2 Notation and state space

- Bags $b = 1,\dots,B$ (a fold's own training bags), each with a **known**
  label $Y_b \in \{0,\dots,K-1\}$ ($K=2$ in the real fit) and $n_b$
  instances.
- $M = 3$ shared **null** components (`N_NULL_COMPONENTS`), indexed
  $0,\dots,M{-}1$ — stroma/necrosis/normal tissue, shared across *every*
  bag regardless of class.
- Total discriminative states $S = M + K$ (`n_states`). State $M+k$ is
  class $k$'s diagnostic state; bag $b$'s own diagnostic state index is
  $M + Y_b$ (`diagnostic_state`).
- **Model (C)'s hard support restriction** (`allowed_states_mask`):
  instance $(b,i)$'s latent state $z_{bi}$ may only take a value in
  $\mathcal{A}_b = \{0,\dots,M-1\} \cup \{M+Y_b\}$ — never another class's
  diagnostic state. $|\mathcal{A}_b| = M+1$ regardless of $K$.

### 0.3 Generative model

$$
\begin{aligned}
a, b &\sim \mathrm{Gamma}(1,1) &&\text{(independent, weak hyperprior)}\\[2pt]
\boldsymbol\mu_k, \boldsymbol\Sigma_k &\sim \mathrm{NIW}(\mathbf{m}_0,\kappa_0,\nu_0,\boldsymbol\Psi_0) && k = 0,\dots,S-1 \text{ (i.i.d.)}\\[2pt]
\pi_b &\sim \mathrm{Beta}(a,b) && \text{for each bag } b \text{ (i.i.d. given } a,b)\\[2pt]
z_{bi} \mid \pi_b &\sim \mathrm{Categorical}\big(\mathcal{A}_b,\; p(\cdot\mid\pi_b)\big) && \text{for each instance}\\[2pt]
\mathbf{x}_{bi} \mid z_{bi}{=}k &\sim \mathcal{N}(\boldsymbol\mu_k, \boldsymbol\Sigma_k)
\end{aligned}
$$

where the instance-state prior, given $\pi_b$, is

$$
p(z_{bi}=k \mid \pi_b) =
\begin{cases}
\pi_b & k = M+Y_b \\[2pt]
(1-\pi_b)/M & k \in \{0,\dots,M-1\} \\[2pt]
0 & \text{otherwise (Model (C))}
\end{cases}
$$

and the NIW prior (`_default_niw_prior`) is weakly informative, fixed
once per fold from the training data: $\mathbf{m}_0 = \mathbf{0}$,
$\kappa_0 = 0.01$, $\nu_0 = d+2$ (the smallest value giving a finite
Inverse-Wishart mean), $\boldsymbol\Psi_0 = s^2 \mathbf I_d$ with
$s^2 = \frac{1}{d}\sum_{j=1}^d \mathrm{Var}(x_{\cdot j})$ (mean per-feature
variance of that fold's training data — a plug-in scale, not itself a
random hyperparameter).

**Joint density**, one fold, bags $b=1,\dots,B$:

$$
p(X,z,\{\boldsymbol\mu_k,\boldsymbol\Sigma_k\},\{\pi_b\},a,b) =
p(a)\,p(b) \prod_{k=0}^{S-1} \mathrm{NIW}(\boldsymbol\mu_k,\boldsymbol\Sigma_k)
\prod_{b=1}^{B}\Big[\mathrm{Beta}(\pi_b;a,b) \prod_{i=1}^{n_b} p(z_{bi}\mid\pi_b)\,\mathcal N(\mathbf x_{bi};\boldsymbol\mu_{z_{bi}},\boldsymbol\Sigma_{z_{bi}})\Big]
$$

### 0.4 Gibbs sweep (`fit_bayes_mil`, one iteration)

Three exact conjugate draws and one Metropolis-Hastings step, in this
order:

**(i) Component parameters, given $z$** — for each state $k=0,\dots,S{-}1$,
let $X_k = \{\mathbf x_{bi} : z_{bi}=k\}$, $n_k = |X_k|$. If $n_k=0$, draw
directly from the prior: $\boldsymbol\Sigma_k \sim
\mathrm{InvWishart}(\nu_0,\boldsymbol\Psi_0)$,
$\boldsymbol\mu_k\mid\boldsymbol\Sigma_k \sim \mathcal N(\mathbf
m_0,\boldsymbol\Sigma_k/\kappa_0)$ (an emptied component is never dropped —
the state space stays fixed sweep to sweep). Otherwise, with sample mean
$\bar{\mathbf x}_k$ and scatter $\mathbf S_k = \sum_{\mathbf x\in
X_k}(\mathbf x-\bar{\mathbf x}_k)(\mathbf x-\bar{\mathbf x}_k)^\top$:

$$
\kappa_n = \kappa_0+n_k,\quad \nu_n=\nu_0+n_k,\quad
\mathbf m_n = \frac{\kappa_0\mathbf m_0 + n_k\bar{\mathbf x}_k}{\kappa_n},\quad
\boldsymbol\Psi_n = \boldsymbol\Psi_0+\mathbf S_k+\frac{\kappa_0 n_k}{\kappa_n}(\bar{\mathbf x}_k-\mathbf m_0)(\bar{\mathbf x}_k-\mathbf m_0)^\top
$$
$$
\boldsymbol\Sigma_k \sim \mathrm{InvWishart}(\nu_n,\boldsymbol\Psi_n),\qquad
\boldsymbol\mu_k \mid \boldsymbol\Sigma_k \sim \mathcal N(\mathbf m_n,\boldsymbol\Sigma_k/\kappa_n)
$$

the standard NIW conjugate posterior — an exact draw from
$p(\boldsymbol\mu_k,\boldsymbol\Sigma_k\mid z,X)$, independent across $k$
given $z$.

**(ii) Latent states, given $\boldsymbol\mu,\boldsymbol\Sigma,\pi$** — for
every instance independently (one vectorised pass, no per-instance
Python loop):

$$
p(z_{bi}=k \mid \mathbf x_{bi},\pi_b,\boldsymbol\mu,\boldsymbol\Sigma) \;\propto\;
p(k\mid\pi_b)\,\mathcal N(\mathbf x_{bi};\boldsymbol\mu_k,\boldsymbol\Sigma_k),
\quad k \in \mathcal A_b
$$

sampled exactly via the Gumbel-max trick (add i.i.d. $\mathrm{Gumbel}(0,1)$
noise to each allowed state's log-weight, take the arg max — distributionally
identical to direct categorical sampling from the normalised conditional,
with every disallowed state at $-\infty$).

**(iii) Diagnostic fraction, given $z$** — for each bag independently, let
$n_{b}^{\text{diag}} = \#\{i : z_{bi}=M+Y_b\}$,
$n_b^{\text{null}} = n_b - n_b^{\text{diag}}$:

$$
\pi_b \mid z_b, a, b \;\sim\; \mathrm{Beta}\big(a+n_b^{\text{diag}},\; b+n_b^{\text{null}}\big)
$$

the standard Beta-Bernoulli conjugate update (treating $[z_{bi}{=}M{+}Y_b]$
as a Bernoulli$(\pi_b)$ draw), independent across bags given $a,b$.

**(iv) Hyperparameters, given $\{\pi_b\}$** — the only non-conjugate step:
random-walk Metropolis on $(\log a,\log b)$, step size $0.2$. Unnormalised
log target:

$$
\log\pi(a,b\mid\{\pi_b\}) = \sum_{b=1}^B\Big[\log\Gamma(a{+}b)-\log\Gamma(a)-\log\Gamma(b) + (a{-}1)\log\pi_b + (b{-}1)\log(1{-}\pi_b)\Big] \;-\; a - b
$$

(the trailing $-a-b$ is the $\mathrm{Gamma}(1,1)$ log-prior on $a$ and $b$,
up to an additive constant). Propose $a' = a\,e^{\epsilon_a}$, $b' =
b\,e^{\epsilon_b}$, $\epsilon_a,\epsilon_b\stackrel{\text{iid}}{\sim}\mathcal
N(0,0.2^2)$; accept with probability

$$
\min\Big(1,\;\exp\big[\log\pi(a',b') - \log\pi(a,b) + \log(a'/a) + \log(b'/b)\big]\Big)
$$

(the $\log(a'/a)+\log(b'/b)$ term is the Jacobian correction for proposing
on the log scale); on rejection, $(a,b)$ are unchanged for this sweep.

### 0.5 Posterior summary (Rao-Blackwellised)

After a burn-in of $B_0$ sweeps (600 iterations, 200 burn-in in the real
fit), every $T$-th sweep ($T=5$) is retained. At each retained sweep $t$,
rather than storing the hard draw $z_{bi}^{(t)}$, the exact conditional
responsibility

$$
w_{bi}^{(t)}(k) = p\big(z_{bi}{=}k \mid \mathbf x_{bi},\pi_b^{(t)},\boldsymbol\mu^{(t)},\boldsymbol\Sigma^{(t)}\big), \qquad k \in \mathcal A_b
$$

(the same normalised weights as step (ii)) is computed in closed form and
accumulated; the reported per-instance posterior is $\frac{1}{N}\sum_t
w_{bi}^{(t)}$ over the $N$ retained sweeps — lower-variance than averaging
hard $z$ samples for the same compute (Rao-Blackwellisation). Collapsing to
the project-wide $(K{+}1)$-column convention (`posterior_to_K1`): column 0
is $\sum_{m<M} w_{bi}(m)$ (pooled null mass), columns $1,\dots,K$ are
$w_{bi}(M{+}0),\dots,w_{bi}(M{+}K{-}1)$ — at most one of which can be
nonzero, by construction (§4).

### 0.6 Multi-chain selection and convergence

Run $C$ independent chains ($C=3$), each from an independent
structure-aware initialisation of $z$ (cluster all of the fold's instances
into $S$ groups by $k$-means, assign each group to the state whose class's
bags use it most — a starting point, not a constraint the sampler must
keep) and an independent RNG stream (chain $c$ seeded at
$\text{seed}_0+1000c$). For chain $c$, with its own final retained draw
$(\boldsymbol\mu^{(c)},\boldsymbol\Sigma^{(c)},\{\pi_b^{(c)}\})$, define

$$
L_c = \sum_{b,i} \log \sum_{k\in\mathcal A_b} p(k\mid\pi_b^{(c)})\,\mathcal N(\mathbf x_{bi};\boldsymbol\mu_k^{(c)},\boldsymbol\Sigma_k^{(c)})
$$

— the observed-data log-likelihood with $z$ marginalised **at that one
point estimate of the continuous parameters** (`marginal_log_likelihood`).
**This is a plug-in marginal likelihood, not a fully Bayes-integrated one**
(which would also integrate out $\boldsymbol\mu,\boldsymbol\Sigma,\pi,a,b$,
e.g. via bridge sampling); it is used only to rank chains against each
other — to tell which basin a trapped sampler landed in — not as a
model-evidence computation in the Bayesian-model-comparison sense. The
chain with the highest $L_c$ is kept; its own posterior is what's reported.

Convergence diagnostic (`gelman_rubin`): the standard Gelman-Rubin $\hat R$
on the $C$ chains' own history of the population mean
$\bar\pi^{(t)}=\frac1B\sum_b\pi_b^{(t)}$, using each chain's second-half
sweeps. With $W$ the mean within-chain variance of $\bar\pi^{(t)}$ and $B_{\text{between}}$
the (scaled) between-chain variance of each chain's own mean:

$$
\widehat{\mathrm{var}} = \frac{n-1}{n}W + \frac{B_{\text{between}}}{n}, \qquad \hat R = \sqrt{\widehat{\mathrm{var}}/W}
$$

$\hat R > 1.1$ conventionally flags non-convergence — but see §2: this only
checks a fold's chains against each other, not against another fold's
independent fit.

### 0.7 Scoring held-out bags (posterior predictive)

A held-out bag $b^*$ has no fitted in-sample $\pi_{b^*}$. At each retained
sweep $t$ of the **winning** chain, draw a fresh
$\pi_{b^*}^{(t)}\sim\mathrm{Beta}(a^{(t)},b^{(t)})$ from that sweep's own
hyperparameters, compute $w_{b^*i}^{(t)}(k)$ exactly as in §0.5 using that
sweep's own $(\boldsymbol\mu^{(t)},\boldsymbol\Sigma^{(t)})$, and average
over all $N$ retained sweeps (`score_bayes_mil`). This is a genuine
posterior-predictive average over the retained draws — never a plug-in at
a single point, and never a reused in-sample $\pi_b$ value, since $b^*$
never had one.

---

## 1. Why this tier exists

Tier 2 (EM) fits the same Model (C) latent structure by coordinate-ascent
point estimation. Your own framing for wanting a Bayesian alternative:
Tier 2 "has too many components and possible failing points" and a Bayesian
fit is "simpler and much more informative" — specifically because its
*failure mode* is more useful: reporting high posterior uncertainty or a
non-converged chain, rather than confidently relabelling noise as
diagnostic the way an EM point estimate can. §2 below is exactly that
failure mode showing up on real data, which is the intended outcome of
building this tier, not a defect in it.

## 2. Per-fold convergence — R-hat alone is the wrong cut

| fold | n bags | $\hat R$ | Metropolis accept | PCA EVR | R-hat verdict |
|---|---|---|---|---|---|
| fold_0 | 45 | **1.0027** | 0.477 | 0.7945 | converged |
| fold_1 | 47 | **1.0021** | 0.502 | 0.7860 | converged |
| fold_2 | 45 | **9.0900** | 0.537 | 0.8434 | **not converged** |
| fold_3 | 48 | **11.8082** | 0.510 | 0.8086 | **not converged** |
| fold_4 | 47 | **1.0009** | 0.493 | 0.7947 | converged |

$\hat R$ (Gelman-Rubin, across the 3 chains' mean-$\pi_b$ trace) only checks
whether a fold's *own* 3 chains agree with each other — it cannot see
whether the mode they agree on is the same one another fold's independent
fit would find. Checked directly: since every real bag is a *training* bag
in 4 of the 5 folds, the same bag gets an independent $\hat\pi_b$ from up
to 4 different folds' models. This is **not** a check on a bag-level
diagnostic/non-diagnostic label — every bag's class $Y_b$ is already known
and is never in dispute (§1 of the methodology). What's being compared is
each fold's own independent *estimate of the diagnostic fraction*
$\hat\pi_b$ for a bag both folds happen to train on — i.e. whether two
independently-fit models agree on *how much* of that bag's tissue they
each think is diagnostic, not on whether it is.

**Implemented** as `src/mil/bayes.py::cross_fold_pi_b_consistency` (pure
function, reads `{fold: {bag: pi_b_mean}}`, no retraining and no shared
parameters — pooling fit parameters across folds *would* be leakage, since
the union of all 5 folds' training sets is every bag in the dataset; this
only compares each fold's already-independent, already-saved results) and
`scripts/tier5_cross_fold_consistency.py` (reads a real fit's
`bayes_diagnostics.json`, writes `bayes_cross_fold_consistency.json`).
Run on `exp_608` — full $\binom{5}{2}=10$ pairs:

| fold pair | corr | mean $|$diff$|$ |
|---|---|---|
| fold_0 vs fold_4 | **+0.985** | 0.021 |
| fold_0 vs fold_3 | **+0.979** | 0.028 |
| fold_3 vs fold_4 | **+0.939** | 0.036 |
| fold_1 vs fold_3 | +0.500 | 0.153 |
| fold_0 vs fold_1 | +0.406 | 0.181 |
| fold_1 vs fold_4 | +0.310 | 0.218 |
| fold_1 vs fold_2 | +0.026 | 0.166 |
| fold_2 vs fold_3 | −0.298 | 0.338 |
| fold_0 vs fold_2 | −0.393 | 0.405 |
| fold_2 vs fold_4 | −0.417 | 0.429 |

| fold | $\hat R$ | mean corr vs. the other 4 folds |
|---|---|---|
| fold_3 | 11.8082 | **+0.530** (highest) |
| fold_0 | 1.0027 | +0.494 |
| fold_4 | 1.0009 | +0.455 |
| fold_1 | 1.0021 | +0.310 |
| fold_2 | 9.0900 | **−0.270** (only negative one) |

**{fold_0, fold_3, fold_4} agree with each other almost perfectly** —
including fold_3, whose own within-fold $\hat R$ (11.81) says it shouldn't
be trusted, and whose mean cross-fold correlation is in fact the *highest*
of all five folds. **fold_1 sits in between**: positive but weak
correlation with every other fold (0.03–0.50), despite a textbook-good
$\hat R$ (1.0021) — its own 3 chains agree with each other (that's all
R-hat checks), on an answer only loosely consistent with the rest.
**fold_2 is the only fold with a negative mean correlation against
everyone else**, and its R-hat flags it too, though — see §2.1 — not for
the reason R-hat's own internal logic would suggest. Neither fold_1's nor
fold_2's disagreement is a simple global null/diagnostic relabelling
(checked: correlation against $1-$consensus is nowhere near $-1$ for
either), so this is genuine partial, bag-specific disagreement in the
*fraction* each fold estimates, not a label-swap artifact.

**The corrected picture replaces "3 converged, 2 failed" with: 3 folds
(0, 3, 4) that cross-validate each other and are probably closest to
right; fold_1, which passed the only check this tier previously ran and
is only weakly consistent with the rest; and fold_2, the one fold that is
net-negatively correlated with everyone.**

> **Standing disagreement, not resolved.** You've said directly that you're
> "not convinced this tells us much" — recorded here rather than smoothed
> over. The check demonstrates internal *consistency* among a subset of
> folds; it does not, on its own, prove that subset is *correct* in any
> absolute sense (there is no instance-level ground truth to check against
> at all, §1) — three independently-wrong fits agreeing with each other is
> not logically distinguishable from this analysis alone, if whatever makes
> them wrong is shared across folds rather than fold-specific. Decision on
> how much weight this check should carry is deliberately left open.

### 2.1 Mechanistic diagnosis — re-running every individual chain, not just the winner

`fit_bayes_mil_chains` discards every chain but the best-by-marginal-
log-likelihood, so `bayes_diagnostics.json` only ever has the winner. To
see *what* the losing chains actually found, folds 2 and 3 were refit chain
by chain with the exact original seeds (`seed=42/1042/2042`), confirmed to
bit-reproduce the saved `chain_log_liks` exactly (sorted, atol=1.0).

**fold_2 — not really a mixing problem.** All 3 chains agree with *each
other* almost perfectly at the bag level (pairwise corr 0.995–1.000; the
single largest bag-level disagreement found between the winning and
losing chain was $\hat\pi_b=0.0000$ vs $0.1544$ — small). Yet the
~195,000-nat gap between chain 0 (842,547) and chains 1/2 (1,037,641 /
1,037,657) is real and large. Matching the 5 component means between the
winning and losing chain (Hungarian assignment) shows why: 2 of the 5
components moved substantially (distances 0.90 and 1.36, against a
typical 2.68 inter-component spacing in the winning fit), while the other
3 barely moved (0.01–0.42). **The huge likelihood gap is almost entirely
about how well the *null/background* components describe the bulk of the
spectra, not about which bags get called diagnostic** — and all 3 chains,
regardless of which basin they're in for the nulls, land on essentially
the same (and still consensus-disagreeing) bag-level answer. So fold_2's
disagreement with {0,3,4} is not sampler bad luck — it is a property of
*fold_2's own training-data composition* (which 45 bags end up in it,
which one is held out), reproduced deterministically across 3 dispersed
starts. R-hat flags fold_2, but for a reason (null-component instability)
that is mostly irrelevant to the diagnostic question it's actually being
used for.

**This is not a fold_2 data-quality problem** — checked directly by
reloading `exp_501`'s real TRTR models (reused unchanged by `exp_608`) and
scoring each fold's own held-out bags: fold_2 is RF/XGBoost's *best*
fold of all five (bag acc 0.929, their highest), and Apollo's second-best
(0.857). If anything the pattern runs backwards from a "bad data" story —
fold_0 and fold_4, the two folds Tier 5 cross-validates as trustworthy,
are Tier 0's two *hardest* folds (RF/XGBoost bag acc 0.58–0.64), while
fold_2, Tier 5's problem fold, is Tier 0's easiest. A plausible mechanism
consistent with this and with the component-matching result above: cleaner
class separation gives the Gibbs sampler more near-equally-good ways to
place the 3 null components' boundaries through tightly-clustered bulk
data — small shifts there cost little in overall likelihood (hence the
chains agreeing with *each other*) but can still swing which specific
instances end up called diagnostic (hence disagreeing with the other
folds). A messier fold may have less of that flat, multi-way-splittable
null structure to get lost in. Not confirmed beyond this one fold-pair
comparison — see §6.

| fold | Apollo bag acc | RF bag acc | XGBoost bag acc | CNN bag acc | Tier 5 verdict |
|---|---|---|---|---|---|
| fold_0 | 0.643 | 0.643 | 0.643 | 0.571 | consensus |
| fold_1 | 0.750 | 0.833 | 0.833 | 0.833 | disagrees |
| fold_2 | 0.857 | **0.929** | **0.929** | 0.571 | disagrees (worst) |
| fold_3 | 0.818 | 0.909 | 0.909 | 0.818 | consensus |
| fold_4 | 0.583 | 0.583 | 0.583 | 0.583 | consensus |

(Tier 0's own bag-level majority-vote rule, real TRTR data/models, each
fold's own held-out bags — docs/mil_tiers_methodology.md §4.)

**fold_3 — genuine two-basin multimodality, and the existing selection
rule handles it correctly.** Chains 1 and 2 agree with each other almost
exactly (corr 1.000, log-liks 716,114/716,081) and are a real alternate
basin from chain 0 (corr 0.946–0.947 with them, log-lik 727,956 — the
higher, selected mode). Component-matching shows a smaller, partial shift
than fold_2's (2 of 5 components move 0.65/0.78 against a 1.94 typical
spacing; the other 3 move little). All three of fold_3's chains, including
the two "losers," correlate reasonably well with the external fold_0/4
consensus (0.94–0.98) — so this is the ordinary multimodality the module
docstring describes, the selected (highest-likelihood) chain is a
reasonable choice here, and cross-validating against fold_0/4 confirms it.

**Net: folds 2 and 3 fail for different reasons, and only one of them is
the "normal MCMC got trapped, pick a better chain" story the methodology
anticipated.** Fold_3 is; fold_2 is a more fundamental sensitivity to which
bags are in the training set, which more chains or more iterations would
not obviously fix.

### 2.2 The reported credible interval is confident, not honest, on disputed bags

Pulling the bags where fold_1 and fold_0/fold_4 disagree most:

```
HF-2802_V3T_1.h5_9:  fold_0 mean=0.906 CI=[0.894,0.914]   fold_1 mean=0.003 CI=[0.0015,0.0047]
HF-2548_V1T_1.h5_7:  fold_0 mean=0.764 CI=[0.752,0.774]   fold_1 mean=0.0001 CI=[0.0000,0.0004]
HF-442_V4BB_1.h5_12: fold_4 mean=0.633 CI=[0.612,0.653]   fold_1 mean=0.0001 CI=[0.0000,0.0005]
```

Every interval is narrow. The model is exactly as confident when it is
wrong as when it is right, because the reported 95% CI is the **within-
mode** conjugate-Beta posterior, conditioned on whichever labelling the
*selected* chain settled into — it carries no information about
uncertainty over *which labelling is correct*, which is precisely what is
driving the fold-to-fold disagreement. Winner-take-all chain selection
discards exactly the uncertainty that matters most here. See §7's
recommendation.

**Metropolis accept rate** for the one non-conjugate step ($(a,b)$
hyperparameters) sits in the healthy 0.48–0.54 range in all 5 folds,
including the two R-hat-flagged ones — whatever is going on in §2.1 is a
$z$/component-identity problem, not a degenerate proposal-tuning problem.

## 3. What $\pi_b$ (diagnostic fraction) looks like

| fold | median $\hat\pi_b$ | mean $\hat\pi_b$ | frac bags $\hat\pi_b{<}0.01$ | frac bags $\hat\pi_b{>}0.5$ | trust (§2) |
|---|---|---|---|---|---|
| fold_0 | 0.126 | 0.241 | 29% | 16% | consensus |
| fold_1 | 0.011 | 0.123 | 47% | 11% | **disagrees with consensus** |
| fold_4 | 0.142 | 0.277 | 30% | 23% | consensus |

fold_1's own numbers are included for completeness, not because §2 found
them trustworthy — note it already looks different on its own terms (the
lowest mean/median of the three, the most bags pinned near zero), which in
hindsight is visible without the cross-fold check, just not diagnostic on
its own. 95% credible intervals (`pi_b_95ci` in the diagnostics file) are
tight for bags near either extreme within fold_0/fold_4 (e.g. fold_0's
`HF-1043_V1AM_1.h5_0`: $\hat\pi_b=0.783$, CI $[0.771, 0.793]$) and can span
3+ orders of magnitude for bags the sampler is genuinely unsure about
*within a single mode* (e.g. fold_0's `HF-2876_V1T_1.h5_10`: CI
$[5.5\times10^{-8}, 0.008]$) — informative in a way Tier 2's point estimate
has no analogue for, but see §2.2 for where this within-mode credible
interval is not enough.

### 3.1 Distribution family — is a single Beta($a$,$b$) the right amount of flexibility?

$(a,b)$ is a single population-level hyperparameter pair, shared by every
bag in a fold — not bag-specific, not class- or size-dependent. Fitted
values:

| fold | $a$ | $b$ | prior mean | shape |
|---|---|---|---|---|
| fold_0 | 0.239 | 0.815 | 0.227 | U-shaped ($a,b{<}1$: mass at both 0 and 1) |
| fold_1 | 0.234 | 1.366 | 0.146 | monotone decreasing (peaked at 0 only) |
| fold_4 | 0.239 | 0.737 | 0.245 | U-shaped |

$a\approx0.23$–$0.24$ is strikingly stable across all three, including
fold_1 — whatever governs how sharply bags pile up near 0 is well
identified regardless of which mode the fold landed in; $b$ (controlling
the high end) varies more.

Checked whether a richer family is actually warranted by fitting a
2-component Beta mixture to each fold's realised $\hat\pi_b$ values
(Nelder-Mead MLE, multiple restarts) and comparing AIC against the single
Beta:

| fold | single-Beta AIC | mixture AIC | $\Delta$AIC (mixture better if $+$) |
|---|---|---|---|
| fold_0 | −102.33 | −96.50 | −5.82 (single Beta wins) |
| fold_1 | −177.79 | **−194.69** | **+16.90 (mixture wins)** |
| fold_4 | −108.26 | −103.65 | −4.60 (single Beta wins) |

For fold_0/fold_4 — the two folds independently cross-validated as
trustworthy — **the single Beta is already adequate**; the mixture's extra
3 parameters aren't earning their keep. fold_1 is the one fold where a
mixture wins decisively, and the fitted mixture is revealing: a 25%-weight
component squeezed to $a{=}0.577,b{=}47.8$ (mean $\approx0.012$, extremely
concentrated at the floor) plus a broader 75%-weight component at
$a{=}1.15,b{=}1.36$ (mean $\approx0.46$). Given §2.1's finding that fold_1
is the "good R-hat, wrong mode" fold, this almost certainly reflects the
mode-confusion leaking into the marginal shape — a subset of bags
mislabelled toward the floor — rather than genuine bimodality in the true
diagnostic-fraction population. **Conclusion: the evidence does not
support making $\pi_b$'s marginal family more flexible as the priority
fix.** The more flexible, better-supported Beta shows up specifically in
the one fold already known to be wrong for an unrelated reason.

What the evidence *does* support making more flexible is the
**chain-selection/uncertainty-reporting procedure** (§2.2, §7) — that is
where a real, demonstrated problem lives.

## 4. Bag-level classification — vacuous, same as Tier 2 (retracted finding)

A first pass computed bag accuracy from this tier's posterior and got
0.9153 (95 CI [0.714, 0.984], vs. CNN TRTR 0.7119, p=0.000; also
significant vs. Apollo). **This looked like a real win and is not one.**
`k_diagnostic_columns_competition_frac=0.0000` (`stage6_bag_eval`'s own
automatic check) confirms the posterior never places competing mass across
diagnostic classes — Model (C) is enforced at scoring, exactly as for
Tier 2 (docs/mil_tiers_methodology.md §6.2/§8.2) — so this number recovers
the label the model was handed, not one it inferred. **Excluded from this
tier's real evidence**, kept here only as a record of the mistake and the
check that caught it before it was reported as a finding.

### 4.0 Precisely where this comes from — not leakage, a required argument

Worth being exact about, since "circular" and "leaky" are different
failure modes with different fixes. **There is no train/test leakage** — a
held-out bag's instances never appear in `fit_bayes_mil`'s training data
(§0.1 — `Z_train`/`fd.train_bag_ids`/`fd.y_train` are training-side only);
nothing about a held-out bag's spectra reaches the Gibbs sampler.

What actually happens is that $Y_b$ — the bag's **already-known** label,
known for every bag, train or held-out, under the MIL setup this whole
project uses — is a *required input* to `score_bayes_mil`, not something
the function is being asked to recover:

```python
def score_bayes_mil(fit, X, bag_ids, y_bag_per_instance, n_classes):
    diag_state = diagnostic_state(y_bag_per_instance)        # from Y_b
    mask = allowed_states_mask(y_bag_per_instance, n_classes) # from Y_b
    ...
    log_prior[np.arange(n), diag_state] = np.log(pi_draw)
    lw = np.where(mask, log_prior + log_lik, -np.inf)         # every OTHER class: -inf
```

called as `score_bayes_mil(fit, Z_val, fd.val_bag_ids, fd.y_val, n_classes)`
(`pipeline/mil_fit_stage.py::_fit_one_fold_bayes_mil`) — `fd.y_val`, the
validation bags' true labels, passed directly in. The mask zeroes out
every diagnostic state except the bag's own before any likelihood is
evaluated, so when `stage6_bag_eval.py` later does `argmax` over the
diagnostic columns, there was only ever one column with nonzero mass —
not because the data pointed to it, the alternative was never on the
table.

This is *correct* for the task this tier actually solves — estimate
$\pi_b$, the diagnostic fraction, **given** the already-known $Y_b$
(§0.3's generative model needs $Y_b$ to even define $\mathcal A_b$,
following `docs/MIL_REFRAME_PLAN.md` §1's own stated assumption that "a
slide graded $k$ contains only non-diagnostic tissue plus grade-$k$
tumour"). The mismatch is that `stage6_bag_eval.py` is a *generic* script
built for Tier 1 (mi-SVM, independent one-vs-rest, never given $Y_b$) and
Tier 4 (CbR/encoder_head, a genuine free $K$-way softmax, never given
$Y_b$ either), and produces a real number for those — `competition_frac`
0.97–0.99, genuine competition between classes happened. Pointed at this
tier's posterior, it's measuring whether an input equals itself.

### 4.1 Per-fold breakdown — the same circularity, with one instructive exception

`scripts/tier5_per_fold_accuracy.py` (reuses `stage6_bag_eval.py`'s own
data-loading/circularity-check functions, aggregating per fold instead of
pooled):

| fold | n bags | n out-of-sample instances | competition_frac | bag_acc |
|---|---|---|---|---|
| fold_0 | 14 | 52,593 | 0.0000 | 1.0000 |
| fold_1 | 12 | 60,196 | 0.0000 | 1.0000 |
| fold_2 | 14 | 76,560 | 0.0000 | **0.6429** |
| fold_3 | 11 | 59,864 | 0.0000 | 1.0000 |
| fold_4 | 12 | 53,968 | 0.0000 | 1.0000 |

`competition_frac=0.0000` in every fold individually, not just pooled.
Four folds read exactly 1.0, as the circularity argument predicts (an
instance can only ever vote for its own bag's true class, so pooling can't
be wrong). **fold_2 is not a counterexample to the circularity — it's a
different symptom of it.** Every bag's mean-pooled score is exactly
`[1,0]`, `[0,1]`, or `[0,0]` (the "other" column is structurally zero
regardless, so no partial score is possible). fold_2's 5 wrong-predicted
bags are all true class 1, and in 4 of them *every one* of several
thousand out-of-sample instances scored exactly `[0,0]` —
`np.argmax` ties to index 0, flipping the bag to a wrong class-0 call.
This traces to `score_bayes_mil`'s posterior-predictive scoring: a
held-out bag has no fitted $\pi_b$, so each retained draw samples a fresh
one from the fold's own $\mathrm{Beta}(a,b)$ — and fold_2's fitted $a=0.059$
is the smallest of all 5 folds (§3.1) by a wide margin, drawing $\pi_b$ so
close to 0 that it overwhelms whatever per-instance signal the Gaussian
likelihoods carry, for every instance in these bags simultaneously. This
is a **third, independent** confirmation that fold_2's fitted population
prior is an outlier among the 5 folds (alongside §2's cross-fold
correlation and §2.1's component-matching) — not evidence that bag
accuracy is any more meaningful here than the trivial 1.0 everywhere else.

**A fair challenge raised against this table, checked rather than waved
off**: per-fold aggregation above does not merge the 4 cross-fold
duplicate bags (docs/mil_tiers_methodology.md §2) into one combined
decision the way the project's standard, properly-pooled evaluation does
— each is counted once per fold it's a test bag in, unmerged. Checked
directly: all 4 duplicates are shared only between fold_0 and fold_4, and
both folds score all 4 bit-identically (`scripts/tier5_per_fold_accuracy.py`
now prints this check on every run) — so merging them changes nothing
here. The properly pooled, duplicate-averaged bag accuracy across all 59
unique bags (`results/exp_608/mil_eval/stage6_bag_eval_bayes_mil_gibbs.json`,
already in this file at the top of §4) is **0.9153** — not 1.0 — pulled
down from the trivial ceiling by exactly fold_2's anomaly above, and it is
the same already-retracted number, not a new one. The per-fold table is a
genuinely different, complementary cut (it's what makes fold_2's anomaly
visible at all — pooled into one n=59 number, fold_2's 5 wrong bags are
invisible as anything other than a slightly-less-than-1.0 overall figure),
not a competing or more authoritative one.

### 4.2 Is this genuinely better than every other tier? No — full comparison

Checked directly rather than assumed, against every bag-eval artifact saved
anywhere in the project (`results/*/mil_eval/stage6_bag_eval_*.json`):

| tier | method/variant | exp | bag_acc | competition_frac | valid? | vs best Tier 0 |
|---|---|---|---|---|---|---|
| 0 | Apollo (TRTR) | exp_501 | 0.763 | — | reference | — |
| 0 | **RF / XGBoost (TRTR)** | exp_501 | **0.814** | — | reference | — |
| 0 | CNN (TRTR) | exp_501 | 0.712 | — | reference | — |
| 1 | mi-SVM | exp_501 | 0.690 | 0.970 | valid | **worse**, p=0.046 |
| **2** | **EM (constrained)** | exp_501 | **1.000** | **0.000** | **circular** | — |
| 3 | attention | exp_501 | 0.644 | n/a (free softmax) | valid | worse, p=0.002–0.012 |
| 3 | attention | exp_607 | 0.525 | n/a (free softmax) | valid | worse, n.s. (p=0.07) |
| 4 | cbr/encoder_head | exp_501 | 0.797 | n/a (free softmax) | valid | n.s. vs all 3 |
| 4 | cbr/reconstruction | exp_501 | 0.712 | n/a (free softmax) | valid | n.s. vs all 3 |
| 4 | cbr/encoder_head | exp_607 | 0.763 | n/a (free softmax) | valid | n.s. |
| 4 | cbr/reconstruction | exp_607 | 0.729 | n/a (free softmax) | valid | n.s. |
| 4 | latent_rf | exp_607 | 0.729 | n/a (free softmax) | valid | n.s. vs CNN, **worse** vs RF/XGB p=0.008 (docs/stage6_generative_mil.md §9.1) |
| **5** | **Bayesian (Gibbs)** | exp_608 | **0.915** | **0.000** | **circular** | — |

**Among every valid (non-circular) number in the project, nothing beats
Tier 0's own RF/XGBoost (0.814, exp_501).** Tier 1/3/4 are each flat or
significantly worse. The two numbers that look best — Tier 2's 1.000 and
Tier 5's 0.915 — are excluded for the identical, independently-confirmed
reason: `competition_frac≈0.0000`, Model (C) enforced at scoring, so
neither number reflects anything learned (§4, docs/mil_tiers_methodology.md
§6.2/§8.2). **This is the same finding, found twice, for the same
structural reason** — Tier 5 is not a second, independent piece of
evidence that bag accuracy can be beaten; it is the same mechanism
re-discovered on a different formulation. Running the bootstrap comparison
again, as asked, reproduces `p=0.000, significant=True` against CNN
exactly as before (`results/exp_608/mil_eval/stage6_bag_eval_bayes_mil_gibbs.json`)
— the number doesn't change on a re-run, because nothing about the
circularity is stochastic. **It is not a measurement that a significance
test can validate**, so re-running it again would not change this
conclusion either.

**A genuine, non-circular bag-accuracy comparison is possible** — it just
cannot be built from `score_bayes_mil` (which requires $Y_b$ as an
argument by construction, §4.0). §4.3 builds and runs the version that
doesn't: a real experiment, not a restatement of the mechanism argument
above. Separately, the key-instance utility test
(docs/mil_tiers_methodology.md §10.2, already run for Tier 4's
`reconstruction` variant and failed there — docs/stage6_generative_mil.md
§9.3) still has not been run against this tier's posterior and remains
open (§6) — a different question (are the *selected instances* useful to
a different classifier) from §4.3's (can this tier's own fitted model
classify at all when not told the answer).

### 4.3 The genuine, non-circular bag classification — built and run

`src/mil/bayes.py::score_bayes_mil_blind` (new, tested in
`tests/test_bayes_mil.py` — 100% recovery on a well-separated planted
signal with fresh held-out bags, plus a signature-level guard that it
*cannot* accept a label argument at all) never receives $Y_b$. For each
held-out bag and each candidate class $k$, it evaluates the
posterior-predictive log-likelihood of that bag's instances under "this
bag's class is $k$" (§0.6's same machinery, evaluated once per candidate
instead of being told which one is true) and predicts by $\arg\max_k$ —
exactly the ordinary held-out-classifier evaluation this whole
investigation's back-and-forth converged on.

`scripts/tier5_blind_bag_classification.py` reconstructs each fold's exact
chains (same seeds; confirmed bit-exact — every fold's `chain_rhat` and
`chain_log_liks` below match the originally saved values exactly, so this
is the same fit, not a different one) and scores that fold's own held-out
bags blind:

| fold | n bags | majority baseline | **blind bag_acc** | one-sided $p$ (> baseline) | Tier 0 RF/XGB (same fold, §4.1) | Tier 5 circular (§4.1) |
|---|---|---|---|---|---|---|
| fold_0 | 14 | 0.5714 | 0.6429 | 0.399 | 0.643 | 1.0000 |
| fold_1 | 12 | 0.6667 | 0.7500 | 0.393 | 0.833 | 1.0000 |
| fold_2 | 14 | 0.5714 | **0.5714** | 0.611 | **0.929** | 0.6429 |
| fold_3 | 11 | 0.5455 | **1.0000** | **0.0013** | 0.909 | 1.0000 |
| fold_4 | 12 | 0.5000 | 0.5833 | 0.387 | 0.583 | 1.0000 |

**mean per-fold: 0.7095. Pooled (63 fold-bag rows, duplicates not
merged — see below): 0.6984.** Both well below Tier 0's RF/XGBoost
(0.814, exp_501) and far below the circular Tier 5 figure (0.915 pooled,
0.929 mean-per-fold) — **this is the real answer to §4.2's question,
from an actual experiment rather than an argument about mechanism: no,
not better, meaningfully worse once tested honestly.**

Two things worth being precise about rather than just reporting the
headline number:

1. **Only fold_3 reaches significance** — a perfect 11/11, $p=0.0013$
   against its own majority baseline. This is the same fold that every
   other check in this document (the cross-fold $\hat\pi_b$ correlation,
   §2; the component-matching, §2.1) independently flagged as
   cross-validated-correct *despite* its bad within-fold $\hat R$
   (11.81) — a third, independent line of evidence, now from a genuine
   classification test rather than an internal-consistency check, that
   fold_3's selected chain really did land somewhere real.
2. **fold_2 scores *exactly* at its majority baseline — zero bags above
   chance** ($p=0.611$, i.e. worse than a coin flip in direction). This
   is the fold Tier 0's own RF/XGBoost classifies *best* of all five
   (0.929, §4.1) — the clearest possible demonstration that whatever this
   tier's Gibbs-fitted mixture is doing in fold_2, it is not capturing the
   same signal a simple classifier finds trivially there. A fourth
   independent confirmation that fold_2 is broken for this model
   specifically (alongside §2's cross-fold correlation, §2.1's
   component-matching, and §4.1's all-zero-vote pathology).
3. **Cross-fold duplicate bags** (the same 4 as §4.1, fold_0/fold_4 only):
   both folds agree with each other on the predicted class for all 4 — but
   agree on the *wrong* answer for 3 of the 4 (`HF-2104_#5_1.h5_0`,
   `HF-2104_#9_1.h5_1`, `HF-2104_V1T_1.h5_2`: true class 0, both folds
   confidently predict class 1). Not a source of noise in the comparison
   above (no disagreement to resolve), but a concrete example of these
   two folds sharing a genuine, consistent blind spot rather than making
   independent random errors.

folds 0/1/4 show a modest, non-significant edge over baseline (+1 bag out
of 12–14) — not distinguishable from chance at this sample size, and not
a result to build any claim on either way.

## 5. Spatial validation — present, pooled across all 5 folds including the untrustworthy ones

Pooled Moran's I and permutation null, 54 slides with reprojectable coords
(of 59 total), lag ≥3 headline (§10 of the methodology — lags 1–2 excluded,
contaminated by the 3×3 median filter):

| lag | Moran's I | mean perm-$p$ | mean perm-$z$ |
|---|---|---|---|
| 3 | +0.134 | 0.046 | 33.7 |
| 4 | +0.109 | 0.084 | 32.6 |
| 5 | +0.095 | 0.115 | 32.0 |

Variogram range (median across slides): 3 px. External agreement against
`viable_legacy`: ARI=0.0071, AMI=0.0002 (n=288,678 covered instances) —
essentially no agreement, in line with every other tier's external-agreement
number (docs/mil_tiers_methodology.md §10).

**Two things to flag about this table, not yet resolved:**

1. **It pools all 5 folds, and §2/§2.1 now say the right exclusion set is
   not simply "the 2 that failed R-hat."** fold_3 (bad R-hat) is actually
   cross-validated as trustworthy; fold_1 (good R-hat) is not; fold_2 is
   untrustworthy by both checks. Whether lag≥3 coherence survives pooling
   only the genuinely-trustworthy {fold_0, fold_3, fold_4} (dropping
   fold_1 and fold_2 instead of fold_2/fold_3) has not been checked —
   queued, per your own deferral ("we'll come back to them shortly"), now
   with the correct 3-fold set identified for when that's done.
2. **The permutation $z$-scores (30+) are far larger than any other tier in
   this project has produced** for a comparable statistic. This has not
   been explained; a plausible mechanism is that this posterior has many
   near-0/near-1 extreme values (§3's $\hat\pi_b$ spread, which the
   non-diagnostic mass reprojected here inherits), and Moran's I z-scores
   are known to be sensitive to a map's value distribution independent of
   how much the pattern "matters" (docs/mil_tiers_methodology.md §10's own
   warning about Moran's I being scale-free). Not confirmed; flagged as an
   open question before this number is quoted anywhere as comparable to
   another tier's z-scores.
3. **§4's circularity means this coherence, even if it survives the above,
   cannot yet be read as "this tier found diagnostic structure"** — the
   same gap Tier 4's key-instance utility test (§10.2) was built to close
   for Tier 4. That test has not been run against this tier's posterior.

## 6. Open items

- Re-run §5's pooled spatial numbers on {fold_0, fold_3, fold_4} only
  (deferred, queued — correct fold set now identified, §2/§2.1).
- Shuffled-bag-label control — not run (docs/mil_tiers_methodology.md §11).
- Key-instance utility test (§10.2 of the methodology) against this tier's
  posterior — not run; given §4's retraction and §5's open questions, this
  is the next real falsification step before any claim this tier has found
  diagnostic structure.
- Explain the large permutation z-scores in §5 before quoting them next to
  another tier's.
- fold_2's own instability (§2.1 — a robust, reproducible disagreement, not
  sampler bad luck, and confirmed **not** a data-quality problem: fold_2 is
  RF/XGBoost's *best*-performing fold under Tier 0's own ordinary majority
  vote, 0.929 bag accuracy, their highest of any fold) is only explained
  at the level of "cleaner class separation gives the null components more
  near-equally-good ways to split the bulk data" (§2.1) — not yet tied to
  a specific mechanism (one held-out bag, fold_2's particular 45-bag
  composition, or its PCA basis, which has the highest EVR of the 5 at
  0.843) — not investigated further.

## 7. Candidate fixes considered, and their status

1. **Cross-fold $\hat\pi_b$ consistency check — implemented, run, status
   open (§2).** `src/mil/bayes.py::cross_fold_pi_b_consistency` +
   `scripts/tier5_cross_fold_consistency.py`, tested
   (`tests/test_bayes_mil.py`), run on `exp_608`. It demonstrably separates
   folds R-hat alone mislabels (fold_1 false-passed, fold_3 false-failed).
   **What it does not do**: decide the question for you. It's a
   consistency check, not a ground-truth check — three folds agreeing
   with each other is not logically distinguishable, from this analysis
   alone, from three folds sharing the same mistake. You've flagged this
   directly (§2's boxed note) and asked to decide later how much weight it
   carries; this is not resolved and shouldn't be read as resolved.
2. **Likelihood-weighted model averaging across chains — considered and
   retracted.** The original proposal: instead of hard-selecting the best
   chain, blend chains' posteriors weighted by relative marginal
   likelihood, so bags where modes disagree get an honestly wide interval.
   Checked numerically before building it: the real nat gaps between
   chains (11,875 for fold_3; 195,110 for fold_2) are so far past
   float64's underflow point (~745 nats) that a softmax over them is
   **bit-exact hard selection** — not an approximation of something
   softer. Converting to a per-instance basis shows why this isn't a
   numerics bug to patch around: fold_3's gap is only 0.052 nats/instance
   (a 1.05× likelihood ratio per observation) but accumulates over
   228,814 i.i.d. instances into an astronomical total — a principled
   Bayesian treatment would *also* put ~100% weight on the winner at that
   scale, so there is no real ambiguity left to blend. More fundamentally,
   it targets the wrong failure mode: it can only act where chains
   disagree *within* a fold, but fold_1 and fold_2 — the two folds that
   actually matter — each have internally self-consistent chains (that's
   why fold_1 passes R-hat, and why fold_2's 3 chains agree with each
   other at the bag level, §2.1); there is no within-fold disagreement in
   either case for this method to act on. Retracted; not implemented.
3. **Share component parameters $(\mu_k, \Sigma_k)$ across folds —
   considered and rejected, outright leakage.** Proposed as a way to stop
   one held-out bag from swinging the whole partition. Rejected
   immediately on inspection: every bag is a training bag in 4 of the 5
   folds, so the union of all 5 folds' training sets is literally every
   bag in the dataset — "shared" components would be fit on data that
   includes every bag some fold later scores as held out. This is exactly
   what the `in_sample` leakage discipline (docs/mil_tiers_methodology.md
   §2) exists to prevent, and there's no obvious variant of the idea that
   survives the standard 5-fold held-out protocol. Not implemented, and
   not a direction to revisit without a fundamentally different evaluation
   design.
4. **A spatially-informed graph (Potts/MRF coupling on the pixel grid) —
   built and tested (§8); answers a different question from #1–#3.** Every
   other tier (0 through 5) treats instances within a bag as exchangeable;
   if real diagnostic tissue is spatially contiguous, an exchangeable
   mixture is structurally incapable of finding it. This does not address
   fold_1/fold_2's cross-fold disagreement (§2) — that's a different
   problem — it tests whether spatial coupling can detect contiguous
   structure at all. See §8 for the build, the two real bugs testing
   caught before any real data was touched, and the planted-data
   validation results.
5. **Do not prioritise a richer $\pi_b$ marginal family** (§3.1) — the
   evidence available does not support it, and the one fold where a
   2-component Beta mixture fits better (fold_1) is already known to be
   the "good R-hat, wrong mode" fold for an unrelated reason.
6. **Investigate fold_2's specific sensitivity** (§6) — confirmed not a
   data-quality issue (it's Tier 0's *best*-performing fold), narrowed to
   "cleaner class separation gives the null components more
   near-equally-good ways to split the bulk data" (§2.1), not tied to a
   specific mechanism. Lower priority than the above since #1 already
   flags it without needing to know why.

## 8. The spatial extension — build, tests, and the two bugs testing caught

`src/mil/bayes_spatial.py`, `tests/test_bayes_spatial.py`. Built on
explicit instruction to test on known/trivial data *before* touching real
data — §8.2 is exactly why that instruction mattered.

### 8.1 What changed, what didn't

Every component of §0's model is reused unchanged — same NIW prior/
conjugate update, same $\mathrm{Beta}(a,b)$ hyperprior and Metropolis step,
same Model (C) hard support restriction, same structure-aware init. The
only change: instances within a bag are no longer conditionally
independent given $\pi_b$. A fixed (not inferred — see
`docs/tier5_bayesian_mil.md` §7 item 4's reasoning on why learning a Potts
coupling is out of scope) global scalar $J$ adds a pairwise term rewarding
8-connected spatial neighbours (same bag only, never across bags — each
bag is a physically separate slide) for sharing a state:

$$p(z) \propto \exp\Big(J\sum_{(i,j)\in E}\mathbb 1[z_i{=}z_j]\Big)\prod_i p(z_i\mid\pi_{b(i)})$$

**The z-step** can no longer be one vectorised categorical draw (§0.4(ii)),
since each site now depends on its neighbours' current values. Fixed via
4-colour checkerboard Gibbs: `four_color_grid` assigns colour
$2(\text{row}\bmod 2)+(\text{col}\bmod 2)$, a pure graph-theory fact (any
actual 8-connected neighbour offset changes row mod 2 or col mod 2, so
same-coloured sites are never adjacent) that makes each colour class a
valid vectorised block-Gibbs update conditioned on everything else's
current value; sweeping all 4 colours composes into one valid full-sweep
transition. `build_spatial_adjacency` builds the sparse within-bag-only
8-neighbour graph once per fold (not per iteration).

### 8.2 Two real bugs, found by testing before touching real data

1. **Init deadlock in blind scoring.** The first version of
   `score_bayes_mil_spatial_blind` started each candidate-class mini-chain
   at random null states. With $J{>}0$, a site can only reach the
   diagnostic state by copying a neighbour already there — starting every
   site at null gives the diagnostic state zero neighbour support
   anywhere, for *both* candidate hypotheses equally. Result: **0%**
   accuracy (worse than the 25%/50% chance level) on an otherwise
   trivially-separable planted blob (`class_offset=10`, the same setting
   the non-spatial `score_bayes_mil_blind` recovers at 100%). Fixed with a
   proper MAP warm start (per-instance likelihood + $\pi_b$ prior, ignoring
   the spatial term, before any Gibbs sweeps run).
2. **Conceptual scoring bug.** After fixing (1), accuracy was *exactly*
   0% — not noisy, which is itself a clue. Tracing the raw per-state
   log-likelihoods confirmed the fit was perfect (the true diagnostic
   component sat at log-lik $-5.66$ for the matching hypothesis vs.
   $-234.6$ for the mismatched one — completely unambiguous), so the bug
   was downstream, in how candidates were *compared*: the original version
   scored each hypothesis by the **normalised** conditional probability of
   whichever state the sampler settled on. That's the wrong quantity —
   under the *wrong* hypothesis, the sampler can "explain away" the blob
   with a mediocre-but-best-available null state (log-lik $\approx{-}12$)
   and report *high confidence* in that forced local choice, which is
   indistinguishable, after normalisation, from the correct hypothesis's
   genuinely good fit. Fixed by switching to the **unnormalised**
   pseudo-likelihood numerator (`_unnormalized_log_weight_at_z`), which
   preserves the absolute fit-quality gap the normalised version discarded.
   After both fixes: the same planted blob recovers at 100%, with a clean,
   large margin (e.g. one bag: $-606$ vs. $-1386$ nats for the right vs.
   wrong hypothesis).

Neither bug would have been caught by going straight to real data — both
produced confident, clean-looking (if wrong) numbers, exactly the kind of
result that would have been tempting to report without the planted,
known-answer check this section exists to require.

### 8.3 Behavioural validation on planted data (all passing)

| test | what it checks | result |
|---|---|---|
| 4-colouring | no two 8-adjacent cells ever share a colour | exact, pure graph property |
| adjacency | hand-verified edge counts on a 3×3 grid (corner=3, edge=5, centre=8); no cross-bag edges even at identical coords; no phantom edges to filtered-out cells | exact match |
| $J{=}0$ reduction | must recover as well as the non-spatial model | iid 0.90+ vs. spatial($J{=}0$) 0.90+, within 0.1 of each other |
| **contiguous blob, weak signal** | does spatial coupling help when the per-instance signal alone is noisy but the truth is a clean contiguous region | **yes** — iid 0.897, spatial($J{=}1.5$) 0.968 — but **not monotonic**: $J{=}0.5$ (0.849) and $J{=}1.0$ (0.796) both scored *below* the iid baseline at this iteration budget; only $J\ge1.5$ clearly helped. Recorded as a real finding, not smoothed over by only reporting the $J$ that worked. |
| **collapse at high $J$** | the failure mode predicted from the parameter-count/edge-count argument (§ the turn that proposed this build) | **confirmed directly**: on pure noise (no planted structure at all), $J{=}4.0$ collapses noticeably toward a single dominant state relative to $J{=}0$ |
| blind classification | genuine, never-given-the-label classification end to end | 100% on a well-separated planted blob, after the two fixes above |

### 8.4 Real-data run — the $J{=}0$ control failed, and why that's informative rather than fatal

`scripts/tier5_spatial_blind_bag_classification.py` — same genuine blind
protocol as §4.3, swept over $J\in\{0,1,2,4\}$ per fold (3 chains per
fold per $J$, selected by `marginal_log_likelihood`, same convention as
`fit_bayes_mil_chains`). Per-sweep cost measured directly before
launching (not estimated): **~1.01× the non-spatial model's** at real
fold scale (225k instances) — the checkerboard z-step's sparse-matrix
operations are cheap relative to the NIW component-fitting step that
dominates both models' cost equally; the 3–8× theoretical slowdown
estimated before building this was too pessimistic. Smoke-tested end to
end on real `fold_0` data before the full launch.

**Results** (blind bag accuracy, mean across folds):

| $J$ | fold_0 | fold_1 | fold_2 | fold_3 | fold_4 | mean |
|---|---|---|---|---|---|---|
| 0.0 | 0.571 | 0.750 | 0.857 | 0.636 | 0.333 | 0.630 |
| 1.0 | 0.714 | 0.750 | 0.857 | 0.545 | 0.500 | 0.673 |
| 2.0 | 0.571 | 0.500 | 0.786 | 0.545 | 0.500 | 0.581 |
| 4.0 | 0.714 | 0.500 | 0.643 | 0.545 | 0.500 | 0.581 |

**$J{=}0$ was included specifically as a control — it should reproduce
§4.3's already-validated numbers (fold_0 0.643, fold_1 0.750, fold_2
0.571, fold_3 1.000, fold_4 0.583, mean 0.710). It does not.** fold_1
matches almost exactly (0.750 vs. 0.750); fold_0, fold_2, fold_3, and
fold_4 all diverge substantially, fold_2 and fold_3 most of all (0.857
vs. 0.571; 0.636 vs. 1.000).

**Diagnosed before drawing any conclusion from the sweep, not assumed.**
First checked whether the spatial pseudo-likelihood scorer itself
disagrees with the exact closed-form scorer: on a well-separated planted
signal (the same setup §8.3's blind-classification test uses), both
score every held-out bag identically and correctly — the scorer is not
broken, and the decision it produces is robust to the magnitude
differences between the two approaches when the signal is unambiguous.
That rules out a scoring bug and points at the other candidate: **the fit
itself lands in a different mode.** `fit_bayes_mil_spatial`'s checkerboard
Gibbs consumes random numbers in a completely different pattern from
`fit_bayes_mil`'s one-shot vectorised draw — even at an identical seed,
it walks a different path through the posterior. For fold_1 — already
established (§2) as having three *mutually agreeing* chains, i.e. no real
multimodality to be sensitive to — this makes no difference. For
**fold_2 and fold_3 — the two folds this entire investigation has
repeatedly found to be acutely multimodal** (§2's cross-fold consistency
check, §2.1's chain-by-chain reinspection) — a different RNG path lands
somewhere genuinely different, and the numbers diverge sharply, exactly
where the rest of this document would predict they should.

**Consequence: the $J{=}1/2/4$ vs. $J{=}0$ comparisons in this sweep are
confounded by the same mode-selection sensitivity documented everywhere
else in this tier, and the pattern in the table above (a mild uptick at
$J{=}1$, a decline at $J{=}2/4$) cannot be attributed to the spatial
coupling with any confidence** — it could equally be more of the same
noise that made the $J{=}0$ control itself fail to reproduce. This is not
a new problem specific to the spatial extension; it is the fourth
independent surfacing of the same real-data multimodality problem this
whole document has been documenting (the cross-fold check, §2; the
chain-by-chain reinspection, §2.1; the Tier-0 cross-check, §2.1's
addendum) — now showing up as "two different, mathematically-equivalent
code paths land in different modes," rather than being fixed by building
a more capable model on top.

**What a trustworthy version of this comparison would need**: either many
more chains per (fold, $J$) combination to make mode-selection noise
small relative to any real $J$ effect, or using §2's cross-fold
consistency check itself as an additional filter to identify, per fold,
which $J$'s fit is landing in a cross-validated-trustworthy mode before
comparing accuracy at all. Neither has been done — this section reports
what was run and why it should not be over-interpreted, not a final
verdict on whether spatial coupling helps on real data.
