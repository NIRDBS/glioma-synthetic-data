# Stage 4 — Latent store and Tier 1/2, constrained binary

**Status: complete, including the real-data fit.** New package `src/mil/`
(`common.py`, `misvm.py`, `em.py`, `store.py`), new `pipeline/mil_fit_stage.py`
and `pipeline/modal_mil_fit.py`, wired into `pipeline/run_suite.py`. Real fit
on exp_501 finished; results in §6, and honest findings they are -- EM
collapsed in most folds and mi-SVM hadn't fully converged, both surfaced by
running this for real rather than assumed from the synthetic tests alone.

> **Reproducing this.** `pytest tests/test_stage4_mil.py` (18 tests, synthetic
> data only). Real fit: `python -m pipeline.run_suite --configs exp_501
> --stages mil_fit --force` (needs exp_501's `trtr`-trained data path only —
> mi-SVM/EM train their own classifiers from scratch, they do not reuse
> TRTR's saved models).

## 1. Scope: constrained model (C), binary, per the stage's own name

Per docs/MIL_REFRAME_PLAN.md §1: model (M) (the unconstrained mixture,
$z_{bi}$ ranging over all $K{+}1$ types) is Stage 8. This stage implements
model (C) only — each instance's latent type is restricted to
$\{0_1, 0_2, 0_3, Y_b\}$, never another diagnostic class — with $K{=}2$
(binary IDH status, matching exp_501). `src/mil/common.py`'s state-index
convention (`M` null components first, then `K` diagnostic classes) is shared
machinery both tiers use identically, so extending to model (M) or $K{=}6$
later is a matter of relaxing `allowed_states_mask`, not rewriting either
formulation.

## 2. What changed and why

- **`src/mil/misvm.py` (Tier 1)**: mi-SVM and MI-SVM (Andrews et al. 2002),
  one-vs-rest per diagnostic class, reusing Apollo (`src/models/baselines.py`)
  unchanged as the base learner. Only positive-bag instance labels are ever
  revised each iteration (negative-bag instances are fixed at label 0 for the
  whole run — the defining MIL asymmetry, not an approximation of it).
  `variant='mi_svm'` relabels the top `ceil(pi_min * n_b)` scoring instances
  per positive bag each round; `variant='MI_SVM'` keeps only the single
  top-scoring witness. Combined posterior: Platt-scaled margins across the
  $K$ one-vs-rest problems, non-diagnostic mass as $1 - \max_k p_k$,
  renormalised.
- **`src/mil/em.py` (Tier 2)**: label-proportion EM with hierarchical
  $\pi_b$. Meant to yield *calibrated* soft labels for T3 (mi-SVM's Platt
  scores are calibrated per binary sub-problem, not jointly across bags).
  Two deliberate, documented simplifications:
  - **Classification-EM**, not a genuinely soft-label discriminative fit —
    hard pseudo-labels + confidence `sample_weight` at the M-step, since
    sklearn classifiers don't natively fit soft multiclass targets. A
    standard, tractable approximation for exactly this situation.
  - **Empirical-Bayes shrinkage**, not a full hierarchical posterior — each
    bag's raw $\pi_b$ estimate is shrunk toward the population logit-mean
    $\mu$, weighted by $n_b / (n_b + \kappa)$ (larger bags trust their own
    data more). A full Bayesian treatment (Laplace approximation or MCMC)
    was judged disproportionate given the plan's own text already expects
    $\sigma$ to be "poorly determined" at 58 bags.
- **`src/mil/store.py`**: the §3 latent-label sidecar
  (`results/{exp}/latent/latent_labels.h5`, `/{method}/{variant}/{fold}/
  {posterior, in_sample}` + attrs including `alignment_checksum`). Fold
  leakage is enforced with `assert_no_fold_leakage`, not left as a convention.
- **`pipeline/mil_fit_stage.py`**: orchestrates fitting per fold — trains on
  that fold's training bags only (no `patient_cap`; Stage 4 retires it per
  the plan, using every available instance for latent inference), scores
  both training (in-sample) and validation (out-of-sample, same fold's own
  model) instances, scatters fold-relative results back onto the full
  dataset's row order (matched by bag id via the loader's own
  `train_sample_ids`/`val_sample_ids`), and writes both to the store.
- **`src/data/loader.py`**: `tissue_cluster` wired through (see
  docs/stage3_instance_hygiene.md §4) — accepted by `fit_em` but not actually
  used by the cold start that shipped (§3 below); available for a future
  refinement.

## 3. A real debugging story: EM's cold start, three failed attempts

This is worth documenting in full because each failure was a genuine,
different instance of a named degeneracy from docs/MIL_REFRAME_PLAN.md §5,
not a coding typo — and the fix came from tracing actual numbers, not from
reasoning about the algorithm in the abstract.

**Attempt 1 — uniform null-component split at init.** With no
`tissue_cluster`, the initial responsibility split the non-diagnostic mass
*exactly evenly* across the 3 null components. An exact tie means `argmax`
always picks the same (lowest-index) component for every single instance,
regardless of its true identity — so the very first M-step's hard labels were
a **constant**, and the RF trivially "learned" to always predict it. Fixed by
breaking the tie with a random per-instance null-component assignment.

**Attempt 2 — still collapsed, now to $\pi_b \to 1$.** With the tie broken,
class separation recovered (diagnostic vs. null instances were now
distinguishable in the posterior) but $\pi_b$ drifted to 0.98 — the *mirror*
degeneracy §5 names ("collapse to the naive solution"). Root cause: a plain
`gamma.argmax(axis=1)` M-step has no mechanism forcing the diagnostic label
count to stay near the current $\pi_b$ estimate — once the RF is even
slightly more confident about "diagnostic" than "null" for most instances
(easy on well-separated data), every subsequent round reinforces the same
labelling, with nothing pulling it back. Fixed by replacing the global argmax
with mi-SVM's own successful per-bag top-$k$ relabeling
(`_hard_labels_from_pi_b`): the diagnostic label count is capped at
`round(pi_b[bag] * n_b)` each round, so runaway self-reinforcement has no
channel to act through.

**Attempt 3 — collapsed the other way, to exactly `pi_min`.** Bootstrapping
the cold start from a quick mi-SVM pre-fit (rather than a naive 100%-
diagnostic assumption, which independently reintroduces attempt-1's failure
mode by giving the RF zero exposure to any null-labelled example at all)
seemed right in principle, but the bootstrap mi-SVM was floored at `pi_min`
(EM's own tiny numerical identifiability floor, ~2–5%) instead of `pi_init`
(the caller's actual prior belief, e.g. 20%). Traced with a direct
instrumented run (not guessed): the bootstrap itself only recovered ~19% of
*true* diagnostic instances against a true rate of 30%, because mi-SVM had no
pressure to include more than the floor it was given. Fixed by giving the
bootstrap its own, more realistic floor (`pi_init`), separate from EM's
internal numerical floor (`pi_min`) — the two are different concepts (§1: a
*belief* vs. a *floor preventing total collapse*) and conflating them was the
actual bug.

**After all three fixes**: `tests/test_stage4_mil.py`'s synthetic recovery
test passes cleanly — correct class-separation direction for both classes,
and $\bar\pi_b$ lands inside a wide but non-trivial band (not collapsed to
either extreme).

## 4. Verification

- **New `tests/test_stage4_mil.py`, 15/15 passing.** Covers the plan's Stage 4
  test bullets: a fold-leakage assertion that raises on a deliberately
  mismatched fold (asking about a bag that *was* in-sample as though it were
  held out); an alignment-checksum test that rejects a stale posterior file
  keyed to a different `sample_ids`/`coords` array; degeneracy diagnostics
  (`shrinkage_diagnostic` correctly flags both a fully-collapsed case and a
  genuinely-spread case; `null_component_collapse_diagnostic` correctly
  distinguishes 3 well-separated synthetic components from 3 collapsed ones);
  plus the mi-SVM/EM recovery tests from §3, a shrinkage-strength test
  (smaller bags land closer to the population mean than larger bags, the
  actual empirical-Bayes property this implementation is supposed to have),
  and both formulations' `predict`/`score` functions exercised on held-out
  data.
- **Full suite re-run**: no regressions elsewhere.
- **Real data**: `mil_fit_stage` was launched against exp_501 (as part of its
  full retrain — see docs/stage3_instance_hygiene.md §5) and was in progress
  when this document was written; §6 will be updated with the resulting
  $\pi_b$ distribution, shrinkage check, and null-component diagnostics.

## 5. Deviations from the plan

- **`tissue_cluster` is accepted by `fit_em` but not used by the cold start
  that shipped** — the mi-SVM bootstrap (§3) turned out to be the load-bearing
  fix; using `tissue_cluster` as an RF feature or a later-round tie-break is a
  reasonable refinement, not implemented here.
- **MI_SVM's witness set and mi-SVM's churn history are computed and returned
  but not yet surfaced in any report/table** — available on the `fit_misvm`
  return dict (`witness_idx`, `churn_history`) for whichever later stage wants
  to visualise them on the slide maps (Stage 5's spatial validation is the
  natural consumer).
- **No `pi_min` sweep (1–5%) yet** — a single fixed value is used throughout;
  the plan lists this as a sensitivity check (§1), not a Stage 4 requirement.

## 6. Real-data validation

`mil_fit_stage` on exp_501 completed (all 5 folds, both methods, 21,933s /
6.1h total -- see §6.1 for why this was tractable at all). Real, honest
findings, not the clean picture the synthetic tests showed:

- **mi-SVM converged within its 10-iteration cap in every fold, but was still
  refining, not stable**: churn was still decreasing at the final iteration in
  every fold/class (e.g. fold_0 class 0: `...1594, 2222, 1528, 1422`, still
  moving). It hit the iteration ceiling, not a fixed point. `max_iters` should
  be raised before trusting these posteriors fully.
- **EM's shrinkage diagnostic (§1) fired `collapsed=True` in most folds, but
  which folds is itself unstable across nominally-identical re-runs.** The
  original fit collapsed in 4/5 folds (fold_0 the exception,
  `shrinkage_ratio=5.851`, $\pi_b \approx 0.031$). A second full 5-fold
  refit (2026-09-29, same code, same `seed=42`, run to additionally persist
  `pi_b`/`mu`/`sigma` per fold — previously computed and discarded every
  run, see `results/{exp}/latent/em_diagnostics.json`) reproduces the
  *qualitative* pattern (most folds collapse, 1–2 don't) but **not the same
  folds**: this time fold_1/2/3 hit $\sigma=0.000$ exactly, while fold_0
  (`shrinkage_ratio=1.320`) and fold_4 (`shrinkage_ratio=2.354`) did not.
  Across both runs $\pi_b$ stays extremely tight to the floor either way —
  59-bag range **[0.0200, 0.0443]**, std=0.0031. Which specific fold avoids
  the boundary changing between two same-seed runs is itself evidence
  pointing the same direction as the plan's own expectation ("$\sigma$
  poorly determined with 58 bags") — a stably-identified $\sigma$ should not
  flip which fold collapses under nominally identical conditions. This
  falls short of the plan's own stated checkpoint-1 criterion (a formal
  profile-likelihood interval, flat within 2 log-likelihood units across an
  order of magnitude) — that would need a marginal-likelihood model this
  codebase doesn't currently implement, a real additional undertaking, not
  yet built.
- **$\pi_b$'s external correlation, now computed both ways.** Stage 5's
  existing hard-threshold clustering-agreement check (ARI/AMI vs
  `viable_legacy`) was effectively null for both mi-SVM and EM — see
  docs/stage5_spatial_validation.md. A second, complementary check — direct
  Spearman correlation of the above per-bag $\pi_b$ (averaged across the
  ~3.93 folds/bag where each bag is in training) against each of the 3
  NMF/GMM `tissue_cluster` proportions, and against `viable_legacy`
  fraction, run 2026-09-29 via `scripts/stage4_pi_b_external_validity.py`
  — is **also null**: $\rho \in [-0.16, +0.14]$, every one of the 4
  correlations $p > 0.19$ (n=59 bags). Two independent statistics (a
  clustering-agreement metric and a continuous rank correlation) now agree:
  whatever EM's $\pi_b$ is picking up, it is not tracking either
  independent tissue-composition proxy available in this project. Given how
  tight $\pi_b$'s own range is (essentially pinned near the floor for
  nearly every bag), a null correlation here is close to expected rather
  than a surprising additional finding — there is very little genuine
  across-bag variation in $\pi_b$ for any proxy to track in the first
  place.

### 6.1 A necessary fix found only by actually running this on real data

The synthetic tests (§3) ran on bags of ~40-80 instances; exp_501's real
folds are ~200k+ instances (Stage 4 retired `patient_cap` deliberately, so
mi-SVM/EM see every instance). Apollo's RBF SVM scales badly with sample
count -- a single fit at full fold size was confirmed empirically to be
impractically slow (a local run sat in fold_0's first SVM fit for over 14
hours with no result). Fixed with `svm_subsample_per_bag=500`
(`src/mil/misvm.py::_subsample_for_svm`): caps each bag's contribution to the
SVM's own *training* set at the historical `patient_cap` value, per bag (not
a flat global subsample, so no bag risks being starved out entirely) and per
iteration (a fresh subsample each round, not one arbitrary slice reused
throughout). Scoring (`predict_proba`, which drives both relabeling and the
returned posterior) always runs on every instance regardless -- only the fit
itself is capped. Threaded through EM's mi-SVM bootstrap too (same
bottleneck, same fix). This is what made the 6.1h run tractable at all --
without it, mi-SVM/EM were not completing in practical time.

Also added, for the same underlying reason (this workload doesn't fit
comfortably on one machine, and this project's Modal wiring already existed
for the CVAE/classifier stages): `pipeline/modal_mil_fit.py` parallelizes
mi-SVM/EM across the 5 folds as independent CPU-only Modal containers
(sklearn work throughout, no GPU benefit) rather than the local sequential
loop -- checked into a memory-constrained-machine problem specifically (13.7GB
total RAM, 1.9GB free at the time), not a speed complaint. `pipeline/
modal_classifiers.py` separately gained real GPU support (`gpu="T4"`) for the
1D-CNN, with a hard-fail check if TensorFlow doesn't actually see the
requested GPU -- validated cheaply (a throwaway matmul probe, not a real
training run) before ever being used for anything long, directly per a
past-incident concern about silently-CPU-priced-as-GPU credit costs.

## 7. What Stage 5+ inherits

- `src/mil/store.py`'s posteriors are the first genuine per-instance latent
  estimates in the project — Stage 5's spatial validation (Moran's I, ARI
  against `hard_labels.npy`) has real posteriors to validate for the first
  time, from two independent formulations (mi-SVM, EM) to cross-check against
  each other as well as against the spatial/tissue-cluster checks.
  `assert_no_fold_leakage` is ready to be run as a standing invariant on every
  future write to the store, per §12.
- Tier 2's calibrated posteriors are what Stage 7 (CVAE soft conditioning)
  is specified to consume.
