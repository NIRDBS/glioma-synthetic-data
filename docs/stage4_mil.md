# Stage 4 — Latent store and Tier 1/2, constrained binary

**Status: complete (code + tests); real-data fit in progress.** New package
`src/mil/` (`common.py`, `misvm.py`, `em.py`, `store.py`), new
`pipeline/mil_fit_stage.py`, wired into `pipeline/run_suite.py`.

> **Reproducing this.** `pytest tests/test_stage4_mil.py` (15 tests, synthetic
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

## 6. Real-data validation (to be filled in once the fit completes)

`mil_fit_stage` on exp_501 was in progress when this document was written.
Once complete: the fitted $\pi_b$ distribution across the 58 bags, the
shrinkage-check verdict, the null-component collapse diagnostic, and — the
plan's own priority ("worth prioritising") — $\pi_b$'s external correlation
against the `viable_legacy` fraction and NMF cluster proportions, not yet
computed here.

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
