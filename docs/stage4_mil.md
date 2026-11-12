# Stage 4 — Latent store and Tier 1, constrained binary

**Status: complete, including the real-data fit.** New package `src/mil/`
(`misvm.py`, `store.py`), new `pipeline/mil_fit_stage.py` and
`pipeline/modal_mil_fit.py`, wired into `pipeline/run_suite.py`. Real fit on
exp_501 finished; results in §5 -- mi-SVM hadn't fully converged, surfaced by
running this for real rather than assumed from the synthetic tests alone.

> **Reproducing this.** `pytest tests/test_stage4_mil.py` (11 tests, synthetic
> data only). Real fit: `python -m pipeline.run_suite --configs exp_501
> --stages mil_fit --force` (needs exp_501's `trtr`-trained data path only —
> mi-SVM trains its own classifiers from scratch and does not reuse TRTR's
> saved models).

## 1. Scope: constrained model (C), binary, per the stage's own name

Per docs/MIL_REFRAME_PLAN.md §1: model (M) (the unconstrained mixture,
$z_{bi}$ ranging over all $K{+}1$ types) is Stage 8. This stage implements
model (C) only — each instance's latent type is restricted to
$\{0_1, 0_2, 0_3, Y_b\}$, never another diagnostic class — with $K{=}2$
(binary IDH status, matching exp_501).

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
  docs/stage3_instance_hygiene.md §4); not used by mi-SVM, available for a
  future refinement.

## 3. Verification

- **`tests/test_stage4_mil.py`, 11/11 passing.** Covers the plan's Stage 4
  test bullets: a fold-leakage assertion that raises on a deliberately
  mismatched fold (asking about a bag that *was* in-sample as though it were
  held out); an alignment-checksum test that rejects a stale posterior file
  keyed to a different `sample_ids`/`coords` array; mi-SVM's synthetic
  class-separation recovery (with and without per-bag SVM subsampling); its
  `predict`/`score` functions exercised on held-out data; and the store's
  `config_hash` provenance (stable across processes).
- **Full suite re-run**: no regressions elsewhere.
- **Real data**: `mil_fit_stage` was launched against exp_501 (as part of its
  full retrain — see docs/stage3_instance_hygiene.md §5) and was in progress
  when this document was written; results in §5.

## 4. Deviations from the plan

- **MI_SVM's witness set and mi-SVM's churn history are computed and returned
  but not yet surfaced in any report/table** — available on the `fit_misvm`
  return dict (`witness_idx`, `churn_history`) for whichever later stage wants
  to visualise them on the slide maps (Stage 5's spatial validation is the
  natural consumer).
- **No `pi_min` sweep (1–5%) yet** — a single fixed value is used throughout;
  the plan lists this as a sensitivity check (§1), not a Stage 4 requirement.

## 5. Real-data validation

`mil_fit_stage` on exp_501 completed (all 5 folds; see §5.1 for why this was
tractable at all). mi-SVM converged within its 10-iteration cap in every fold,
but was still refining, not stable: churn was still decreasing at the final
iteration in every fold/class (e.g. fold_0 class 0: `...1594, 2222, 1528,
1422`, still moving). It hit the iteration ceiling, not a fixed point.
`max_iters` should be raised before trusting these posteriors fully.

### 5.1 A necessary fix found only by actually running this on real data

The synthetic tests (§3) ran on bags of ~40-80 instances; exp_501's real
folds are ~200k+ instances (Stage 4 retired `patient_cap` deliberately, so
mi-SVM sees every instance). Apollo's RBF SVM scales badly with sample
count -- a single fit at full fold size was confirmed empirically to be
impractically slow (a local run sat in fold_0's first SVM fit for over 14
hours with no result). Fixed with `svm_subsample_per_bag=500`
(`src/mil/misvm.py::_subsample_for_svm`): caps each bag's contribution to the
SVM's own *training* set at the historical `patient_cap` value, per bag (not
a flat global subsample, so no bag risks being starved out entirely) and per
iteration (a fresh subsample each round, not one arbitrary slice reused
throughout). Scoring (`predict_proba`, which drives both relabeling and the
returned posterior) always runs on every instance regardless -- only the fit
itself is capped. This is what made the run tractable at all -- without it,
mi-SVM was not completing in practical time.

Also added, for the same underlying reason (this workload doesn't fit
comfortably on one machine, and this project's Modal wiring already existed
for the CVAE/classifier stages): `pipeline/modal_mil_fit.py` parallelizes
mi-SVM across the 5 folds as independent CPU-only Modal containers
(sklearn work throughout, no GPU benefit) rather than the local sequential
loop -- checked into a memory-constrained-machine problem specifically (13.7GB
total RAM, 1.9GB free at the time), not a speed complaint. `pipeline/
modal_classifiers.py` separately gained real GPU support (`gpu="T4"`) for the
1D-CNN, with a hard-fail check if TensorFlow doesn't actually see the
requested GPU -- validated cheaply (a throwaway matmul probe, not a real
training run) before ever being used for anything long, directly per a
past-incident concern about silently-CPU-priced-as-GPU credit costs.

## 6. What Stage 5+ inherits

- `src/mil/store.py`'s posteriors are the first genuine per-instance latent
  estimates in the project — Stage 5's spatial validation (Moran's I, ARI
  against `hard_labels.npy`) has real posteriors to validate for the first
  time.
  `assert_no_fold_leakage` is ready to be run as a standing invariant on every
  future write to the store, per §12.
