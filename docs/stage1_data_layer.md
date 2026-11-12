# Stage 1 — data layer and re-export

**Status: complete.** Branch `mil-reframe`. Producer: `src/data/preprocessing.py` via
`scripts/build_glioma_mil.py`. Output: `data/processed/glioma_mil.h5`.

> **Reproducing this.** `python scripts/build_glioma_mil.py --workers 12` rebuilds the
> export from raw cubes + the precomputed per-slide clustering artifacts (~5 min on
> 12 cores). `python scripts/stage1_lambda_sensitivity.py` reproduces the λ study in
> §3. Tests: `pytest tests/test_stage1_export.py` (fast tests always run; a handful
> that recompute from raw data are skipped if `data/raw/` is absent, and the
> export-dependent ones skip if `glioma_mil.h5` hasn't been built).

## 1. What changed and why

Stage 1's job was to fix `src/data/preprocessing.py`, which could not run at all —
`gaussian_filter` was used without being imported, `_build_file_to_map_correspondence`
was called but never defined, and the tumour-map key lookup appended `.npy` to a key
that never had it, so every slide's mask always resolved to `None` and was skipped.
The actual producer for every dataset that has ever existed on disk was notebook 01
section 4; that logic is folded into `src/data/preprocessing.py` here, so the
notebook goes back to being exploratory rather than a build dependency.

Three deliberate departures from what that notebook did, beyond the bug fixes:

1. **The export mask widened from "manually-selected tumour clusters" to "all
   strong-signal pixels".** The old selection was never a tumour label — it is a
   biologically-viable-signal mask: NMF(100) → GMM(3) *independently per slide*,
   then a person picked which of the 3 resulting clusters to call "tumour" in
   `data/processed/file_names.csv`. That this is a judgement call, not a
   measurement, is directly visible in the data: `metadata.json`'s own automated
   `tissue_identification` heuristic disagrees with the manual selection on the very
   first slide checked (`HF-1002_V1AT_1.h5_0`: the heuristic's highest-mean-intensity
   cluster is `2`, but `file_names.csv` selected cluster `0` as "tumour" for that
   slide). The per-slide cluster ids have no cross-slide semantic meaning either — a
   fresh, independent GMM fit per slide assigns cluster labels in arbitrary order, so
   `tissue_cluster` is exported as a **weak, per-slide-only prior**, not a portable
   label; the plan's downstream ARI/AMI validation (Stage 5) is permutation-invariant
   to this by design.

   Practically: the old, narrower selection survives exactly as the boolean
   `viable_legacy` column (verified to reproduce `tumor_mask.npy` pixel-for-pixel on
   every slide checked — §4), so anything that depended on the old subset still can,
   while the wider export makes non-diagnostic tissue available to the MIL models
   that are the point of this reframing.

2. **Baseline correction is IARPLS**, not airPLS(λ=100) — the Stage 0 finding.
   Still strictly per-spectrum (§3 covers what remains open about it).

3. **Clustering is not re-run.** NMF(100)/GMM(3) per slide is expensive (~500 NMF
   iterations per slide) and already exists on disk
   (`data/processed/<slide>/{hard_labels,strong_signal_mask,tumor_mask}.npy`,
   produced by notebook 01 sections 2–3). This module reads those artifacts; it does
   not recompute them. If a slide's directory is missing them, that slide is skipped
   and logged, not silently dropped.

## 2. Schema

`data/processed/glioma_mil.h5`, **288,678 instances, 59 slides** (5.0 min to build,
12 workers; 1.52 GB):

| dataset | shape | dtype | notes |
|---|---|---|---|
| `spectra` | (288678, 1738) | float32 | uncropped; IARPLS-corrected + min-max normalised per spectrum |
| `coords` | (288678, 2) | int32 | `(col, row)` pixel position; convention fixed and tested in `src/data/spatial.py` |
| `sample_ids` | (288678,) | str | bag id, e.g. `HF-2104_V1T_1.h5_2` |
| `patient_ids` | (288678,) | str | **changed**: `"HF-<N>"` string, was a bare int32 |
| `y_bag` | (288678,) | int32 | **renamed** from `y_class` — a bag label broadcast to instances, never an instance label |
| `tissue_cluster` | (288678,) | int32 | per-slide GMM(3) cluster id — weak prior, **not comparable across slides** (§1) |
| `viable_legacy` | (288678,) | bool | was this pixel in the old manual "tumour" cluster selection? **227,344 True**, exactly the old production total |
| `slide_names` | (59,) | str | side table |
| `slide_shapes` | (59, 2) | int32 | `(n_rows, n_cols)` per slide — previously recorded only in per-slide `metadata.json`, now queryable in one place |
| `wavenumbers` | (1738,) | float32 | unchanged |

`y_tumor` is **dropped entirely** — it was a misnomer (the mask it recorded was
viability, not tumour status) and, in every `results/exp_*/data/glioma_dataset.h5`
export, a constant `1` carrying no information.

HDF5 attrs record provenance: `baseline_method="iarpls"`, `baseline_lambda=10000.0`,
`baseline_lambda_status="provisional"` (§3), `median_kernel=3`, `schema_version="mil-v1"`.

**Deferred**: `nmf_W` (the plan's schema table marks it optional, ~91 MB). Its flat
per-slide layout would need a verified row-major/column-major convention check before
export, and nothing in Stages 1–4 consumes it — deferred until a consumer exists
rather than exported speculatively.

## 3. λ remains provisional — properly investigated, not re-deferred

Stage 0 flagged `IARPLS_LAMBDA = 1e4` as provisional because its synthetic generator
was measurably smoother than real baselines and its real-data check used only ~40–60
spectra from a handful of slides. Carrying that forward as another unexamined
"provisional" flag would be a cop-out, so two things were actually done.

### 3.1 Per-spectrum L-curve — the plan's first candidate resolution, tested and rejected

`scripts/stage1_lambda_sensitivity.py` implements a per-spectrum L-curve selector for
IARPLS: the same trade-off-curve-corner method that works cleanly for airPLS
(0% edge-pinning in Stage 0). It does not transfer, and the reason is architectural,
not incidental. My airPLS L-curve computes the trade-off curve from the
**reweighted** residual, which is near-zero at peak locations by convergence — so it
never reads the peaks as "residual to shrink". IARPLS is a black box from this side
(`pybaselines`/`ramanspy` expose no intermediate weights), so the only available
trade-off curve uses the **raw, unweighted** residual `y - baseline`, which the peaks
dominate. The corner finder reads that as "the fit is too stiff" and collapses λ —
the same failure mode as GCV, for a closely related reason.

Measured on 60 real spectra, stratified across slides (grid × N = 7 × 60 = 420 IARPLS
fits):

| | value |
|---|---|
| median selected λ | 1000 (vs. the fixed default 10⁴) |
| at grid edge | 0% (not simply broken, but systematically low) |
| median retention vs. fixed λ=10⁴ | 0.79 (loses ~21% of fingerprint amplitude) |
| fraction with retention < 0.7 | 33% |

**Rejected.** A smaller first pass (n=15, one slide) had shown a more dramatic
collapse (median λ=10, retention 0.33–0.63) — noise from too small a sample, not a
different effect; the corrected, larger-sample numbers still show clear, systematic
under-smoothing, just less extreme than the first pass suggested. This is the value
of running the larger sample rather than reporting the first result.

The plan's second candidate resolution — "a small set of manually curated reference
baselines" — needs domain-expert curation and is out of scope for this stage to do
unilaterally.

### 3.2 A properly powered real-data sweep

1,180 spectra, ~20 per slide, all 59 slides (vs. Stage 0's ~40–60 from a handful):
add synthetic fluorescence (8× signal amplitude), correct with and without it, and
measure both **drift** (Stage 0's criterion — 0 = invariant) and **retention**
(fingerprint amplitude retained relative to the signal's own true scale):

| λ | drift (median) | drift (p90) | retention (median) | retention (p10) |
|---|---|---|---|---|
| 100 | 0.0000 | 0.0053 | 0.414 | 0.273 |
| 316 | 0.0000 | 0.0116 | 0.456 | 0.330 |
| 1,000 | 0.0001 | 0.0130 | 0.529 | 0.427 |
| 3,160 | 0.0004 | 0.0156 | 0.603 | 0.535 |
| **10,000** | **0.0015** | **0.0211** | **0.750** | **0.617** |
| 31,600 | 0.0025 | 0.0172 | 0.879 | 0.824 |
| 100,000 | 0.0034 | 0.0170 | 0.907 | 0.874 |

**This is a genuine trade-off, not a value to be discovered.** Drift and retention
move in opposite directions with λ: a low λ tracks (and removes) *any* baseline
closely, including a newly-added one, so drift is near zero — but at the cost of also
eating real signal, so retention craters to 0.41–0.53. A high λ is too stiff to fully
track a new fluorescence hump, so some of it survives as apparent signal drift — but
genuine structure is preserved, so retention climbs to 0.88–0.91. There is no λ where
both are simultaneously minimised/maximised; which failure mode matters more depends
on the downstream task, which this stage does not evaluate.

`λ=1e4` sits past the point of diminishing drift returns to be more of the way toward
useful retention: 0.0015 median drift is an order of magnitude below what the
*fixed λ=100 arm ever achieves at 31,600 or 100,000*, while retention (0.750) clears
half. It also matches the Stage 0 synthetic oracle and the smooth/structured-family
optimum. It is kept as the **documented production default under this trade-off**,
not claimed as a solved optimum — a downstream-accuracy-driven re-evaluation
(training classifiers at two or three λ values and comparing bag-level accuracy) is
the natural way to actually settle it, and is better done once Stage 4's models exist
to be the judge, reusing the existing reconstruction-loss-ablation harness pattern
(configs 201–203) rather than another proxy metric.

Per-slide drift at fixed λ shows no persistent problem slide: the worst-5 lists at
λ=100 and λ=1e4 barely overlap (Pearson r=0.17 between a slide's mean drift at the two
λ), consistent with sampling noise at n≈20 spectra/slide rather than a data-quality
signal. Even the worst slides at λ=1e4 (drift ≈0.03–0.04) sit an order of magnitude
below the *old* airPLS(λ=100) pipeline's typical drift (0.16–0.55, Stage 0 §4) — the
open question here is a refinement among options that already dominate the previous
pipeline, not a question of whether the new corrector is better.

Full data: `results/stage1_lambda_sensitivity/`.

## 4. Verification

- **Reprojection round-trip, all 59 slides, against the actual written export**:
  `viable_legacy` reprojected via `coords` reproduces `tumor_mask.npy` pixel-for-pixel
  on every slide (`tests/test_stage1_export.py::test_export_reprojection_roundtrip_all_slides`).
  A second, from-raw-cubes version of the same check runs on a representative subset
  (smallest, largest, square, rectangular, plus the ex-`ignore_samples` slide) rather
  than recomputing all 59 with real IARPLS work on every test run.
- **`viable_legacy` count**: exactly 227,344 — the old production total, cross-checked
  earlier against `results/exp_201/data/glioma_dataset.h5`.
- **Purity**: `process_slide` calls the already-tested `correct_iarpls` per spectrum
  (Stage 0); an added test confirms two independent calls on the same slide are
  bit-identical.
- **`slide_shapes` registry**: covers every `sample_id` exactly (set equality
  asserted).
- **The 59-vs-58 discrepancy is resolved, not just flagged.** `ignore_samples` is
  dead config — read only into `ConfigManager`'s defaults, never applied by loader or
  pipeline code — and excluding `HF-1887_via-t_2_1.h5_3` from the old
  `results/exp_102` export was an artifact of the old
  `RamanPreprocessor.IGNORE_SAMPLES` class attribute, not a considered decision. The
  new export includes it (matching `results/exp_201`, which was built via notebook 01
  and also has 59); a test pins this (`test_export_includes_all_59_slides_not_58`).
  **Update, 2026-09-28**: the export having 59 slides turned out not to be enough —
  `data/splits/manual_splits.json` and `data/splits/strict_splits.json` (what
  `GliomaDataLoader.__iter__` actually uses to build each fold's train/val mask)
  independently omitted the slide from every fold's train/test list, so it was
  never trained on or scored anywhere regardless of the export. This is the reason
  Stage 5 found "zero covered instances" for it (docs/stage5_spatial_validation.md
  §2). On inspection the raw spectrum has a different baseline fluorescence level
  than the rest of the cohort but is otherwise sound, and IARPLS correction is
  expected to normalise that; both split files now assign it to one fold's test
  set and train in the other four, the same pattern every other non-duplicated
  slide has (`test_hf1887_reachable_through_both_split_files`). The dataset is
  therefore 59 biopsies going forward, not 58 — anything computed before this fix
  (exp_501's stored Stage 4/5 results included) still reflects 58 and was not
  re-run for this one slide.
- **Loader**: `coords` is now loaded and kept aligned through the tumour filter and
  patient cap (both previously silent no-ops for `coords`, since it was never loaded
  at all); `y_bag` is read with a fallback to `y_class` so pre-Stage-1 exports still
  open. `__iter__` yields `train_coords`/`val_coords`/`val_groups` symmetrically with
  the pre-existing `train_groups`.
- **`prepare_fold`** now returns a `FoldArrays` dataclass that iterates as the
  historical 4-tuple (`X_train, y_train, X_val, y_val = prepare_fold(...)` — all three
  existing call sites in `pipeline/{trtr,tstr,tsrtr}_stage.py` needed **zero**
  changes) while exposing `.train_coords`/`.val_coords`/`.train_bag_ids`/
  `.val_bag_ids` by attribute for the MIL stages that need them from Stage 4 on. The
  patient cap subsets coords/bag-ids with the exact same index array as X/y; a test
  checks this alignment survives the cap rather than assuming it.
- **Full suite**: `pytest tests/` — 37 passed, 2 pre-existing failures in
  `test_config_manager.py` unrelated to this stage (confirmed failing before Stage 0
  began, independent of anything here).

## 5. Deviations from the plan

- **`nmf_W` deferred** (§2, "Deferred").
- **No experiment-config (YAML) migration in this stage.** The ~30 experiment configs
  under `configs/` still point at the old per-experiment
  `results/exp_*/data/glioma_dataset.h5` files, which stay valid and are left alone.
  `data/processed/glioma_mil.h5` is a new, standalone dataset not yet wired into any
  experiment config; that wiring belongs with re-running each suite against it
  (§11's "Suites" section already scopes this to when each suite is actually re-run,
  not to the data-layer stage).
- **`pipeline/preprocessing_stage.py` rewritten**, not patched — the old
  `mask_strategy`/`tumor_maps_path`/`use_ramanspy`/`config=` interface is retired
  along with the class it drove; the new interface takes exactly what the new
  producer needs (`raw_data_dir`, `processed_dir`, `calibration_path`,
  `baseline_lambda`).
- **`tests/test_data_integrity.py` and `tests/test_pipeline.py` replaced, not
  fixed.** Both were already non-functional before this stage (confirmed by running
  them on `main`): the former pointed at `data/tumor_maps.npy`, a path that has never
  existed; the latter called `GliomaDataLoader.get_cross_validation_generator()`, a
  method that does not exist on the class. Ad hoc debugging scripts, not intentional
  regression tests. Replaced by `tests/test_stage1_export.py`.
- **Reprojection helper lives in `src/data/spatial.py`**, not inline in the producer —
  it is also the natural home for Stage 5's spatial-validation code, which needs the
  same function.

## 6. What Stage 2 inherits

- `GliomaDataLoader` and `prepare_fold` are both ready to hand out `coords`/bag-ids;
  Stage 2's evaluation reframing does not need to touch either for that.
- `tissue_cluster` and `viable_legacy` are in the export, ready for Stage 4's latent
  store to use as weak priors / external validation targets, per the original plan.
- The λ trade-off in §3 is a candidate ablation axis once classifiers exist
  (Stage 3+), not a blocker — flagged in `docs/MIL_REFRAME_PLAN.md`.
