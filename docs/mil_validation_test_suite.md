# MIL Validation Test Suite — reference list

Consolidated, complete list of every method used in this project to validate
whether a MIL tier's per-instance posterior reflects genuine diagnostic
structure, rather than an artefact of the objective, the formulation, or (for
a spatially-regularized model) the model's own construction. Extracted
directly from `docs/stage5_spatial_validation.md` and
`docs/mil_tiers_methodology.md` §9–10, not re-derived. This is a methods
reference for drafting the paper from — not a findings doc (see
`docs/stage5_spatial_validation.md`, `docs/stage6_attention_mil.md`,
`docs/stage6_generative_mil.md` for results) and
not a coverage map (see `docs/experiment_inventory.md`).

---

## 0. Why two independent test families, not one

Spatial coherence (family A, below) asks: *does this tier's posterior have
structure on the slide?* Key-instance utility (family B) asks a different
question: *are the instances the structure selects actually the diagnostic
ones?* These are not redundant, and a tier can pass one while failing the
other for principled reasons:

- A formulation whose objective has no access to geometry (every MIL tier in
  this project — `inspect.signature` enforces no `coords`-shaped parameter
  can reach any fit function) can still show lag-band spatial structure
  purely because real tissue has real spatial autocorrelation in its raw
  spectral features, independent of which label a region is being fit toward
  — confirmed to be exactly what happened for all four tested claims in this
  project (§2).
- **Conversely, and the reason family B is mandatory, not optional, going
  forward**: a model that spatially *regularizes* its own posterior by
  construction — e.g. one that couples each instance's latent state to its
  neighbours via an explicit pairwise term — will show lag-band
  coherence *because the model enforces it*, not because it recovered real
  diagnostic structure. For such a model, spatial-coherence family A becomes
  close to circular (it is testing whether the smoothing prior worked, not
  whether the result is meaningful), and family B is the test that still has
  teeth: smoothing a posterior does not, by itself, guarantee the smoothed
  region is the diagnostically-correct one. **Both families must be run for
  any spatially-regularized formulation; family A alone is insufficient.**

---

## A. Spatial coherence (geometric, lag-based)

Code: `src/evaluation/spatial.py`, `scripts/stage5_spatial_validation.py`,
`scripts/stage6_shuffled_label_control.py`. Tests: `tests/test_stage5_spatial.py`
(14/14, synthetic fields of known correlation length, no real-data
dependency).

**Precondition enforced by construction, not convention**: no MIL fit
function anywhere in this project accepts a `coords`-shaped parameter
(`inspect.signature` guard, permanent). Any spatial structure found is
therefore evidence about what the model recovered, not about what it was
given.

**Reprojection**: stored `(col, row)` coordinates place each instance's
non-diagnostic-mass posterior back onto its slide's $(H,W)$ pixel grid.

1. **Moran's I**, by Chebyshev lag 1–10 — scale-free (normalized by total
   variance). Good at detecting pattern; can mislead about how much it
   matters: a numerically-constant map can still report large Moran's I
   z-scores, purely because the statistic is scale-free.
2. **Empirical variogram**, same lag bands,
   $\gamma(h) = \tfrac12\,\mathrm{mean}((x_i-x_j)^2)$ — scale-*sensitive*;
   reads exactly 0.0000 for a genuinely constant map. **Trust the variogram
   over Moran's I whenever they disagree.**
3. **Variogram range** (first lag reaching ~90% of the plateau value) —
   reported as a descriptive diagnostic in its own right, not just a
   byproduct of #2: ~1–2px is squarely in filter-artefact territory (the
   project's 3×3 median filter provably cannot manufacture correlation
   beyond Chebyshev distance 2); ~10–20px would be consistent with real
   tissue architecture. No independent validated ground truth exists for the
   "correct" size, so this is reported descriptively, not scored against a
   target.
4. **Permutation null**: shuffle in-mask values within each slide's own mask
   (geometry held fixed), 200 repeats, per-lag p-value/z-score for Moran's I.
5. **Shuffled-bag-label control** (the definitive test) —
   `src/mil/shuffle_control.py::permuted_bag_label_map` computes an **exact
   permutation of the true label multiset across bags** (not independent
   resampling, so class balance is identical to the real run; one
   permutation computed once, so a bag spanning two CV folds keeps the same
   permuted label in both), substituted for the loader's own label array
   *before* any fold splitting, then the target formulation is refit
   completely unchanged. **Reading it**: if lag ≥ 3 coherence is smaller
   under permuted labels than under the true ones, the real finding is
   credited to the true label. If a permuted-label refit produces
   comparable-or-larger coherence, the real finding cannot be attributed to
   the true label at all.
6. **External agreement** (ARI/AMI vs `viable_legacy`) — not geometric or
   lag-based; a different validation axis (cluster agreement with the
   project's earlier unsupervised tumour selection), explicitly **not**
   ground truth, bundled into the same reporting pass for convenience.

**Lag convention, applies to every result in this family**: lags 1–2 are
contaminated by the project's 3×3 spatial median filter (shown, never used
to support a claim); **lag ≥ 3 is the headline**.

**Two statistics considered and deliberately not implemented**: join-count/
boundary-length statistics and a qualitative RGB overlay — left out because
Moran's I + variogram already gave a clear, mutually-consistent answer;
these would only have strengthened the same conclusion, not offered an
independent second opinion.

**Project-wide result (as of 2026-09-30, `exp_501`)**: four
spatial-coherence claims were made (mi-SVM, attention, CbR `reconstruction`,
CbR `encoder_head`); all four were tested with the shuffled-bag-label
control; **all four failed** — permuted-label refits produced comparable-or-
larger lag≥3 coherence in every case. See
`docs/stage5_spatial_validation.md` §6 for the full synthesis.

**Extended, 2026-10-12, to the full Suite 600 β sweep (`607`/`608`/`609`,
β=0.25/0.5/0.75)**: attention and both CbR variants, all three β values —
8 of 9 (config × variant) cells replicate the `exp_501` failure exactly.
**The one exception: `607`'s attention (β=0.25)**, where the real fit shows
*more* lag-3 coherence than its shuffled counterpart — the only case in
the v1 sweep where this direction has been observed. Flagged, not yet
reported as a surviving claim (see §B below — it needs to clear the
key-instance test too before being investigated further). Full numbers:
`docs/stage6_attention_mil.md` §10.1, `docs/stage6_generative_mil.md` §10.1.

**Extended further, same day, to v2 preprocessing (`exp_2.607` β=0.25,
`exp_2.608` β=0.5 — mi-SVM/attention/CbR both variants, both configs)**:
mi-SVM (shared across both — dataset-level, confirmed bit-identical, §4.3
of `docs/experiment_inventory.md`) and attention both fail at both β. CbR
`reconstruction` fails at both β too (shuffled ≈ real). **CbR
`encoder_head` is a second, independent exception — real beats shuffled at
*both* v2 β values**, a more consistent signal than `607`'s single v1 data
point since it reproduces across two independent fits. No v1 config shows
this for `encoder_head`. Full numbers: `docs/experiment_inventory.md` §4.4.

---

## B. Key-instance utility test

Code: `scripts/cvae_weighted_rf_vote.py`. Asks whether the instances a
formulation selects as "key" are actually the diagnostically informative
ones — a question spatial coherence cannot answer, and (per §0) the one that
remains meaningful even for a formulation that spatially regularizes its own
posterior by construction.

1. Binarize the formulation's own per-instance posterior into "key" vs
   "not key." Threshold is formulation-specific: for a K=2 joint-softmax
   formulation, use a **median split** on non-diagnostic mass, not an
   absolute 0.5 (the project's own §7 bound makes 0.5 structurally
   unreachable for such formulations, which would silently turn the filter
   into a no-op).
2. Fit Tier 0's own RF — same hyperparameters, same raw spectral features,
   nothing else changed — on **only** the key training instances.
3. At inference, apply the same threshold to flag key validation instances,
   score only those, pool to a bag decision with Tier 0's ordinary majority
   vote.
4. Compare against Tier 0's unmodified RF (all instances, equal vote) via the
   same bag-level bootstrap used everywhere else in this project (§9 of
   `docs/mil_tiers_methodology.md`).

**Reading it**: if the selected instances carry the diagnostic signal, a
classifier restricted to them should match or beat the one that votes over
everything. If it comes out worse, the selection is actively removing
diagnostic content. A null result (no difference) is weak evidence either
way — a strong classifier voting over thousands of instances per bag is
robust to losing half of them, so "no difference" is also consistent with
"the selection is uninformative but harmless."

**Two cheaper variants**, reusing the *already-trained* Tier 0 RF (removes
confound (i) below entirely, at the cost of testing a weaker claim — does
the posterior improve aggregation, rather than does it identify the
informative instances):
- **Weighted vote** — each instance's vote weighted by the formulation's
  per-instance class probability.
- **Key-instance vote** — only key instances' existing votes are counted.

**Confounds to state whenever this is reported:**
- **(i) Training-set-size confound.** The restricted RF trains on fewer
  instances (half, under a median split), so a deficit conflates "selection
  removes signal" with "less training data hurts." **Fix, committed to but
  not yet run**: a random-subset control of the same size, run before any
  strong claim based on this test.
- **(ii) Empty key-set fallback.** A bag whose key set is empty needs a
  documented fallback (revert to all its instances), and the number of bags
  that hit it must be reported, not silently absorbed.

**Current coverage, updated 2026-10-12**: run for `607`/`608`/`609`
(β=0.25/0.5/0.75), both attention and CbR (both variants) — 9 (config ×
variant) cells total. The confound-(i) random-subsample control (above) is
now implemented and run for all 9. Result: no cell shows CVAE-based
selection beating the size-matched random control, **except `607`'s
attention** (0.8305 vs 0.7966) — the same lone exception as Family A, and
the only candidate currently flagged for a closer look (gated on surviving
both families, per the project's own standing decision). One cell
(`609`'s CbR `reconstruction`) shows selection significantly *hurting*
(p=0.022). Full numbers: `docs/stage6_attention_mil.md` §10.2,
`docs/stage6_generative_mil.md` §10.2.

**Extended further, same day, to v2 preprocessing (`exp_2.607`/`exp_2.608`)**:
less clean than Family A here — `exp_2.608` shows selection beating random
for all three tested tiers (attention, both CbR variants); `exp_2.607` only
for CbR `encoder_head` (0.8305 vs 0.7966). `encoder_head` is again the one
result consistent at both β, agreeing with its own Family A exception
(§A above) — the strongest combined signal in the project so far, still not
individually significant on either family's own bootstrap. Full numbers:
`docs/experiment_inventory.md` §4.5. **Still not run**: `exp_501`.

---

## C. Standing policy going forward

Per the reasoning in §0: **both families are run together for any new MIL
formulation going forward**, not family A alone — especially for any
spatially-regularized model, where family A alone risks confirming the
model's own smoothing prior rather than testing anything about recovered
diagnostic structure. The random-subsample control for family B's confound
(i) is also now standard practice, not an optional strengthening.
