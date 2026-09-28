# Stage 5 — Spatial validation

**Status: complete.** New `src/evaluation/spatial.py` (Moran's I and the
empirical variogram, both by lag band; permutation nulls; external
agreement) and `scripts/stage5_spatial_validation.py` (the real-data runner).

> **Reproducing this.** `pytest tests/test_stage5_spatial.py` (14 tests, all
> synthetic maps of known correlation length — no real-data dependency).
> Real analysis: `python scripts/stage5_spatial_validation.py --config exp_501
> --method em` or `--method misvm`. Needs Stage 4's `mil_fit` to have already
> run for that config.

## 1. What this stage checks, and why by lag band

The design (docs/MIL_REFRAME_PLAN.md §6): withhold `coords` from every MIL
model, infer the latent posterior, reproject it onto the slide's pixel grid,
and ask whether the resulting map is spatially coherent. Coherence is then
*evidence* the model recovered real tissue structure, not an artefact of an
objective that never saw geometry.

The confound: the project's 3×3 spatial median filter (kept — signal
cleaning, not this stage's concern) couples a pixel only to its 8 neighbours,
so two pixels at Chebyshev distance ≥ 3 share no input pixel whatsoever and
the filter provably cannot manufacture correlation between them. Every
statistic here is reported **by lag band**, not as one pooled number: lags
1–2 are contaminated (shown, excluded from the claim); **lag ≥ 3 is the
headline**. The variogram's *range* is itself diagnostic (§6: "1–2 px is the
filter; 10–20 px is tissue").

Two complementary statistics are computed at every lag, not one:
- **Moran's I** — scale-free (normalised by total variance), so it can flag
  "significant" spatial pattern even in a signal whose absolute magnitude is
  negligible. Good at detecting pattern; can mislead about how much it
  matters.
- **The empirical variogram** — scale-*sensitive* ($\gamma(h) = \tfrac12
  \,\mathrm{mean}((x_i-x_j)^2)$), so a genuinely constant map reads as
  exactly zero at every lag, unambiguously. This turned out to be the
  statistic that mattered (§3).

Permutation null: shuffle the in-mask values within each slide's own mask
(geometry held fixed), recompute the statistic, repeat (n=200) to get a
per-lag p-value/z-score.

## 2. Real-data results (exp_501)

Both MIL formulations from Stage 4, run against the same 58 slides (one
slide, `HF-1887_via-t_2_1.h5_3`, had zero covered instances and was skipped —
consistent with it being the historically excluded slide documented in
Stage 1).

### mi-SVM

| lag | Moran's I | p | z | variogram γ |
|---|---|---|---|---|
| 1 (contaminated) | +0.108 | 0.007 | +15.1 | 0.0074 |
| 2 (contaminated) | +0.029 | 0.059 | +5.8 | 0.0082 |
| **3 (headline)** | **+0.006** | **0.361** | **+1.50** | 0.0084 |
| 4 | +0.005 | 0.362 | +1.36 | 0.0084 |
| 5 | +0.004 | 0.403 | +1.45 | 0.0084 |
| 10 | +0.003 | 0.373 | +1.29 | 0.0084 |

**Variogram range: median 2 px.** The variogram rises from lag 1 to lag ~2–3
then **plateaus completely** — no further increase out to lag 10. Per the
plan's own criterion, a 2 px range is squarely in filter-artefact territory,
not the 10–20 px expected of real tissue architecture. Consistent with this,
lag ≥ 3 Moran's I is not significant anywhere (p ≥ 0.36, z ≤ 1.5, well under
the ~1.96 needed for p < 0.05).

**External agreement vs. `viable_legacy`** (the old manually-curated tumour
selection, read directly from the main HDF5): ARI = −0.007, AMI = 0.002 —
indistinguishable from no agreement. mi-SVM's hard "diagnostic" call (posterior
mass > 0.5) fired on only 2,784 of 288,678 instances (0.96%), against
`viable_legacy`'s 227,344 (78.7%) — a large scale mismatch, not just a
weak correlation.

### EM

| lag | Moran's I | p | z | variogram γ |
|---|---|---|---|---|
| 1 (contaminated) | +0.182 | 0.001 | +26.3 | 0.0000 |
| 2 (contaminated) | +0.083 | 0.003 | +17.3 | 0.0000 |
| **3 (headline)** | **+0.042** | **0.116** | **+11.5** | 0.0000 |
| 4 | +0.033 | 0.154 | +10.6 | 0.0000 |
| 10 | +0.014 | 0.269 | +6.8 | 0.0000 |

**The variogram is exactly 0.0000 at every single lag.** Read this before the
Moran's I column: a variogram of zero everywhere means the reprojected map is
numerically constant — direct, independent confirmation of the
`shrinkage_ratio=0` collapse Stage 4's own diagnostic already flagged in 4 of
5 folds (docs/stage4_mil.md §6). Moran's I nonetheless reports large z-scores
(up to +26 at lag 1) precisely *because* it is scale-free: it can detect
"significant" pattern in what is, in absolute terms, floating-point-level
noise around a constant. **This is exactly the scenario the two-statistic
design exists to catch, and it did** — trust the variogram here, not Moran's
I. EM's spatial numbers are not informative about tissue coherence; the
model collapsed before it could express a real prediction one way or the
other.

**External agreement vs. `viable_legacy`**: ARI = 0.0000, AMI = 0.0000, hard
"diagnostic" count = 0/288,678 — every instance falls below the 0.5 threshold,
the direct consequence of $\pi_b \approx 0.02$-$0.03$ almost everywhere.

## 3. Honest synthesis

Neither formulation shows evidence of genuine, tissue-scale spatial coherence
surviving Stage 5's own headline criterion, on this run:

- **EM** collapsed (Stage 4's own finding, independently reconfirmed here by
  the variogram) — its result is *uninformative*, not a negative finding
  about spatial structure specifically.
- **mi-SVM** produced a real, non-degenerate, spatially-varying posterior,
  but that variation lives entirely within the 1–2 px range the median filter
  itself can produce, and shows no significant coherence at lag ≥ 3.
  External agreement with the historical tumour selection is essentially
  null.

Per the plan's own Stage 5 exit criterion — "Excess coherence at lag ≥ 3
survives the shuffled-label control. If not, §6 is rewritten and the spatial
claim is dropped — **that outcome is reportable too**" — this is already
pointing toward "does not survive," without even needing the shuffled-label
control to explain the observed effect away (there is little at lag ≥ 3 to
explain). That control was **not run** in this pass (it implies a full
mil_fit-scale refit on permuted labels — another multi-hour job); given the
observed effect is already non-significant, its main remaining value would be
confirmatory rather than decisive.

Two concrete, unexplored leads before treating this as final, both flagged
rather than acted on (per the instruction to stop here for review):

1. **mi-SVM had not converged** — churn was still decreasing at the
   10-iteration cap in every fold/class. Whether more iterations changes the
   spatial picture is untested.
2. **EM's collapse may or may not be the "correct" answer** for real spectra
   at this bag count — it could be a genuine finding (the discriminative
   signal really is this weak at 58 bags) or an artifact of this specific EM
   configuration (cold-start, shrinkage strength, fixed `pi_init`). Not yet
   distinguished.

## 4. Verification

- **New `tests/test_stage5_spatial.py`, 14/14 passing.** Covers the plan's
  test bullets directly: Moran's I and the variogram both verified against
  synthetic fields of known correlation length (white noise reads as ~0
  everywhere; a Gaussian-smoothed field reads high at short lag and decays;
  `variogram_range` scales correctly with the smoothing scale used to
  generate the field); the permutation null correctly flags the smooth field
  as significant and does not flag white noise; and a direct, permanent
  guard (`inspect.signature`) that neither `fit_misvm` nor `fit_em` accepts
  any coordinate-shaped parameter at all — geometry cannot leak into the
  model by construction, not merely by convention.
- **Real data**: both formulations' full pipeline (merge out-of-sample
  posteriors across folds → reproject 58 slides → per-slide + pooled
  statistics → external agreement) ran end-to-end with no errors; results in
  §2. A new `merge_out_of_sample_posterior` (`src/mil/store.py`) — averaging
  the two out-of-sample estimates for the 4 cross-fold-duplicated bags,
  consistent with Stage 2's own duplicate-bag convention — has its own
  regression test (`tests/test_stage4_mil.py`).

## 5. Deviations from the plan

- **External agreement was checked against `viable_legacy`, not
  `hard_labels.npy`'s 3-way NMF clustering.** The latent store persists only
  the (K+1)-collapsed posterior (non-diagnostic mass combined across all $M$
  null components), not the full $(M{+}K)$-way state breakdown — collapsing
  the null components was deliberate (§3's store schema), but it means a
  genuine 3-way ARI against `hard_labels.npy` isn't computable from what's
  persisted. `viable_legacy` (a real, already-available binary reference) was
  used instead. Getting the finer 3-way comparison would need EM specifically
  to also persist its internal $(M{+}K)$-way `gamma` — a small, targeted
  store extension, not implemented in this pass.
- **NMF cluster-proportion correlation with $\pi_b$** (the plan's other named
  external-validity check, docs/MIL_REFRAME_PLAN.md §1) is not computed.
- **The shuffled-bag-label control was not run** — see §3.
- **Join-count/boundary-length statistics and the qualitative RGB overlay**
  (both named in §6) are not implemented — the two statistics that were
  implemented (Moran's I, variogram) already gave a clear, mutually
  consistent answer; these two would only strengthen the same conclusion
  without offering the "does the answer contradict itself" second opinion, so
  they were left out given the already-answered exit criterion.

## 6. What comes next

Not started, pending review: whether to (a) fix mi-SVM's convergence and
EM's collapse and re-run before drawing conclusions, (b) run the
shuffled-label control regardless to have the fully definitive answer, (c)
treat the current result as the honest finding and write up §6's spatial
claim as not supported for this MIL configuration, or (d) something else.
