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

> **Update, 2026-09-28**: the root cause was `data/splits/manual_splits.json`
> omitting this slide from every fold's train/test list — not a property of
> the slide's spectra, which on inspection differ only in baseline
> fluorescence level, expected to be handled by IARPLS. Fixed in the split
> file (docs/stage1_data_layer.md §4); the dataset is 59 slides going
> forward. The numbers in this section were not re-run and still reflect 58 —
> per the reasoning at the time, one slide out of 58 was not expected to
> change the mi-SVM/EM conclusions above, but a from-59 re-fit would now
> include a 59th out-of-sample posterior this report never had.

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

### mi-SVM: the shuffled-bag-label control (run 2026-09-29)

The one check named as "primary" from the start (§3) and, until now, never
run. `scripts/stage6_shuffled_label_control.py` permutes which label goes
with which bag — an exact permutation of the true label multiset, so class
balance is identical to the real run — holding every instance's features,
bag membership, and the CV split completely fixed, then refits mi-SVM
exactly as `pipeline/mil_fit_stage.py` does (reused unchanged; the
permutation is a single substitution on the loader's own label array before
any fold splitting). Full 5-fold refit, same `max_iters=10`, same
convergence pattern as the real run (every class in every fold hit the
iteration cap). Run against 59 slides (post-HF1887-fix — the real mi-SVM
numbers above are still 58; see the update note in §2).

| lag | Moran's I | p | z | variogram γ |
|---|---|---|---|---|
| 1 (contaminated) | +0.166 | 0.001 | +23.8 | 0.0081 |
| 2 (contaminated) | +0.054 | 0.038 | +11.2 | 0.0095 |
| **3 (headline)** | **+0.018** | **0.261** | **+5.01** | 0.0100 |
| 4 | +0.015 | 0.273 | +4.57 | 0.0101 |
| 5 | +0.013 | 0.254 | +4.36 | 0.0100 |
| 10 | +0.008 | 0.391 | +3.70 | 0.0101 |

Variogram range: median 2 px — identical to the real fit. External
agreement: ARI = −0.0075, AMI = 0.0011, hard "diagnostic" count 3,883/288,678
(1.35%) — all indistinguishable from the real fit's own null result.

**This settles the question, and not in the real fit's favour.** The
shuffled-label control does not merely fail to collapse relative to the real
result — it produces *comparable or larger* apparent coherence at every
lag ≥ 3 (z = +5.01 vs the real fit's +1.50 at lag 3; z stays above +3.7 out
to lag 10, where the real fit had already fallen to +1.29). Per the plan's
own pre-registered exit criterion ("excess coherence at lag ≥ 3 survives the
shuffled-label control — if not, the spatial claim is dropped"), mi-SVM's
already-non-significant lag ≥ 3 signal now fails this definitively rather
than just lacking significance on its own: a fit with **no genuine
label-content relationship whatsoever** produces the same or stronger
"coherence." The most likely mechanism is not biological: mi-SVM's per-bag
top-k relabelling can latch onto *any* spectrally-distinct, spatially-
contiguous subcluster within a slide — real tissue has real spatial
autocorrelation in its raw spectral features regardless of which label a
subcluster is being fit to represent — so some lag-band "coherence" is a
close-to-mechanical property of this formulation applied to this kind of
data, not evidence the model recovered the true diagnostic structure. The
spatial claim for mi-SVM is dropped, per the plan's own contingency, on the
strongest available evidence rather than a suggestive p-value alone.

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
claim is dropped — **that outcome is reportable too**" — **the shuffled-label
control has now been run for mi-SVM (2026-09-29, above) and the criterion
fails explicitly**: the permuted-label fit produces comparable-or-larger
apparent coherence at every lag ≥ 3 than the real fit did. This is stronger
than the earlier reading ("little at lag ≥ 3 to explain") — it's not just
that the real effect was small, it's that an equally-sized-or-larger effect
appears with no genuine label-content relationship at all, pointing at a
property of the formulation-on-this-data rather than of the true labels.
**The mi-SVM spatial claim is dropped.**

Two concrete leads from the original pass, one now addressed:

1. ~~**mi-SVM had not converged**~~ — moot for the spatial claim specifically:
   the shuffled control hit the same iteration cap with the same convergence
   pattern, and still showed coherence at least as strong as the real fit, so
   further real-fit convergence would not rescue a claim the control already
   defeats. (Whether more iterations changes mi-SVM's *bag-classification*
   accuracy — a separate question, docs/stage6_generative_mil.md §5 — is
   still untested.)
2. **EM's collapse may or may not be the "correct" answer** for real spectra
   at this bag count — it could be a genuine finding (the discriminative
   signal really is this weak at 58 bags) or an artifact of this specific EM
   configuration (cold-start, shrinkage strength, fixed `pi_init`). Not yet
   distinguished, and EM's own shuffled-label control was not run (Stage 4's
   collapse already makes EM's real-fit result uninformative on its own
   terms, so a shuffled version would be informative about the collapse
   mechanism, not about a spatial claim EM never supported in the first
   place — lower priority than mi-SVM's was).

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
  external-validity check, docs/MIL_REFRAME_PLAN.md §1) — done, 2026-09-30,
  also null (Spearman $\rho \in [-0.16,+0.14]$, all $p>0.19$, n=59 bags):
  `scripts/stage4_pi_b_external_validity.py`, docs/stage4_mil.md §6.
- **The shuffled-bag-label control was not run in this original pass** — run
  for mi-SVM on 2026-09-29, see the new subsection in §2 and the updated §3.
  EM's own shuffled control remains not run (lower priority — its real fit
  was already uninformative via the collapse, independent of this control).
  Attention (Tier 3) and both Tier 4 variants have since had theirs run too
  (2026-09-29/30) — see §6.
- **Join-count/boundary-length statistics and the qualitative RGB overlay**
  (both named in §6) are not implemented — the two statistics that were
  implemented (Moran's I, variogram) already gave a clear, mutually
  consistent answer; these two would only strengthen the same conclusion
  without offering the "does the answer contradict itself" second opinion, so
  they were left out given the already-answered exit criterion.

## 6. What comes next — updated 2026-09-30: every tier's control is now in, and every one failed

**Resolved, and closed out completely.** The user's direction (2026-09-28)
was (c): treat this as the honest finding. The shuffled-label control
(2026-09-29, §2/§3) then went further than (c) alone would have for
mi-SVM — it doesn't just accept a non-significant result, it shows a
permuted-label fit produces comparable-or-larger apparent coherence,
actively defeating the spatial claim rather than merely failing to support
it.

At the time this section was last written, Tier 4's generative-MIL
posteriors looked like "a materially different, more encouraging picture"
and were flagged as "the live spatial result going forward," pending their
own control. **That control has since been run, on Modal GPU (2026-09-30,
a full 5-fold CVAE retrain under permuted labels), and it failed too** —
`encoder_head`'s shuffled result is numerically indistinguishable from its
real one, and `reconstruction`'s shuffled result is *larger* than its real
one at every headline lag (docs/stage6_generative_mil.md §4a, the same
day Tier 3's attention-MIL control also failed — docs/stage6_attention_mil.md
§5a). That "more encouraging picture" language is retracted along with
mi-SVM's — it was true of the raw numbers and wrong about what they meant.

**The complete, final state: four spatial-coherence claims were made across
this project (mi-SVM, attention, `reconstruction`, `encoder_head`); all
four were tested; all four failed.** EM never produced one to test (its
real fit collapsed on its own terms first). There is currently no tier in
this project whose lag≥3 spatial coherence survives a shuffled-label
control. The likely shared mechanism — real tissue has real spatial
autocorrelation in its raw spectral characteristics, present in whatever
representation a formulation builds on top of the spectrum, regardless of
which label a region is being fit toward — is stated identically at each
of the four retractions, not re-derived independently each time, since it
is the same underlying explanation in every case. This is a project-level
finding for the paper's limitations section, not a per-tier footnote: the
spatial-coherence validation strategy itself worked exactly as designed (it
is precisely what caught all four failures), but nothing currently in this
project demonstrates that any MIL formulation's posterior map recovers
genuine, label-driven diagnostic structure at the pixel level.
