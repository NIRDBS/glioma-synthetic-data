# Stage 6 (Tier 4) — Generative MIL from the CVAE

**Status: complete, including the real-data fit and both evaluations.** New
`src/mil/generative.py`, `pipeline/mil_fit_stage.py` gains a `method='cbr'`
dispatch, new `scripts/stage6_bag_eval.py`. Real fit + evaluation on exp_501
finished; results below, and they cut both ways — a genuine, statistically
significant bag-level result against the weakest Tier 0 reference, no such
result against the three strongest ones, real (if modest and short-range)
spatial coherence beyond what Tier 1/2 showed, and two distinct
methodological bugs this pass found by actually running the numbers rather
than trusting them: the external-agreement check is structurally degenerate
for a K=2 softmax source (§4, independent of anything about the CVAE
itself), and — more seriously — a naive bag-accuracy metric applied to EM
(§5) turned out to be vacuously, trivially 1.0000 by construction, not a
real finding, and has been excluded rather than reported.

Per the user's own framing ("Tier 4 CbR-based first, cheap"), this is Tier 4
only. Tier 3 (attention MIL) is not started.

> **Reproducing this.** `pytest tests/test_stage6_generative_mil.py` (13
> tests, synthetic fakes only — no real CVAE weights needed). Real fit:
> `python -m pipeline.mil_fit_stage --config exp_501 --methods cbr` (needs
> exp_501's `cvae` stage already trained — reuses its saved weights, trains
> nothing new). Spatial validation:
> `python scripts/stage5_spatial_validation.py --config exp_501 --method cbr
> --variant reconstruction` (or `--variant encoder_head`). Bag-level eval:
> `python scripts/stage6_bag_eval.py --config exp_501 --method cbr --variant
> reconstruction --reference-model cnn` (`--reference-model` also takes
> `apollo`, `rf`, `xgb`).

## 1. What this tier is, and why it's "cheap"

Per docs/MIL_REFRAME_PLAN.md §4, Tier 4 promotes two things the project
already had, at zero additional training cost:

- **'reconstruction'**: Classification by Reconstruction's per-instance,
  class-conditional decoder MSE (`pipeline/evaluation_stage.py::_cbr_predictions`),
  turned into a genuine posterior via `softmax(-mse/T)` instead of a hard
  `argmin`.
- **'encoder_head'**: the CVAE encoder's own auxiliary softmax classifier
  (`src/models/cvae.py`'s `class_output`, trained with the `alpha=0.2` loss
  term as part of ordinary CVAE fitting) — discarded by CbR today
  (`evaluation_stage.py` takes `encoder.predict(...)[0]`, the z_mean, never
  `[3]`, the class prediction).

Both score with the exact same per-fold CVAE weights the `cvae` pipeline
stage already saved
(`results/exp_501/models/cvae/{encoder,decoder}_fold_N.weights.h5`) — no
training happens anywhere in this stage. The real fit (all 5 folds, both
variants, ~280k instances) completed in **well under a minute**, confirming
the "cheap" framing used to reorder Stage 6 ahead of Tier 3.

Neither variant models a non-diagnostic class explicitly (both are K-way,
over diagnostic classes only), so both are lifted to the store's (K+1)-way
convention the same way `src/mil/misvm.py`'s Platt margins are:
non-diagnostic mass = `1 - max_k p_k`, renormalised
(`src/mil/generative.py::_to_K1_posterior`). **This convention turns out to
have a real, discovered problem for K=2 softmax sources — see §4.**

## 2. Picking the softmax temperature from real data, not a guess

`softmax(-mse/T)` needs a temperature, and the plan doesn't specify one. A
smoke test on real exp_501 fold_0 validation data (2026-09-28, 2000
instances) showed why `T=1` (the naive default) is wrong for this data: the
real per-class MSE values are tiny (median ≈0.003, inter-class gap median
≈0.0008), so at `T=1` the softmax is within floating-point noise of uniform
(mean top-class probability ≈0.50 for K=2 — no usable signal at all, and
every instance's non-diagnostic mass would be a near-constant ≈0.5,
manufacturing the exact same "spatially uninformative because numerically
constant" failure Stage 4 found in EM's collapse, but from a temperature
choice rather than a model degeneracy).

| T | mean(max class prob) | reading |
|---|---|---|
| 1.0 | 0.500 | uniform — no signal |
| 0.05 | 0.504 | still effectively uniform |
| 0.01 | 0.518 | barely graded |
| 0.005 | 0.535 | weak grading |
| **0.001** | **0.665** | genuinely graded, not near-hard — **chosen default** |
| 0.0005 | 0.779 | approaching a hard decision |

`T=0.001` (≈ the real median inter-class MSE gap) was chosen as a moderate,
documented, provisional value — the same treatment as IARPLS λ or CVAE
epochs elsewhere in this project: not swept, not claimed optimal, recorded
with the reasoning that produced it (`configs/exp_501.yaml`'s `mil:` block,
`src/mil/generative.py::mse_to_posterior`'s docstring).

## 3. Bag-level accuracy vs Tier 0 — the headline number depends entirely on which baseline

`scripts/stage6_bag_eval.py` is new: it reads any stored `(method, variant)`
from the latent-label store, pools each fold's out-of-sample instances to a
bag-level decision (reusing `aggregate_bag_scores_across_folds` unchanged,
the exact machinery Tier 0's own `mil_eval_stage.py` uses), and runs a
class-stratified bootstrap against one classifier's Tier 0 majority-vote
pooling. **This closes a real gap**: `docs/stage4_mil.md` §6 never computed a
bag-accuracy number for mi-SVM/EM — their story stayed at π_b/degeneracy
diagnostics and spatial checks. Filling that gap for Tier 1/2 too (§5 below)
is what makes this table a genuine cross-tier comparison, not just a Tier-4
one-off.

All numbers: exp_501, n=59 bags (the HF-1887 split-file fix applies to
everything computed in this stage — see docs/stage1_data_layer.md §4), 1000-
resample class-stratified bootstrap, seed 42.

**Tier 0 references (TRTR, majority vote), all four classifiers:**

| model | bag accuracy |
|---|---|
| 1D-CNN | 0.5932 |
| Apollo (RF+SVM) | 0.7627 |
| Random Forest | 0.8136 |
| XGBoost | 0.8136 |

These span a 22-point range — the CNN is, by a wide margin, the weakest of
the four at bag level (consistent with docs/MIL_REFRAME_PLAN.md §7.2's own
framing of it as undertrained relative to the tree ensembles even after
Stage 3's leak fix); the two tree ensembles are essentially tied at the top.
**Which one "Tier 0" means changes the entire verdict below — reporting only
the CNN comparison, the one that happens to look best, would be selective.**

**Tier 4 vs every reference:**

| variant | bag acc (F1, κ) | Δ vs CNN | Δ vs Apollo | Δ vs RF | Δ vs XGBoost |
|---|---|---|---|---|---|
| reconstruction | 0.7119 (0.711, 0.423) | **+0.119**, p=0.150 | −0.051, p=0.474 | −0.102, p=0.104 | −0.102, p=0.104 |
| encoder_head | 0.7966 (0.786, 0.577) | **+0.203**, p=0.006 ✓ | +0.034, p=0.410 | −0.017, p=0.694 | −0.017, p=0.694 |

(✓ marks the one 95% CI excluding zero, CI[+0.085,+0.339]; every other
comparison's CI spans zero. Full CIs in each `stage6_bag_eval_*.json`.)

**Read this honestly, not selectively.** Against the CNN, `encoder_head`
clears the plan's Stage 6 exit criterion ("beats Tier 0 at bag level")
decisively — a 20-point, p=0.006 improvement, at zero extra training cost,
from a classifier that was already being trained and simply never reported
before. Against every other Tier 0 classifier, **neither variant shows a
significant difference in either direction** — `encoder_head` is essentially
tied with Apollo/RF/XGBoost (all within 3.4 points, none significant), and
`reconstruction` is numerically somewhat behind all three (again, not
significantly). The honest summary is **"beats the weakest Tier 0 baseline
by a wide, significant margin; statistically indistinguishable from the
three strongest ones"** — not "beats Tier 0" unqualified. The tree ensembles
(Apollo/RF/XGBoost) remain the strongest classifiers in this project at bag
level, Tier 4 included.

One more calibration point: `reconstruction`'s mean-pooled softmax bag
accuracy (0.712) is *lower* than the existing hard-argmin CbR majority-vote
result already in the paper (0.741, exp_201's old-data run, restored to
`paper/main.tex` earlier this session). Converting a hard decision to a soft,
temperature-scaled probability and mean-pooling it is not a strict
improvement on voting the hard decisions directly — a real, reportable
nuance, not a red flag (the two numbers are also not on identical data:
0.741 is exp_201/airPLS/58-bags, 0.712 is exp_501/IARPLS/59-bags).

## 4. Spatial validation — real coherence, and a discovered convention bug

Both variants' posteriors were run through Stage 5's unchanged spatial
pipeline (`scripts/stage5_spatial_validation.py`), covering **all
288,678/288,678 instances** (59/59 slides — the HF-1887 fix applies here
too; Stage 4's mi-SVM/EM runs, fit before that fix, covered 58).

### Moran's I and the variogram, by lag

| | lag 3 (headline) | lag 4 | lag 5 | variogram range |
|---|---|---|---|---|
| **reconstruction** | I=+0.122, p=**0.028**, z=+29.4 | I=+0.092, p=**0.041**, z=+25.8 | I=+0.076, p=0.072, z=+23.8 | median 3.0 px |
| **encoder_head** | I=+0.049, p=0.057, z=+11.1 | I=+0.038, p=0.088, z=+10.0 | I=+0.031, p=0.104, z=+9.2 | median 3.0 px |
| *(for comparison)* mi-SVM (Stage 5) | I=+0.006, p=0.361, z=+1.5 | — | — | median 2 px |
| *(for comparison)* EM (Stage 5) | variogram ≡ 0 (collapsed) | — | — | n/a |

**This is a materially different picture from Tier 1/2.** `reconstruction`
clears p<0.05 at lag 3 *and* lag 4 — genuine excess correlation at distances
the 3×3 median filter provably cannot produce, with z-scores an order of
magnitude larger than mi-SVM's. `encoder_head` is directionally the same but
weaker, sitting just above the conventional 0.05 threshold.

**The "10–20 px is tissue" bar from the plan was framing language, not a
measured quantity, and this project has no validated ground truth for tumour
region size to check it against.** That needs to be stated plainly, because
an earlier version of this section got it wrong: it computed the variogram
range of `viable_legacy` and reported that map as "the old manual tumour
annotation," "tissue we're confident is real" — treating it as ground truth.
**It is not.** Per the project's own foundational finding
(docs/MIL_REFRAME_PLAN.md's opening section, and the limitation already
written into `paper/main.tex`'s Data Preprocessing discussion): `viable_legacy`
is the output of an *unsupervised* per-slide pipeline — NMF(100) then a
3-component GMM — where a human then looked at each cluster's mean spectrum
and picked whichever looked like "the strongest signal," manually labelling
that "tumor." It is a **signal-quality judgment, not a histopathological
one**, and it is exactly the kind of instance-level label this whole MIL
reframing exists to stop treating as ground truth. Reusing it as a tumour-size
reference would have quietly reintroduced the same false assumption Stage 1
retired.

What the earlier computation actually measured, stated correctly: the
variogram range of *which per-slide GMM cluster got manually picked as
strongest-signal*, pooled across 41/59 slides with a real mixed selection
(2026-09-28) — median 7.0 px, mean 10.5 px, p10=3 px, p90=21 px. This is a
real, independently-computed number (different pipeline, different math from
any MIL posterior), but it calibrates nothing about tumour size, and no
conclusion should be drawn from comparing it to Tier 4's 3 px or mi-SVM's
2 px — that comparison, in the earlier version of this section, is retracted.
**Whether 3 px reflects real, small-scale tumour coherence or residual filter
structure remains genuinely unresolved** by anything currently in this
project. The shuffled-bag-label control (still not run for any tier, per
Stage 5's own §3) is the one check that could actually distinguish these —
not a comparison to `viable_legacy`, which was never fit for that purpose.

### External agreement vs `viable_legacy`: degenerate for a real, discovered reason

Both variants: **ARI=0.0000, AMI=0.0000, "diagnostic": 288,678/288,678 —
every single instance.** This is not the CVAE finding nothing, and not a bug
in Stage 5's own code (unchanged, already trusted for mi-SVM/EM). It is a
**structural property of the `1 - max_k p_k` convention applied to a genuine
K-way softmax, specific to K=2**:

The convention was designed for mi-SVM, where each class's score comes from
an *independent* one-vs-rest classifier — `max_k p_k` can legitimately fall
below 0.5 when neither one-vs-rest model is confident, which is exactly how
mi-SVM's own hard-diagnostic count (2,784/288,678, 0.96%) stayed
non-degenerate. But both Tier 4 variants' K=2 class probabilities come from a
*joint* softmax that always sums to exactly 1 — so `max_k p_k ≥ 0.5` by
construction, and after the K→K+1 lift and renormalisation, the resulting
non-diagnostic mass is provably bounded: `(1-max_p)/(2-max_p)`, which for
`max_p ∈ [0.5, 1]` ranges over **`[0, 1/3]`, strictly below the 0.5
hard-diagnostic threshold at every possible confidence level, including
maximal uncertainty.** The external-agreement check can therefore never call
anything "non-diagnostic" for a K=2 softmax source, regardless of what the
posterior actually encodes. (The bound loosens toward, but never reaches,
0.5 as K grows — `(K-1)/(2K-1)` — so this would also affect a K=6 softmax
source, just less severely; worth flagging for Tier 3 if its attention head
is a plain softmax too.)

**What this does and doesn't affect**: the *hard-threshold* ARI/AMI check is
uninformative for both variants as implemented — not fixed in this pass,
since a principled replacement threshold (e.g. a distribution-relative cut
rather than a fixed 0.5 inherited from a differently-shaped score source)
needs its own justification, not an ad hoc patch chosen to make a number look
better. The *continuous* non-diagnostic mass used by the lag-band statistics
above is unaffected — real per-instance variation exists within `[0, 1/3]`,
and that is exactly what Moran's I and the variogram measured.

## 5. Closing the gap: Tier 1/2 bag-level accuracy — and a second, more serious
   circularity bug this pass found and had to exclude

`scripts/stage6_bag_eval.py` is generic, so it was also run against the
already-fitted Tier 1/2 store entries — the bag-accuracy number
docs/stage4_mil.md §6 never computed. Doing so surfaced a real methodological
bug, this time serious enough to invalidate a result outright rather than
just narrow a comparison (§4's K=2 convention issue).

**mi-SVM (`misvm/mi_svm`): bag_acc=0.6897 (F1=0.689, κ=0.385), n=58 bags**
(pre-HF1887-fix coverage, unchanged from Stage 4's original fit — not
re-run). Against the CNN reference restricted to the same 58 bags
(bag_acc=0.6034 — this reproduces, exactly, the historical
`docs/stage4_mil.md`-era exp_501 TRTR number, since it is the same trained
model scored on the same bags): Δ=+0.0862, CI[−0.017,+0.190], p=0.126, *not
significant*. Independently re-verified bag-by-bag, not just at the
aggregate: mi-SVM and the CNN **disagree on 9 of 58 bags**
(49/58 agreement) and land on the same overall count by coincidence, not by
one reusing the other's predictions — checked directly because the exact
numeric match to a previously-seen delta was, correctly, suspicious enough
to verify before trusting it.

**EM (`em/constrained`): EXCLUDED — bag_acc=1.0000 is not a real result.**
The raw number this script computed is a perfect, trivial 1.0000
(Δ=+0.3966, "p<0.001, significant") — and it is **meaningless, not a
finding**. Checked directly against the real store: on held-out validation
rows, **0% of instances ever have both diagnostic columns simultaneously
nonzero** (`k_diagnostic_columns_competition_frac=0.0000`). This is Model
(C) working exactly as designed, not a bug in `src/mil/em.py`: `score_em`
correctly restricts every instance's state to `{null_1..null_M, Y_b}` —
literally never allowing mass on any diagnostic class *other than* the
instance's own bag's already-known true label — so
`posterior[:, 1:].argmax()` recovers the bag label the model was *given*,
not one it *inferred*. This bag-classification metric is well-posed only
for methods that don't hard-enforce the bag label throughout scoring
(confirmed: mi-SVM's independent one-vs-rest scores show real competition in
16.5% of instances; Tier 4's plain-softmax variants in ~100%) — it cannot,
even in principle, be a meaningful evaluation of EM specifically, or of any
future formulation that implements the same Model-(C) restriction end to end
(a flag for Tier 3, if its attention head is ever built the same way).
`scripts/stage6_bag_eval.py` now computes and prints this competition
fraction on every run and warns loudly when it is near zero, so this
mistake — mine, in this script, not in EM's own fit or in Stage 4/5's prior
work — cannot silently repeat.

## 6. Verification

- **New `tests/test_stage6_generative_mil.py`, 13/13 passing.** Pure-numpy
  checks on `mse_to_posterior`/`class_probs_to_posterior`/`_to_K1_posterior`
  (hand-built arrays, including the numerical-stability case and the
  temperature-sharpness relationship); `instance_posteriors_from_cvae`
  end-to-end against a fully deterministic fake encoder/decoder pair (no real
  CVAE weights — `z_mean = X`, true class = sign of the first feature,
  decoder reconstructs exactly on a match and badly otherwise, so every
  output is hand-checkable); `mil_fit_stage._score_one_fold_cbr`'s
  scatter-back and `in_sample` bookkeeping, monkeypatched to avoid touching
  real weights, including the case of a bag absent from a fold (stays NaN,
  not silently zero); a guard that `'cbr'` is available but not in the
  default `METHODS` tuple `run_suite.py` relies on; and
  `diagnostic_column_competition_frac` (§5's circularity guard) correctly
  flagging a hand-built hard-masked posterior at 0.0 and a genuinely
  competitive one at 1.0, so this specific mistake is pinned against
  regression, not just narrated here.
- **Full suite re-run**: 124 passed (114 prior + these 13, net of the 3
  added after the first 10 were already counted), same 2 pre-existing
  `test_config_manager.py` failures, unrelated to this stage.
- **Real data, full pipeline**: fit → store → spatial validation → bag-level
  eval ran end-to-end with no errors, for both cbr variants and (retroactively)
  misvm/em; the em run's console output was checked directly to confirm the
  new competition-fraction warning actually fires on the real degenerate case,
  not just in the synthetic test.
- **Leakage**: unchanged machinery (`write_posteriors`'s `in_sample` flag,
  the same train/val fold masks `_fit_one_fold` already uses) — every
  out-of-sample instance is scored by a CVAE that never trained on its bag,
  same as CbR's existing, already-trusted result.
- **The bag-accuracy numbers were sanity-checked, not taken on faith.** Before
  writing anything into §3/§5, every number that looked too good (an exact
  match to a historical figure; a perfect 1.0000) was independently
  re-derived by a different code path before being trusted or reported —
  this is what caught §5's EM circularity bug in the first place, rather than
  it silently entering the record as a "significant" finding.

## 7. Deviations from the plan / honest gaps

- **The external-agreement convention bug (§4) is documented, not fixed.**
  A principled K-appropriate hard-diagnostic threshold is future work.
- **`reconstruction`'s temperature (T=0.001) was calibrated from one fold's
  smoke-test sample, not swept.** A sensitivity check across T is not done.
- **The K-diagnostic-column bag-accuracy metric itself (§5) is now known to
  be unusable for any method that implements Model (C)'s hard support
  restriction throughout scoring** — not a Tier 4 limitation, but a limitation
  of `scripts/stage6_bag_eval.py` as a general-purpose tool. It remains valid
  for mi-SVM and for Tier 4's two variants (verified per-method, not assumed),
  and the script now warns automatically rather than requiring the caller to
  remember which methods are safe.
- **The shuffled-bag-label control** (Stage 5's own named, not-yet-run
  check) still hasn't been run for any tier, Tier 4 included — the lag≥3
  significance in §4 is suggestive, not definitive, without it.
- **mi-SVM/EM were not re-run against Apollo/RF/XGBoost** (only vs the CNN,
  §5) — mi-SVM's number is genuine and the comparison is cheap to extend;
  not done in this pass since Tier 4 was the actual ask.
- **Tier 3 (attention MIL) is now complete** — see
  docs/stage6_attention_mil.md. Its instance head is indeed a plain K=2
  softmax (a deliberate design choice, made independently of this
  prediction), so §4's K=2 convention bound was confirmed to apply, exactly
  as flagged here. §5's EM-specific circularity check does **not** apply to
  it: Tier 3's instance head was deliberately built *without* Model (C)'s
  hard support restriction (docs/mil_tiers_methodology.md §6.3), precisely
  so its bag classification would not be circular the way EM's is.

## 8. What Stage 6+ inherits

- **Correction to a prediction made here**: it turned out `scripts/stage6_bag_eval.py`
  could *not* simply be reused for Tier 3 as this bullet originally
  predicted. Tier 3's shared-store posterior is deliberately attention-free
  (right for spatial validation, wrong for bag classification — a bug
  found and fixed during Tier 3's own smoke-testing, docs/stage6_attention_mil.md
  §2), so bag-level evaluation needed a dedicated
  `scripts/stage6_attention_bag_eval.py` reading a separate,
  already-attention-weighted artifact instead. The circularity guard this
  bullet mentions did carry over unchanged (reused via
  `stage6_bag_eval.py::_tier0_reference`, imported directly by the new
  script) — only the posterior-pooling step needed to be separate.
- The K=2 softmax external-agreement bound (§4) was confirmed to apply to
  Tier 3 as well (ARI=AMI=0.0000, identical mechanism) — see
  docs/stage6_attention_mil.md §5.
- `mil_fit_stage.py`'s `method='cbr'` dispatch pattern
  (`_score_one_fold_cbr` returning a *list* of `(variant, posterior,
  in_sample)` from one shared per-fold computation) is the template for any
  future tier that produces more than one variant per fold cheaply.
