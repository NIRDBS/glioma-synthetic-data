# Stage 6 (Tier 4) — Generative MIL from the CVAE

**Status: complete, including the shuffled-bag-label control — and both
variants' spatial claims are dropped (§4a).** New `src/mil/generative.py`,
`pipeline/mil_fit_stage.py` gains a `method='cbr'` dispatch, new
`scripts/stage6_bag_eval.py`. Real fit + evaluation on exp_501 finished;
results below, and they cut both ways — a genuine, statistically
significant bag-level result against the weakest Tier 0 reference, no such
result against the three strongest ones; real-looking (if modest and
short-range) lag≥3 coherence that does **not** survive a full CVAE retrain
under permuted labels (§4a, 2026-09-30) — `encoder_head`'s shuffled result
is numerically indistinguishable from its real one; and two distinct
methodological bugs this pass found by actually running the numbers rather
than trusting them: the external-agreement check was structurally
degenerate for a K=2 softmax source (§4, fixed 2026-09-29, independent of
anything about the CVAE itself), and — more seriously — a naive
bag-accuracy metric applied to EM (§5) turned out to be vacuously, trivially
1.0000 by construction, not a real finding, and has been excluded rather
than reported.

This tier's spatial retraction completes the pattern across the whole
project: mi-SVM (Tier 1), attention (Tier 3), and now both Tier 4 variants
have all had their spatial claims tested against a shuffled-label control,
and all four have failed (§4a's closing paragraph). No tier in this project
currently has a spatial-coherence finding that survives this control.

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

## 4. Spatial validation — real coherence, a discovered convention bug, and (§4a) a dropped claim

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

**This looked like a materially different picture from Tier 1/2 — until the
shuffled-label control (§4a) showed it wasn't.** `reconstruction` clears
p<0.05 at lag 3 *and* lag 4 — genuine excess correlation at distances the
3×3 median filter provably cannot produce, with z-scores an order of
magnitude larger than mi-SVM's. `encoder_head` is directionally the same but
weaker, sitting just above the conventional 0.05 threshold. **Both numbers
are real, and both are now known not to be evidence of label-driven
coherence — see §4a.**

### 4a. The shuffled-bag-label control — both variants fail, and the project-wide pattern is now complete

Per docs/MIL_REFRAME_PLAN.md §6/§13's pre-registered check, run 2026-09-30
on Modal GPU (this tier's control needed a full CVAE retrain under permuted
labels — the trunk itself, not a downstream head, since Tier 4 conditions
directly on the CVAE's own weights): `scripts/stage6_shuffled_label_control.py
--method cvae` (5 folds, full 100-epoch schedule, permutation_seed=123,
44/59 bags relabeled, weights written to `results/exp_501/models/cvae_shuffled/`
— the real weights, confirmed by file timestamp, were never touched), then
`--method cbr` to score those shuffle-trained weights exactly as the real
fit was scored (same `cbr_temperature=0.001`, same code path, pure
inference).

| | lag 3 real | lag 3 shuffled | lag 4 real | lag 4 shuffled | lag 5 real | lag 5 shuffled |
|---|---|---|---|---|---|---|
| **reconstruction** | I=+0.122, z=+29.4 | **I=+0.125, z=+32.3** | I=+0.092, z=+25.8 | I=+0.100, z=+29.8 | I=+0.076, z=+23.8 | I=+0.086, z=+29.2 |
| **encoder_head** | I=+0.049, z=+11.1 | **I=+0.049, z=+11.0** | I=+0.038, z=+10.0 | I=+0.041, z=+10.8 | I=+0.031, z=+9.2 | I=+0.036, z=+10.1 |

(shuffled variogram range: `reconstruction` median 3.0px — identical to the
real fit's; `encoder_head` median 2.0px, vs the real fit's 3.0px. Full
per-lag detail: `results/exp_501/spatial_validation/stage5_cbr_{reconstruction,encoder_head}_shuffled.json`.)

**Both variants fail — one narrowly, one so completely there is no
meaningful gap left to explain.** `reconstruction`'s shuffled coherence is
slightly *larger* than its real fit's at every one of lags 3–5 (the same
direction of failure mi-SVM's and Tier 3's controls showed). `encoder_head`
goes further: its shuffled numbers are **essentially indistinguishable
from its real ones** (I=+0.049 vs +0.049 at lag 3; +0.041 vs +0.038 at lag
4) — not "comparable," but numerically the same result to two decimal
places, from a model trained on labels that have zero relationship to the
truth. Per the plan's own pre-registered exit criterion, **both of Tier 4's
spatial claims are dropped**, on the same strength of evidence as mi-SVM's
and Tier 3's retractions.

**This closes out every control this project has run, and the pattern is
now unanimous.** Four real spatial-coherence claims were made across this
project's four tiers (mi-SVM, attention, `reconstruction`, `encoder_head`
— EM's was never made, since it collapsed on its own terms before spatial
validation could say anything). All four have now been tested against a
shuffled-label control. **All four failed.** The likely mechanism,
articulated identically at each retraction (docs/stage5_spatial_validation.md
for mi-SVM, docs/stage6_attention_mil.md §5a for attention, here for Tier
4): every one of these posteriors is a function of *some* representation of
the raw spectrum — RF-selected bands for mi-SVM, a frozen CVAE embedding
for attention, the CVAE's own encoder/decoder for both cbr variants — and
real tissue has real spatial autocorrelation in its raw spectral
characteristics, entirely independently of which diagnostic label a region
is being fit to represent. A model with enough flexibility to fit *any*
label well enough (§3's 88–93% training "accuracy" on `encoder_head`'s
auxiliary classifier, achieved under complete label noise, is direct
evidence this flexibility exists) will produce a lag≥3-coherent
non-diagnostic-mass map regardless of the label, because the coherence was
never coming from the label in the first place. **The spatial-validation
pipeline itself (Moran's I + variogram by lag, docs/mil_tiers_methodology.md
§9) worked exactly as designed at catching this** — it is the pipeline that
found all four failures, not a limitation of it. What it has shown is that
none of the four tiers' apparent spatial structure, as measured here,
demonstrates recovery of the true diagnostic label; each is better
explained as smooth, label-independent structure already present in the
underlying spectral representation.

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

### External agreement vs `viable_legacy`: degenerate for a real, discovered reason — now fixed

Both variants, under the *original* flat 0.5 threshold: **ARI=0.0000,
AMI=0.0000, "diagnostic": 288,678/288,678 — every single instance.** This
was not the CVAE finding nothing, and not a bug in Stage 5's own code
(unchanged, already trusted for mi-SVM/EM). It was a **structural property
of the `1 - max_k p_k` convention applied to a genuine K-way softmax,
specific to K=2**:

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
0.5 as K grows — `(K-1)/(2K-1)` — this was flagged as a concern for Tier 3
too, and confirmed directly once Tier 3 was built — docs/stage6_attention_mil.md §5.)

**Fixed 2026-09-29**, not just documented: `hard_diagnostic_threshold()`
(`scripts/stage5_spatial_validation.py`) rescales the cutoff for a
joint-softmax source to half the maximum non-diagnostic mass the
construction can ever produce, `(K-1)/(2·(2K-1))` = 1/6 at K=2, instead of a
flat 0.5 inherited from mi-SVM's differently-shaped independent-score
source (which is left on 0.5, unaffected — already non-degenerate there).
Recomputed under the corrected threshold (threshold=0.1667):

| variant | ARI | AMI | "diagnostic" |
|---|---|---|---|
| reconstruction | −0.0148 | +0.0009 | 26,221/288,678 (9.1%) |
| encoder_head | −0.0458 | +0.0089 | 246,678/288,678 (85.5%) |

No longer structurally degenerate (no longer 100%/0.0000 for either), but
also not a positive finding — both ARI values are slightly *negative*
(worse than chance agreement with `viable_legacy`), and the two variants
now disagree sharply with each other on how much of the dataset is
"diagnostic" (9% vs 86%), which the flat-0.5 version's shared 100% had
been masking. The *continuous* non-diagnostic mass used by the lag-band
statistics above was never affected by any of this — real per-instance
variation exists within `[0, 1/3]` regardless of the threshold, and that is
exactly what Moran's I and the variogram measured throughout.

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

**Extended to the other three Tier 0 references (2026-09-29), closing the
gap this section originally left open:**

| reference | Tier 0 bag_acc | mi-SVM Δ | 95% CI | p | significant? |
|---|---|---|---|---|---|
| 1D-CNN | 0.6034 | +0.0862 | [−0.017, +0.190] | 0.126 | no |
| Apollo (RF+SVM) | 0.7586 | −0.0690 | [−0.190, +0.035] | 0.238 | no |
| Random Forest | 0.8103 | −0.1207 | [−0.225, −0.017] | 0.046 | **yes, worse** |
| XGBoost | 0.8103 | −0.1207 | [−0.225, −0.017] | 0.046 | **yes, worse** |

Same qualitative shape as Tier 3's later table (docs/stage6_attention_mil.md
§4) and Tier 4's own table above: not distinguishable from the CNN or from
Apollo, but significantly behind the two tree ensembles specifically. Across
every tier evaluated in this project so far — Tier 0 itself, mi-SVM, Tier 4
both variants, and Tier 3 — RF and XGBoost have never once been beaten by a
statistically significant margin. **mi-SVM's own spatial claim is
separately dropped** (docs/stage5_spatial_validation.md, shuffled-label
control) — this bag-accuracy result stands on its own and is unaffected by
that, since it was never a spatial claim to begin with.

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
- **§4a's Modal GPU run was verified directly, not assumed from exit code
  0** (2026-09-30): the real, pre-existing CVAE weights' file timestamps
  were confirmed unchanged after the shuffled run completed; the shuffled
  run's own history CSVs were read directly, confirming a real 100-epoch
  loss trajectory (not a crash-and-silently-return-early); the parameter
  chain carrying `permutation_seed` from the driver script through to
  `run_local` was re-read end-to-end by inspection, since Modal's `.remote()`
  call did not stream the container's own per-epoch print statements back
  to the local terminal the way a local subprocess would.

## 7. Deviations from the plan / honest gaps

- **The external-agreement convention bug (§4) is fixed (2026-09-29)**, not
  just documented — `hard_diagnostic_threshold()` in
  `scripts/stage5_spatial_validation.py` rescales to half the theoretical
  max for a joint-softmax source. No longer an open item.
- **`reconstruction`'s temperature (T=0.001) was calibrated from one fold's
  smoke-test sample, not swept.** A sensitivity check across T is not done.
  Moot in one sense given §4a's retraction, but still relevant to the
  bag-level numbers in §3, which stand independently of the spatial claim.
- **The K-diagnostic-column bag-accuracy metric itself (§5) is now known to
  be unusable for any method that implements Model (C)'s hard support
  restriction throughout scoring** — not a Tier 4 limitation, but a limitation
  of `scripts/stage6_bag_eval.py` as a general-purpose tool. It remains valid
  for mi-SVM and for Tier 4's two variants (verified per-method, not assumed),
  and the script now warns automatically rather than requiring the caller to
  remember which methods are safe.
- **The shuffled-bag-label control is done for every tier that can produce
  one, and every one has failed (§4a, 2026-09-30).** No longer an open
  item — the answer is now known, and it is negative across the board.
- **mi-SVM was re-run against Apollo/RF/XGBoost (2026-09-29)** — not
  significant vs Apollo, significantly worse vs RF/XGBoost (docs/
  stage6_generative_mil.md §5). EM's own bag-accuracy remains excluded
  (circular, §5) regardless of reference model, so extending it further
  would not be informative.
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
- **§4a's cross-tier pattern (all four spatial claims failed) is a project-level
  finding, not a Tier-4-specific one** — any future MIL formulation (Stage 8's
  mixture model, in particular) inherits the same burden of proof: a lag≥3
  Moran's I/variogram result is not reportable as evidence of label-driven
  structure until it has been run through its own shuffled-label control,
  regardless of how large the z-score looks.
- `mil_fit_stage.py`'s `method='cbr'` dispatch pattern
  (`_score_one_fold_cbr` returning a *list* of `(variant, posterior,
  in_sample)` from one shared per-fold computation) is the template for any
  future tier that produces more than one variant per fold cheaply.

## 9. Update, 2026-10-02 — re-run against the β-tuned CVAE (`exp_607`), plus a new variant and a new test

Tier 4 was re-run end to end against `exp_607`'s CVAE (β=0.25, the
least-collapsed generator from the Suite 600 β sweep,
docs/beta_sweep_findings.md). The motivation was that §3/§4's original
numbers came from `exp_501`'s β=2.0 CVAE, whose decoder was later shown to
be badly mode-collapsed — the question was whether a materially better
generator changes Tier 4's verdict. **It does not.**

No code changes were needed to re-point Tier 3/4 at a different experiment's
CVAE (`--config exp_607`); both already resolve weights from
`paths.models_cvae_dir`.

### 9.1 Bag-level accuracy (n=59 bags, TRTR references)

A third variant was added this round — **`latent_rf`**: a plain
RandomForest (Tier 0's own hyperparameters) fit on the frozen encoder's
`z_mean`, i.e. the same classifier Tier 0 uses on raw spectra, fed the
CVAE's 32-d latent instead. Unlike `reconstruction`/`encoder_head` it does
train something per fold, so it is not "zero-training-cost"
(`src/mil/generative.py::latent_rf_posteriors`,
`pipeline/mil_fit_stage.py --methods latent_rf`).

| Variant | bag_acc | vs CNN (0.712) | vs Apollo (0.763) | vs RF/XGB (0.814) |
|---|---|---|---|---|
| `encoder_head` | 0.763 | +0.051, n.s. | ±0.000, n.s. | −0.051, p=0.072 |
| `reconstruction` | 0.729 | +0.017, n.s. | −0.051, n.s. | −0.085, p=0.070 |
| `latent_rf` | 0.729 | +0.017, n.s. | −0.034, n.s. | **−0.085, p=0.008** |

Same verdict as §3, now with a *correctly trained* CNN reference (exp_501's
TRTR CNN was refit — it had been badly undertrained at 0.593, inflating
every earlier CNN-referenced delta; it is 0.712 here). Against the two
strongest Tier 0 classifiers, all three variants are flat-to-worse, and
`latent_rf` is significantly worse. The CVAE's latent space does not beat
the raw spectrum through the same classifier.

### 9.2 Spatial validation — coherence is present, and §9.3 shows it is not diagnostic

Pooled Moran's I, 59 slides, lag ≥3 headline (lags 1–2 contaminated by the
3×3 median filter, §4):

| Variant | lag 3 | lag 4 | lag 5 | variogram range |
|---|---|---|---|---|
| `reconstruction` | I=+0.144, p=0.0003 | I=+0.107, p=0.021 | I=+0.085, p=0.048 | 4 px |
| `encoder_head` | I=+0.056, p=0.059 | I=+0.044, p=0.100 | I=+0.037, p=0.132 | 3 px |

`reconstruction` shows real, significant coherence through lag 4–5;
`encoder_head` does not reach significance at any lag. **No shuffled-label
control has been run for `exp_607`** — the control that killed every
equivalent claim for `exp_501` (§4a). Until it is, this is an uncontrolled
result, and §4a's cross-tier pattern is the prior.

### 9.3 Key-instance utility test — the coherence is not diagnostic structure

The new falsification test defined in docs/mil_tiers_methodology.md §10.2,
run with `reconstruction` as the selection signal (the only variant with
significant coherence, §9.2) and Tier 0's own RF as the classifier.
Implementation: `scripts/cvae_weighted_rf_vote.py`. Baseline is Tier 0's
unmodified RF, all instances, equal vote (bag_acc 0.814):

| Variant | bag_acc | Δ | p |
|---|---|---|---|
| Weighted vote (same RF, CVAE P1 as vote weight) | 0.814 | ±0.000 | 1.000 |
| Key-instance vote (same RF, median split, 0 fallbacks) | 0.814 | ±0.000 | 1.000 |
| **Key-instance train** (new RF on key-only instances) | **0.763** | **−0.051** | 0.230 |

**Correction, 2026-10-12**: the key-instance-vote row originally read 0.797
(Δ−0.017, p=0.680), computed by thresholding the shared RF's
`predict_proba()` at 0.5 instead of calling its own `.predict()` — these can
genuinely disagree on real data (confirmed: sklearn's RandomForest flips
69/3686 instances for one bag here, enough to change that bag's majority).
Fixed in `scripts/cvae_weighted_rf_vote.py` (`_collect_per_fold` now returns
and uses the model's own `.predict()` output); re-run on all 15
(config × tier/variant) cells this project has used this script for (§10.2,
§11, `docs/stage6_attention_mil.md` §10.2) and confirmed this is the
**only** cell anywhere that the fix actually changed — every other baseline/
weighted/key-instance-vote number already matched `.predict()` by
coincidence. See `tests/test_cvae_weighted_rf_vote.py::test_collect_per_fold_returns_predict_not_thresholded_proba`
for the regression test.

Weighted vote and key-instance vote both tie equal-vote pooling exactly
(neither helps nor hurts), and the strongest form of the test — actually
training on the selected instances — is the only one that underperforms.
**This is the substantive finding of this update**: the
instances `reconstruction`'s posterior selects are not the instances the
decision depends on. A posterior can be spatially coherent at lag ≥3
(§9.2) and still be selecting on something the classification task does not
use — exactly the gap §9.2 of the methodology exists to close, and the
reason spatial coherence alone should not be reported as evidence that a
MIL formulation has found diagnostic instances.

Caveats, per the methodology's own instruction: the key-instance-train RF
sees ~50% of the training instances (118k vs 236k per fold), so its deficit
conflates "selection removes signal" with "less training data hurts" — **a
random-50% control has not been run** and is required before the strong
reading. The key/non-key split was 50/50 on training data by construction
(median split) but swung from 25.5% to 62.7% key on validation across the
five folds, which is itself a cross-fold inconsistency worth noting: the
five independently-trained CVAEs do not place their non-diagnostic mass on
a common scale.

### 9.4 A second CbR convention, discovered here

`evaluation_stage.py`'s automatic CbR (the number that appears inside
`stats_eval`'s TRTR-vs-TSTR/TSRTR output) and `scripts/stage6_bag_eval.py`'s
Tier 4 CbR use **different bag-decision rules** and therefore report
different numbers for the same model and data: the former takes a hard
argmin(MSE) per instance then a majority vote (bag_acc 0.746 for exp_607);
the latter mean-pools the softmax posterior across the bag then takes the
argmax (0.729). Both are defensible; only the second is the project's
documented Tier 0/1/2/4 shared machinery (docs/mil_tiers_methodology.md §9),
so it is the canonical one for cross-tier comparison. The `stats_eval`
side-computation should not be quoted alongside Tier 4 numbers without
saying which convention produced it.

## 10. Update, 2026-10-12 — the two gaps §9.2/§9.3 flagged are now closed, and extended to 608/609

§9.2 explicitly flagged `exp_607`'s CbR spatial result as uncontrolled
("No shuffled-label control has been run for `exp_607`... until it is,
this is an uncontrolled result"). §9.3 flagged the key-instance test's
training-size confound as unresolved ("a random-50% control has not been
run and is required before the strong reading"). Both are now closed, and
both tests are extended to `608`/`609` (β=0.5/0.75) — requiring a full
second CVAE per config, trained from scratch under permuted labels
(`scripts/stage6_shuffled_label_control.py --method cvae`, since CbR's
posterior is the CVAE's own encoder/decoder output — there is no CbR model
independent of the CVAE weights themselves), before CbR's posterior could
even be computed under the shuffled condition.

### 10.1 Spatial — shuffled-label control, closing §9.2's gap

Pooled Moran's I, lag 3 (headline), real vs. shuffled fit:

| config (β) | variant | real I / z | shuffled I / z | verdict |
|---|---|---|---|---|
| `607` (0.25) | `reconstruction` | +0.1436 / +34.54 | +0.1497 / +37.09 | shuffled ≥ real — **fails** |
| `607` (0.25) | `encoder_head` | +0.0557 / +12.93 | +0.0561 / +13.40 | ~identical — **fails** |
| `608` (0.5) | `reconstruction` | +0.1570 / +38.88 | +0.1710 / +42.08 | shuffled > real — **fails** |
| `608` (0.5) | `encoder_head` | +0.0594 / +13.79 | +0.0565 / +13.59 | ~identical — **fails** |
| `609` (0.75) | `reconstruction` | +0.1577 / +38.56 | +0.1641 / +40.77 | shuffled > real — **fails** |
| `609` (0.75) | `encoder_head` | +0.0612 / +14.12 | +0.0581 / +13.57 | ~identical — **fails** |

§9.2's own prior ("given this pattern [mi-SVM, attention both failed], that
result should be treated as provisional, not assumed to survive, until its
own control is run") is confirmed exactly: `reconstruction`'s apparently
strong real coherence (the largest raw Moran's I of any tier/variant in
this project, I≈0.14–0.16) is matched or exceeded by its shuffled
counterpart at all three β values, with no exception. `encoder_head`
repeats the `exp_501` `encoder_head` pattern (real and shuffled agree to
within noise) at all three β values too. **All six (config × variant)
cells fail — this closes the "uncontrolled result" caveat definitively,
in the same direction as every other tier tested in this project except
one** (`607`'s attention — docs/stage6_attention_mil.md §10.1).

### 10.2 Key-instance utility test — §9.3's confound control added, extended to 608/609

The missing control: a fresh RF trained on a **random**, size-matched
subset of training instances (not CVAE-selected), scored on the identical
CVAE-selected validation instances `key_instance_train` uses — isolating
"does the CVAE's selection help" from "does less training data hurt."
Implementation added to `scripts/cvae_weighted_rf_vote.py`
(`_fit_and_score_random_subsample_rf`, tested in
`tests/test_cvae_weighted_rf_vote.py`, 6/6 passing). Shared baseline
(the same Tier 0 RF reused from `exp_501` in every run) ≈0.8136. Numbers
below are post-fix (see §9.3's correction note); `607`/`reconstruction`'s
key-instance-vote cell is the only one that changed (0.7966 → 0.8136):

| config (β) | variant | weighted | key-instance vote | **key_instance_train** | **random_subsample** (new) |
|---|---|---|---|---|---|
| `607` (0.25) | `reconstruction` | 0.8136 | 0.8136 | 0.7627 | 0.8136 |
| `607` (0.25) | `encoder_head` | 0.7797 | 0.7966 | 0.7966 | 0.7966 |
| `608` (0.5) | `reconstruction` | 0.8136 | 0.7966 | 0.7627 | 0.7627 |
| `608` (0.5) | `encoder_head` | 0.7627 | 0.8136 | 0.8136 | 0.8136 |
| `609` (0.75) | `reconstruction` | 0.8136 | 0.7797 | **0.7458** | 0.7627 |
| `609` (0.75) | `encoder_head` | 0.7627 | 0.8136 | 0.8136 | 0.8136 |

§9.3's original concern is resolved, and resolved against the selection
hypothesis: `607`'s `reconstruction` `key_instance_train` (0.7627) is now
directly compared to its size-matched random control (0.8136) rather than
only to the full-data baseline, and the random control is **better**, not
worse — the apparent deficit §9.3 found was not a training-size artifact
masking a real selection benefit; a same-size random subset does strictly
better than the CVAE's own choice of instances. `608`'s `reconstruction`
ties its random control exactly (0.7627 both). `609`'s `reconstruction`
is the clearest case in either direction found so far: `key_instance_train`
(0.7458) is *significantly worse* than baseline (p=0.022) and also worse
than its random-subsample control (0.7627, not independently significant,
p=0.056) — CbR's `reconstruction` posterior is actively selecting instances
that hurt a classifier trained on them, not merely failing to help.
`encoder_head` shows no selection effect at any β — tied or worse than
random at every config.

**Conclusion, replacing §9.3's provisional reading**: CbR (both variants,
all three tested β values) shows no case where CVAE-based key-instance
selection beats a same-size random subset — one case (`609`
`reconstruction`) shows selection significantly *hurting*. Unlike Tier 3's
attention (docs/stage6_attention_mil.md §10.2), CbR has no combination that
survives both this test and its own spatial shuffled-label control.

## 11. Update, 2026-10-12 — v2 preprocessing (`exp_2.607`/`exp_2.608`): the one combination that DOES survive both tests

Same two tests, extended to the new fully-automated preprocessing
(`src/data/preprocessing_v2.py`), β=0.25/0.5. `reconstruction` fails its
shuffled-label control at both β (real/shuffled z: 30.40/30.50 at β=0.25,
34.90/35.07 at β=0.5 — shuffled ≈ real each time) and shows no key-instance
advantage. **`encoder_head` is different: real spatial coherence beats
shuffled at *both* β (10.89/8.46 at β=0.25, 10.54/7.98 at β=0.5), and its
key-instance test also shows selection beating the size-matched random
control at both β (0.8305 vs 0.7966 at β=0.25, 0.8136 vs 0.7797 at β=0.5).**
This is the only tier/variant/preprocessing combination in the entire
project where *both* independent falsification tests point the same
direction at more than one data point — a more consistent signal than
`607`'s v1 attention exception (one data point, `docs/stage6_attention_mil.md`
§10). Neither family's real-vs-shuffled or selection-vs-random gap has been
checked for its own statistical significance yet. Full numbers:
`docs/experiment_inventory.md` §4.4/§4.5. Queued for a closer look
alongside `607`'s attention (`docs/experiment_inventory.md` §9, gap #10),
not yet investigated.
