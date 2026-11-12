# Stage 6 (Tier 3) — Attention MIL

**Status: complete, including the shuffled-bag-label control — and the
spatial claim is dropped.** New `src/mil/attention.py`,
`pipeline/mil_attention_stage.py`, `scripts/stage6_attention_bag_eval.py`.
Full architecture and design rationale: docs/mil_tiers_methodology.md §6.3–6.4
(methodology only, no results — this file is the results counterpart,
mirroring docs/stage6_generative_mil.md's treatment of Tier 4).

The honest summary: a real, non-degenerate attention model that trains
cleanly on all 5 folds and is a **significantly worse bag-level classifier
than three of the four Tier 0 references** (Apollo, RF, XGBoost), not
significantly better than the fourth (the CNN, the weakest Tier 0 reference
by a wide margin) — and whose apparent lag≥3 spatial coherence **does not
survive the shuffled-bag-label control** (§5a): a fit under completely
permuted labels produces *larger* coherence than the real fit at every
headline lag, the same failure mode mi-SVM's control found
(docs/stage5_spatial_validation.md). Per the plan's own pre-registered exit
criterion, Tier 3's spatial claim is dropped, same as mi-SVM's. This tier's
own debugging process also surfaced three real, sequentially-discovered bugs
before any of the numbers below could be trusted — recorded here because
each is a property of attention MIL on this data, not a one-off
implementation slip.

> **Reproducing this.** `pytest tests/test_stage6_attention_mil.py
> tests/test_stage6_attention_bag_eval.py`. Real fit:
> `python -m pipeline.mil_attention_stage --config exp_501` (needs exp_501's
> `cvae` stage already trained — reuses its saved per-fold encoder weights,
> frozen). Spatial validation:
> `python scripts/stage5_spatial_validation.py --config exp_501 --method
> attention`. Bag-level eval:
> `python scripts/stage6_attention_bag_eval.py --config exp_501
> --reference-model {cnn,apollo,rf,xgb}`.

## 1. What this tier is, briefly

Gated attention (Ilse et al. 2018) over instance logits, computed on each
fold's frozen CVAE encoder trunk, trained on randomly-sampled 228-instance
sub-bags (sized from the π_min identifiability floor, per
docs/mil_tiers_methodology.md §6.3). Full rationale for every design choice
— why logits rather than embeddings, why the trunk is frozen, why
sub-bagging, the regularization, the two separate posterior methods — is in
the methodology doc; not repeated here.

## 2. Three real bugs, found and fixed before trusting any number

All three were caught by smoke-testing before the real 5-fold run, each
diagnosed from first principles and verified empirically rather than
guessed at:

1. **Bag-identity overfitting.** The first, unregularized smoke test showed
   sub-bag training loss falling steadily while held-out bag accuracy got
   *worse* with more training — the signature of the model memorizing which
   of the ~38 real training bags it was looking at, rather than learning
   diagnostic content, given thousands of sub-bag draws per epoch from so
   few real bags. Fixed with dropout (0.3), L2 (1e-3), a smaller attention
   dimension (32), and per-(sub-)bag feature centering.
2. **A fold-specific vanishing-gradient stall.** fold_1's smoke test froze
   at `loss ≈ ln(2)` (chance-level, for a 2-class cross-entropy) for 70+
   epochs. Traced — by direct inspection of gradients and activations, not
   guessed — to each fold's independently-trained CVAE encoder producing
   `z_mean` on a different absolute scale (fold_1's raw standard deviation
   measured ~7.5× smaller than fold_0's). Per-bag centering could not fix
   this by construction (it changes location, not scale — confirmed
   directly: the before/after-centering std ratio was 1.0000 exactly).
   Fixed with `LayerNormalization` applied unconditionally after centering.
3. **An evaluation-methodology mismatch, not a training bug.** With both
   fixes above, training looked excellent, but *naive* pooling of the
   attention-free `instance_posterior` (majority vote or mean, the same
   pooling Tier 0/1/2/4 use) gave chance-level bag accuracy on a model whose
   own attention-weighted training loss was excellent. Root cause: an
   instance attention learns to downweight during training receives
   near-zero training gradient regardless of what it individually predicts,
   so it can develop an arbitrary individual posterior — exactly the
   instances a naive vote lets dominate. Fixed by adding `bag_posterior`
   (attention-weighted, over a bag's true full instance count) as a
   separate scoring path, verified against `call()`'s own batched
   computation and a "resists dilution by noise instances" behavioral test.
   This also meant the shared latent-label store (which holds only the
   attention-free posterior, by design — Stage 5 needs one context-free
   value per instance) could not be the source for bag-level evaluation, so
   `mil_attention_stage.py` additionally persists each fold's true
   held-out test bags' attention-weighted decision directly, while the
   fold's model is still in memory, to
   `results/{exp}/latent/attention_bag_level.json`
   (`scripts/stage6_attention_bag_eval.py` reads this artifact, not the
   store — see that script's own docstring for why this is a deliberate,
   documented split rather than an inconsistency).

## 3. The real 5-fold fit

All 5 folds trained cleanly, no stalls, `sub_bag_null_rate=0.0000` in every
fold (target ε=0.01 — sub-bags essentially always contained at least one
diagnostic instance, well inside tolerance):

| fold | fit bags | inner-val bags | best epoch inner_val_loss | inner_val bag_acc | sub_bag_null_rate |
|---|---|---|---|---|---|
| fold_0 | 38 | 7 | 0.6031 | 0.8571 | 0.0000 |
| fold_1 | 39 | 8 | 0.1189 | 1.0000 | 0.0000 |
| fold_2 | 38 | 7 | 0.2766 | 0.8571 | 0.0000 |
| fold_3 | 40 | 8 | 0.4715 | 0.8750 | 0.0000 |
| fold_4 | 39 | 8 | 0.2042 | 0.8750 | 0.0000 |

**Read the inner-validation numbers as a training-health check, not as the
headline result.** Mean inner-val bag accuracy across folds is 0.893 — but
this is computed on 7–8 held-out-from-training bags per fold (still drawn
from the same slides the model's early stopping was tuned against,
group-aware but small-N), not the true CV test fold. §4's pooled n=59
true-test-bag number is substantially lower (0.644) — worth stating
plainly rather than letting the better-looking inner number stand as if it
were the real result. Some of that gap is expected (7–8-bag accuracy has a
wide sampling range — one bag is worth ~13 points), but its direction and
size are consistent with the model still not generalizing as well to
genuinely unseen bags as the inner-validation loop suggested during
training.

## 4. Bag-level accuracy vs Tier 0 — all four references

`scripts/stage6_attention_bag_eval.py`, exp_501, n=59 bags, 1000-resample
class-stratified bootstrap, seed 42. Tier 0 references reproduced from
docs/stage6_generative_mil.md §3 (same numbers, same script family):

| reference | Tier 0 bag_acc | attention bag_acc | Δ (attention − Tier0) | 95% CI | p | significant? |
|---|---|---|---|---|---|---|
| 1D-CNN | 0.5932 | 0.6441 | +0.0508 | [−0.102, +0.203] | 0.604 | no |
| Apollo (RF+SVM) | 0.7627 | 0.6441 | −0.1186 | [−0.220, −0.034] | 0.012 | **yes, worse** |
| Random Forest | 0.8136 | 0.6441 | −0.1695 | [−0.288, −0.068] | 0.002 | **yes, worse** |
| XGBoost | 0.8136 | 0.6441 | −0.1695 | [−0.288, −0.068] | 0.002 | **yes, worse** |

(attention: bag_F1=0.6228, bag_kappa=0.2567, n=59 throughout — one
`bag_posterior` per bag, duplicate cross-fold bags already averaged per
`_load_bag_level_across_folds`.)

**Read this the same way docs/stage6_generative_mil.md §3 reads Tier 4's
table — by every reference, not just the flattering one.** Against the
CNN, attention MIL shows a positive but non-significant delta, the same
qualitative read Tier 4's `reconstruction` variant got against the CNN
(+0.119, p=0.150) — a hint, not a result. Against the three stronger, tree-
based Tier 0 classifiers, attention MIL is **significantly worse**, not
merely tied — the one tier, so far, whose comparison table has no ✓ at all.
The tree ensembles (Apollo/RF/XGBoost) remain the strongest bag-level
classifiers in this project across every tier evaluated to date (Tier 0
themselves, Tier 1 mi-SVM, Tier 4 both variants, and now Tier 3).

## 5. Spatial validation

Full lag sweep, all 59/59 slides, 288,678/288,678 instances (the HF-1887
fix applies here — this is a fresh fit, not inherited from before that
fix), 200 within-slide permutations/slide, seed 42
(`results/exp_501/spatial_validation/stage5_attention_attention.json`):

| lag | Moran's I (mean ± std) | permutation p (mean ± std) | permutation z (mean ± std) | variogram γ |
|---|---|---|---|---|
| 1 [contaminated] | +0.152 ± 0.070 | 0.0004 ± 0.003 | +20.47 ± 9.61 | 0.00536 |
| 2 [contaminated] | +0.058 ± 0.039 | 0.0015 ± 0.007 | +11.20 ± 7.72 | 0.00599 |
| **3 [headline]** | **+0.0236 ± 0.028** | **0.1105 ± 0.210** | **+5.634 ± 7.255** | 0.00622 |
| 4 | +0.0176 ± 0.026 | 0.1749 ± 0.286 | +5.003 ± 7.802 | 0.00627 |
| 5 | +0.0148 ± 0.023 | 0.1873 ± 0.283 | +4.637 ± 7.313 | 0.00629 |
| 6–10 | declining, +0.0068 to +0.0122 | rising, 0.19–0.28 | declining, +3.2 to +4.2 | ≈0.0063, flat |

Variogram range: median 2.0px (n=59).

**External agreement vs `viable_legacy`, using the corrected threshold**
(2026-09-29 fix — see below): **ARI=0.0164, AMI=0.0008** (n=288,678,
threshold=0.1667; model calls 243,692/288,678 = 84.4% "diagnostic", vs
`viable_legacy`'s own 227,344/288,678 = 78.8%). Weak but no longer
structurally degenerate. The original computation under a flat 0.5
threshold gave **ARI=0.0000, AMI=0.0000, 288,678/288,678 (100%) called
"diagnostic"** — the identical structural degeneracy
docs/stage6_generative_mil.md §4 diagnoses for Tier 4's two variants:
attention's instance head is a genuine, unconstrained K=2 softmax (docs/
mil_tiers_methodology.md §6.3 states this was a deliberate design choice,
made *before* this consequence was confirmed), so `max_k p_k ≥ 0.5` by
construction and the lifted non-diagnostic mass is provably bounded to
`[0, 1/3]` — never crossing a flat 0.5 threshold at all, regardless of what
the posterior encodes. This is the K=2-softmax convention bug §4 of that
doc predicted would apply to Tier 3 "if its attention head is a plain
softmax too" — confirmed directly rather than left as a flag, and now
**fixed** (not just documented) via `hard_diagnostic_threshold()`
(`scripts/stage5_spatial_validation.py`): for a joint-softmax source the
cutoff is rescaled to half of the maximum non-diagnostic mass the
construction can ever produce, `(K-1)/(2·(2K-1))` = 1/6 at K=2, rather than
a flat 0.5 inherited from a differently-shaped (independent-score) source
like mi-SVM's — which is deliberately left on its original 0.5 (already
non-degenerate, 0.96% hard-diagnostic — this fix does not retroactively
touch it). Provisional, documented, not swept, the same treatment as this
project's other provisional constants. As with Tier 4, the *continuous*
non-diagnostic mass driving the Moran's I/variogram numbers above was never
affected by this — only the hard-threshold external-agreement check was.

### 5a. The shuffled-bag-label control — the spatial claim is dropped

Per docs/MIL_REFRAME_PLAN.md §6/§13's pre-registered check: does lag≥3
coherence survive a fit under bag labels that are an exact permutation of
the true ones (same class balance, everything else — features, bag
membership, CV splits, architecture, hyperparameters — held fixed)? For
Tier 3 this is cheap to run: the frozen CVAE trunk needs no retrain, only
the attention head refits (`scripts/stage6_shuffled_label_control.py
--method attention`, 2026-09-29; permutation_seed=123, 44/59 bags received
a different label). All 5 folds trained without incident (early stop
between epoch 15–30, vs. the real fit's 25–79 — a *faster* stall, exactly
what training on labels with no real signal should look like).

| lag | real fit: I / z | shuffled fit: I / z |
|---|---|---|
| 3 (headline) | +0.0236 / +5.63 | **+0.0613 / +14.70** |
| 4 | +0.0176 / +5.00 | **+0.0478 / +13.38** |
| 5 | +0.0148 / +4.64 | **+0.0403 / +12.45** |
| 10 | +0.0068 / +3.24 | +0.0231 / +10.08 |

(shuffled fit's own variogram range: median 2.0px, identical to the real
fit's — `results/exp_501/spatial_validation/stage5_attention_attention_shuffled.json`.)

**This does not merely fail to support the real fit's coherence — it beats
it, at every headline lag, by more than 2×.** Per the plan's own
pre-registered exit criterion ("excess coherence at lag≥3 survives the
shuffled-label control — if not, the spatial claim is dropped"), Tier 3's
already-ambiguous lag-3 signal (§5's own mean-p/mean-z tension) now fails
definitively: a fit with **no genuine label-content relationship
whatsoever** produces *stronger* apparent coherence than the real one. This
is the same failure mode mi-SVM's own control found
(docs/stage5_spatial_validation.md), and the same likely mechanism applies:
the frozen CVAE trunk's `z_mean` features are a real, continuous embedding
of each pixel's spectrum, and real tissue has real spatial autocorrelation
in its raw spectral characteristics *regardless of which label a region
happens to be fit toward* — general tissue continuity, acquisition
smoothness, or both, not diagnostic content. An attention head trained on
permuted labels has nothing else to latch onto, so it latches onto
*whatever direction in that shared feature space happens to vary smoothly
across space* — and that is apparently enough to produce lag≥3 "coherence"
on its own. **The spatial claim for Tier 3 is dropped**, per the plan's own
contingency, on the same strength of evidence as mi-SVM's retraction: a
head-to-head comparison, not a borderline p-value.

Two controls have now been run (mi-SVM, attention); both failed the same
way. That is no longer a one-off — it is starting to look like a property
of doing MIL spatial validation on top of *any* representation built from
raw spectral features (RF-selected bands for mi-SVM, a frozen CVAE
embedding for attention) on this dataset, independent of which MIL
formulation sits on top. Tier 4's `reconstruction` variant had the
strongest real spatial result of any tier so far (lag3 p=0.028) but its own
control has not been run yet (needs a full CVAE retrain) — given this
pattern, that result should be treated as provisional, not assumed to
survive, until its own control is run.

**A real tension worth stating plainly: mean-p and mean-z disagree at the
headline lag.** At lag 3, the mean permutation p-value (0.1105) does not
clear the conventional 0.05 bar, while the mean z-score (+5.634) would
imply an extremely small p under a normal approximation. This is not a
contradiction in the underlying test — permutation p is bounded below by
`1/(n_perm+1) ≈ 0.005` per slide while z is not, and the large std on both
columns (0.210 on p, 7.255 on z, both close to or exceeding their own
means) signals a **highly heterogeneous per-slide effect**: some slides
show strong, low-p coherence and others show none, and naively averaging
p-values across 59 such slides is a conservative, easily-dominated-by-
null-slides summary compared to averaging z (which the large positive
slides pull up more). Neither the mean-p nor the mean-z framing alone is a
clean verdict; both are reported rather than picking the one that reads
better. The unambiguous parts: Moran's I is positive and monotonically
declining from lag 3 to lag 10 (not noise-like), and every lag from 3–10
has a positive mean z. Whether that adds up to "real spatial coherence"
survives a stricter bar than this file settles — see §7.

## 6. Cross-tier spatial comparison, lag 3

Consolidating every tier's headline lag-3 number in one place (mi-SVM and
Tier 4 numbers reproduced from docs/stage5_spatial_validation.md and
docs/stage6_generative_mil.md §4 respectively — not re-run here):

| method | Moran's I (lag 3) | mean p | mean z | variogram range | shuffled control |
|---|---|---|---|---|---|
| mi-SVM (Tier 1) | +0.006 | 0.361 | +1.5 | median 2px | **failed — z=+5.01 shuffled vs +1.50 real; dropped** |
| **attention (Tier 3)** | **+0.024** | **0.111** | **+5.6** | median 2.0px | **failed — z=+14.70 shuffled vs +5.63 real; dropped** |
| CbR reconstruction (Tier 4) | +0.122 | 0.028 | +29.4 | median 3.0px | not run yet (needs CVAE retrain) |
| CbR encoder_head (Tier 4) | +0.049 | 0.057 | +11.1 | median 3.0px | not run yet (needs CVAE retrain) |

**Both controls run so far have failed the same way — the raw Moran's I/z
numbers in this table are no longer a reliable guide to which tier's
spatial claim is real.** Tier 3 sat between mi-SVM and Tier 4 by every
column here, but "sits in the middle of two numbers" turned out not to
predict "closer to holding up": mi-SVM's weaker-looking real signal and
Tier 3's stronger-looking one both collapsed identically under their
controls. Until Tier 4's own control is run, its table position (strongest
of all five) should not be read as evidence it will survive — see §5a's
closing paragraph.

## 7. Honest overall read

Attention MIL is a genuine, working model — not degenerate, not overfit
(after the fixes in §2), trains reproducibly across all 5 folds. It is
**not**, on the evidence now in hand, a source of real spatial information:
§5a's shuffled-label control shows its apparent lag≥3 coherence is fully
explained by the frozen CVAE trunk's own feature-space smoothness, not by
anything the attention head learned about the true label. And as a
bag-level classifier — the task's actual clinical objective — it is **the
first tier in this project whose comparison table shows a significant
loss** against multiple Tier 0 references, not merely a non-significant
tie. Three things are true at once and none should be softened to fit the
others: the architecture and its debugging were done correctly (§2, §3,
verified in §9), the resulting model is, on the evidence gathered so far,
not competitive with the simpler Tier 0 tree ensembles at the one task that
matters most, and its one apparent point of genuine interest — spatial
coherence — does not survive the one test built specifically to check it.
This is, so far, the most negative honest result of any tier in this
project, and it is reported as such rather than emphasizing §3–§4's
legitimate engineering work over §5a's outcome.

## 8. Deviations from the plan / honest gaps

- **The inner-validation/true-test generalization gap (§3) is unexplained.**
  Mean inner-val bag accuracy (0.893) substantially exceeds the true
  pooled test accuracy (0.644). Plausible contributors — small inner-val
  N (7–8 bags), residual overfitting the current regularization doesn't
  fully remove, or a genuine distribution shift between training-fold
  bags and held-out bags — have not been distinguished from one another.
  Explicitly deferred (2026-09-29): would need several more real Tier 3
  refits under different regularization to chase down, and the tier's
  spatial claim is already dropped (§5a) regardless of the outcome.
- **No hyperparameter sweep.** Dropout/L2/attn_dim/sub-bag size were fixed
  from the smoke-testing process (§2) and the π_min-derived formula
  respectively, not tuned against the true test folds (which would itself
  be a leakage risk given how few bags exist).

## 9. Verification

- **New `tests/test_stage6_attention_mil.py`, `tests/test_stage6_attention_bag_eval.py`,
  `tests/test_stage6_shuffle_control.py`** — the shuffle-control test file
  covers `src/mil/shuffle_control.py`'s permutation logic (pure numpy);
  `scripts/stage6_shuffled_label_control.py`'s own attention dispatch (added
  2026-09-29, `pipeline/mil_attention_stage.py` gaining optional
  `loader`/`store_path`/`bag_level_path` overrides) was smoke-tested on
  fold_0 before the real 5-fold shuffled run, confirmed correct wiring
  (separate output paths, 44/59 bags relabeled) and a directionally sensible
  result (early stop at epoch 15 vs. the real fit's 40, `sub_bag_null_rate`
  jumping from 0.0000 to 0.3368) before being trusted for the full run.
- **New `tests/test_stage5_spatial.py` cases for `hard_diagnostic_threshold`**
  (2026-09-29): confirms mi-SVM stays at a flat 0.5, confirms the K=2
  joint-softmax cutoff is exactly 1/6, confirms the cutoff loosens (but
  stays below 0.5) as K grows.
- **Full suite re-run against `tests/` (this pass, 2026-09-29): 152
  passed, 2 failed** — the same two pre-existing `test_config_manager.py`
  failures present throughout this project's history, unrelated to Tier 3
  or to anything touched this stage.
- **Real data, full pipeline**: fit (5 folds) → store + `attention_bag_level.json`
  → spatial validation → bag-level eval against all four Tier 0 references
  → shuffled-label control (5 folds) → shuffled spatial validation ran
  end-to-end with no errors.
- **`bag_posterior` correctness** (§2, bug 3) verified two ways before
  being trusted for the real run: consistency against `call()`'s own
  batched computation, and a behavioral test that a bag's decision resists
  dilution by added noise instances (an attention model that had not
  actually learned to downweight noise would fail this).

## 10. Update, 2026-10-12 — extended to the β sweep (607/608/609): shuffled-label control + the new key-instance utility test

§5a's shuffled-label control was `exp_501`-only (β=2.0, the original,
badly-collapsed CVAE). Closing that gap: the same control, plus the new
key-instance utility test (docs/mil_tiers_methodology.md §10.2,
`scripts/cvae_weighted_rf_vote.py`, extended this round to accept any
tier's posterior via `--method`/`--variant`, not just Tier 4's), now run
for all three Suite 600 β values (0.25/0.5/0.75).

### 10.1 Spatial — shuffled-label control, all three β values

Pooled Moran's I, lag 3 (headline), real vs. shuffled fit:

| config (β) | real I / z | shuffled I / z | verdict |
|---|---|---|---|
| `607` (0.25) | +0.0742 / +19.10 | +0.0650 / +16.14 | **real > shuffled — the one exception** |
| `608` (0.5) | +0.0541 / +13.10 | +0.0657 / +16.25 | shuffled > real — fails, same as `exp_501` |
| `609` (0.75) | +0.0428 / +10.75 | +0.0610 / +15.24 | shuffled > real — fails, same as `exp_501` |

`608` and `609` replicate `exp_501`'s finding exactly (§5a): shuffled
coherence meets or beats real coherence, so the spatial claim fails at
both of these β values too. `607` is the only config, across every tier and
every β tested in this project to date, where the real fit shows *more*
lag-3 coherence than its shuffled counterpart. The gap is real but modest
(not the >2× margins seen in the clear failures) and no formal test has
been run on the real-minus-shuffled difference itself — the project's
existing convention only ever compared point estimates. Not yet treated as
a surviving spatial claim; see §10.2 for the second, independent check
before deciding whether this is worth a closer look.

### 10.2 Key-instance utility test — first-ever run for Tier 3

This test previously existed only for Tier 4 (CbR)'s `reconstruction`
posterior (`docs/stage6_generative_mil.md` §9.3). It had never been run
with attention's own posterior as the selection signal before this round,
and a confound-(i) control (a same-size random-subsample, isolating
"selection helps" from "less training data hurts") did not exist at all
until this round. Baseline is the same shared Tier 0 RF reused from
`exp_501` in every run (bag_acc≈0.8136) — identical across all three
configs below, since only the posterior used for selection/weighting
changes, not the classifier being scored:

| config (β) | weighted vote | key-instance vote | **key_instance_train** | **random_subsample** (new) |
|---|---|---|---|---|
| `607` (0.25) | 0.7966 | 0.7966 | **0.8305** | 0.7966 |
| `608` (0.5) | 0.8136 | 0.8136 | 0.7966 | 0.8136 |
| `609` (0.75) | 0.7797 | 0.7966 | 0.7797 | 0.8136 |

**Verified, 2026-10-12**: a `.predict()`-vs-`predict_proba()`-threshold bug
was found and fixed in `scripts/cvae_weighted_rf_vote.py` (corrected one
cell in `docs/stage6_generative_mil.md` §9.3/§10.2). All 9 (config × tier)
key-instance cells, including this table's 9 attention/CbR values, were
re-run post-fix; **none of this table's numbers changed** — re-confirmed
identical to the above.

`607` is again the only config where `key_instance_train` (a fresh RF
fit on only the attention-selected key instances) beats its own
size-matched random-subsample control — 0.8305 vs 0.7966, and also above
the shared baseline (0.8136). `608` and `609` show no selection advantage:
`608` ties baseline via random subsampling and selection does no better;
`609`'s selection (0.7797) is actually below its own random-subsample
control (0.8136).

**Taken together with §10.1: `607`'s attention is the only result, across
this entire project, where both independent falsification tests (spatial
coherence and key-instance utility) point the same direction rather than
failing.** Neither individually reaches the dramatic margins that
characterize a clean failure elsewhere in this project, and neither has
been checked for its own statistical significance beyond the existing
per-variant bootstrap (none of the four key-instance variants' deltas vs.
baseline are significant at `607`: p=0.69–0.79). This is flagged as the one
candidate worth a closer look, not yet reported as a positive finding.

## 11. Update, 2026-10-12 — v2 preprocessing (`exp_2.607`/`exp_2.608`)

Same two tests, extended to the new fully-automated preprocessing
(`src/data/preprocessing_v2.py`, no NMF/GMM/manual curation),
β=0.25/0.5 (mirroring `607`/`608`). Attention fails its shuffled-label
control at **both** v2 β values (real z=13.73/shuffled z=14.50 at β=0.25;
11.56/14.38 at β=0.5) — does **not** replicate `607`'s v1 exception (§10.1).
Key-instance: `exp_2.608` shows selection beating random
(0.7966 vs 0.7797); `exp_2.607` does not (0.7966 vs 0.8136). Full numbers
and the cross-tier comparison (CbR `encoder_head` is the exception under
v2, not attention): `docs/experiment_inventory.md` §4.4/§4.5.
