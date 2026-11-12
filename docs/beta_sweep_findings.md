# CVAE β Sweep — Findings

> **Note (`review-refactor`).** Files referenced below that paper v2.3 does not use were removed from the repository; they remain in the `mil-reframe` history.

Results companion to docs/beta_sweep_methodology.md (methodology only, no
numbers). Covers the downstream classification results (TRTR/TSTR/TSRTR
bag-level accuracy, plus Classification-by-Reconstruction) across all five
tested β values: 2.0 (`exp_501`, the original production config), 1.0
(`exp_601`), 0.75/0.5/0.25 (`exp_609`/`608`/`607`). Reconstruction-loss,
KL-loss, and synth/real std-ratio results from the same sweep live as
generated plots in `results/suite600_beta_sweep/` and are referenced, not
reproduced, here.

> **Correction notice (2026-10-02).** Every number below was recomputed from
> scratch after two bugs were found and fixed in this exact dataset:
> 1. `exp_501`'s TRTR (the shared baseline every comparison here is measured
>    against) was trained on a stale 58-slide split (one biopsy was silently
>    missing from `manual_splits.json` until a since-fixed bug) with
>    pre-Modal CNN settings (100 epochs, patience 10) — refit clean on the
>    current 59-slide data with 200 epochs/patience 20/Modal-offloaded CNN.
>    Its CNN bag accuracy moved from 0.593 to **0.712** — the single biggest
>    source of the earlier (now superseded) "significant" deltas, nearly all
>    of which were significant only *because* that baseline was artificially
>    weak, not because of anything about the CVAE.
> 2. `exp_601`'s numbers were, for most of this investigation, silently
>    sourced from a disposable smoke-test config (`exp_601_pipeline_smoke`)
>    that was only ever meant to validate pipeline plumbing — its CNN also
>    predated the Modal fix. A second, narrower bug (`exp_601.yaml`'s
>    `models_baseline_dir` pointing at its own, nonexistent `trtr/` instead
>    of the shared one) then caused one re-run attempt to silently skip all
>    four classifier comparisons. Both are fixed; the numbers below are from
>    a real `exp_601` run under its own name, against the corrected baseline.
>
> The take-away from finding these two bugs is itself a finding: **the
> previously reported "clean, monotonic β trend" in the downstream task does
> not survive.** See §3.

> **Reproducing this.** Bag-level numbers below come from each config's
> `statistical_evaluation_TRTR_vs_{TSTR,TSRTR}_*.json` /
> `statistical_summary_*.csv` under `results/exp_{501,601,607,608,609}/statistical_eval/`,
> produced by `python -m pipeline.run_suite --configs exp_60N --stages tstr
> tsrtr stats_eval` (TRTR reused from `exp_501` via `models_baseline_dir` —
> see `beta_sweep_methodology.md` §6). All bootstrap comparisons: 1000
> resamples, class-stratified over bag decisions, n=59 bags.

## 1. Bag-level classification accuracy, by regime and β

**TRTR (real-only, shared reference — `exp_501`, reused identically across
every comparison below):**

| Apollo | RF | XGBoost | CNN |
|---|---|---|---|
| 0.763 | 0.814 | 0.814 | 0.712 |

**TSTR (train on synthetic only, test on real):**

| Model | β=2.0 | β=1.0 | β=0.75 | β=0.5 | β=0.25 |
|---|---|---|---|---|---|
| Apollo | 0.559 | 0.593 | 0.576 | 0.695 | 0.576 |
| RF | 0.729 | 0.729 | 0.763 | 0.831 | 0.797 |
| XGBoost | 0.763 | 0.746 | 0.763 | 0.831 | 0.814 |
| CNN | 0.542 | 0.661 | 0.712 | 0.678 | 0.746 |

**TSRTR (train on real+synthetic mix, `synth_ratio=1.0`, test on real):**

| Model | β=2.0 | β=1.0 | β=0.75 | β=0.5 | β=0.25 |
|---|---|---|---|---|---|
| Apollo | 0.831 | 0.831 | 0.814 | 0.831 | 0.831 |
| RF | 0.831 | 0.831 | 0.831 | 0.831 | 0.831 |
| XGBoost | 0.814 | 0.814 | 0.814 | 0.814 | 0.814 |
| CNN | 0.729 | 0.712 | 0.729 | 0.746 | 0.814 |

**Classification by Reconstruction (bag_acc, automatic side computation —
see beta_sweep_methodology.md §6; this is the `evaluation_stage.py`
majority-vote convention, not the official Tier 4 mean-posterior convention
documented in docs/stage6_generative_mil.md — the two differ, see that
doc's own numbers for the canonical ones):**

| β=2.0 | β=1.0 | β=0.75 | β=0.5 | β=0.25 |
|---|---|---|---|---|
| 0.678 | 0.729 | 0.780 | 0.780 | 0.746 |

Monotonic-ish and consistent with the reconstruction-side diagnostics:
β=2.0 (most collapsed) gives CbR's worst bag accuracy of the five; the
lower-β variants are all comfortably better, though not perfectly ordered
among themselves (0.5 and 0.75 tie at the top, 0.25 sits slightly below
them).

## 2. Statistical significance (bootstrap p-values, n=59 bags), full sweep

**TSRTR CNN delta vs TRTR:**

| β | Δ | p |
|---|---|---|
| 2.0 | +0.017 | 0.868 |
| 1.0 | +0.000 | 1.000 |
| 0.75 | +0.017 | 0.862 |
| 0.5 | +0.034 | 0.670 |
| 0.25 | **+0.102** | **0.028** |

Only β=0.25 is significant. **This is not the clean monotonic trend
previously reported** — that claim was an artifact of the stale TRTR CNN
baseline (§0). With the corrected baseline, four of five β values show a
small, non-significant CNN gain and only the lowest tested β clears p<0.05.

**TSRTR Apollo delta vs TRTR:**

| β | Δ | p |
|---|---|---|
| 2.0 | +0.068 | **0.022** |
| 1.0 | +0.068 | **0.022** |
| 0.75 | +0.051 | 0.078 |
| 0.5 | +0.068 | **0.022** |
| 0.25 | +0.068 | **0.022** |

Significant at four of five β values, essentially the **same magnitude
every time** — this effect is present even at β=2.0, the most collapsed
CVAE tested. It does not track β at all; augmentation modestly helps Apollo
regardless of how collapsed the generator is.

**TSTR Apollo delta vs TRTR (synthetic-only, always ≤0):**

| β | Δ | p |
|---|---|---|
| 2.0 | −0.203 | **0.012** |
| 1.0 | −0.170 | **0.004** |
| 0.75 | −0.186 | **0.004** |
| 0.5 | −0.068 | 0.240 |
| 0.25 | −0.186 | **0.000** |

Significantly worse at four of five β values — β=0.5 is the one exception,
not β=0.25 as might be expected if lower β were cleanly better.

**RF, XGBoost — flat across the entire sweep, both regimes.** XGBoost's
TSRTR bag decisions are *identical* to TRTR at every single β (Δ=0.000,
p=1.000 five times in a row) — the clearest null result in this table.
RF's TSRTR delta is a constant +0.017 (p=0.670) at every β. Neither model's
TSTR delta is ever significant either. Full per-β numbers in §1's tables.

## 3. Interpretation

- **The clean "lower β helps the downstream task, monotonically" story does
  not survive the corrected baseline.** It was real for the
  reconstruction-loss/KL/std-ratio diagnostics (`results/suite600_beta_sweep/`
  plots) but does not transfer cleanly to bag-level classification across
  the full five-point sweep. Only isolated points are significant, and they
  don't form a trend.
- **RF/XGBoost are flat** — saturated, insensitive to β or regime, at every
  tested value, not just the ones originally checked.
- **Apollo's TSRTR gain is real but β-independent** — present at virtually
  the same magnitude (Δ≈0.068, p≈0.022) whether the CVAE is badly collapsed
  (β=2.0) or not (β=0.25). Augmentation modestly helps Apollo; this isn't
  something the beta sweep bought.
- **Apollo's TSTR loss is also real and widespread** — synthetic-only
  training significantly hurts Apollo at 4 of 5 β values. Training on
  synthetic alone is not a viable substitute for real data for this model,
  regardless of β.
- **β=0.25 is the only β with a significant CNN TSRTR gain** (p=0.028) —
  worth taking seriously as a real, single data point, but it's one
  significant result out of a 5×2×4 = 40-cell comparison grid, not
  confirmation of a trend the other 39 cells mostly don't support.
- **CbR's bag accuracy (§1) is the cleanest β-tracking result in this whole
  document** — worst at β=2.0 (0.678), clearly better at every lower β
  tested (0.73–0.78). This lines up with the reconstruction-side diagnostics
  because CbR's mechanism (argmin reconstruction MSE) is directly downstream
  of decoder collapse in a way the trained classifiers (Apollo/RF/XGBoost/CNN)
  are only indirectly affected by.

## 4. Open items

- No consolidated multi-seed or multi-split repetition exists for any config
  here — every number is a single run / single CV split, the same standing
  caveat as the rest of this project (docs/beta_sweep_methodology.md §7,
  docs/mil_tiers_methodology.md §9). With a 40-cell comparison grid and
  α=0.05 uncorrected, 1-2 nominally "significant" cells are plausible by
  chance alone — none of the single-cell results above (β=0.25's CNN TSRTR
  gain in particular) should be read as confirmed without replication.
- Tier 3 (attention MIL) and Tier 4 (CbR/encoder_head) were also evaluated
  against `exp_607`'s CVAE; results live in docs/stage6_generative_mil.md and
  docs/stage6_attention_mil.md, not duplicated here. None beat Tier 0's
  RF/XGBoost baseline.
- `exp_608`/`609` have not had Tier 3/4 run against their CVAEs yet (only
  `exp_607` has) — would need the same recipe if the β=0.25 CbR/encoder_head
  pattern is worth confirming across the sweep.

## 5. Update, 2026-10-06 — 6-class exploratory extension (`exp_661`)

A single exploratory run at $K{=}6$ (the methylation-derived LGm1–LGm6
grades, Stage 8 of docs/MIL_REFRAME_PLAN.md), β=0.5 (the best-performing
binary β), own TRTR/CVAE/TSTR/TSRTR/stats_eval — not reusing exp_501's
binary baseline, since the task itself is different. **One run, one CV
split — explicitly exploratory, not a conclusion**, per the same
single-split caveat as §4.

**Severe, structural class imbalance.** The 6-class stratification over 59
bags gives LGm1–LGm6 only 5/12/9/13/15/5 bags respectively — LGm1 and LGm6
have 5 bags each, a hard statistical-power floor no amount of modelling
fixes (`configs/exp_661.yaml`'s own notes record this going in).

**Bag-level accuracy (n=59, chance ≈ 0.167 for 6 balanced classes, though
classes here are not balanced):**

| Model | TRTR | TSTR | TSRTR ($r{=}1.0$) |
|---|---|---|---|
| Apollo | 0.322 | 0.271 | 0.373 |
| RF | 0.288 | 0.203 | 0.271 |
| XGBoost | 0.288 | 0.305 | 0.305 |
| CNN | 0.203 | 0.203 | 0.288 |
| CbR | — | 0.305 | 0.305 |

No TSTR or TSRTR delta against TRTR is significant at any model
(`bag_significant_95=False` throughout, bootstrap CIs all straddling zero) —
unlike the binary sweep (§2), synthetic data neither clearly helps nor hurts
here; the dominant signal is the class-imbalance floor above, not β or the
synthetic regimen.

**Was training class-balanced?** Checked directly rather than assumed,
because an unweighted fit on this imbalance (5/12/9/13/15/5 bags, and
`patient_cap` — §3 of docs/mil_tiers_methodology.md — caps *per slide*, not
per class, so instance counts still scale with each class's bag count)
could produce exactly a "always predict the biggest class" artifact that
would masquerade as a genuine confusion finding. Two of the four
classifiers use explicit inverse-frequency `class_weight`, two do not:

| Model | class-weighted? | where |
|---|---|---|
| Apollo (RF+SVM) | **yes** — hardcoded, not config-driven | `src/models/baselines.py:29,43` (`balanced_subsample` / `balanced`) |
| RF (standalone) | **yes** | `configs/exp_661.yaml:88` (`class_weight: "balanced"`), read at `pipeline/_common.py:239` |
| XGBoost | **no** — no `class_weight`/`sample_weight` param at all | `pipeline/_common.py:252-258` |
| 1D-CNN | **no** — no `class_weight`/`sample_weight` passed to `.fit()`; only within-class mixup (doesn't rebalance across classes) | `pipeline/_common.py:382-388` |

No oversampling/SMOTE anywhere in this pipeline for any model.

**Confusion-matrix analysis, per model, not pooled** (pooling across
balanced and unbalanced classifiers would hide exactly the artifact this
check is for). Predicted-class share vs. true share (8%/20%/15%/22%/25%/8%
for LGm1–LGm6) and per-class recall:

| Model | weighted | LGm1 pred-share | LGm2 pred-share | LGm1 recall | LGm6 recall |
|---|---|---|---|---|---|
| Apollo | yes | 0.05 | 0.24 | 0.00 | 0.20 |
| RF | yes | 0.02 | 0.27 | 0.00 | 0.00 |
| XGBoost | no | 0.00 | 0.24 | 0.00 | 0.00 |
| CNN | no | 0.00 | **0.63** | 0.00 | 0.00 |

**The CNN alone shows a majority-class collapse** — 63% of all its
predictions land on LGm2 against a 20% true share, far beyond the other
three (24–27%, all close to LGm2's own base rate). This is consistent with
your hypothesis, but specifically for the CNN: **XGBoost is equally
unweighted and does not show it** (24%, same as the two weighted models),
so "unweighted training" alone does not predict the collapse — something
particular to the CNN (already documented elsewhere as this task's weakest
classifier by a wide margin) rather than a general property of skipping
`class_weight`.

**LGm1/LGm6 unrecoverability and the LGm2↔LGm3 / LGm4↔LGm5 confusions
survive explicit class-weighting**, which argues against reading them as
pure imbalance artifacts:
- LGm1 recall is 0.00 for **all four** models, weighted or not — balancing
  did not rescue it.
- LGm6 recall is 0.00 for three of four; only Apollo (weighted) recovers a
  single bag (1/5).
- The LGm2↔LGm3 confusion is, if anything, **strongest in RF** (weighted):
  67% of true LGm3 bags are called LGm2, more than in either unweighted
  model. Explicit inverse-frequency reweighting should have suppressed
  this if it were mainly a frequency artifact, and it did not.

Pooled across all four for reference (236 bag-decisions, mixing weighted
and unweighted models — read with the above caveat, the CNN's column
disproportionately inflates the LGm2 counts):

| true \ pred | LGm1 | LGm2 | LGm3 | LGm4 | LGm5 | LGm6 |
|---|---|---|---|---|---|---|
| **LGm1** (n=20) | 0 | 8 | 3 | 0 | 8 | 1 |
| **LGm2** (n=48) | 2 | **27** | 7 | 5 | 7 | 0 |
| **LGm3** (n=36) | 0 | 21 | **3** | 7 | 4 | 1 |
| **LGm4** (n=52) | 1 | 10 | 3 | **15** | 19 | 4 |
| **LGm5** (n=60) | 1 | 11 | 2 | 23 | **19** | 4 |
| **LGm6** (n=20) | 0 | 4 | 1 | 6 | 8 | **1** |

**Tentative reading, not acted on.** The LGm2↔LGm3 and LGm4↔LGm5
confusions and the LGm1/LGm6 floor look like genuine properties of this
task (spectral similarity between adjacent grades, and a hard 5-bag power
floor) rather than artifacts of skipped class-weighting — raising
coarser-grouping as a post-hoc question worth asking, *not* a
recommendation to re-run anything yet. This is a single-run, single-split
finding at a task with a 5-bag class-size floor; it has not been checked
against a different seed/split or a per-class-balanced CNN/XGBoost
variant (the one re-run that would most directly test the remaining
imbalance-artifact concern), and `mil_fit`/`mil_eval` (Tiers 1–5) were
deliberately not run for this config (`exp_661.yaml`'s own scope note) —
none of this project's MIL instance-selection findings have been tested at
$K{=}6$ at all.
