# CVAE β Sweep — Findings

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
  docs/mil_tiers_methodology.md §8). With a 40-cell comparison grid and
  α=0.05 uncorrected, 1-2 nominally "significant" cells are plausible by
  chance alone — none of the single-cell results above (β=0.25's CNN TSRTR
  gain in particular) should be read as confirmed without replication.
- Tier 3 (attention MIL) and Tier 4 (CbR/encoder_head/latent_rf — a new
  variant added this session) were also evaluated against `exp_607`'s CVAE;
  results live in docs/stage6_generative_mil.md and
  docs/stage6_attention_mil.md, not duplicated here. None beat Tier 0's
  RF/XGBoost baseline.
- `exp_608`/`609` have not had Tier 3/4 run against their CVAEs yet (only
  `exp_607` has) — would need the same recipe if the β=0.25 CbR/encoder_head
  pattern is worth confirming across the sweep.
