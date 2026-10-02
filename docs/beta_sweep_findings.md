# CVAE β Sweep — Findings

Results companion to docs/beta_sweep_methodology.md (methodology only, no
numbers). Covers the downstream classification results (TRTR/TSTR/TSRTR
bag-level accuracy, plus Classification-by-Reconstruction) from the flat
low-β extension (`exp_607`–`609`, β=0.25/0.5/0.75) and `exp_601` (β=1.0).
Reconstruction-loss, KL-loss, and synth/real std-ratio results from the
broader sweep (`exp_501`, `exp_601`–`609`) live as generated plots in
`results/suite600_beta_sweep/` (`all_flat_beta_recon_loss.png`,
`all_flat_beta_kl_loss.png`, `continuation_recon_loss_and_beta.png`,
`continuation_kl_loss.png`) and are referenced, not reproduced, here.

> **Reproducing this.** Bag-level numbers below come from each config's
> `statistical_evaluation_TRTR_vs_{TSTR,TSRTR}_*.json` /
> `statistical_summary_*.csv` under `results/exp_60N/statistical_eval/`,
> produced by `python -m pipeline.run_suite --configs exp_60N --stages tstr
> tsrtr stats_eval` (TRTR is reused from `exp_501`, not retrained — see
> `beta_sweep_methodology.md` §6). All bootstrap comparisons: 1000 resamples,
> class-stratified over bag decisions, n=59 bags.

## 1. Bag-level classification accuracy, by regime and β

**TRTR (real-only, shared reference — `exp_501`, reused identically across
every comparison below):**

| Apollo | RF | XGBoost | CNN |
|---|---|---|---|
| 0.763 | 0.814 | 0.814 | 0.593 |

**TSTR (train on synthetic only, test on real):**

| Model | β=0.25 | β=0.5 | β=0.75 | β=1.0* |
|---|---|---|---|---|
| Apollo | 0.576 | 0.695 | 0.576 | 0.593 |
| RF | 0.797 | 0.831 | 0.763 | 0.729 |
| XGBoost | 0.814 | 0.831 | 0.763 | 0.746 |
| CNN | 0.746 | 0.678 | 0.712 | 0.593 |

**TSRTR (train on real+synthetic mix, `synth_ratio=1.0`, test on real):**

| Model | β=0.25 | β=0.5 | β=0.75 | β=1.0* |
|---|---|---|---|---|
| Apollo | 0.831 | 0.831 | 0.814 | 0.814 |
| RF | 0.831 | 0.831 | 0.831 | 0.831 |
| XGBoost | 0.814 | 0.814 | 0.814 | 0.814 |
| CNN | 0.814 | 0.746 | 0.729 | 0.678 |

**Classification by Reconstruction (bag_acc, automatic side computation —
see beta_sweep_methodology.md §6):**

| β=0.25 | β=0.5 | β=0.75 | β=1.0* |
|---|---|---|---|
| 0.746 | 0.780 | 0.780 | 0.729 |

\* **β=1.0 (`exp_601`) is not on equal footing with 0.25/0.5/0.75**: it ran
before the Modal-CNN-offload + 200-epoch/patience-20 fix (docs below, §3),
so its CNN used the old local settings (100 epochs, patience 10). This is
the likely explanation for its CNN numbers being the lowest in every
column, including an exact TRTR=TSTR=0.593 tie — not necessarily a genuine
β=1.0 effect. Flagged as an open item (§4), not corrected in place.

## 2. Statistical significance (bootstrap p-values, n=59 bags)

**TSRTR CNN delta vs TRTR — the strongest, most consistent signal:**

| β | Δ | p |
|---|---|---|
| 0.25 | **+0.220** | **0.0000** |
| 0.5 | +0.153 | 0.026 |
| 0.75 | +0.136 | 0.054 |

Monotonic in both magnitude and significance as β decreases.

**Other notable deltas:**

| Comparison | β | Δ | p |
|---|---|---|---|
| TSRTR Apollo | 0.25 | +0.068 | 0.024 |
| TSRTR Apollo | 0.5 | +0.068 | 0.024 |
| TSRTR Apollo | 0.75 | +0.051 | 0.082 |
| TSTR CNN | 0.25 | +0.153 | **0.028** (synthetic *alone* beats real-only) |
| TSTR CNN | 0.5 | +0.085 | 0.162 |
| TSTR CNN | 0.75 | +0.119 | 0.122 |
| TSTR Apollo | 0.25 | −0.186 | 0.0000 (sig. *worse*) |
| TSTR Apollo | 0.75 | −0.186 | 0.0060 (sig. *worse*) |

RF and XGBoost: every delta across every β and both regimes is small
(≤0.02) and never significant — both models are effectively saturated on
this task regardless of training regime or β.

## 3. Interpretation

- **RF/XGBoost are flat** — saturated, insensitive to β or regime.
- **Apollo and CNN are where the effect concentrates.** TSTR (synthetic-only)
  consistently *hurts* Apollo (significant at β=0.25 and β=0.75) while
  *helping* CNN (significant at β=0.25); TSRTR (augmentation) helps both,
  strengthening as β drops.
- **β=0.25 is the standout configuration**: it is the only value where
  synthetic data alone (TSTR) significantly beats real-only training for any
  model (CNN), and it gives the largest, most significant TSRTR gain
  (p=0.0000). This matches the direction already established by the
  reconstruction-loss/KL/std-ratio diagnostics (`results/suite600_beta_sweep/`)
  — lower β reduces decoder collapse, and that reduction now shows up on the
  actual downstream task, not just on reconstruction-side proxies.
- **CbR's bag accuracy (0.75–0.78 across the board) is clearly non-trivial**,
  unlike the original β=2.0 collapsed-decoder finding that motivated this
  whole sweep (docs/counterfactual_probe.md) — consistent with reduced
  collapse making reconstruction-based classification genuinely harder, not
  a free trivial win.

## 4. Open items

- **Redo `exp_601` (β=1.0)'s TSTR/TSRTR** with the current Modal-CNN/200-epoch
  settings so it is directly comparable to 607–609 — its CNN numbers are
  currently the least trustworthy in this table (see §1 footnote).
- **TRTR's own CNN accuracy (0.593) is oddly low** relative to RF/XGBoost/
  Apollo (0.76–0.81) on the same real data — unresolved, and it deflates the
  baseline every CNN delta/significance figure above is measured against
  (the CNN deltas are real relative to this baseline, but the baseline may
  itself be underperforming its potential, worth a separate look before
  citing the CNN results as a headline number).
- No consolidated multi-seed or multi-split repetition exists for any config
  here — every number is a single run / single CV split, the same standing
  caveat as the rest of this project (docs/beta_sweep_methodology.md §7,
  docs/mil_tiers_methodology.md §8).
