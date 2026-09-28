# Stage 2 — evaluation reframing and Tier 0

**Status: complete.** Branch `mil-reframe`. Core: `src/utils/statistical_evaluation.py`.
Wired through `pipeline/evaluation_stage.py`, new `pipeline/mil_eval_stage.py`,
`pipeline/run_suite.py`, `patch_cbr_metrics.py`, `paper/gen_appendix_tables.py`.

> **Reproducing this.** `pytest tests/test_stage2_evaluation.py` (13 tests, no data
> dependency beyond `data/splits/manual_splits.json`, already in the repo).
> `python -m pipeline.run_suite --configs exp_201 --stages stats_eval mil_eval --force`
> regenerates one experiment's real bag-level numbers (~70 min at the production
> `bootstrap_samples: 1000`). `python paper/gen_appendix_tables.py` refreshes
> `paper/appendix_extended_results.tex` from whatever JSONs currently exist — safe to
> run any time; stale (pre-Stage-2) JSONs degrade to `--` rather than crash or show a
> wrong number.

## 1. What changed and why

Every metric the pipeline reported was computed at the **instance level** and silently
presented as if it answered the **bag-level** clinical question — the exact confusion
the MIL reframing exists to retire (docs/MIL_REFRAME_PLAN.md §7). Concretely, in
`src/utils/statistical_evaluation.py` as it stood before this stage:

- `accuracy`, `balanced_accuracy` and `recall_macro` were **literally the same float**
  — mean within-bag instance accuracy, flattened over (class, slide) pairs. Not a bag
  decision at all.
- `precision_macro` was a **constant ≈0.5**: every bag is single-class, so a bag's own
  instance precision for its own class is always exactly 1.0; macro-averaging that
  against the *other* class's structurally-undefined precision landed near 0.5 for
  every single run, regardless of model quality.
- `kappa` was **always 0.0** — `cohen_kappa_score` computed against a `y_true` that is
  constant within every group it's evaluated over.
- `f1_macro` was **upward-biased and non-monotone in accuracy**: a slide with 0%
  within-bag accuracy was silently excluded from the average (a `rec is not None and
  prec is not None` guard), so a bad slide's badness never reached the reported number.
  This is exactly how `results/exp_401` could report `f1_macro=0.926` beside
  `accuracy=0.431` before this stage.

None of these are new observations — they're what the plan's own investigation (§7)
found by reading the code before Stage 2 began. This stage replaces them rather than
patching around them:

- **`bag_decisions(preds, bag_ids, y_true=None, rule='majority')`** — the single place
  that collapses per-instance predictions to one hard decision per bag (mode vote,
  ties broken toward the lowest label index). Previously this exact computation was
  inlined only to build a confusion matrix and thrown away otherwise; it's now
  factored out and reused everywhere a bag decision is needed.
- **`_slide_level_metrics`** rewritten to compute **two separate metric sets in one
  pass, explicitly not mixed**: genuine bag-level metrics (`bag_accuracy`,
  `bag_balanced_accuracy`, `bag_f1_macro`, `bag_precision_macro`, `bag_recall_macro`,
  `bag_kappa`, `bag_sensitivity`/`bag_specificity` for binary tasks,
  `bag_auroc`/`bag_auprc`) computed by standard sklearn calls on the `n_bags` majority
  decisions against `n_bags` true labels — non-degenerate by construction, since there
  is no per-bag filtering anywhere in that path. And within-bag instance diagnostics
  (`within_bag_instance_recall` — the renamed, demoted old headline statistic — plus
  `f1_weighted`), kept for continuity but explicitly documented as validating nothing
  (no instance-level ground truth exists in this dataset).
- **The duplicate-bag rule (§7.1) fell out of `bag_decisions` for free**: it keys
  purely by bag id, so a bag seen in two CV folds (4 of them, forced by 6-class
  stratification over classes with only 4 patients — verified directly against
  `data/splits/manual_splits.json`: 63 (bag, fold) test pairs, 59 unique bags as of
  2026-09-28 (`HF-1887_via-t_2_1.h5_3` added, single test fold, not a duplicate;
  was 62/58 before — docs/stage1_data_layer.md §4),
  `HF-2104_#5_1.h5_0`/`HF-2104_#9_1.h5_1`/`HF-2104_V1T_1.h5_2`/`HF-592_V3T_1.h5_4` each
  appearing in exactly two folds) contributes one entry, not two. No special-casing
  needed for majority vote specifically.
- **Tier 0 naive-MIL baselines** (`pool_instance_scores`, `aggregate_bag_scores_across_
  folds`) — mean-posterior, majority-vote-as-fraction, log-sum-exp, and top-q(as a
  *fraction* of bag size) pooling, each producing a valid per-bag probability vector.
  `aggregate_bag_scores_across_folds` does genuine per-fold aggregation before
  averaging a duplicated bag's fold-level scores, rather than pooling instances first
  — required for top-q/LSE (non-linear in the instance set) and validated rather than
  assumed equivalent to pooling-then-computing-once for the linear rules (§4).
- **`bag_level_bootstrap`/`bag_level_bootstrap_cbr`** reuse the existing
  `stratified_slide_bootstrap`/`_cbr` machinery with each bag as its own singleton
  "slide" group (`bag_ids = np.arange(n_bags)`) — an exact degenerate case of the
  existing cluster bootstrap, not new resampling logic.
- **New `pipeline/mil_eval_stage.py`** — the Tier 0 stage proper. For each of the 4
  models and each of the 3 regimens (TRTR/TSTR/TSRTR), computes all four pooling
  rules' bag-level metrics plus a bootstrap comparison against majority vote (the
  reference rule). Writes `mil_tier0_{regimen}_{timestamp}.json/csv`, a distinct
  filename prefix from `stats_eval`'s `statistical_evaluation_*.json` so
  `run_suite.py`'s skip-if-done check for one stage never fires off the other's output
  (the plan explicitly flagged this collision risk in §9's pipeline-wiring note).
- **`pipeline/evaluation_stage.py`** rewritten to compute and print both levels, bag
  first and labeled as such, within-bag instance recall clearly labeled
  `[diagnostic]`. **`patch_cbr_metrics.py`** updated for the new key names.
  **`paper/gen_appendix_tables.py`** rewritten (§4).

## 2. Real-data validation (exp_201, n_boot=1000, full production rigor)

Not a synthetic check — the rewritten pipeline run end-to-end against exp_201's
actual trained models (the source of the paper's main-text Table 2) and CbR's actual
CVAE weights, both statistical comparisons (TRTR vs TSTR, TRTR vs TSRTR) plus all four
Tier-0 pooling rules for all three regimens.

**Before (pre-Stage-2 JSON, `results/exp_201/statistical_eval/statistical_summary_TRTR_vs_TSTR_20260625_114210.csv`,
the exact numbers the submitted paper's Table 2 was built from) vs after:**

| model | metric | before (instance-level, mislabeled) | after: `bag_accuracy` | after: `within_bag_instance_recall` |
|---|---|---|---|---|
| Apollo (RF+SVM) | TRTR | 0.6880 | **0.7414** | 0.6880 |
| Apollo | TSTR | 0.4310 | **0.4310** | 0.4310 |
| Apollo | `precision_macro`/`kappa` | 0.500 / 0.000 (constant, both regimens) | `bag_precision_macro`: 0.7434 (TRTR) — genuinely varies | `bag_kappa`: see below |
| Apollo | `f1_macro` (old) vs `bag_f1_macro` (new) | TSTR: **0.9260** beside accuracy 0.431 | TSTR: **0.3012** beside bag_accuracy 0.431 | — |

The last row is the concrete instance of the bug the plan named directly (§7): the old
`f1_macro` was *higher* for the *worse* regimen because it silently dropped 0%-accuracy
slides. The new `bag_f1_macro` moves the same direction as `bag_accuracy` instead —
0.7277 (TRTR) → 0.3012 (TSTR), tracking the real 0.7414 → 0.4310 accuracy collapse
rather than masking it. `within_bag_instance_recall` (0.688) is now visibly a
*different number* from `bag_accuracy` (0.7414) for every model in both comparisons —
the two were mathematically forced to be identical before this stage; that they now
diverge is the core fix, not a side effect.

**Full bag-level headline, all 4 models + CbR, exp_201, n=58 bags, n_boot=1000:**

| model | TRTR | TSTR | Δ (TSTR) | 95% CI | TRTR | TSRTR | Δ (TSRTR) | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Apollo | 0.7414 | 0.4310 | −0.3103 | [−0.4138, −0.2069]* | 0.7414 | 0.7586 | +0.0172 | [−0.0517, +0.0862] |
| RF_Standard | 0.8103 | 0.7069 | −0.1034 | [−0.2069, −0.0172]* | 0.8103 | 0.8103 | +0.0000 | [+0.0000, +0.0000] |
| XGBoost | 0.7931 | 0.6897 | −0.1034 | [−0.2241, +0.0172] | 0.7931 | 0.8103 | +0.0172 | [+0.0000, +0.0517] |
| 1D_CNN | 0.6207 | 0.3966 | −0.2241 | [−0.3448, −0.1034]* | 0.6207 | 0.6034 | −0.0172 | [−0.1207, +0.0690] |
| CbR | — | bag_acc=0.7414, CI[0.6207, 0.8448] | | | | (same CbR row both comparisons — model-independent) | | |

(*) excludes zero at 95%. Qualitatively unchanged from the pre-Stage-2 story (TSTR
degrades sharply, TSRTR roughly matches or slightly improves on TRTR) — exactly the
plan's Stage 2 exit criterion ("bag-level numbers reproduce the existing qualitative
story at the correct level of analysis, with wider and honest CIs"). The CIs *are*
wider: n=58 bag-level resampling units instead of instance counts in the hundreds of
thousands per slide.

**Tier 0 pooling-rule sensitivity, exp_201 TRTR** — the naive-MIL reference table the
plan asked for. Apollo is the only model where the four rules disagree at all:

| model | mean | majority (ref) | lse | topq |
|---|---|---|---|---|
| Apollo | 0.7586 | 0.7414 | 0.7586 | 0.7759 |
| RF_Standard | 0.8103 | 0.8103 | 0.8103 | 0.8103 |
| XGBoost | 0.7931 | 0.7931 | 0.7931 | 0.7931 |
| 1D_CNN | 0.6207 | 0.6207 | 0.6207 | 0.6207 |

## 3. A real discrepancy, tracked down rather than waved off

The two independent code paths that both compute "majority vote" —
`bag_decisions` (drives `evaluation_stage.py`'s primary bag-level metrics, via each
model's `.predict()`) and `pool_instance_scores(rule='majority')` (drives Tier 0's
"majority" reference rule, via `argmax` of `.predict_proba()`) — disagreed on 2 of 58
bags for Apollo specifically (`bag_f1_macro` 0.7277 vs 0.7228, a ~0.005 difference),
even though they should be two views of the same computation.

Traced to ground truth, not left as a hand-wave: **the two disagreeing bags are not
among the 4 cross-fold-duplicated bags** (confirmed: those 4 have exactly equal
instance counts in both folds they appear in, `[(fold, n)]` pairs like
`[(0, 2315), (4, 2315)]`, so the duplicate-averaging equivalence proof in §4 holds for
them specifically). The actual cause is `src/models/baselines.py:39-45`: Apollo's SVM
is `SVC(..., probability=True)`, and scikit-learn's own documentation warns that Platt
scaling's probability estimates are a separate fit from the decision function and "the
argmax of the scores may not be the argmax of the probabilities." `bag_decisions` uses
`.predict()` (the decision function); Tier 0's majority rule uses `argmax(.predict_proba())`
— so for the small fraction of instances where Apollo's own `.predict()` and
`.predict_proba()` disagree, the two paths can reach different per-instance labels,
occasionally flipping a close within-bag vote. Measured directly on exp_201's real
validation data (238,814 instances, TRTR models): Apollo disagrees on 6,632 instances
(**2.777%**); RF_Standard, XGBoost and 1D_CNN disagree on **exactly 0** — confirming
this is specific to Apollo's internal SVC, not a property of the pooling code. This is
exactly consistent with what the pooling-rule table above shows: Apollo is the *only*
model where the four Tier-0 rules diverge from each other at all — RF_Standard/
XGBoost/1D_CNN don't have this predict/predict_proba split, so all four rules collapse
to the same decisions for them, every time.

Not a bug in this stage's code — a pre-existing property of `SVC(probability=True)`
that this stage's separation of "decision-based" and "probability-based" majority
vote made newly visible. Documented rather than silently reconciled: `bag_decisions`
is the primary metric (uses the model's actual `.predict()`, what a deployed model
would output); Tier 0's majority rule is a diagnostic comparison across pooling
strategies, not required to match it bit-for-bit for models with this property.

## 4. Verification

- **New `tests/test_stage2_evaluation.py`, 13/13 passing.** Covers exactly the plan's
  three Stage 2 test bullets: hand-built single-class bags showing
  `bag_precision_macro`/`bag_kappa`/`bag_f1_macro` are no longer degenerate (a 4-bag
  fixture hand-verified against sklearn directly: `bag_accuracy=0.75`,
  `bag_precision_macro=0.8333`, `bag_kappa=0.5`, `bag_f1_macro=0.7333`, all far from
  the old constants, plus an explicit check that `bag_f1_macro` is *lower* than the
  "drop the 0%-accuracy bag" alternative computation the old code implicitly
  performed); the 62-pairs-to-58-bags duplicate rule, both via plain pooling and via
  the genuine per-fold `aggregate_bag_scores_across_folds` path; and Tier-0
  majority-pooling reproducing `bag_decisions`'s confusion matrix exactly on
  no-tie synthetic data. Plus the linear-rule cross-fold equivalence
  `aggregate_bag_scores_across_folds`'s own docstring claims (verified bit-exact,
  not assumed).
- **Full suite**: `pytest tests/` — 65 passed, 2 pre-existing failures in
  `test_config_manager.py` (confirmed unrelated: that file tests
  `src/utils/config_manager.py`, untouched by this stage, and the failures are a
  required-fields-vs-type-check ordering issue present before Stage 0).
- **Real end-to-end, twice**: `evaluation_stage()` and `mil_eval_stage()` both run to
  completion against exp_401's actual trained models and exp_201's actual trained
  models + CVAE weights (§2), at both a fast smoke-test `n_boot` and the full
  production `n_boot=1000`. No exceptions, sane numbers, `n_bags=58` throughout.
- **`mil_eval_stage.py`'s own standalone CLI entry point was broken** (`sys.path`
  insertion placed after the top-level `pipeline.evaluation_stage` import it depends
  on, so `python pipeline/mil_eval_stage.py --config ...` — its own documented usage —
  could never work). Caught by actually running it that way rather than only via
  `run_suite.py`'s dynamic import (which happens to already have the repo root on
  `sys.path` and so never exercised the bug). Fixed to match the working pattern in
  `patch_cbr_metrics.py`.
- **`paper/gen_appendix_tables.py` was writing to a file the build never read**
  (`paper/appendix (5).tex`; the real build does `\input{appendix}` →
  `paper/appendix.tex`) — an independent bug found while wiring Stage 2's numbers into
  the paper, unrelated to the metric fixes themselves. Fixed by splitting
  `appendix.tex`'s auto-generated section into `paper/appendix_extended_results.tex`
  (`\input{}`-ed from the hand-written shell) and repointing the generator at it;
  verified via a full `latexmk` rebuild (exit 0, zero undefined references).

## 5. Deviations from the plan

- **No changes to `precision`/`recall`/`f1` for the *within-bag* diagnostic beyond the
  rename.** The plan's §7 bug list is about the bag-level metrics; the within-bag
  numbers (`within_bag_instance_recall`, `f1_weighted`) are carried over
  computationally unchanged from the old `stratified_slide_accuracy`, just relabeled
  and re-documented as non-validating diagnostics rather than fixed.
- **The predict-vs-predict_proba discrepancy (§3) was not in the plan** — an
  unplanned but directly relevant finding surfaced by having two independent
  majority-vote code paths for the first time. Documented rather than "fixed" because
  there is no single correct answer: both `.predict()` and `argmax(.predict_proba())`
  are legitimate decision rules, and forcing them to agree would mean discarding one
  of Apollo's two outputs arbitrarily.
- **Appendix regeneration is a large compute job, run in two batches rather than
  deferred.** Unlike Stage 1's full data export (deferred, documented, left for later),
  here the real per-call cost was measured directly (a `stratified_slide_bootstrap`
  call at production `n_boot=1000` over the full ~240k-instance validation set costs
  ~300s; a bag-level bootstrap over 58 bags costs ~40s) before committing to a plan:
  exp_201 (~70 min, needed for the paper's main-text Table 2) was run first and is
  reflected in §2 above; the remaining 10 appendix experiments (~15h estimated, ~11h
  actual) were started as a second background batch after checking with the user on
  the time commitment. **Completed**: all 20 stage-runs (10 experiments × stats_eval +
  mil_eval) report `ok`, zero failures despite `--continue-on-error` being enabled.
  `paper/appendix_extended_results.tex` now carries real bag-level numbers for all 11
  experiments; rebuilt via `latexmk`, exit 0, no undefined references.

## 7. Two claims from the pre-Stage-2 paper, checked against the full bag-level rerun

Both `paper/main.tex` claims that cite appendix tables (deliberately left unedited
pending this rerun — see §5) were checked against the real numbers rather than assumed:

- **6-class TR/TR accuracy "approximately 30%"** (exp_204/205/206): bag-level TR/TR
  accuracy is 0.259–0.293 across the four models (RF: 0.2931, Apollo: 0.2759, XGBoost:
  0.2586, 1D-CNN: 0.2241–0.2414 depending on loss variant) — **the claim still holds**,
  unchanged in substance at the bag level.
- **"1D-CNN showed a tendency of improvement with larger training datasets"**
  (exp_401/402/403, the TSR/TR ratio ablation): bag-level 1D-CNN TSR/TR delta vs. TRTR
  is **+0.052 (1:0.5), +0.052 (1:1), +0.155\* (1:2)** — monotonically increasing with
  more synthetic data, and the highest ratio is now statistically significant at the
  bag level (n=58; not true of any single-ratio point estimate under the old
  instance-level pooling). This is a case where the bag-level correction *strengthens*
  a pre-Stage-2 qualitative claim rather than weakening it, unlike Table 2's 1D-CNN
  TSR/TR sign flip at the default ratio (exp_201: −0.017, not significant) — the two
  are consistent (exp_201 uses a different default ratio than the 401/402/403 sweep,
  and the effect is ratio-dependent, not contradictory).

Not folded into `paper/main.tex` proper: the user is rewriting the paper's framing and
narrative directly (see project memory), so these are reported here as inputs to that
rewrite rather than applied as further automatic edits on top of the Table 2 change
already made mid-Stage-2.

## 6. What Stage 3 inherits

- `bag_decisions`, `pool_instance_scores`, `aggregate_bag_scores_across_folds`, and
  `bag_level_bootstrap` are the stable primitives every later MIL tier (mi-SVM, EM,
  attention, generative) is measured against — Tier 0's numbers in §2 are the
  reference row.
- The `SCALAR_METRIC_KEYS` / `build_summary_rows` schema (`bag_*` primary,
  `within_bag_instance_recall`/`wbir_*`/`f1_weighted` diagnostic,
  `cbr_bag_*`/`cbr_*` for CbR) is now the canonical JSON/CSV shape later stages should
  extend, not replace.
- The predict-vs-predict_proba property (§3) is worth checking for any future
  classifier added to `MODEL_SPECS` that uses Platt-scaled SVM probabilities or
  similar internally-calibrated-separately-from-decision estimators.
