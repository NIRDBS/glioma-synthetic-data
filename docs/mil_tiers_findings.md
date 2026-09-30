# MIL Tiers — Findings

Consolidated, structured inventory of what this project's experiments actually establish —
for drafting the paper's Results/Discussion from. This is analysis, not paper prose: numbers,
tables, exact statistical support, and honest characterization of what is and isn't proven.
Companion to docs/mil_tiers_methodology.md (methods only, no results). Full per-tier detail
and derivations: docs/stage4_mil.md (Tier 1/2), docs/stage5_spatial_validation.md (spatial,
Tier 1/2 + cross-tier synthesis), docs/stage6_attention_mil.md (Tier 3),
docs/stage6_generative_mil.md (Tier 4).

All numbers below are read directly from persisted result files
(`results/exp_501/mil_eval/*.json`, `results/exp_501/spatial_validation/*.json`), not
transcribed from console logs or recalled from memory.

---

## 0. What was run, on what

Four MIL tiers (naive pooling excepted, which is the Tier 0 reference itself, not a tier under
test) — mi-SVM (Tier 1), EM (Tier 2), gated-attention MIL (Tier 3), generative/CbR-based MIL
(Tier 4, two variants) — each fit on exp_501 (binary IDH-mutant vs wildtype), evaluated at bag
level against all four Tier 0 classifiers and, where a real posterior existed, against a
lag-banded spatial-coherence test with a shuffled-bag-label control.

**A real, load-bearing asterisk on comparability**: mi-SVM/EM's fits predate the discovery that
slide `HF-1887_via-t_2_1.h5_3` was being silently excluded from every prior analysis. Their
bag-level numbers below are computed on **n=58 bags**; Tier 3/Tier 4's are computed on the
corrected **n=59 bags**. The two n's use slightly different Tier 0 reference numbers as a
result (shown in both tables below) — a real discrepancy, not an inconsistency to paper over.
mi-SVM was not re-fit after the fix because its spatial claim was independently dropped
(§2) before that would have mattered enough to justify the re-run.

---

## 1. Bag-level classification — the clinical task

Class-stratified bootstrap, 1000 resamples, seed 42, majority-vote pooling (TRTR) for every
Tier 0 reference.

### Tier 0 references (the baseline every MIL tier is compared against)

| model | bag acc, n=58 (mi-SVM's comparison) | bag acc, n=59 (Tier 3/4's comparison) |
|---|---|---|
| 1D-CNN | 0.6034 | 0.5932 |
| Apollo (RF+SVM) | 0.7586 | 0.7627 |
| Random Forest | 0.8103 | 0.8136 |
| XGBoost | 0.8103 | 0.8136 |

The two tree ensembles are the strongest classifiers at either n, by a wide margin over the
CNN specifically (~20 points).

### Every MIL tier vs every Tier 0 reference

| tier | n | bag acc (F1, κ) | Δ vs CNN | Δ vs Apollo | Δ vs RF | Δ vs XGBoost |
|---|---|---|---|---|---|---|
| mi-SVM (Tier 1) | 58 | 0.6897 (0.689, 0.385) | +0.086, CI[−0.017,+0.190], p=0.126 | −0.069, CI[−0.190,+0.035], p=0.238 | **−0.121, CI[−0.225,−0.017], p=0.046** | **−0.121, CI[−0.225,−0.017], p=0.046** |
| EM (Tier 2) | — | *excluded — see §1.1* | — | — | — | — |
| attention (Tier 3) | 59 | 0.6441 (0.623, 0.257) | +0.051, CI[−0.102,+0.203], p=0.604 | **−0.119, CI[−0.220,−0.034], p=0.012** | **−0.170, CI[−0.288,−0.068], p=0.002** | **−0.170, CI[−0.288,−0.068], p=0.002** |
| CbR `reconstruction` (Tier 4) | 59 | 0.7119 (0.711, 0.423) | +0.119, CI[−0.051,+0.254], p=0.150 | −0.051, CI[−0.169,+0.051], p=0.474 | −0.102, CI[−0.203,0.000], p=0.104 | −0.102, CI[−0.203,0.000], p=0.104 |
| CbR `encoder_head` (Tier 4) | 59 | 0.7966 (0.786, 0.577) | **+0.203, CI[+0.085,+0.339], p=0.006** | +0.034, CI[−0.034,+0.102], p=0.410 | −0.017, CI[−0.051,0.000], p=0.694 | −0.017, CI[−0.051,0.000], p=0.694 |

**Bold = statistically significant at the 95% level.** Everything else is a non-significant
tie in either direction, not a demonstrated win or loss.

### 1.1 A metric that is provably not a result: EM's bag "accuracy"

A naive bag-classification accuracy computed the same way as the table above gives EM a
perfect **1.0000** — and this is provably meaningless, not a finding. Model (C)'s hard support
restriction confines every instance's state to `{null_1, null_2, null_3, Y_b}` — its own bag's
already-known label, never another bag's — so the metric recovers the label EM was *given*,
not one it *inferred*. Verified directly: 0% of instances, even held out, ever show
competition between diagnostic columns. Excluded from every comparison, with a permanent code
guard (`diagnostic_column_competition_frac`) so this can't silently reappear for any future
Model-(C)-constrained formulation.

### 1.2 What this table proves

- **RF and XGBoost were never significantly beaten by any MIL tier, in either direction, at any
  point in this project.** The strongest MIL result (`encoder_head`) is a statistical tie with
  both, not a win.
- **The CNN is a weak reference specifically** — every positive-looking MIL delta in this table
  is against the CNN, and only one clears significance.
- **The one significant win in the entire project**: `encoder_head` vs. CNN (p=0.006). Read
  precisely: "ties the strongest conventional classifiers, significantly beats the weakest
  one" — not "beats Tier 0" unqualified.
- **Two tiers show a significant *loss*, not merely a non-significant tie**: mi-SVM loses to
  RF/XGBoost; attention loses to Apollo *and* RF/XGBoost. Attention is the only tier whose
  comparison table has no non-significant-or-better cell against the three strongest
  references at all.

---

## 2. Spatial validation — every claim tested, every claim failed

Lag-banded Moran's I + empirical variogram (Chebyshev lag 1–10; lags 1–2 contaminated by the
3×3 median filter and excluded from any claim; lag≥3 is the headline), 200 within-slide
permutations, pooled across all 59 slides for every tier below. The shuffled-bag-label control
is an exact permutation of the true label multiset (same class balance), holding every
instance's features/bag membership/CV split fixed, refit end-to-end under the permuted labels.

| tier | real lag-3 (I, z) | shuffled lag-3 (I, z) | variogram range (real / shuffled) | verdict |
|---|---|---|---|---|
| mi-SVM (Tier 1) | +0.006, z=+1.5 | +0.018, z=+5.01 | 2px / 2px | **failed — dropped** |
| EM (Tier 2) | variogram ≡ 0 (collapsed) | not run (moot) | n/a | never produced a claim to test |
| attention (Tier 3) | +0.0236, z=+5.63 | +0.0613, z=+14.70 | 2.0px / 2.0px | **failed — dropped** |
| CbR `reconstruction` (Tier 4) | +0.122, z=+29.4 | +0.1253, z=+32.31 | 3.0px / 3.0px | **failed — dropped** |
| CbR `encoder_head` (Tier 4) | +0.049, z=+11.1 | +0.0485, z=+11.04 | 3.0px / 2.0px | **failed — numerically identical, dropped** |

**This is the single most rigorous result in the project**, precisely because it is a
pre-registered, adversarial test rather than a post-hoc interpretation of a p-value: labels
carrying zero true information produce comparable-or-larger apparent lag≥3 coherence than the
true labels, for every tier that could be tested. `encoder_head`'s case removes any residual
ambiguity — real and shuffled agree to two decimal places at every lag.

**What this proves, precisely**: not "there is no spatial structure" (there plainly is, at
every lag, in every tier) — but that **none of the detected structure is attributable to the
true diagnostic label**. The much better-supported explanation, identical across all four
failures: real tissue has real spatial autocorrelation in its raw spectral characteristics —
present in whatever representation a formulation builds on top of the spectrum (RF-selected
bands for mi-SVM, a frozen CVAE embedding for attention, the CVAE's own encoder/decoder for
both cbr variants) — entirely independently of which label a region happens to be fit toward.
A model flexible enough to fit *any* label reasonably well (§1's `encoder_head` reaching
88–93% training "accuracy" under complete label noise over 100 epochs is direct, independent
evidence this flexibility exists) will produce a lag≥3-coherent map regardless of the label,
because the coherence was never coming from the label.

**The validation methodology itself is vindicated, not undermined, by this result**: the
two-statistic design (scale-free Moran's I + scale-sensitive variogram) is what caught EM's
collapse being misread as "significant" purely from floating-point noise around a constant; the
shuffled-label control is what caught all four remaining false positives. Nothing about the
pipeline failed — it is functioning exactly as designed, and what it found is negative.

### 2.1 External agreement vs. `viable_legacy` (secondary, weaker check)

Hard-threshold ARI/AMI against the old unsupervised cluster selection — explicitly **not**
ground truth (no tumor annotation of any kind exists in this dataset; see §4).

| tier | ARI | AMI | "diagnostic" fraction |
|---|---|---|---|
| mi-SVM (real) | −0.0072 | 0.0015 | 0.96% (2,784/288,678) |
| attention (real, corrected threshold) | 0.0164 | 0.0008 | 84.4% |
| CbR `reconstruction` (real, corrected threshold) | −0.0148 | 0.0009 | 9.1% |
| CbR `encoder_head` (real, corrected threshold) | −0.0458 | 0.0089 | 85.5% |

Uniformly weak-to-null; two of four ARIs are slightly *negative* (worse than chance agreement).
Not a positive finding under any threshold tested. (A structural bug made this check read as
perfectly degenerate — ARI=AMI=0.0000, 100% "diagnostic" — for any joint-softmax source
[attention, both cbr variants] before a 2026-09-29 fix; mi-SVM's independent-score construction
was never affected and is shown here unchanged.)

---

## 3. π_b external validity (Tier 2 / EM)

- π_b (inferred per-bag diagnostic fraction) stays pinned near the numerical identifiability
  floor across **two independent full refits**: range **[0.0200, 0.0443]** over 59 bags,
  std=0.0031.
- Spearman correlation against 3 independent NMF/GMM tissue-cluster proportions and against
  `viable_legacy` fraction: **ρ ∈ [−0.16, +0.14], every p > 0.19** (n=59). Null — agreeing with
  §2.1's ARI/AMI-based null result via a completely different statistic.
- σ (the population spread of logit π_b) hits the 0.000 boundary in most folds on both refits,
  and *which specific folds* collapse changes between the two nominally-identical reruns (same
  seed, same code) — informal evidence of poor identifiability at this bag count, consistent
  with the plan's own pre-registered expectation, though short of a formal profile-likelihood
  criterion (not built).

**What this proves**: EM finds essentially no genuine, externally-verifiable per-bag signal at
58–59 bags — π_b is close to a single global constant, not a informative per-slide estimate.

---

## 4. Foundational fact this entire project rests on

**No instance-level (pixel-level) tumor annotation exists anywhere in this dataset, at any
point in this project's history.** `viable_legacy` and `tissue_cluster` are both the output of
an unsupervised pipeline (NMF(100) → GMM(3) → a human picking whichever cluster's mean spectrum
looked like "the strongest signal") — a signal-quality judgment, not a histopathological one. A
still-earlier PCA/DBSCAN-based attempt was superseded before this pipeline was adopted, for the
same reason. Every classifier in this project prior to the MIL reframing was trained under the
false assumption that a slide's grade label applies to every one of its spectra. This is not a
new finding from the experiments above — it is the premise the whole reframing was built to
correct — but it is a directly verifiable, citable fact (from the data-generation code itself),
not an assumption.

---

## 5. Real methodological bugs found and fixed (supports a rigor/methods narrative)

Each independently verifiable from the code/tests, not merely asserted:

- Three metric bugs in the pre-MIL evaluation code: `precision_macro` was a constant ≈0.5,
  `kappa` was always 0.0, `f1_macro` silently dropped 0%-accuracy bags (upward-biased).
- The CNN's early-stopping monitor was reading the true test fold (a real leak, not just
  optimism) — fixed with a group-aware inner-validation split.
- A slide (`HF-1887_via-t_2_1.h5_3`) was silently excluded from every prior analysis via a stale
  `ignore_samples` entry that was never actually active — found, fixed, all splits and configs
  corrected, dataset grew from 58→59 slides / 227,344→288,678 instances.
- The K=2-joint-softmax external-agreement threshold was structurally unreachable (§2.1) —
  found, explained from first principles, fixed with a rescaled, documented-as-provisional cut.
- Tier 3's naive (attention-free) posterior pooling gave chance-level bag accuracy despite
  excellent attention-weighted training loss on the same model — found via smoke-testing,
  fixed with a dedicated attention-weighted `bag_posterior` method.
- Tier 3's fold_1 training stall (frozen at ln(2) loss for 70+ epochs) was traced to
  independently-trained per-fold CVAE encoders producing latent features on different absolute
  scales (one fold's raw std measured 7.5× smaller than another's) — fixed with
  `LayerNormalization`.

---

## 6. Summary — what can and cannot be stated as a result

**Defensible, positive claims:**
- The MIL reframing itself was necessary and is now justified by direct evidence, not just
  principle (§4; every prior instance-level label was unsupervised-derived).
- `encoder_head` (Tier 4) is a real bag-level classifier statistically tied with the strongest
  conventional methods (Apollo/RF/XGBoost) and significantly ahead of the weakest one (CNN), at
  zero additional training cost.
- The spatial-validation methodology (lag-banded two-statistic design + shuffled-label control)
  correctly detects both a scale-free-statistic artifact (EM) and four label-independent
  false positives (mi-SVM, attention, both cbr variants) — it is doing its job.
- Multiple real, independently-verifiable bugs in the pre-MIL evaluation pipeline were found
  and fixed (§5), meaning the numbers now being reported were arrived at carefully.

**Defensible, negative claims — equally rigorous, not lesser findings:**
- No MIL tier significantly beats the strongest conventional classifiers (RF, XGBoost) at
  bag-level diagnosis; two tiers are significantly *worse*.
- No MIL tier's posterior map demonstrates spatial coherence attributable to the true
  diagnostic label — all four testable claims failed an adversarial shuffled-label control.
- π_b shows no detectable relationship to any independent tissue-composition proxy available
  in this project, and stays essentially pinned at its numerical floor.

**Genuinely open — not established in either direction:**
- Whether 58–59 bags is fundamentally too few for pixel-level latent inference to succeed here
  at all, versus these five specific formulations simply not finding it. The project's own
  designed checks say "not demonstrated," not "impossible."
- EM's collapse root cause (genuine weak signal vs. this specific config) — deferred, not
  chased further (2026-09-29 decision).
- Tier 3's inner-validation-vs-true-test generalization gap (0.893 vs 0.644 bag accuracy) —
  deferred, not chased further (2026-09-29 decision).
- σ's formal identifiability (a profile-likelihood interval, the plan's own stated criterion)
  — informal evidence only (§3); not built.

---

## 7. Status of Stages 7–8

**Not pursued, by deliberate decision (2026-09-30), given the results above** — not merely
unstarted. Stage 7 (CVAE soft conditioning) would need to pick a posterior to condition
generation on; every candidate's spatial claim failed (§2), removing the one property
(demonstrated label-driven structure) that would have made a particular choice clearly
justified over the others, and the remaining candidates split between circular
(`encoder_head`, self-referential for this use) and weaker classifiers (mi-SVM, attention).
Stage 8 (mixture model (M) + 6-class) was judged not to add further evidence given the
constrained model (C) already shows no tier recovering label-driven instance-level structure
under 5× less demanding conditions (binary, hard support) than the full mixture would require.
The experimental programme is considered complete for the purposes of writing up Stages 0–6's
findings; docs/MIL_REFRAME_PLAN.md §13 carries the full backlog if either stage is revisited
later.

**One further, free-standing analysis was added after this section was written**: a CVAE
counterfactual probe (docs/counterfactual_probe.md) — decode the same latent `z` under both
class labels and treat the difference as a learned "IDH effect" spectrum. Unlike every result
above, its shuffled-label control came back **genuinely mixed**, not uniformly negative: the
Δ spectrum's magnitude and cross-fold consistency fail the same way every tier above did
(shuffled ≥ real), but an independent classifier's counterfactual-flip rate does not
(real ≫ shuffled, 99.87%/91.16% vs. 41.13%/52.84%) — the first statistic in this project to
survive an adversarial shuffled-label control. A follow-up α=0 ablation (retraining the CVAE
with its auxiliary class head's loss weight zeroed, real GPU cost, run 2026-09-30) then
settled the architectural concern that motivated checking any of this in the first place:
every one of these statistics is unchanged, within noise, whether that head trains or not —
the flip-to-target finding does not depend on it. See that doc for the full, precise write-up
of what does and doesn't hold up.
