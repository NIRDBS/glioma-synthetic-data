# MIL Tiers — Findings

Synthesis of what this project's experiments establish, intended as source material for the
paper's Results and Discussion sections. Analysis, not publication prose: tables, exact
statistical support, and precise characterization of what is and is not established. Companion
to `docs/mil_tiers_methodology.md` (methods only, no results). Full per-tier derivations:
`docs/stage4_mil.md` (Tiers 1–2), `docs/stage5_spatial_validation.md` (spatial validation,
Tiers 1–2, cross-tier synthesis), `docs/stage6_attention_mil.md` (Tier 3),
`docs/stage6_generative_mil.md` (Tier 4).

All values below are read directly from persisted result files
(`results/exp_501/mil_eval/*.json`, `results/exp_501/spatial_validation/*.json`), not
transcribed from console logs or recalled from memory.

---

## 0. Experiment Overview

Four MIL tiers — mi-SVM (Tier 1), EM (Tier 2), gated-attention MIL (Tier 3), generative/CbR-based
MIL (Tier 4, two variants) — each fit on `exp_501` (binary IDH-mutant vs. wildtype), evaluated at
bag level against all four Tier 0 classifiers and, where a real posterior existed, against a
lag-banded spatial-coherence test with a shuffled-bag-label control. Naive pooling (Tier 0) is
the reference, not a tier under test. §1–§3 below cover `exp_501` only; §8 extends this to five
further experiments (the CVAE β sweep and the v2-preprocessing replication), and §9 maps which
remaining experiments in the current paper's family (`exp_5xx`–`exp_7xx`) do and do not have MIL
results.

**Comparability constraint.** mi-SVM/EM's fits predate the discovery that slide
`HF-1887_via-t_2_1.h5_3` was silently excluded from every prior analysis. Their bag-level
numbers are computed on **n=58** bags; Tier 3/Tier 4's are computed on the corrected **n=59**.
The two sample sizes correspond to slightly different Tier 0 reference numbers, reported
separately in both tables below. mi-SVM was not refit after the correction: its spatial claim
was independently dropped (§2) before a refit would have been justified.

---

## 1. Bag-Level Classification

Class-stratified bootstrap, 1000 resamples, seed 42, majority-vote pooling (TRTR) for every
Tier 0 reference.

### 1.0 Tier 0 References

| model | bag acc, n=58 (mi-SVM comparison) | bag acc, n=59 (Tier 3/4 comparison) |
|---|---|---|
| 1D-CNN | 0.6034 | 0.5932 |
| Apollo (RF+SVM) | 0.7586 | 0.7627 |
| Random Forest | 0.8103 | 0.8136 |
| XGBoost | 0.8103 | 0.8136 |

The two tree ensembles are the strongest classifiers at either $n$, by a wide margin over the
CNN specifically (~20 points).

### 1.1 MIL Tiers vs. Tier 0

| tier | n | bag acc (F1, κ) | Δ vs CNN | Δ vs Apollo | Δ vs RF | Δ vs XGBoost |
|---|---|---|---|---|---|---|
| mi-SVM (Tier 1) | 58 | 0.6897 (0.689, 0.385) | +0.086, CI[−0.017,+0.190], p=0.126 | −0.069, CI[−0.190,+0.035], p=0.238 | **−0.121, CI[−0.225,−0.017], p=0.046** | **−0.121, CI[−0.225,−0.017], p=0.046** |
| EM (Tier 2) | — | excluded — §1.2 | — | — | — | — |
| attention (Tier 3) | 59 | 0.6441 (0.623, 0.257) | +0.051, CI[−0.102,+0.203], p=0.604 | **−0.119, CI[−0.220,−0.034], p=0.012** | **−0.170, CI[−0.288,−0.068], p=0.002** | **−0.170, CI[−0.288,−0.068], p=0.002** |
| CbR `reconstruction` (Tier 4) | 59 | 0.7119 (0.711, 0.423) | +0.119, CI[−0.051,+0.254], p=0.150 | −0.051, CI[−0.169,+0.051], p=0.474 | −0.102, CI[−0.203,0.000], p=0.104 | −0.102, CI[−0.203,0.000], p=0.104 |
| CbR `encoder_head` (Tier 4) | 59 | 0.7966 (0.786, 0.577) | **+0.203, CI[+0.085,+0.339], p=0.006** | +0.034, CI[−0.034,+0.102], p=0.410 | −0.017, CI[−0.051,0.000], p=0.694 | −0.017, CI[−0.051,0.000], p=0.694 |

Bold indicates statistical significance at the 95% level. All other cells are non-significant
ties, in either direction, not demonstrated wins or losses.

### 1.2 Invalidity of EM's Bag Accuracy

A naive bag-classification accuracy, computed identically to §1.1, gives EM a perfect
**1.0000**. This value is provably meaningless, not a finding: Model (C)'s hard support
restriction confines every instance's state to `{null_1, null_2, null_3, Y_b}` — its own bag's
already-known label, never another bag's — so the metric recovers the label EM was given, not
one it inferred. Verified directly: 0% of instances, including held-out ones, ever show
competition between diagnostic columns. Excluded from every comparison in §1.1, with a
permanent code guard (`diagnostic_column_competition_frac`) preventing recurrence for any future
Model-(C)-constrained formulation.

### 1.3 Interpretation

- RF and XGBoost are not significantly beaten by any MIL tier, in either direction, anywhere in
  this project. The strongest MIL result (`encoder_head`) is a statistical tie with both, not a
  win.
- The CNN is a weak reference specifically: every positive MIL delta in §1.1 is measured against
  the CNN, and only one reaches significance.
- The sole significant win in the project is `encoder_head` vs. CNN ($p=0.006$): it ties the
  strongest conventional classifiers and significantly beats only the weakest one, not Tier 0 as
  a whole.
- Two tiers show a significant *loss*, not a non-significant tie: mi-SVM loses to RF/XGBoost;
  attention loses to Apollo and to RF/XGBoost. Attention is the only tier with no
  non-significant-or-better cell against the three strongest references.

---

## 2. Spatial Coherence Validation

Lag-banded Moran's I and empirical variogram (Chebyshev lag 1–10; lags 1–2 are contaminated by
the 3×3 median filter and excluded from any claim; lag≥3 is the headline statistic), 200
within-slide permutations, pooled across all 59 slides for every tier below. The shuffled-bag-
label control is an exact permutation of the true label multiset (same class balance), holding
every instance's features, bag membership, and CV split fixed, and refitting end-to-end under
the permuted labels.

### 2.1 Real vs. Shuffled-Label Fits

| tier | real lag-3 (I, z) | shuffled lag-3 (I, z) | variogram range (real / shuffled) | verdict |
|---|---|---|---|---|
| mi-SVM (Tier 1) | +0.006, z=+1.5 | +0.018, z=+5.01 | 2px / 2px | failed — dropped |
| EM (Tier 2) | variogram ≡ 0 (collapsed) | not run (moot) | n/a | no claim produced |
| attention (Tier 3) | +0.0236, z=+5.63 | +0.0613, z=+14.70 | 2.0px / 2.0px | failed — dropped |
| CbR `reconstruction` (Tier 4) | +0.122, z=+29.4 | +0.1253, z=+32.31 | 3.0px / 3.0px | failed — dropped |
| CbR `encoder_head` (Tier 4) | +0.049, z=+11.1 | +0.0485, z=+11.04 | 3.0px / 2.0px | failed — numerically identical, dropped |

This is the most methodologically stringent result in the project: a pre-registered, adversarial
test rather than a post-hoc interpretation of a $p$-value. Labels carrying zero true information
produce comparable or larger apparent lag≥3 coherence than the true labels, for every tier
tested. The `encoder_head` result removes residual ambiguity: real and shuffled statistics agree
to two decimal places at every lag.

**Remark.** Spatial structure is present, at every lag, in every tier; none of it is
attributable to the true diagnostic label. The better-supported explanation, identical across
all four failures: real tissue carries
genuine spatial autocorrelation in its raw spectral characteristics, present in whatever
representation a formulation builds on top of the spectrum (RF-selected bands for mi-SVM, a
frozen CVAE embedding for attention, the CVAE's own encoder/decoder for both CbR variants),
independent of which label a region is fit toward. A model flexible enough to fit any label
reasonably well — `encoder_head` reaches 88–93% training "accuracy" under complete label noise
over 100 epochs, direct independent evidence of this flexibility — produces a lag≥3-coherent map
regardless of the label, since the coherence never originated from the label.

**Methodological note.** The two-statistic design (scale-free Moran's I + scale-sensitive
variogram) is what distinguishes EM's collapse (floating-point noise around a constant,
initially misread as significant) from a genuine signal; the shuffled-label control is what
identifies the remaining four false positives. The validation pipeline performed as designed;
the result it returns is negative.

### 2.2 Agreement with Unsupervised Tissue Clustering

Hard-threshold ARI/AMI against the legacy unsupervised cluster selection (`viable_legacy`) — not
ground truth (no tumor annotation of any kind exists in this dataset; §4).

| tier | ARI | AMI | "diagnostic" fraction |
|---|---|---|---|
| mi-SVM (real) | −0.0072 | 0.0015 | 0.96% (2,784/288,678) |
| attention (real, corrected threshold) | 0.0164 | 0.0008 | 84.4% |
| CbR `reconstruction` (real, corrected threshold) | −0.0148 | 0.0009 | 9.1% |
| CbR `encoder_head` (real, corrected threshold) | −0.0458 | 0.0089 | 85.5% |

Uniformly weak to null; two of four ARIs are negative (worse than chance agreement). Not a
positive finding under any threshold tested. (A structural bug produced a degenerate read —
ARI=AMI=0.0000, 100% "diagnostic" — for any joint-softmax source [attention, both CbR variants]
before a 2026-09-29 fix; mi-SVM's independent-score construction was unaffected and is reported
here unchanged.)

---

## 3. π_b External Validity — Tier 2 (EM)

- $\pi_b$ (the inferred per-bag diagnostic fraction) remains pinned near the numerical
  identifiability floor across two independent full refits: range **[0.0200, 0.0443]** over 59
  bags, $\text{std}=0.0031$.
- Spearman correlation against three independent NMF/GMM tissue-cluster proportions and against
  the `viable_legacy` fraction: $\rho \in [-0.16, +0.14]$, every $p>0.19$ ($n=59$) — null,
  agreeing with §2.2's ARI/AMI-based null result via an independent statistic.
- $\sigma$ (the population spread of logit $\pi_b$) hits the 0.000 boundary in most folds on both
  refits, and which folds collapse differs between the two nominally identical reruns (same
  seed, same code) — informal evidence of poor identifiability at this bag count, consistent
  with the plan's pre-registered expectation, though short of a formal profile-likelihood
  criterion (not constructed).

**Conclusion.** EM recovers no genuine, externally verifiable per-bag signal at 58–59 bags:
$\pi_b$ is close to a single global constant, not an informative per-slide estimate.

---

## 4. Absence of Instance-Level Ground Truth

No instance-level (pixel-level) tumor annotation exists anywhere in this dataset, at any point
in this project's history. `viable_legacy` and `tissue_cluster` are both the output of an
unsupervised pipeline (NMF(100) → GMM(3) → manual selection of the cluster whose mean spectrum
resembled the strongest signal) — a signal-quality judgment, not a histopathological one. An
earlier PCA/DBSCAN-based attempt was superseded before this pipeline was adopted, for the same
reason. Every classifier in this project prior to the MIL reframing was trained under the
assumption that a slide's grade label applies to every one of its spectra.

**Remark.** This fact predates the experiments above — it is the premise the MIL reframing was
built to correct — and it is directly verifiable from the data-generation code itself, not an
assumption.

---

## 5. Methodological Corrections

Each item independently verifiable from the code or tests, not merely asserted:

- Three metric bugs in the pre-MIL evaluation code: `precision_macro` was a constant ≈0.5,
  `kappa` was always 0.0, `f1_macro` silently dropped 0%-accuracy bags (upward-biased).
- The CNN's early-stopping monitor read the true test fold — a genuine leak, not merely
  optimism — fixed with a group-aware inner-validation split.
- Slide `HF-1887_via-t_2_1.h5_3` was silently excluded from every prior analysis via a stale
  `ignore_samples` entry that was never actually active; found, fixed, all splits and configs
  corrected, dataset grew from 58→59 slides / 227,344→288,678 instances.
- The K=2 joint-softmax external-agreement threshold (§2.2) was structurally unreachable; found,
  explained from first principles, fixed with a rescaled, documented-as-provisional cut.
- Tier 3's naive (attention-free) posterior pooling gave chance-level bag accuracy despite strong
  attention-weighted training loss on the same model; found via smoke-testing, fixed with a
  dedicated attention-weighted `bag_posterior` method.
- Tier 3's fold_1 training stall (loss frozen at $\ln 2$ for 70+ epochs) was traced to
  independently trained per-fold CVAE encoders producing latent features on different absolute
  scales (one fold's raw std measured 7.5× smaller than another's); fixed with
  `LayerNormalization`.

---

## 6. Summary of Claims

**Established, positive.**
- The MIL reframing is justified by direct evidence, not principle alone (§4): every prior
  instance-level label was unsupervised-derived.
- `encoder_head` (Tier 4) is a bag-level classifier statistically tied with the strongest
  conventional methods (Apollo/RF/XGBoost) and significantly ahead of the weakest one (CNN), at
  zero additional training cost.
- The spatial-validation methodology (lag-banded two-statistic design + shuffled-label control)
  correctly detects a scale-free-statistic artifact (EM) and four label-independent false
  positives (mi-SVM, attention, both CbR variants).
- Multiple independently verifiable bugs in the pre-MIL evaluation pipeline were found and fixed
  (§5); the numbers reported here reflect the corrected pipeline.

**Established, negative — equally rigorous, not lesser findings.**
- No MIL tier significantly beats the strongest conventional classifiers (RF, XGBoost) at
  bag-level diagnosis; two tiers are significantly worse.
- No MIL tier's posterior map demonstrates spatial coherence attributable to the true diagnostic
  label; all four testable claims failed an adversarial shuffled-label control.
- $\pi_b$ shows no detectable relationship to any independent tissue-composition proxy available
  in this project, and remains pinned near its numerical floor.

**Open.**
- Whether 58–59 bags is fundamentally insufficient for pixel-level latent inference in this
  setting, versus these five specific formulations simply not finding it. The project's checks
  establish "not demonstrated," not "impossible."
- EM's collapse root cause (genuine weak signal vs. this specific configuration) — deferred, not
  pursued further (2026-09-29 decision).
- Tier 3's inner-validation-vs-true-test generalization gap (0.893 vs. 0.644 bag accuracy) —
  deferred, not pursued further (2026-09-29 decision).
- $\sigma$'s formal identifiability (a profile-likelihood interval, the plan's stated criterion)
  — informal evidence only (§3); not constructed.

---

## 7. Stages 7–8: Disposition

Not pursued, by deliberate decision (2026-09-30), given the results above — not merely
unstarted. Stage 7 (CVAE soft conditioning) requires selecting a posterior to condition
generation on; every candidate's spatial claim failed (§2), removing the one property
(demonstrated label-driven structure) that would justify a particular choice over the others,
and the remaining candidates split between circular (`encoder_head`, self-referential for this
use) and weaker classifiers (mi-SVM, attention). Stage 8 (mixture model (M), 6-class) was judged
not to add further evidence: the constrained model (C) already shows no tier recovering
label-driven instance-level structure under substantially less demanding conditions (binary,
hard support) than the full mixture would require. The experimental programme is considered
complete for writing up Stages 0–6; `docs/MIL_REFRAME_PLAN.md` §13 carries the full backlog if
either stage is revisited.

**Addendum.** A free-standing analysis was added after this section was written: a CVAE
counterfactual probe (`docs/counterfactual_probe.md`) — decode the same latent $z$ under both
class labels and treat the difference as a learned "IDH effect" spectrum. Unlike every result
above, its shuffled-label control is genuinely mixed, not uniformly negative: the Δ spectrum's
magnitude and cross-fold consistency fail the same way every tier above did (shuffled ≥ real),
but an independent classifier's counterfactual-flip rate does not (real ≫ shuffled, 99.87%/91.16%
vs. 41.13%/52.84%) — the first statistic in this project to survive an adversarial shuffled-label
control. A follow-up $\alpha=0$ ablation (retraining the CVAE with its auxiliary class head's
loss weight zeroed, real GPU cost, run 2026-09-30) settled the architectural concern that
motivated this check: every statistic is unchanged, within noise, whether that head trains or
not — the flip-to-target finding does not depend on it. See that document for the full write-up.

---

## 8. Replication Across CVAE β and Preprocessing Variant

Five further experiments carry MIL results beyond `exp_501`: three CVAE β values under v1
preprocessing (`exp_607`: β=0.25, `exp_608`: β=0.5, `exp_609`: β=0.75 — TRTR reused unchanged
from `exp_501`, so the Tier 0 reference table is §1.0's n=59 column throughout) and two
v2-preprocessing replications of the first two β values (`exp_2.607`: β=0.25, `exp_2.608`:
β=0.5). All five share `exp_501`'s n=59 bag count and evaluation machinery (§9).

**Coverage is uneven across these five experiments** — different runs evaluated against
different subsets of the four Tier 0 references, and not every tier/variant carries a
shuffled-label control. Each table below states exactly which comparisons exist; a cell is
never filled in by inference from another experiment.

### 8.1 mi-SVM and Tier 5 Are Invariant to β

mi-SVM (Tier 1) and Bayesian MIL (Tier 5) fit on PCA-reduced raw spectral features, never the
CVAE latent space (`docs/mil_tiers_methodology.md` §1.1) — their results cannot depend on which
β an experiment otherwise trains, and do not: both are bit-for-bit identical across all three of
`exp_607`/`608`/`609`, and identical again (at different values, from a different dataset) across
both of `exp_2.607`/`2.608`. Reported once per preprocessing version rather than three times.

| tier | preprocessing | acc (F1, κ) | vs Apollo |
|---|---|---|---|
| mi-SVM | v1 (607/608/609) | 0.7627 (0.763, 0.530) | −0.0014, CI[−0.102,+0.102], p=1.000 |
| mi-SVM | v2 (2.607/2.608) | 0.6271 (0.624, 0.250) | **−0.1173, CI[−0.220,−0.017], p=0.030** |

mi-SVM ties Apollo under v1 preprocessing but loses significantly under v2 — the only case in
this replication where a tier's significance verdict flips with the preprocessing variant.

### 8.2 Bayesian MIL (Tier 5): Genuine Blind Classification, Not the Circular Metric

`stage6_bag_eval.py`'s posterior for this tier is scored by `score_bayes_mil`, which — exactly
like Tier 2's EM (§1.2) — is handed each bag's already-known label and asked what fraction of
its instances match it (`docs/mil_tiers_methodology.md` §8.2, §9.1). Confirmed directly:
`k_diagnostic_columns_competition_frac=0.0000` in every one of these files. The resulting
bag accuracy (0.9153 in every v1 config) is excluded here for the same reason EM's 1.0000 was
excluded in §1.2.

The non-circular alternative, `score_bayes_mil_blind` (`tier5_blind_bag_classification.py`),
never receives $Y_b$ and evaluates each candidate class's posterior-predictive likelihood
directly:

| preprocessing | merged bag acc ($n$=59) | vs Apollo |
|---|---|---|
| v1 (607/608/609) | 0.7288 | −0.0327, CI[−0.153,+0.068], p=0.668 |
| v2 (2.607/2.608) | 0.7119 | −0.0339, CI[−0.136,+0.051], p=0.592 |

A non-significant tie with Apollo, both preprocessing versions, invariant to β — consistent with
§8.1's mi-SVM result and with no other tier's win in this project.

### 8.3 Attention MIL (Tier 3): Degrades With β, Significantly Loses to Apollo at Every β

| β | preprocessing | acc (F1, κ) | vs CNN | vs Apollo |
|---|---|---|---|---|
| 0.25 | v1 (`exp_607`) | 0.5254 (0.508, 0.021) | −0.1857, CI[−0.373,0.000], p=0.070 | **−0.2219, CI[−0.373,−0.068], p=0.006** |
| 0.5 | v1 (`exp_608`) | 0.4576 (0.450, −0.100) | not evaluated | **−0.3047, CI[−0.475,−0.153], p=0.000** |
| 0.75 | v1 (`exp_609`) | 0.4068 (0.407, −0.170) | not evaluated | **−0.3529, CI[−0.508,−0.203], p=0.000** |
| 0.25 | v2 (`exp_2.607`) | 0.4576 (0.445, −0.109) | not evaluated | **−0.2878, CI[−0.441,−0.136], p=0.000** |
| 0.5 | v2 (`exp_2.608`) | 0.4915 (0.488, −0.023) | not evaluated | **−0.2544, CI[−0.407,−0.102], p=0.002** |

**Remark.** `exp_501`'s significant loss to Apollo (§1.3) is not a one-off: it replicates at
every β and both preprocessing versions tested, and $\kappa<0$ at β≥0.5 under v1 preprocessing
— worse than chance-weighted agreement, not merely a weak tier.

### 8.4 Generative MIL (Tier 4): CbR Variants

| β | preprocessing | variant | acc (F1, κ) | vs CNN | vs Apollo |
|---|---|---|---|---|---|
| 0.25 | v1 (`exp_607`) | `reconstruction` | 0.7288 (0.719, 0.441) | +0.0188, CI[−0.102,+0.136], p=0.864 | not evaluated |
| 0.5 | v1 (`exp_608`) | `reconstruction` | 0.7288 (0.710, 0.431) | not evaluated | −0.0327, CI[−0.119,+0.051], p=0.584 |
| 0.75 | v1 (`exp_609`) | `reconstruction` | 0.6780 (0.668, 0.339) | not evaluated | −0.0852, CI[−0.186,+0.017], p=0.156 |
| 0.25 | v2 (`exp_2.607`) | `reconstruction` | 0.7458 (0.735, 0.474) | not evaluated | +0.0018, CI[−0.102,+0.102], p=1.000 |
| 0.5 | v2 (`exp_2.608`) | `reconstruction` | 0.7288 (0.722, 0.445) | +0.0679, CI[−0.085,+0.221], p=0.468 | −0.0176, CI[−0.102,+0.068], p=0.842 |
| 0.25 | v1 (`exp_607`) | `encoder_head` | 0.7627 (0.754, 0.511) | +0.0518, CI[−0.051,+0.153], p=0.426 | not evaluated |
| 0.5 | v1 (`exp_608`) | `encoder_head` | 0.7966 (0.789, 0.581) | not evaluated | +0.0345, CI[−0.034,+0.102], p=0.466 |
| 0.75 | v1 (`exp_609`) | `encoder_head` | 0.7797 (0.770, 0.544) | not evaluated | +0.0185, CI[−0.034,+0.068], p=0.722 |
| 0.25 | v2 (`exp_2.607`) | `encoder_head` | 0.7797 (0.773, 0.547) | not evaluated | +0.0339, CI[−0.034,+0.102], p=0.446 |
| 0.5 | v2 (`exp_2.608`) | `encoder_head` | 0.7966 (0.789, 0.581) | **+0.1367, CI[+0.017,+0.271], p=0.048** | +0.0511, CI[0.000,+0.119], p=0.098 |

`encoder_head` is the only tier/variant whose `exp_501` significant win (§1.1: vs. CNN,
p=0.006) replicates under a different preprocessing version: `exp_2.608` (β=0.5, v2) reaches the
same comparison at p=0.048. Every `reconstruction` comparison across both preprocessing versions
is a non-significant tie in either direction.

`latent_rf` (Tier 4's third variant) was evaluated once, at β=0.25 (`exp_607`) only: acc=0.7288
(F1 0.715, κ 0.436), vs CNN +0.0203, CI[−0.102,+0.136], p=0.842 — not replicated at any other β
or preprocessing version.

### 8.5 Spatial Coherence: Same Verdict at Every β and Preprocessing Variant

Same lag-banded Moran's I / variogram design as §2. A tier/variant is marked "no control" where
no shuffled-label fit exists in `results/{exp}/spatial_validation/` — absence of a control, not a
pass.

| tier | β | preprocessing | real lag-3 (I, z) | shuffled lag-3 (I, z) | verdict |
|---|---|---|---|---|---|
| mi-SVM | 0.25 | v1 (607) | +0.0047, z=+1.20 | no control run | n/a |
| mi-SVM | 0.5 | v1 (608) | +0.0046, z=+1.17 | no control run | n/a |
| mi-SVM | 0.75 | v1 (609) | +0.0046, z=+1.17 | no control run | n/a |
| mi-SVM | 0.25 | v2 (2.607) | +0.0056, z=+1.30 | +0.0117, z=+2.61 | failed — dropped |
| mi-SVM | 0.5 | v2 (2.608) | +0.0056, z=+1.30 | +0.0117, z=+2.61 | failed — dropped |
| attention | 0.25 | v1 (607) | +0.0742, z=+19.10 | +0.0650, z=+16.14 | failed — real exceeds shuffled, but both strongly coherent; see §2's remark on label-independent tissue autocorrelation |
| attention | 0.5 | v1 (608) | +0.0541, z=+13.10 | +0.0657, z=+16.25 | failed — dropped |
| attention | 0.75 | v1 (609) | +0.0428, z=+10.75 | +0.0610, z=+15.24 | failed — dropped |
| attention | 0.25 | v2 (2.607) | +0.0580, z=+13.73 | +0.0627, z=+14.50 | failed — dropped |
| attention | 0.5 | v2 (2.608) | +0.0523, z=+11.56 | +0.0647, z=+14.38 | failed — dropped |
| CbR `reconstruction` | 0.25 | v1 (607) | +0.1437, z=+34.54 | +0.1497, z=+37.09 | failed — dropped |
| CbR `reconstruction` | 0.5 | v1 (608) | +0.1570, z=+38.88 | +0.1710, z=+42.08 | failed — dropped |
| CbR `reconstruction` | 0.75 | v1 (609) | +0.1577, z=+38.56 | +0.1641, z=+40.77 | failed — dropped |
| CbR `reconstruction` | 0.25 | v2 (2.607) | +0.1357, z=+30.40 | +0.1345, z=+30.50 | failed — numerically close, dropped |
| CbR `reconstruction` | 0.5 | v2 (2.608) | +0.1524, z=+34.90 | +0.1513, z=+35.07 | failed — numerically close, dropped |
| CbR `encoder_head` | 0.25 | v1 (607) | +0.0557, z=+12.93 | +0.0561, z=+13.40 | failed — numerically close, dropped |
| CbR `encoder_head` | 0.5 | v1 (608) | +0.0594, z=+13.79 | +0.0565, z=+13.59 | failed — dropped |
| CbR `encoder_head` | 0.75 | v1 (609) | +0.0612, z=+14.12 | +0.0581, z=+13.57 | failed — dropped |
| CbR `encoder_head` | 0.25 | v2 (2.607) | +0.0508, z=+10.89 | +0.0410, z=+8.46 | failed — dropped |
| CbR `encoder_head` | 0.5 | v2 (2.608) | +0.0504, z=+10.54 | +0.0385, z=+7.98 | failed — dropped |
| Bayesian MIL (Gibbs) | any | v1 (607/608/609, identical) | +0.1336, z=+33.66 | **no shuffled control exists for this tier in the codebase** | open — see remark |

**Remark.** Every shuffled-label comparison that could be run reaches the same verdict as
`exp_501`'s original five: label-independent tissue autocorrelation, not label-driven structure.
mi-SVM under v1 preprocessing and Bayesian MIL at every β are exceptions only because no
shuffled-label control was ever run for them, not because either passed one — mi-SVM's real
lag-3 $z\approx1.2$ is itself weak evidence of coherence regardless; Bayesian MIL's real lag-3
coherence ($z$=+33.66, the strongest of any tier) is therefore the one spatial result in this
entire project that has never been tested against an adversarial control, and should not be read
as label-driven on that basis (§9's open-items list carries this forward).

### 8.6 External Agreement vs. `viable_legacy`

Only defined for v1 preprocessing — `viable_legacy` does not exist as a column in v2-preprocessed
data, so this check is not applicable (not merely unrun) for `exp_2.607`/`2.608`.

| tier | β | ARI | AMI |
|---|---|---|---|
| mi-SVM | 0.25 (607) | −0.0012 | 0.00002 |
| mi-SVM | 0.5/0.75 (608/609, identical) | −0.0010 | 0.00001 |
| attention | 0.25 (607) | 0.0036 | 0.0079 |
| attention | 0.5 (608) | 0.0817 | 0.0373 |
| attention | 0.75 (609) | −0.0006 | 0.00002 |
| CbR `reconstruction` | 0.25 (607) | 0.0478 | 0.0156 |
| CbR `reconstruction` | 0.5 (608) | 0.0104 | 0.0006 |
| CbR `reconstruction` | 0.75 (609) | 0.0032 | 0.00004 |
| CbR `encoder_head` | 0.25 (607) | −0.0538 | 0.0128 |
| CbR `encoder_head` | 0.5 (608) | −0.0435 | 0.0079 |
| CbR `encoder_head` | 0.75 (609) | −0.0484 | 0.0100 |
| Bayesian MIL (Gibbs) | any (607/608/609, identical) | 0.0071 | 0.00016 |

Uniformly weak to null at every β, consistent with `exp_501`'s §2.2 conclusion — no β recovers a
positive external-agreement signal this check did not already rule out.

### 8.7 Interpretation

- The β sweep and the v2-preprocessing replication change no qualitative conclusion from `exp_501`:
  no tier reaches significant spatial coherence under its shuffled-label control at any β or
  preprocessing variant tested with a control available (§8.5).
- Two tiers are structurally invariant to β (§8.1–§8.2) because neither uses the CVAE latent
  space; this is a property of their formulation, not a finding that needed five reruns to
  discover, and is reported once rather than repeated per β.
- Attention's significant loss to Apollo (`exp_501`, §1.3) replicates and worsens with β under
  v1 preprocessing, and replicates independently under v2 preprocessing at both β values tested
  (§8.3) — the most consistently negative result in this project.
- `encoder_head`'s single significant win (`exp_501` vs. CNN, §1.3) replicates once, at β=0.5
  under v2 preprocessing (§8.4) — not at every β, and not against Apollo at any β tested.
- Bayesian MIL's real spatial coherence is the strongest of any tier measured ($z$=+33.66) and
  the only one with no shuffled-label control at all (§8.5) — a genuine open item, not a result.

---

## 9. Experiment Coverage Map

Scope: the current paper's experiment family (`exp_5xx` onward — earlier suites, `exp_1xx`
through `exp_4xx`, predate the MIL reframing entirely and are out of scope, not "missing"
results). Status as of this writing; `exp_4.608` is mid-run.

| experiment | what it is | MIL results |
|---|---|---|
| `exp_501` | v1 preprocessing, β=2.0 (original) | Full — §1–§3 |
| `exp_607` | v1 preprocessing, β=0.25 | Full — §8 |
| `exp_608` | v1 preprocessing, β=0.5 | Full — §8 |
| `exp_609` | v1 preprocessing, β=0.75 | Full — §8 |
| `exp_2.607` | v2 preprocessing, β=0.25 | Full — §8 |
| `exp_2.608` | v2 preprocessing, β=0.5 | Full — §8 |
| `exp_2.609` | v2 preprocessing, β=0.75 | Not run |
| `exp_3.608` | v1 preprocessing, training restricted to the legacy `viable_legacy` tumour mask, β=0.5 | Not run |
| `exp_4.608` | v3 preprocessing (fixed intensity threshold 0.4), β=0.5 | In progress (fit stage only) |
| `exp_601`–`606` | CVAE β/epoch-schedule sweep without a dedicated MIL follow-up | Not run |
| `exp_701` | latent-dim ablation, dim=16 | Not run |
| `exp_702` | latent-dim ablation, dim=64 | Not run |
| `exp_703` | reconstruction-loss ablation, γ=0.5/δ=0.5 | Not run |
| `exp_704` | reconstruction-loss ablation, γ=0.0/δ=1.0 (pure cosine) | Not run |
| `exp_705` | synth/real ratio ablation, ratio=0.5 | Not run |
| `exp_706` | synth/real ratio ablation, ratio=2.0 | Not run |

Nine experiments in the current paper's family carry no MIL fit of any kind. This is a scope
statement, not an implied finding — §6–§8's conclusions apply only to the six experiments
actually tested.
