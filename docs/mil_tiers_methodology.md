# MIL Tiers — Methodology Reference

Consolidated methods reference for the MIL tiers (docs/MIL_REFRAME_PLAN.md
§4), written for transcription into the paper's Methods section. This file is
**methodology only** — no results, no headline numbers. Results live in
docs/stage4_mil.md (Tier 1), docs/stage5_spatial_validation.md (spatial
validation + the shuffled-label control), docs/stage6_generative_mil.md
(Tier 4) and docs/stage6_attention_mil.md (Tier 3). Update this file when a
formulation's method changes; update the stage docs when a number changes.

Code: `src/mil/misvm.py`, `src/mil/attention.py`, `src/mil/generative.py`,
`src/mil/store.py`, `src/mil/shuffle_control.py`. Fitting:
`pipeline/mil_fit_stage.py` (Tiers 1/4), `pipeline/mil_attention_stage.py`
(Tier 3), `pipeline/mil_eval_stage.py` (Tier 0). Evaluation:
`scripts/stage6_bag_eval.py` (Tiers 0/1/4),
`scripts/stage6_attention_bag_eval.py` (Tier 3), `scripts/stage5_spatial_validation.py`
(all tiers), `scripts/stage6_shuffled_label_control.py`.

---

## 1. Problem formulation

Each biopsy slide $b$ is a **bag** with a known label $Y_b \in \{1..K\}$ (the
methylation-derived tumour grade) and $n_b$ **instances** — the slide's
strong-signal Raman spectra, one per surviving pixel of the slide's spatial
grid. $K=2$ throughout (IDH-mutant vs IDH-wildtype); the 6-class extension is
future work.

**No instance-level label exists anywhere in this dataset.** The spatio-
spectral pipeline that selects which pixels enter the modelling set (median
filter, an intensity threshold, NMF+GMM clustering, manual cluster selection)
is an unsupervised signal-quality judgement, not a histopathological one —
true of every candidate annotation in this project, including an earlier
PCA/DBSCAN-based attempt superseded before this pipeline was settled on. Every
tier below infers, rather than assumes, which instances in a bag are
diagnostic.

**The constrained model (C).** Each instance's latent state is
$z_{bi} \in \{0_1, 0_2, 0_3, Y_b\}$: three shared **null components**
(stroma / necrosis / normal tissue — shared across *all* bags regardless of
class, which is what identifies them at all) plus the bag's own diagnostic
class. An instance can never be assigned another bag's diagnostic class. This
is Model (C) from docs/MIL_REFRAME_PLAN.md §1; the unconstrained mixture
Model (M), which would admit intratumoral heterogeneity across classes, is
out of scope (Stage 8).

**Diagnostic fraction $\pi_b$.** The fraction of bag $b$'s instances that are
genuinely diagnostic, $\pi_b = \tfrac{1}{n_b}\sum_i [z_{bi} = Y_b]$. It varies
per bag and is never fixed globally (Tier 3's sub-bag size is calibrated
against it). $\pi_{\min} = 0.02$ is a **numerical
identifiability floor**, preventing the all-null collapse — a floor, not a
biological claim about minimum tumour content.

**The $(K{+}1)$-way posterior convention.** Every tier ultimately produces,
per instance, a $(K{+}1)$-vector: column 0 is the combined non-diagnostic mass
(summed over the 3 null components), columns $1..K$ are the diagnostic
classes. Every tier builds column 0 the same way, as the lifted
non-diagnostic mass $1 - \max_k p_k$, renormalised: from independent
per-class scores for Tier 1, and from a genuine unconstrained $K$-way softmax
for Tiers 3 and 4 (§6.3, §7).

### 1.1 How each tier represents the latent instance state

The tiers do not all model the same object. They differ along two
independent axes: **(a)** whether the latent state is restricted to Model
(C)'s $(M{+}K)$-state space $\{0_1,0_2,0_3,Y_b\}$ at all, or is an
unconstrained $K$-way diagnostic-class label with no null substructure; and
**(b)** whether the instance-level quantity produced is a hard label, a
point-estimate probability vector, or a full posterior distribution over
that vector. Table below; formal definitions in each tier's own section.

| Tier | Method | Latent object & formal type | Model (C) state space? | How it is produced |
|---|---|---|---|---|
| 1 | mi-SVM | $K$ independent hard binary labels $\ell_i^{(k)}\in\{0,1\}$, one per one-vs-rest sub-problem — **not** a single $K$-way (or $(M{+}K)$-way) categorical | No — no null substructure; "non-diagnostic" is not a state the model ever assigns | Iterative relabelling: per positive bag, top-$\lceil\pi_{\min}n_b\rceil$ (or top-1) Platt-scored instances set to 1 each round, to convergence (§5) |
| 3 | Attention MIL | Unconstrained discrete class $\hat y_i\in\{1..K\}$, represented as **instance logits $\to$ softmax** $\in\Delta^{K-1}$ (point estimate, $K$-way only, no null state) | No, deliberately — masking the instance head would make instance-level evaluation circular (§6.3) | Learned instance head over a frozen CVAE trunk, supervised only through attention-pooled bag loss, never per instance |
| 4 | Generative (CbR / encoder-head) | Unconstrained discrete class $\hat y_i\in\{1..K\}$, as a **softmax** $\in\Delta^{K-1}$ (point estimate, $K$-way only, no null state) | No | Zero-training-cost scoring with already-fit CVAE weights ($-$MSE or the auxiliary head) (§7) |

**Reading the table.** No tier represents "non-diagnostic" as a state:
the non-diagnostic store column (§1's $(K{+}1)$-way convention) is a post-hoc
arithmetic artifact, $1-\max_k p_k$, not something the fitted model assigned
mass to directly.

---

## 2. The latent-label store and leakage discipline

Every formulation's posterior is written to a sidecar HDF5 file, never the
main (immutable) dataset: `results/{exp}/latent/latent_labels.h5`, schema
`/{method}/{variant}/{fold_name}/{posterior, in_sample}` with attributes
`method, variant, K, config_hash, git_sha, timestamp, alignment_checksum`.

**Leakage rule.** Indexed by fold: an instance in fold $f$'s *validation*
bags gets its posterior from the fold-$f$ model, which never trained on it
(`in_sample=False`). An instance in fold $f$'s *training* bags is
necessarily in-sample (`in_sample=True`) — legitimate, since that is
train-side use. `assert_no_fold_leakage` enforces this with a raising
assertion, not merely a convention; `alignment_checksum` (a hash of the
source HDF5's `sample_ids`/`coords`) guards against a silent re-export
invalidating stored posteriors.

**Cross-fold duplicate bags.** Four slides are forced into two CV test folds
each (6-class stratification over classes with only 4 patients — see the
CV-schema appendix). `merge_out_of_sample_posterior` averages a duplicated
bag's two out-of-sample estimates rather than picking one arbitrarily,
giving one posterior per bag regardless of how many test folds it appears
in. Bag-level evaluation (§9) then has exactly $n=59$ bag decisions from 63
(bag, fold) test pairs.

**Two-posterior split for Tier 3.** Tier 3 additionally writes a *second*
artifact outside this store, `results/{exp}/latent/attention_bag_level.json`
— see §6.4 for why one posterior is not sufficient for this tier.

---

## 3. Shared instance-learner stack (feeds Tier 0)

Four classifiers, each trained under the **naive-MIL assumption** — every
instance inherits its bag's label — the assumption every tier past Tier 0
exists to correct:

| model | configuration |
|---|---|
| RF + SVM ("Apollo") | RF feature selector: 300 trees, depth 10, `balanced_subsample`, selects the top 20 bands by importance; RBF SVC, $C{=}1$, `balanced` class weight, Platt-scaled probabilities |
| Random Forest | 100 trees, `balanced` class weight |
| XGBoost | histogram method, default hyperparameters |
| 1D-CNN | Conv1D blocks $[32,64,128]$, kernels $[11,5,3]$, batch norm, max-pool, global average pooling, dense 128, dropout 0.6; Adam $10^{-3}$, batch 256, up to 100 epochs; early stopping (patience 10) on a 15%, slide-grouped inner split carved from *training* slides only — never the true CV test fold |

Training data: cropped to the fingerprint region (700–1800 cm⁻¹, 571 bands),
capped at 500 instances/slide for the classifiers (dynamic per-epoch
resampling for the CNN's augmentation; a fixed draw for the others). Test
bags are always scored on every one of their instances, uncapped.
Augmentation (within-class mixup, fractional spectral shift, baseline
wobble) applies to the CNN only.

---

## 4. Tier 0 — naive-MIL pooling

No new training. Pools the four classifiers' existing per-instance outputs
into a bag decision under four rules, giving the reference every later tier
is compared against:

| rule | bag score |
|---|---|
| mean | $\arg\max_k \bar p_{bk}$ |
| majority (reference) | $\arg\max_k \sum_i [\hat y_{bi}=k]$ |
| log-sum-exp | temperature-scaled soft max, $T{=}1$ |
| top-$q$ | mean of the top 10% of a bag's instances by score — a *fraction*, since bag size varies roughly 8× (1,878–14,674 instances) |

Cross-fold duplicate bags are handled by per-fold aggregation before
averaging (`aggregate_bag_scores_across_folds`), not by pooling instances
first and losing the fold boundary.

---

## 5. Tier 1 — mi-SVM

One-vs-rest per diagnostic class $k$ (Andrews et al. 2002), reusing Apollo
(§3) unchanged as the base learner.

**Setup.** Positive bags = slides labelled $k$; negative bags = every other
slide. Negative-bag instances are fixed at label 0 for the *entire* run,
never revised — the defining MIL asymmetry (a bag that is not class $k$
contributes no evidence for class $k$ from any of its instances).

**Iteration.** (i) Fit Apollo on the current instance labels. (ii) Score
every instance. (iii) Within each positive bag, relabel the top
$\lceil \pi_{\min} \cdot n_b \rceil$ scoring instances positive, the rest
negative. Repeat up to `max_iters` (10 in the real fit). `variant='MI_SVM'`
(not used in the real fit) would instead keep only the single top-scoring
witness per positive bag.

**Scale fix.** With `patient_cap` retired for MIL fitting (every instance is
used), a fold's training set is $\sim$230k instances — an RBF SVM fit at
that scale was confirmed empirically to take 14+ hours with no result. Fixed
by subsampling *only the SVM's own fit* to 500 instances/bag, a fresh random
draw every iteration (seed varies with the iteration count); scoring
(`predict_proba`, which drives both relabelling and the returned posterior)
always runs on every instance regardless.

**Posterior.** Platt-scaled probability per one-vs-rest sub-problem,
combined as $[1-\max_k p_k,\ p_1,\dots,p_K]$, renormalised (§1's non-
diagnostic-mass convention for independent per-class scores).

**Hyperparameters used in the real fit.** $\pi_{\min}=0.02$, `max_iters=10`,
`svm_subsample_per_bag=500`, `variant='mi_svm'`.

---

## 6.3 Tier 3 — attention MIL

Gated attention (Ilse et al. 2018) over instance **logits**, on a **frozen**
per-fold CVAE encoder trunk — the plan's own instruction (docs/
MIL_REFRAME_PLAN.md §4): attention over pooled *embeddings* would give only
a bag score, with no interpretable per-instance posterior; pooling logits
keeps each instance's own $K$-way opinion directly readable, with attention
only deciding how much each instance's opinion counts.

**Why the instance head is a free, unconstrained softmax.** Tier 3's
instance head is never masked to the bag's own class. A head masked to
Model (C)'s support would make bag classification circular (§9.1); moreover,
only the attention-*pooled* bag score
is supervised (the training loss never touches an individual instance's
logits directly), so a genuinely learned, evaluable instance posterior
requires the head to be free to be wrong about any one instance.

**Architecture** (`src/mil/attention.py::AttentionMIL`), per (sub-)bag of
$s$ instances:
1. Frozen encoder: $X \in \mathbb{R}^{s\times n_{\text{features}}} \to$
   $z_{\text{mean}} \in \mathbb{R}^{s\times d}$ ($d{=}32$, the CVAE's own
   latent dimension) — the same feature the CVAE's own $z_{\text{mean}}$
   output is, reused unmodified, gradients never flow into it.
2. **Per-(sub-)bag centering** (`center_by_bag=True`, default): $z_{\text{mean}}$
   is centred by the (sub-)bag's own mean before anything downstream sees
   it — removes a bag-specific acquisition/baseline fingerprint the frozen
   trunk could otherwise leak, while preserving within-bag relative
   structure, which is where diagnostic-vs-null contrast has to live under
   this project's own formulation. (Centering changes *location*, not
   *scale* — see the next point for why a separate fix was still needed.)
3. **LayerNormalization**, unconditionally, after centering — each fold has
   its *own* independently-trained CVAE encoder, and they do not produce
   $z_{\text{mean}}$ on a consistent scale (confirmed: one fold's raw
   $z_{\text{mean}}$ std measured 7.5$\times$ smaller than another's on
   real data, enough on its own to starve gradients at ordinary
   initialization scales). This normalisation is what makes training robust
   to whatever scale a given fold's frozen trunk happens to produce.
4. Instance head: a single Dense layer, $K$-way logits, L2-regularised.
5. Gated attention (Ilse et al. 2018, eq. 9):
   $a_i = w^\top\big(\tanh(Vh_i)\odot\sigma(Uh_i)\big)$, softmax-normalised
   over the $s$ instances in the (sub-)bag.
6. Bag score: $\text{bag\_logits} = \sum_i a_i \cdot \text{instance\_logits}_i$;
   $\text{bag\_probs} = \mathrm{softmax}(\text{bag\_logits})$, trained by
   cross-entropy against $Y_b$ with one-sided label smoothing at $\epsilon$
   (below) — for $K{=}2$, symmetric Keras `label_smoothing=2\epsilon`
   reproduces the one-sided case exactly, since there is only one "other"
   class to smooth toward.

**Regularisation.** Dropout 0.3 (before the instance head and inside the
attention branch), L2 weight decay $10^{-3}$, `attn_dim=32`. Added after a
real smoke test showed the *un*regularised configuration overfitting to bag
identity: sub-bag training loss fell steadily while held-out bag accuracy
got *worse* with more training — the signature of the model learning "which
training bag is this" rather than diagnostic content, given only 38–40 real
bags and thousands of sub-bag exposures per epoch.

**Sub-bagging.** A fresh, uniform-at-random draw of sub-bags every epoch
(never contiguous — preserves the withheld-coordinates property spatial
validation depends on, and is what the binomial sizing calculation below
assumes). Sub-bag size:
$$s \ge \frac{\ln\epsilon}{\ln(1-\pi_{\min})}$$
— the smallest size such that a uniformly-drawn sub-bag from a bag at the
$\pi_{\min}$ floor has probability $\le\epsilon$ of containing zero
diagnostic instances. At $\pi_{\min}{=}0.02$, $\epsilon{=}0.01$: $s\ge228$
(`required_sub_bag_size`, matching the plan's own worked example exactly).
Sampling is without replacement (every real bag is comfortably larger than
228) with a with-replacement fallback for a hypothetically smaller bag.

**Diagnostic — sub-bag null rate.** The empirical fraction of sampled
sub-bags where *no* instance's own-class posterior mass clears 0.5 under the
fitted model, checked against the $\epsilon$ the sizing targeted
(`sub_bag_null_rate`). Materially above $\epsilon$ would mean the sizing
under-covers this bag's real diagnostic fraction.

**Training loop.** Manual per-epoch resampling (matching the pattern already
used for CVAE training, `pipeline/cvae_stage.py`) rather than a `tf.data`
repeat/steps-per-epoch scheme. Early stopping monitors **inner-validation
loss** over freshly-drawn sub-bags from an inner-validation split — not raw
bag accuracy, which is too coarse (7–8 bags, $\sim$14-point jumps) to
reliably separate genuine improvement from noise. The inner-validation split
itself is carved from *training* bags only (15%, group-aware —
`pipeline._common._inner_validation_split`), never the true CV test fold,
the same leak-avoidance Stage 3 put in place for the CNN.

### 6.4 Two posteriors, for two different questions

`AttentionMIL` exposes two scoring methods, and using the wrong one for a
given question was a real, discovered failure mode, not a hypothetical:

- **`instance_posterior`** — attention-free, one $(K{+}1)$-vector per
  instance, independent of which other instances happen to be scored
  alongside it. This is what is written to the shared latent-label store
  (§2) and what Stage 5's spatial validation needs (one value per pixel to
  reproject).
- **`bag_posterior`** — attention-*weighted*, run over a bag's full real
  instance count (not a fixed sub-bag size): the aggregate this model is
  actually trained to optimise. An instance attention learns to downweight
  during training can develop an arbitrary individual prediction, since its
  contribution to the attention-pooled training loss is near zero
  regardless of what it predicts — nothing in the objective ever shapes it.
  Pooling `instance_posterior` naively (majority vote or plain mean) for
  bag classification therefore lets exactly those never-individually-
  supervised instances dominate the aggregate. Confirmed empirically: two
  independently-trained folds showed excellent attention-pooled training
  loss alongside chance-level accuracy from the naive pooling, on the same
  trained model; switching to `bag_posterior` resolved it (one fold reached
  1.00 held-out inner-validation accuracy once correctly evaluated).

Consequently, **bag-level evaluation for Tier 3 does not read the shared
store.** `pipeline/mil_attention_stage.py` computes `bag_posterior` on each
fold's true held-out test bags immediately after training (before the
model is discarded — it cannot be reconstructed from the store afterward)
and writes it to a separate artifact,
`results/{exp}/latent/attention_bag_level.json`, one attention-weighted
decision per held-out bag per fold. `scripts/stage6_attention_bag_eval.py`
reads this directly (with the same cross-fold duplicate-bag averaging as
elsewhere, §2) for the bag-level comparison against Tier 0 — a genuinely
different evaluation path from Tiers 0/1/2/4's shared
`scripts/stage6_bag_eval.py`, for a principled, documented reason, not an
inconsistency.

---

## 7. Tier 4 — generative MIL (CbR-promoted)

Two variants, both pure inference — no training beyond what the `cvae`
pipeline stage already does — reusing the same per-fold CVAE weights
Classification by Reconstruction already uses.

**`reconstruction`.** Classification by Reconstruction's own evidence,
turned into a posterior instead of a hard decision. Per instance: encode to
$z_{\text{mean}}$, decode under each of the $K$ one-hot class conditions,
MSE against the original (cropped) spectrum, then
$\mathrm{softmax}(-\text{MSE}/T)$ over the $K$ classes. Temperature
$T{=}0.001$, calibrated (not guessed) from a real measurement of typical
inter-class MSE gaps on held-out data — at $T{=}1$ the resulting posterior
is within floating-point noise of uniform, since real per-instance MSE
differences at this data's scale are tiny (median gap $\approx 8\times10^{-4}$).

**`encoder_head`.** The CVAE encoder's own auxiliary softmax classifier
head (`src/models/cvae.py`'s `class_output`), trained during ordinary CVAE
fitting via the $\alpha{=}0.2$-weighted multi-task loss and otherwise
discarded by CbR (which takes only the encoder's $z_{\text{mean}}$ output).
Read directly — no decoder pass needed at all.

Both lifted to $(K{+}1)$ via the same $1-\max_k p_k$ convention as Tier 1
(§1), from a genuine $K$-way softmax in both cases.

**A known limitation of that convention at $K{=}2$.** For a joint softmax
(as opposed to Tier 1's independent per-class scores), $\max_k p_k \ge 0.5$
by construction, so the renormalised non-diagnostic mass is provably bounded:
$$\frac{1-\max_k p_k}{2-\max_k p_k} \in \Big[0,\ \tfrac{1}{3}\Big] \quad (K{=}2)$$
— never reaching the conventional 0.5 "is this diagnostic" threshold, at
any confidence level. This makes a *hard*-threshold external-agreement check
(ARI/AMI against an external reference) uninformative for both variants —
not a finding about the CVAE, a property of the threshold. The general bound
is $(K-1)/(2K-1)$, loosening toward but never reaching 0.5 as $K$ grows — a
consideration for any future tier whose instance posterior is a plain
$K$-way softmax, Tier 3 included. The *continuous* non-diagnostic mass this
convention produces is unaffected and is what the spatial-coherence
statistics (§9) use directly.

---

## 9. Bag-level classification evaluation

**Shared machinery** (Tiers 0/1/4, `src/utils/statistical_evaluation.py`):
`aggregate_bag_scores_across_folds` pools each fold's out-of-sample
instances to a bag decision under a chosen rule (§4's four rules for Tier 0;
mean-pooling of the $(K{+}1)$ posterior's diagnostic columns, renormalised,
for Tiers 1/4 — column 0 answers a different question and is excluded
from this comparison specifically), respecting the cross-fold duplicate-bag
averaging (§2) throughout, never pooling instances first and losing the
fold boundary. `bag_level_bootstrap`: a class-stratified bootstrap
(1000 resamples) over *bag* decisions — not instances — for the primary
comparison's CI and $p$-value, implemented by treating each bag as its own
singleton "cluster" in the pre-existing slide-level bootstrap machinery.

**Tier 3's own path** is §6.4's `bag_posterior`-based
`attention_bag_level.json` plus `scripts/stage6_attention_bag_eval.py`,
using the identical bootstrap/duplicate-averaging machinery on that
artifact instead of the shared store.

**Reference model.** By default the 1D-CNN (TRTR, majority vote) — but note
the CNN is, by a wide margin, the weakest of the four Tier 0 classifiers at
this task; Apollo/RF/XGBoost are meaningfully stronger and should also be
reported (`--reference-model {apollo,rf,xgb}` on both bag-eval scripts) —
reporting only the CNN comparison would be selective.

### 9.1 A circularity trap specific to Model (C)-constrained methods

`scripts/stage6_bag_eval.py`/`stage6_attention_bag_eval.py` both compute a
diagnostic fraction — the share of instances where more than one diagnostic
column is simultaneously non-negligible — before trusting a bag-accuracy
number at all. Near zero means the method hard-restricts every instance to
its own bag's already-known class throughout scoring (Model (C) enforced at
inference, not just fitting), making the resulting
"accuracy" a trivial 1.0000 that recovers the given label rather than a
learned one. This is checked automatically, not assumed from the method
name, because it is not visible from the accuracy number alone.

---

## 10. Spatial validation

Applies identically to any tier's stored posterior (`src/evaluation/spatial.py`,
`scripts/stage5_spatial_validation.py`) — models never see `coords` during
fitting (a permanent guard via `inspect.signature` asserts this for every
formulation), so any spatial structure in the reprojected posterior is
evidence the model recovered something real, not an artefact of an
objective that had geometry available to exploit.

**Reprojection.** Stored `(col, row)` coordinates place each instance's
non-diagnostic mass (column 0) back onto its slide's $(H,W)$ pixel grid.

**Statistics, by Chebyshev lag 1–10, not pooled into one number.** The
project's $3\times3$ spatial median filter (kept — signal cleaning, not a
concern here) couples a pixel only to its 8 neighbours, so two pixels at
Chebyshev distance $\ge 3$ share no input pixel and the filter provably
cannot manufacture correlation between them. Lags 1–2 are shown but
excluded from any claim; **lag $\ge 3$ is the headline.**
- Moran's I — scale-free, good at detecting pattern, can mislead about how
  much it matters (confirmed: a numerically-constant map can still report
  large Moran's I z-scores purely because it is scale-free — trust the
  variogram when the two disagree).
- The empirical variogram, $\gamma(h) = \tfrac12\,\mathrm{mean}((x_i-x_j)^2)$
  — scale-*sensitive*, reads exactly zero for a genuinely constant map.
- Permutation null: shuffle values within each slide's own mask (geometry
  held fixed), 200 repeats, per-lag $p$/$z$.
- **Variogram range** (first lag reaching $\sim$90% of the plateau value)
  is itself diagnostic: no independent, validated ground truth exists for
  how large a real diagnostic region should be in pixels (§1), so the range
  is reported as a descriptive fact, not compared against an assumed
  target.

**External agreement.** ARI/AMI between a hard diagnostic/non-diagnostic
call (non-diagnostic mass $<0.5$) and `viable_legacy` — the project's
earlier unsupervised cluster selection, used here only as an independent
*reference* computation from a different pipeline, explicitly **not**
ground truth (§1).

### 10.1 The shuffled-bag-label control

The definitive test of whether a lag $\ge 3$ finding is genuinely tied to
the true label. `src/mil/shuffle_control.py::permuted_bag_label_map`
computes an **exact permutation** of the true label multiset across bags
(not independent resampling, so class balance is identical to the real
run) — one permutation, computed once, so a bag spanning two CV folds keeps
the same permuted label in both. `scripts/stage6_shuffled_label_control.py`
substitutes this for the loader's own label array *before* any fold
splitting, then reuses the target formulation's ordinary fitting code
completely unchanged. If lag $\ge 3$ coherence **survives** — is smaller
under permuted labels than under the true ones — the real finding is
credited to the true label. If a permuted-label refit produces
comparable-or-larger coherence, the real finding cannot be attributed to
the true label at all, and must be explained some other way (e.g. a
formulation-specific mechanism that produces lag-band structure from raw
spatial autocorrelation in the features, independent of which label a
region happens to be fit toward).

### 10.2 The key-instance utility test

A second, independent falsification test, and the one that is *cheapest to
run* — no retraining of the MIL formulation, no GPU, no permutation. It asks
the question spatial coherence cannot: **are the instances a formulation
selects actually the informative ones?**

Spatial coherence says a formulation's per-instance posterior has structure
on the slide. It does not say that structure is *diagnostic* structure.
Raman maps of tissue are spatially autocorrelated in many ways that have
nothing to do with tumour class — acquisition gradients, tissue folds,
stroma/necrosis boundaries — so a posterior can be smooth, significantly
non-random at lag $\ge 3$, and still be selecting on something clinically
irrelevant. This test closes that gap by checking downstream utility
directly:

1. Take the formulation's own per-instance posterior and binarise it into
   "key" vs "not key" (the threshold is formulation-specific; for a $K{=}2$
   joint softmax use a **median split** on non-diagnostic mass, *not* an
   absolute 0.5 — §7's bound makes 0.5 structurally unreachable, which
   silently turns the filter into a no-op).
2. Fit Tier 0's own RF — same hyperparameters, same raw spectral features,
   nothing else changed — on **only** the key training instances.
3. At inference, apply the same threshold to decide which validation
   instances are key, score only those, and pool to a bag decision with
   Tier 0's ordinary majority vote.
4. Compare against Tier 0's unmodified RF (all instances, equal vote) with
   the same bag-level bootstrap as §9.

**Reading it.** If the selected instances carry the diagnostic signal, a
classifier restricted to them should match or beat the one that votes over
everything — the discarded instances were supposed to be noise. If it comes
out **worse**, the selection is actively removing diagnostic content, and
whatever spatial structure the formulation shows is not the structure the
task depends on. A null result (no difference) is weak evidence either way:
it is consistent with "the selection is uninformative but harmless," since
a strong classifier voting over thousands of instances per bag is robust to
losing half of them.

**Two confounds to state whenever this is reported.** (i) The restricted RF
trains on fewer instances (half, under a median split), so a deficit
conflates "selection removes signal" with "less training data hurts" — a
random-subset control of the same size separates them and should be run
before any strong claim. (ii) A bag whose key set is empty needs a
documented fallback (revert to all its instances), and the number of bags
that hit it must be reported, not silently absorbed.

Two weaker variants of the same idea reuse the *already-trained* Tier 0 RF
and change only the pooling, which removes confound (i) entirely at the cost
of testing a weaker claim (does the posterior improve aggregation, rather
than does it identify the informative instances): a **weighted vote**, where
each instance's vote is weighted by the formulation's per-instance class
probability, and a **key-instance vote**, where only key instances' existing
votes are counted. Implementation of all three:
`scripts/cvae_weighted_rf_vote.py`.

---

## 11. What is not yet covered by this reference

- **Stage 7 (CVAE soft conditioning)** is not started; no methodology to
  document yet. **Stage 8 (mixture model / 6-class)** is not in scope; a
  single exploratory 6-class run (docs/beta_sweep_findings.md §5) changed
  only $K$, not the formulation.
- **Tier 3's shuffled-label control** has not been run (would need a full
  per-fold retrain, cheap relative to Tier 4's — no CVAE retrain required,
  since the trunk is frozen and reused as-is).
- **Tier 4's shuffled-label control** would need a full per-fold **CVAE**
  retrain (the trunk itself, not just a downstream head) — this project's
  first real GPU training run; not started, pending a cheap Modal GPU
  validation first per standing project practice.
