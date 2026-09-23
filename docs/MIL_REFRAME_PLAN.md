# Reframing as weakly-supervised MIL with latent tissue composition

> **Living document.** This is the execution plan for the MIL reframing, worked one stage at a time
> per the workflow in §9. Update the Progress table and the affected stage's notes as each stage
> lands — including deviations from the plan and why.
>
> - **Branch**: `mil-reframe`, cut from `thesis` at `517db47` (dissertation-era state frozen).
> - **Current stage**: 1 — data layer and re-export. (Stage 0 complete, see
>   [stage0_findings.md](stage0_findings.md); its verdict changed the corrector, so
>   §2.1 below records the original reasoning and the findings doc records what won.)
> - **Thesis is out of scope.** The paper is in scope and is updated stage by stage, never batched.

## Context

The paper was rejected. The reframing below keeps the existing work (CVAE generation, the
four-classifier benchmark, the slide-level bootstrap) but fixes the assumption underneath it.

**The false assumption.** Every classifier in the repo is trained on `(spectrum, slide_label)`
pairs — `pipeline/_common.py:45-96` hands each spectrum its slide's grade verbatim. That asserts
every spectrum in a slide carries that slide's tumour grade. It does not. The mask that produced
the modelling data is a **biologically-viable-signal** mask (NMF(100) → GMM(3) → manual cluster
selection in `data/processed/file_names.csv`), not a tumour mask. The `y_tumor` dataset name is a
misnomer, and in `results/exp_*/data/glioma_dataset.h5` it is a constant `1` for all 227,344 rows —
carrying no information. **There is no instance-level label anywhere in this dataset. Only bag
labels.**

So the problem is Multiple-Instance Learning with a latent per-instance tissue composition, and
that is what the reframed paper should say.

**What we gain by saying it.** Three contributions instead of one:

| | Task | Output | Why it is new |
|---|---|---|---|
| T1 | Bag classification | slide grade | the actual clinical task; currently never evaluated as such |
| T2 | Latent instance inference | per-spectrum posterior over {non-diagnostic, LGm1..6} | a tumour-probability map per slide, validated without spatial input |
| T3 | Conditional generation | CVAE conditioned on the inferred latent | the rejected contribution, made correct — the current CVAE is asked to generate "LGm-1" from a set that includes non-diagnostic tissue |

**Decisions taken.** Thesis unchanged, paper reframed. Re-export the HDF5 under the new formulation,
and revisit baseline correction while it is open (§2.1). The latent distribution is **not** inferred
at preprocessing time but by the MIL models, with **every formulation's posterior stored** for later
comparison (§3). Constrained latent support first, full mixture as extension; binary Mutant/WT
first, 6-class as extension. The diagnostic fraction $\pi_b$ is **inferred hierarchically**, not
fixed (§1). The non-diagnostic component is **multi-modal**, $M{=}3$ (§1). The **CV schema is
retained unchanged** and documented rather than rebuilt (§7.1). The spatial median filter **stays** —
it is signal cleaning, and the spatial confound is handled by a lag-stratified statistic instead
(§6). Baseline λ is selected **per spectrum and label-free**, preserving the project rule that no
parameter is ever fitted across the dataset (§2.1). The 1D-CNN is **fixed, not dropped** (§7.2).
Bag-level CIs are the headline uncertainty. `patient_cap` is retired in favour of explicit
bag-size-aware pooling (§4). This runs on a dedicated branch with a codebase-wide terminology change
(§11).

---

## 1. Problem statement

Slide $b$ (a bag) has a known label $Y_b \in \{1..K\}$ and contains $n_b$ viable spectra
$\{x_{b1}..x_{bn_b}\}$ (instances). Each instance has an unobserved tissue type
$z_{bi} \in \{0_1..0_M, 1..K\}$, where $0_m$ are **$M$ non-diagnostic components**.

**The null is multi-modal.** Stroma, necrosis and normal brain are spectrally distinct, so a single
null component is too rigid. Set $M = 3$, matched to the existing NMF/GMM tissue clusters — which
also gives the ARI validation in §6 a component-wise target rather than a binary one. If the
multi-modality assumption does not hold the components collapse into near-duplicates, which is a
benign failure and an observable one: report pairwise cosine between null-component means and the
effective number of components, so collapse is visible rather than silent.

- **Constrained model (C).** $z_{bi} \in \{0_1..0_M, Y_b\}$ — a slide graded $k$ contains only
  non-diagnostic tissue plus grade-$k$ tumour. Diagnostic/non-diagnostic is binary per instance;
  the null is subdivided into $M$ components shared across all bags. Well-posed with 58 bags.
- **Mixture model (M).** $z_{bi}$ ranges over all $K+1$ types — admits intratumoral heterogeneity.
  Weakly identified from 58 bag labels; needs the priors in §5.

Both are the same machinery at different support, so (C) is (M) with hard zeros — implement (M)'s
interface and constrain it, rather than writing two codebases.

### Diagnostic fraction $\pi_b$ — inferred, not fixed

$\pi_b = \tfrac{1}{n_b}\sum_i [z_{bi} = Y_b]$ is the fraction of slide $b$ that is diagnostic
tumour. It genuinely varies across slides — plausibly 2% to 30% — so fixing one global value would
be the wrong model, not merely a conservative one. **Infer it, hierarchically:**

$$\operatorname{logit}(\pi_b) \sim \mathcal{N}(\mu, \sigma^2), \qquad (\mu, \sigma) \text{ estimated across the 58 bags}$$

The shared prior is what makes per-bag inference possible at this sample size: individual $\pi_b$
are weakly identified, and partial pooling borrows strength across bags instead of pretending each
one is estimable alone.

$\pi_{\min}$ survives only as a **numerical identifiability floor** (sweep 1–5%) preventing the
all-null collapse in §5. It is not a biological claim and should not be described as one in the
paper. Under (C), $\pi_{bk} = 0$ for $k \notin \{0_1..0_M, Y_b\}$.

**Two diagnostics decide whether inference actually worked:**

1. **Shrinkage check** — compare the spread of inferred $\pi_b$ against the prior spread. If
   shrinkage is near-total, the model is returning $\mu$ for every bag and there is no per-bag
   signal; report this honestly rather than presenting prior means as findings.
2. **External correlation** — regress inferred $\pi_b$ against independent proxies: the fraction of
   the slide in the legacy manually-selected clusters (`viable_legacy`), the per-slide NMF cluster
   proportions, and any pathology annotation available. **If inferred $\pi_b$ tracks an independent
   proxy, that is stronger evidence than spatial coherence** — it is the one place a real external
   quantity is available. Worth prioritising.

Expect $\sigma$ to be poorly determined with 58 bags; report a profile-likelihood interval on both
$\mu$ and $\sigma$ rather than point estimates.

---

## 2. Data layer

### 2.1 Baseline correction — replace the inherited constant

> **SUPERSEDED by the Stage 0 result.** The reasoning below (per-spectrum GCV, IARPLS
> "not taken") is kept as the record of why the study was run. The study rejected it:
> GCV pinned at the grid floor in 83% of spectra, and IARPLS beat airPLS's best
> achievable fixed λ while being inert over λ∈[1e2,1e4]. Production path is now
> `correct_iarpls` at λ=1e4. See [stage0_findings.md](stage0_findings.md).

airPLS with `lambda=100, porder=1` is currently inherited from Lita et al. (Apollo) and was tuned
by eye on a handful of spectra. Since the export is being redone, fix this properly. Baseline
choice changes every downstream number, so it must be settled **before** the export, not swept
afterwards.

**The governing constraint: no parameter is ever fitted across the dataset.** Verified — the current
pipeline already honours this. `_preprocess_spectrum` (`src/data/preprocessing.py:141-153`) runs
airPLS and min-max **per spectrum**; no statistic crosses spectra. The only aggregate is the 5th-
percentile threshold in `_create_bio_mask:88-89`, computed per slide and label-free. Preprocessing
is therefore fold-independent and leakage-clean today, and λ selection must preserve that.

That rules out a globally-fitted λ, and it rules out per-training-fold tuning too (correct in
principle, but it makes preprocessing fold-dependent, forces five exports, and is hard to explain).
**So λ becomes per-spectrum and label-free.**

**Decided: per-spectrum λ by generalised cross-validation.** Settled before anything else, using an
established method rather than a bespoke one.

airPLS's inner smoother is a Whittaker smoother, and GCV for Whittaker smoothers is standard and
citable — Craven & Wahba (1979) for GCV itself, Eilers (2003, *A perfect smoother*) for the
Whittaker case, which gives the closed-form GCV score for exactly this penalised-least-squares
system. So this is not a new selector, it is the textbook one applied where an arbitrary constant
currently sits.

Procedure per spectrum: evaluate the GCV score over a log-spaced λ grid, take the minimiser, then
run airPLS's asymmetric reweighting at that λ. L-curve (Hansen) as the fallback if GCV proves
unstable on low-SNR spectra — check this during the Phase 0 validation, since GCV is known to
occasionally under-smooth at low SNR.

**Compute**: this multiplies the export cost by roughly the grid size. Each solve is a banded sparse
system over 1738 points and already runs 15 reweighting iterations, so a 10–12 point grid is on the
order of a few times the current cost for a one-time batch job, and it parallelises per slide.
Budget for it rather than being surprised.

**Not taken: IARPLS.** `_preprocess_batch_ramanspy:164` already uses
`rp.preprocessing.baseline.IARPLS(lam=10000)`, so the two code paths in this repo currently disagree
on λ by a factor of 100 — which is itself good evidence that a fixed λ is not well determined. Keep
IARPLS as a comparator arm in the Phase 0 study, not as the production path.

**This dissolves the comparability problem.** With a self-tuning per-spectrum procedure there is no
global constant whose change invalidates old results — just a documented method, with **airPLS
λ=100 retained as a fixed-λ ablation arm** for continuity with Lita et al.

**The simulation study changes role.** It no longer picks a constant; it **validates the selector**.
Take high-SNR corrected spectra, add synthetic baselines from a realistic family (low-order
polynomials plus broad Gaussians mimicking fluorescence, spanning the fluorescence levels actually
present in the data), and measure recovery RMSE plus **peak fidelity** — height and area stability
at 1004 (Phe), 1440 (CH₂), 1660 (amide I), 2850–2950 (CH). Does the per-spectrum selector recover
the known baseline across that range, and does it beat fixed λ=100? That is a cleaner claim than
"we picked λ=100" and it makes a publishable methods figure.

**Delete the RamanSPy path.** `_preprocess_batch_ramanspy` is unused; it also applies SavGol spectral
denoising, contradicting the spatial-only design (§6). Remove it rather than maintaining two
philosophies. Its `IARPLS` call remains useful as a one-off comparator in the Phase 0 study, then
goes.

**Settled**: baseline correction runs over the **full 1738-band range, before** the 700–1800 crop —
current behaviour, confirmed. A wider crop for learning is a later-phase question (§9.1).

### 2.2 Re-export

`src/data/preprocessing.py` is currently non-functional (four independent defects: `gaussian_filter`
not imported at `:268`; `_build_file_to_map_correspondence` at `:209` does not exist; the
`tumor_maps` key lookup appends `.npy` and never matches; `pipeline/preprocessing_stage.py:36`
passes a `config=` kwarg `__init__` does not accept). The real producer is notebook 01 §4, mirrored
in `_archive/dev_scripts/export_hdf5_section4.py`.

**Fix `src/data/preprocessing.py` into the single producer** and re-export with a wider mask and
richer side tables. Export **all strong-signal pixels** (`strong_signal_mask.npy`, already on disk
per slide), not just the manually-selected clusters — masking becomes a load-time decision like
cropping, so mask choice can be ablated without re-exporting.

New schema (`data/processed/glioma_mil.h5`):

| dataset | shape | notes |
|---|---|---|
| `spectra` | (N, 1738) | uncropped; crop stays a load-time op |
| `coords` | (N, 2) int32 | **already exists**, convention `(col, row)` = `(axis1, axis0)` |
| `sample_ids` | (N,) str | bag id, e.g. `HF-2104_V1T_1.h5_2` |
| `patient_ids` | (N,) str | **change**: currently int32; store the real `HF-\d+` id |
| `y_bag` | (N,) int32 | **rename** from `y_class` — names the bag-level nature |
| `tissue_cluster` | (N,) int32 | NMF/GMM 3-cluster id from `hard_labels.npy` — weak prior + external validation target |
| `viable_legacy` | (N,) bool | was this pixel in the old manual "tumour" cluster selection? preserves the old subset exactly |
| `slide_names` | (S,) str | **new** side table |
| `slide_shapes` | (S, 2) int32 | **new** — `(H, W)` per slide; currently recorded only in `data/processed/<slide>/metadata.json` |
| `nmf_W` | (N, 100) float32 | optional (~91 MB); compact features for MIL + spatial analysis |

Drop `y_tumor` entirely — it is a misnomer and is constant.

### 2.3 Loader

`src/data/loader.py` drops `coords` on the floor (`_load_data:152-200` never opens it). Add it, and
**apply every existing mask to it** — `:192-195` (tissue filter) and `:340-345`
(`apply_patient_cap`) — or it silently desynchronises. Yield `val_coords`, `val_groups`,
`train_coords` from `__iter__`; `train_groups` is already yielded at `:386`, add `val_groups`
symmetrically. Row order is preserved throughout (`keep_indices` is `np.sort`ed at `:337`, no
shuffle anywhere), so positional alignment holds.

`prepare_fold` (`pipeline/_common.py:45-96`) returns a fixed 4-tuple; widen it to a dataclass or
dict so coords/bag-ids propagate without threading more positional returns.

### 2.4 Reprojection helper (verified correct)

```python
def reproject(values, coords, shape, fill=np.nan):
    img = np.full(shape, fill, dtype=float)
    img[coords[:, 1], coords[:, 0]] = values      # row = coords[:,1], col = coords[:,0]
    return img
```
Round-trip verified exact against the tumour masks on three slides of differing aspect ratio. Do
**not** copy the indexing in `_archive/dev_scripts/consolidate_tumor_masks.py:82-83` — it transposes
row/col and is only correct for square slides.

---

## 3. Latent-label store

Every MIL formulation produces its own estimate of the latent distribution. Store all of them.

**Sidecar file** `results/{exp}/latent/latent_labels.h5`, not the main dataset — the main HDF5 is
shared across experiments and must stay immutable.

```
/{method}/{variant}/{fold_name}/
    posterior   (N, K+1) float32     # rows aligned 1:1 to the main HDF5 row order
    in_sample   (N,)     bool        # was this spectrum in this fold's MIL training bags?
  attrs: method, variant, K, constrained, config_hash, git_sha, timestamp,
         alignment_checksum          # hash of (sample_ids, coords) from the source HDF5
```

**Leakage rule — this is the load-bearing part.** Index by fold. A spectrum in fold $f$'s
*validation* slides gets its posterior from the fold-$f$ model, which never saw it. The CVAE for
fold $f$ conditions on fold-$f$ posteriors over fold-$f$'s *training* slides (in-sample by
necessity, and legitimate — it is train-side). Any cross-fold reuse of posteriors reintroduces the
leakage the CV design was built to prevent. Enforce with an assertion, not a convention.

`alignment_checksum` guards against a silent re-export invalidating stored posteriors.

---

## 4. The model family

### Bag size is heterogeneous — and `patient_cap` goes away

Measured on `results/exp_201/data/glioma_dataset.h5`: 59 bags, 227,344 instances, bag size
**865 → 14,536, a 16.8× spread** (q1 2,799 / median 3,600 / q3 4,078).

`patient_cap = 500` was papering over this. It is retired — it was a preprocessing hack standing in
for a modelling decision, and under the reframing the decision has to be explicit:

- **Mean pooling** over 865 instances and over 14,536 instances have very different variance. An
  unweighted mean treats them as equally reliable bag estimates; they are not.
- **Top-$q$ pooling** changes meaning entirely with bag size — fixed-$q$ favours large bags
  (more chances at a high-scoring instance), fractional-$q$ does not. Use a **fraction**, and say so.
- **$\pi_b$ estimation** (§1) is much better determined for large bags. The hierarchical prior
  should shrink small bags harder — which it does automatically if the likelihood is written with
  the correct per-bag $n_b$, and not if bags are pre-capped to equal size.

Replace `cv_strategy.patient_cap` with an explicit, model-side **instances-per-bag budget** used
only where compute demands it (mi-SVM iterations), plus sub-bagging for the neural tier. Record the
true $n_b$ regardless, so bag-size-aware pooling and the $\pi_b$ likelihood both see it. Report bag
size as a covariate in the results — if accuracy tracks bag size, that is worth knowing.

### Tier 0 — aggregation baselines (no new training)

The existing TRTR/TSTR/TSRTR models already emit per-instance posteriors; `_gather_probas`
(`pipeline/evaluation_stage.py:92-98`) already collects them as `(N, K)`. Pool them:

| rule | bag score |
|---|---|
| mean-posterior | $\arg\max_k \bar p_{bk}$ |
| majority vote | $\arg\max_k \sum_i [\hat y_{bi} = k]$ — **already computed** at `statistical_evaluation.py:242`, then discarded except for the confusion matrix |
| log-sum-exp | smooth max, one temperature parameter |
| top-$q$ mean | mean of the $q$ highest-scoring instances — the "witness fraction" baseline |

Near-free, and it gives the naive-MIL reference row the reframed paper needs. Register the instance
posteriors as method `naive` in the latent store: they *are* a latent estimate, just one made under
the false assumption. Everything else must beat it.

### Tier 1 — mi-SVM / MI-SVM (Andrews et al. 2002)

One-vs-rest per class $k$: positive bags = slides labelled $k$, negative bags = all others. This is
classic MIL and maps cleanly onto the constrained model. Reuse **Apollo** (`src/models/baselines.py`
— RF top-20 band selection + RBF SVM) as the base learner, so the feature pipeline is unchanged and
comparable to the existing benchmark.

- **mi-SVM**: alternate (i) fit on current instance labels, (ii) relabel instances subject to
  $\ge \pi_{\min}$ positives per positive bag and zero in negative bags.
- **MI-SVM**: bag represented by its witness (most positive instance) — sparser, yields an explicit
  witness set that is directly interpretable on the slide map.

Instance posterior: Platt-scaled margins across the $K$ one-vs-rest problems, with the
non-diagnostic mass as $1 - \max_k p_k$ (or a learned null score — see §5).

Scale: with `patient_cap=500` the training set is ~29k instances for 58 slides, which an RBF SVM
handles per iteration. Cap iterations and log the instance-label churn per round.

### Tier 2 — EM / label-proportion model

Explicit responsibilities $\gamma_{ik}$ with a discriminative M-step, constrained by the bag
proportions. This is the formulation that yields **calibrated** soft labels, which is what T3 needs
— mi-SVM's Platt scores are not calibrated across bags. Expect this to be the one that feeds the
CVAE.

### Tier 3 — neural attention MIL (Ilse et al. 2018, gated)

**Use attention over instance *logits*, not over embeddings.** Embedding-level attention gives you
attention weights, which are a relevance score, not a class posterior — and T2 needs a posterior.
Additive/CLAM-style instance heads keep the instance layer interpretable.

Two mitigations are mandatory with 58 bags, not optional:

1. **Sub-bagging.** Sample random sub-bags per bag per epoch → thousands of pseudo-bags. This is the
   difference between a trainable attention model and one that memorises 58 labels, and it reuses
   the per-epoch resampling already at `pipeline/cvae_stage.py:142-152`. **But see §4.1 — a sub-bag
   can contain zero diagnostic instances, which makes its inherited label simply wrong.**
2. **Frozen pretrained trunk.** Use the per-fold CVAE encoder (`encoder_{fold}.weights.h5`, already
   on disk) as the instance encoder. Reuses existing artifacts and removes most of the parameters
   from the bag-supervised gradient.

### 4.1 Sub-bagging breaks the bag label — size it from $\pi_b$

A sub-bag drawn from a bag labelled $k$ can contain **zero** instances of class $k$, in which case
its inherited label is simply wrong. The full bag is safe by the §1 constraint (it contains at least
$\pi_{\min} n_b$ diagnostic instances); a sub-bag is not. **Sub-bagging trades label fidelity for bag
count**, and that trade has to be priced rather than assumed away.

For a uniform sub-bag of size $s$, $\;P(\text{all non-diagnostic}) \approx (1-\pi_b)^s$:

| $\pi_b$ | $s=64$ | $s=256$ |
|---|---|---|
| 5% | 3.8% | ~0% |
| 2% | 27% | 0.6% |
| 0.5% | 73% | 28% |

At the low end this is not a rounding error — **more than a quarter of sub-bags carry a wrong label**.
Four consequences for the design:

1. **Sample uniformly at random across the whole bag — never contiguous spatial patches.** Diagnostic
   instances are spatially clustered, so a contiguous patch is far more likely to be entirely
   non-diagnostic than the table suggests; random sampling is what makes the binomial rate apply at
   all. It also preserves the withheld-coordinates property the §6 validation depends on.
2. **Size $s$ from the floor, not by guesswork**: $s \ge \ln \varepsilon / \ln(1-\pi_{\min})$. With
   $\pi_{\min} = 2\%$ and $\varepsilon = 1\%$ this gives $s \ge 228$ — so a 256-instance sub-bag is
   calibrated to a 2% floor, not an arbitrary choice. This ties the sweep's centre to §1.
3. **Size adaptively per bag once $\hat\pi_b$ exists.** Tier 1/2 runs before Tier 3 in the staging,
   so set $s_b \ge \ln\varepsilon / \ln(1-\hat\pi_b)$ — bags estimated to be mostly non-diagnostic
   get larger sub-bags. This falls out of the phase order already chosen.
4. **Model the residual noise.** Even at $\varepsilon = 1\%$ some sub-bags are mislabelled, and
   always in one direction — a positively-labelled sub-bag with no positives, never the reverse
   under (C). That is asymmetric label noise at a **computed, not guessed**, rate: apply label
   smoothing on the bag target at $\varepsilon$, or an explicit one-sided noise model.

**Risk to avoid**: importance-sampling sub-bags by the current instance posteriors would cut
$\varepsilon$ sharply, but it is confirmation bias — the model reinforcing its own beliefs. If used
at all, floor it with a fixed fraction of uniform sampling and report both arms.

**Diagnostic to report**: the empirical fraction of sub-bags with zero diagnostic mass under the
final model. If it materially exceeds $\varepsilon$, the sizing was miscalibrated and the Tier-3
numbers are not trustworthy.

Note this pushes sub-bag size well above typical MIL practice. Whole-slide-image MIL usually works
with small bags; here $\pi_b$ is small, so bags must be large.

### Tier 4 — generative MIL (promotes the existing CbR)

`_cbr_predictions` (`pipeline/evaluation_stage.py:101-146`) already computes
`mse (n, K)` = per-instance class-conditional reconstruction error. `softmax(-mse/T)` is an instance
posterior for free; pool it over the bag. This turns the existing Classification-by-Reconstruction
into a principled MIL method and ties the generative contribution to the MIL story.

Note the CVAE encoder also has an unused supervised softmax head (`src/models/cvae.py:91`, trained
with $\alpha{=}0.2$) that CbR discards by taking `[0]` at `evaluation_stage.py:131`. That is a
second ready-made instance classifier — register it as a Tier-4 variant at zero training cost.

---

## 5. Degeneracies to guard against

The likely reason a referee kills this. Name each failure mode and its guard explicitly in the paper.

| degenerate solution | guard |
|---|---|
| all mass on non-diagnostic ($\pi_b \to 0$) | identifiability floor $\pi_{\min}$ (1–5%, swept); this is the floor's only job |
| collapse to the naive solution ($\pi_b \to 1$) | the hierarchical prior on $\operatorname{logit}\pi_b$ plus its shrinkage check (§1) |
| a null component absorbs a whole tumour class | the $M$ null components are **shared across bags of all classes** — sharing is what identifies them |
| the $M$ null components collapse into duplicates | benign, but must be visible: report pairwise cosine between component means and the effective component count |
| mixture model (M) permutes components arbitrarily | anchor by the bag-label constraint $\arg\max_k \pi_{bk} = Y_b$ |

Report the fitted $\pi_b$ per slide as a headline figure, with its external correlation (§1). If the
model says 95% of every slide is diagnostic tumour, it has learned nothing beyond the naive model —
and if it returns the same $\pi_b$ for every slide, it has learned nothing at all.

---

## 6. Validation — and the confound that must be addressed

The design is: withhold `coords` from every model, infer the latent posterior, reproject onto the
slide grid, and ask whether the map is spatially coherent. Coherence is then *evidence* rather than
an artefact of the objective.

- **Moran's I** on the posterior map, permutation null by permuting values within the slide's own
  mask (preserves mask geometry). Per slide and pooled.
- **Join-count / boundary length** of the thresholded map against the same null.
- **External agreement**: ARI/AMI between the learned hard assignment and the NMF+GMM 3-cluster
  `hard_labels.npy` — an unsupervised tissue segmentation from an entirely separate pipeline. This
  is the strongest available check and needs no new labels.
- **Qualitative**: overlay against `data/rgb_img_bio_markers/<slide>.png`.

### The confound, and why it does not require dropping the filter

The 3×3 spatial median filter (notebook 01 cells 4/15/21; `Thesis/4-methodology.tex:64`,
`paper/main.tex:377`) is there for **signal cleaning** — Gaussian/shot noise and cosmic rays — and a
median preserves edges far better than a mean would. It stays. But it does couple neighbouring
pixels, so a naive Moran's I against a permutation null would be significant for trivial reasons.

**The fix is a lag-stratified statistic, not an ablation.** A 3×3 median couples a pixel only to its
8 neighbours. Two pixels at Chebyshev distance ≥ 3 share **no input pixel whatsoever**, so the
filter provably cannot manufacture any correlation between them. Therefore:

- Report the **empirical variogram / Moran's I by lag band**, not a single pooled number.
- Lags 1–2 are contaminated (windows overlap at Chebyshev distance ≤ 2) — show them, but exclude
  them from the claim.
- **The headline statistic is coherence at lag ≥ 3**, which the filter cannot produce.
- Real tissue architecture operates at tens of pixels. If the variogram range is 1–2 px it is the
  filter; if it is 10–20 px it is tissue. The range itself becomes the reported evidence.

This is strictly stronger than an ablation would have been, and costs nothing.

**Still run the shuffled-bag-label control as the primary control.** Permute slide labels across
slides and refit. It holds preprocessing *completely* fixed and breaks only the label→latent link,
so it is the cleanest null available — and unlike a filter ablation it also catches coherence
arising from spectral smoothness, intensity gradients, or acquisition artefacts.

### Why spatial rather than spectral denoising — the methods-section argument

This reasoning is sound and should be written up explicitly rather than left implicit:

- A **cosmic ray** strikes one detector position in one exposure. In a 3×3 window it is 1 value among
  9, so the median rejects it completely. A median is precisely the optimal rejector of impulsive
  outliers that are sparse within the filtering window.
- A **genuine Raman band** is present in most or all of the 9 neighbours, so the median preserves it
  exactly — including sharp, narrow peaks. A mean would blur them; the median does not.
- **This is information a per-spectrum method does not have.** A spectral despiker must decide
  "artefact or real narrow band?" from one spectrum alone. The spatial filter separates *narrow and
  isolated* (artefact) from *narrow and shared with neighbours* (real). Strictly more information,
  and the right way to exploit the fact that neighbouring pixels are near-replicate measurements.

One correction to an earlier suggestion of mine: **Savitzky–Golay is a smoother, not a despiker.**
It does not remove spikes — it smears them across neighbouring bands, converting a localised
artefact into a broad distortion. That was a conflation on my part, and it argues *against* the
spectral route rather than for it. (Whitaker–Hayes despiking is a detector plus interpolator, a
different thing, but it still has strictly less information available than the spatial median.)

**State the cost plainly and accept it**: a 3×3 median requires a feature to occupy ≥5 of 9 pixels
to survive, so features confined to 4 pixels or fewer are removed. This is the intended behaviour —
it is what rejects cosmic rays — and biology of interest spans blobs far larger than 4 px. One
sentence in methods; no defensiveness needed.

The induced lag-1/2 coupling is already handled by the lag ≥ 3 statistic above, so the filter costs
nothing in validation strength.

Only once this passes should a spatially-regularised variant (CRF / graph smoothing over the pixel
lattice) be fitted — reported as a *model*, never as evidence, exactly as you said.

---

## 7. Evaluation reframing

The current metric stack conflates the two levels and has three outright bugs. Verified in
`src/utils/statistical_evaluation.py:236-305`:

- `accuracy`, `balanced_accuracy` and `recall_macro` are **literally the same float** (`:274-282`),
  equal to mean-within-bag instance accuracy. This is not a bag prediction — it is
  $\mathbb{E}_{\text{slide}}[\text{fraction of spectra correct}]$. Rename to
  `within_bag_instance_recall` and demote it to a diagnostic.
- It is **not class-balanced** despite the docstring at `:170`: `all_recalls` is flattened over
  (class, slide) pairs at `:271`, so each *slide* gets weight 1 and the class with more slides
  dominates. It is a slide-macro, not a class-macro.
- `precision_macro` is a **constant ≈0.5** — every slide is single-class, so per-slide precision for
  the slide's own class is always exactly 1.0.
- `kappa` is **always 0.0** — `cohen_kappa_score` on constant `y_true` (`:246-251`).
- `f1_macro` is **upward-biased and non-monotone in accuracy**: `:264-265` fires only when both
  `rec` and `prec` are non-None, so slides with 0% within-bag accuracy are silently dropped. This
  is how `results/exp_401` reports `f1_macro = 0.926` next to `accuracy = 0.431`. `f1_weighted` is
  the honest version.
- `paper/gen_appendix_tables.py:53-81` derives macro sensitivity/specificity from the
  **slide-count** confusion matrix while the neighbouring columns are instance averages — the
  appendix tables currently mix two units of analysis in one row.

### 7.1 The CV schema is retained — and must be documented, not changed

Same 5 folds, same patient split (`data/splits/manual_splits.json`). The 6-class extension needs
exactly that stratification, and keeping it preserves comparability with the thesis results. What
changes is the *reporting*, not the split. Three things go in the methods section:

1. **Grouping is patient-consistent.** The code groups on slide id, but the authored folds never put
   one patient's slides on both sides of a split — verified across all folds in both split files.
   State this as a property of the curated folds; the code does not enforce it.
2. **Four slides appear in two test folds each** (`HF-2104` ×3, `HF-592`), forced by 6-class
   stratification over classes with only 4 patients (class 2: 9 slides / 4 patients; class 5:
   5 slides / 4 patients). 62 (slide, fold) test pairs over 58 unique slides. Not leakage — a slide
   in fold $k$'s test set is correctly absent from fold $k$'s training set.
3. **This has a new consequence at bag level.** The instance-level code merged duplicates implicitly
   (`slide_to_indices` is keyed by slide id, so both appearances pool into one entry → 58). Forming
   one decision per *(slide, fold)* pair would give 62 bag decisions and double-count 4 slides.
   **Rule: average the duplicated slide's posteriors across its two folds, then decide once → n=58
   bags.** This is the natural extension of what the current code already does, keeps the bootstrap
   unit well-defined, and preserves continuity with existing numbers. It must be stated explicitly —
   a reframed paper describing a clean MIL protocol will otherwise be asked about it.

Keep reporting per-fold results as now, plus the pooled bag-level number with its CI.

**New structure — two clearly separated levels:**

- **Bag level** (the clinical claim): a stated pooling rule → one decision per slide → accuracy,
  balanced accuracy, AUROC over 58 bags. CI from the existing `stratified_slide_bootstrap`, now
  resampling bag *decisions*, which is its natural unit. Slide-level clustering for CIs is
  appropriate here and stays as-is.
- **Instance level** (no ground truth exists): report only spatial coherence, NMF agreement,
  cross-fold and cross-method stability of the posteriors, and downstream effect on generation.
  Never report an instance "accuracy" against the bag label — that number is what the reframing
  exists to retire.

### 7.2 The 1D-CNN baseline — fix it, do not drop it

It is currently undertrained and biased, and four concrete defects explain most of that:

| defect | location | consequence |
|---|---|---|
| early-stops on the **test fold itself** | `pipeline/_common.py:189-201` | leakage, not merely optimism — `restore_best_weights` selects on held-out data |
| `dropout_rate` never forwarded | `pipeline/_common.py:186` | the config's `0.6` is silently ignored; `build_1d_cnn`'s default `0.4` is used |
| compiled twice | `src/models/cnn.py:44-48` then `pipeline/_common.py:187` | `sparse_categorical_crossentropy` replaced by `categorical_crossentropy`; harmless but means the model file's own config is a lie |
| no augmentation at all, 30 epochs, patience 5 | `pipeline/_common.py:189-201` | undertrained on ~29k capped instances |

**Fix rather than remove**, because under the reframing the instance CNN becomes the Tier-3
attention trunk (§4) — the work is needed anyway, and a removed baseline is a question at review.

Fixes, in order:

1. **Inner validation split carved from *training slides*, group-aware.** Never the test fold. This
   is a leakage fix first and a bias fix second.
2. Forward `dropout_rate`; compile once.
3. Train longer, with patience tuned on the inner split.
4. **Turn on augmentation.** `src/data/augmentation.py` is dead code but holds the *better*
   implementations — `spectral_shift` (fractional, cubic interpolation) and `within_class_mixup`.
   The CVAE stage reimplements a cruder variant inline (a single integer `tf.roll` shared across the
   whole batch, `pipeline/cvae_stage.py:65`). Use the good one and delete the duplicate.

**Augmentation worth adding for Raman**, several of which connect to §2.1:

- **baseline wobble injection** — add a random smooth baseline, making the classifier invariant to
  residual baseline error. Directly motivated by the baseline study, and the most valuable of these.
- multiplicative intensity scaling / jitter (laser power, focus variation)
- additive noise at realistic SNR
- band-wise dropout (detector defects, saturated regions)
- at bag level, sub-bagging (§4) is itself augmentation

If after a fair fight the CNN still loses to Apollo/RF/XGB, **that is a reportable result** — 1738
correlated bands with 58 bags of supervision is not a regime where depth should be expected to win
— not an embarrassment. But it has to be a fair fight before that sentence can be written.

---

## 8. CVAE under inferred labels (T3)

The graph is barely coupled to hard labels — the label enters at exactly two float
`(B, K)` tensors:

- `src/models/cvae.py:99` — decoder `Concatenate([z, label])`. A simplex row flows through with
  **zero** shape change and weight-file compatibility preserved.
- `src/models/cvae.py:168` — `categorical_crossentropy(y, class_pred)` accepts soft targets
  natively.

The work is at the integer boundaries:

- **Producing the condition**: `pipeline/cvae_stage.py:39` (`to_categorical(y_s)`) is the single
  manufacturing site. Replace with a lookup into the latent store for (method, variant, fold).
- **Not already present via mixup**: `cvae_stage.py:55` draws mixup partners **within class**, so
  the blended label equals the original one-hot. The soft label is not hiding there.
- **Self-target loop**: if the condition comes from the CVAE's own encoder head, the $\alpha$ term
  becomes degenerate. Stop-gradient, or use an outer EM schedule.
- **Generation**: `pipeline/_cvae_gen.py:110` enumerates simplex corners and returns `int64` labels
  at `:126`. Replace the loop with a sampler over the empirical posterior simplex (or a fitted
  Dirichlet). Downstream sklearn/XGBoost consumers (`pipeline/tsrtr_stage.py:64,74-76`) need hard
  labels — project by argmax at that boundary, or sample near-corner only, and state which.
- **CbR needs rethinking**: `evaluation_stage.py:134-141` is structurally "decode at each one-hot
  corner, pick the lowest MSE". Under soft training the corners may be off-manifold, so CbR could
  degrade for reasons unrelated to generative quality. Replace with mixture-weight optimisation per
  spectrum, or use the encoder head.

**The claim to test**: conditioning on the inferred latent should improve generation, because the
current model is asked to generate "LGm-1" from a set that includes non-diagnostic tissue.
Measurable against `src/evaluation/synthetic_realism.py` plus bag-level TSTR/TSRTR accuracy.

---

## 9. Staged execution

**This plan is executed one stage at a time.** On approval, the first action is to copy this file
into the repo on the new branch (`docs/MIL_REFRAME_PLAN.md`) so it is version-controlled alongside
the work, and updated in place as stages complete.

### Per-stage workflow — applies to every stage without exception

1. **Implement** only the changes listed for that stage. Do not start the next stage's work.
2. **Test** — run that stage's correctness checks. They must pass before anything else.
3. **Consistency check** — re-run the prior stages' checks; a later stage must not silently break an
   earlier guarantee.
4. **Document** — update `docs/` and this file's Progress table with what actually changed,
   including deviations from the plan and why.
5. **Update the paper** if that stage changes a number, a method, or a claim. Do not batch paper
   edits to the end.
6. **Commit** on the branch with the stage number in the message.
7. **Stop and review** before the next stage.

### Progress

| # | stage | status | notes |
|---|---|---|---|
| 0 | Baseline selector | **done** | **Verdict reverses the plan: IARPLS, not GCV.** GCV pinned at the grid floor in 83% of spectra; IARPLS beats an *oracle* airPLS λ and wins 12/12 stress conditions by 2.4–11.5×. Current λ=100 suppresses amide I by 43% and drifts peak heights 16–55% with fluorescence. **λ itself is UNRESOLVED**: the initial "inert over 1e2–1e4" claim was withdrawn after a follow-up study — the inert window is 2.5 decades for smooth baselines but 0.5 for curved ones, and real baselines look more curved than any synthetic family tested (consistent with Kizilov et al. 2026). `IARPLS_LAMBDA=1e4` is provisional; **Stage 1 must carry a λ sensitivity analysis**. See [stage0_findings.md](stage0_findings.md) §2.3 and §7.1, and the notebook [00_Stage0_Baseline_Correction.ipynb](../notebooks/00_Stage0_Baseline_Correction.ipynb) (experiment 1). `preprocessing.py` surgery deferred to Stage 1. |
| 1 | Data layer + re-export | not started | |
| 2 | Evaluation reframing + Tier 0 | not started | |
| 3 | Instance-model hygiene | not started | |
| 4 | Latent store + Tier 1/2 | not started | |
| 5 | Spatial validation | not started | |
| 6 | Tier 3/4 | not started | |
| 7 | CVAE soft conditioning | not started | |
| 8 | Mixture model + 6-class | not started | |

---

### Stage 0 — Per-spectrum baseline selector

**Goal.** Replace the arbitrary `airpls_lambda=100` with per-spectrum GCV (§2.1). Nothing else.

**Changes.** New `src/preprocessing/baseline.py` — GCV score over a log-spaced λ grid for the
Whittaker system, returning λ per spectrum; L-curve fallback. Rewire
`src/data/preprocessing.py::_airPLS:100-139` to accept a selected λ. **Delete**
`_preprocess_batch_ramanspy:155-175` (unused, and contradicts the spatial-only design). New
validation script producing the Phase-0 figure.

**Tests.**
- Synthetic-baseline recovery: known polynomial + broad-Gaussian baselines added to high-SNR
  spectra; report recovery RMSE and peak fidelity at 1004/1440/1660/2850–2950 for GCV vs fixed
  λ=100 vs IARPLS, across the fluorescence range present in the data.
- **Purity assertion**: one spectrum processed alone is bit-identical to the same spectrum processed
  inside a full batch. Guards the no-global-parameter rule.
- GCV stability on the lowest-SNR decile (known failure mode); fall back to L-curve if it
  under-smooths.

**Docs / paper.** Methods → preprocessing. This replaces an inherited constant with a citable
procedure (Craven–Wahba; Eilers 2003) — worth a short paragraph, not a footnote.

**Exit.** GCV beats fixed λ=100 on peak fidelity, or the fallback is adopted and justified.
Preprocessing remains fold-independent.

---

### Stage 1 — Data layer and re-export

**Goal.** One working producer, new schema, `coords` surviving into modelling.

**Changes.** Fix the four defects in `src/data/preprocessing.py` (`gaussian_filter` not imported
`:8/:268`; `_build_file_to_map_correspondence:209` does not exist; `tumor_maps` key lookup appends
`.npy` and never matches; `pipeline/preprocessing_stage.py:36` passes a `config=` kwarg `__init__`
rejects). Fold notebook 01 §4's export into the module so the notebook stops being a build
dependency. Export the §2.2 schema, widening the mask to strong-signal. Add `coords` to
`src/data/loader.py::_load_data:152-200` **and to every mask site** (`:192-195`, `:340-345`). Yield
`val_coords` / `val_groups` / `train_coords` from `__iter__`. Widen `prepare_fold`'s 4-tuple return
(`pipeline/_common.py:45-96`).

**Tests.**
- Reprojection round-trip exact against `tumor_maps.npy` for **all 58 bags** (verified by hand on 3).
- Coords stay row-aligned after crop → cap → tissue filter, in every order.
- `slide_shapes` registry covers every `sample_id`; flag the `results/exp_201` 59-vs-58 discrepancy
  (`HF-1887_via-t_2_1.h5_3` is in `ignore_samples` but present in the file).
- Re-run Stage 0's purity assertion.

**Also carried in from Stage 0 — λ is unresolved.** `IARPLS_LAMBDA = 1e4` is provisional
(stage0_findings.md §7.1). Before this export is treated as final, run a λ sensitivity analysis
on the *real* corrected spectra and report it: the synthetic study cannot settle it, because its
baselines are measurably smoother than the real ones. Candidate resolutions: per-spectrum L-curve
applied to IARPLS, or a small set of manually curated reference baselines to calibrate a combined
drift-plus-retention criterion. Exporting under a provisional λ is acceptable **only** if the
export is cheap to repeat — establish that cost before committing to it.

**Docs / paper.** New data-schema doc. Paper → data/preprocessing section, including the honest
statement that the mask is viability, not tumour, plus the λ sensitivity analysis.

**Exit.** Reprojection passes on all bags; the old and new exports agree on the `viable_legacy`
subset; λ sensitivity is reported.

---

### Stage 2 — Evaluation reframing and Tier 0

**Goal.** Separate bag-level from instance-level reporting; fix the metric bugs; add naive-MIL
baselines. **Highest paper value of any stage** — it corrects numbers already submitted.

**Changes.** `src/utils/statistical_evaluation.py`: rename
`stratified_slide_accuracy` → `within_bag_instance_recall` and demote it; add true bag-level
metrics; fix `precision_macro` (constant 0.5), `kappa` (always 0), `f1_macro` (silently drops
0%-accuracy slides, `:264-265`). Implement the duplicate-bag averaging rule (§7.1) so *n*=58.
New `pipeline/mil_eval_stage.py` for the Tier-0 poolings, reusing `_gather_probas`
(`evaluation_stage.py:92-98`) unchanged. Wire into `run_suite.py` at its five touch points. Fix the
mixed unit-of-analysis in `paper/gen_appendix_tables.py:53-81`.

**Tests.**
- Unit tests on hand-built single-class bags asserting each previously-degenerate metric now
  behaves (precision varies; kappa is defined or absent; f1_macro retains 0%-accuracy bags).
- Duplicate-bag averaging yields exactly 58 bag decisions from 62 (bag, fold) pairs.
- Tier-0 mean-pooling reproduces the existing majority-vote confusion matrix where they should agree.

**Docs / paper.** **Every results table changes.** Bag-level CIs become the headline; the old
statistic appears as a diagnostic under its new name. Budget real time here.

**Exit.** Bag-level numbers reproduce the existing qualitative story at the correct level of
analysis, with wider and honest CIs.

---

### Stage 3 — Instance-model hygiene

**Goal.** A trunk worth benchmarking on, before any MIL model is built on it.

**Changes.** Per §7.2: inner validation split from **training bags only**, group-aware; forward
`dropout_rate` (`pipeline/_common.py:186`); compile once (`src/models/cnn.py:44-48` vs `:187`);
train longer. Wire `src/data/augmentation.py` (currently dead code, but holds the better
`spectral_shift` and `within_class_mixup`) and retire the cruder inline duplicate at
`pipeline/cvae_stage.py:53-72`. Add **baseline-wobble injection**, intensity jitter, SNR-matched
noise, band dropout.

**Tests.**
- Assert no test-fold `sample_id` reaches the early-stopping monitor. This is a leak fix, not a
  tuning fix.
- Assert `dropout_rate` from config reaches the built model.
- Augmentation output stays in-range and label-preserving.

**Docs / paper.** Baseline methods. If the CNN still loses after a fair fight, write that as a
result about the regime (1738 correlated bands, 58 bags), not as an apology.

**Exit.** Leak assertion passes; CNN either becomes competitive or its failure is characterised.

---

### Stage 4 — Latent store and Tier 1/2, constrained binary

**Goal.** First real latent estimates, with the leakage structure enforced.

**Changes.** New `src/mil/` package: a common formulation interface, `misvm.py` (mi-SVM / MI-SVM
one-vs-rest over Apollo from `src/models/baselines.py`), `em.py` (label-proportion EM with
hierarchical $\pi_b$, §1), `store.py` (the §3 sidecar). New `pipeline/mil_fit_stage.py`. Retire
`patient_cap`; implement bag-size-aware pooling (§4).

**Tests.**
- **Fold-leakage assertion**: posteriors scoring a validation bag must come from that bag's fold;
  a deliberately mismatched fold must raise.
- Alignment checksum rejects a stale posterior file after re-export.
- Degeneracy diagnostics (§5): $\pi_b$ distribution, shrinkage check, null-component cosines.
- $\pi_b$ external correlation against `viable_legacy` fraction and NMF proportions.

**Docs / paper.** New methods section — the formulation itself. The $\pi_b$ figure is a headline.

**Exit.** Non-degenerate $\pi_b$ per §5; shrinkage check shows per-bag signal; leakage assertions
pass.

---

### Stage 5 — Spatial validation

**Goal.** Establish that the latent maps mean something, without feeding the models any geometry.

**Changes.** New `src/evaluation/spatial.py`: variogram and Moran's I **by lag band**, permutation
null within the bag's own mask, ARI/AMI against `hard_labels.npy`. A shuffled-bag-label control
runner that refits the whole Stage-4 pipeline on permuted labels.

**Tests.**
- Lag-band computation verified on synthetic maps of known correlation length.
- Assert no model in the chain ever receives `coords`.
- Shuffled-label refit collapses lag ≥ 3 coherence.

**Docs / paper.** **The novel validation section.** Report by lag, exclude lags 1–2 from the claim,
report the variogram range.

**Exit.** Excess coherence at lag ≥ 3 survives the shuffled-label control. If not, §6 is rewritten
and the spatial claim is dropped — that outcome is reportable too.

---

### Stage 6 — Tier 3/4

**Goal.** Attention MIL and generative MIL, with sub-bagging priced correctly.

**Changes.** New `src/mil/attention.py` — gated attention **over instance logits**, frozen per-fold
CVAE encoder trunk. Sub-bag sampler implementing §4.1 (uniform not contiguous; $s$ from
$\pi_{\min}$, then adaptive from $\hat\pi_b$; one-sided label smoothing at $\varepsilon$). Tier 4
promotes CbR: `softmax(-mse/T)` from `evaluation_stage.py:133-139`, plus the unused encoder class
head (`src/models/cvae.py:91`) as a zero-cost variant.

**Tests.**
- Sub-bag null-rate diagnostic against the targeted $\varepsilon$.
- Assert sub-bags are uniform draws, not contiguous patches.
- Re-run Stage 5's spatial checks on Tier-3 posteriors.

**Docs / paper.** Model comparison table across all tiers.

**Exit.** Beats Tier 0 at bag level, with sub-bag label fidelity within target.

---

### Stage 7 — CVAE soft conditioning

**Goal.** Rescue the rejected generative contribution by conditioning on the inferred latent.

**Changes.** `pipeline/cvae_stage.py:39` — replace `to_categorical` with a latent-store lookup keyed
(method, variant, fold). `pipeline/_cvae_gen.py:110,126` — replace the corner-enumeration loop with
a simplex sampler; argmax-project at the sklearn/XGBoost boundary
(`pipeline/tsrtr_stage.py:64,74-76`) and say so. Rework CbR (`evaluation_stage.py:134-141`), which is
structurally tied to one-hot corners. Guard the encoder-head self-target loop with stop-gradient.

**Tests.**
- Soft labels flow end-to-end with no shape change; decoder weights stay load-compatible.
- Generated class balance matches the requested simplex distribution.
- Synthetic realism (`src/evaluation/synthetic_realism.py`) plus bag-level TSTR/TSRTR.

**Docs / paper.** The generative section — reframed as conditional generation under weak
supervision.

**Exit.** Improves on hard-label conditioning by realism *and* bag-level downstream accuracy.

---

### Stage 8 — Mixture model and 6-class

**Goal.** The extensions, reported either way.

**Changes.** Relax (C) to (M) behind the `mil.formulation` config key. Add 6-class configs mirroring
201–203 → 204–206.

**Tests.** The §10.2 pre-registered criterion for whether (M) identified. Wide 6-class CIs are
expected, not a failure — classes 2 and 5 have 4 patients each.

**Docs / paper.** Extension section, or a limitations section if (M) does not identify.

**Exit.** A reported result in either direction.

---

**Why this order.** Stage 3 precedes the MIL models deliberately: augmentation is a prerequisite for
any neural model and the Tier-3 trunk *is* the fixed CNN, so doing it later would mean benchmarking
attention on a trunk known to be undertrained. Stage 4 precedes Stage 6 because $\hat\pi_b$ is what
sizes the sub-bags (§4.1). Stages 1–2 are worth doing regardless of whether the MIL programme
succeeds — they fix reporting errors that are in the submitted paper.

### Pipeline wiring

New stages slot into `pipeline/run_suite.py` at five touch points: `DEFAULT_STAGES:34`,
`CLASSIFIER_STAGES:35`, `STAGE_ALIASES:36-40`, `_stage_outputs_exist:57-82`, and the `_run_stage`
dispatch chain at `:85-138`. Proposed: `mil_fit` (fits a formulation, writes posteriors) and
`mil_eval` (bag-level metrics + spatial validation). Note `_stage_outputs_exist` globs
`statistical_evaluation_*.json` for `stats_eval` — a new stage writing into the same
`statistical_dir` will make that skip-check fire spuriously.

### Dead code — do not wire changes through it

- `src/data/augmentation.py` is currently unused by the live pipeline, which reimplements both
  augmentation and the patient cap inline in `pipeline/cvae_stage.py`. **Note this reverses in Stage
  3**: the module holds the better implementations, so it gets wired in and the inline duplicate at
  `cvae_stage.py:53-72` is retired. Until then, behaviour changes must go in `cvae_stage.py`.
- `src/data/annotator.py` (`ClinicalAnnotator`) is never invoked.
- Unread config keys: `models.cvae.num_classes` (the code uses `models.cnn.num_classes`),
  `reconstruction_loss_mode`, `validation_split`, `cap_strategy`, `use_precomputed_masks`,
  `spatial_median_filter*` (printed only). Do not add a `soft_labels` switch alongside these.
- `src/utils/clinical_inference.py` is an unintegrated standalone, but its `:63-84` is already a
  majority-vote bag classifier — useful as a reference implementation, not as a dependency.

---

## 9.1 Deferred — revisit after the first pass

Recorded so they are not lost, explicitly **not** in scope now:

- **Spectral despiking** (Whitaker–Hayes) as a complement to the spatial median. The spatial
  argument in §6 makes it unnecessary for validation, and it would be a second preprocessing change
  landing in the same export as the baseline change.
- **No-filter export** as a sensitivity check. A one-liner once the lag-stratified statistic is in
  place, not a gate on anything.
- **Spatially-regularised variant** (CRF / graph smoothing) — only after §6 passes, and reported as
  a model, never as evidence.
- **Wider crop region for learning.** Baseline correction stays on the full 1738 bands; whether the
  *model* should see more than 700–1800 cm⁻¹ is a separate later-phase question.

## 10. Settled, and still open

**Settled this round.** $\pi_b$ is inferred hierarchically, with $\pi_{\min}$ demoted to a numerical
floor (§1). The null is multi-modal, $M{=}3$, matched to the NMF/GMM clusters, with collapse made
observable (§1, §5). The CV schema is retained unchanged and documented, with duplicated slides
averaged across folds to keep $n{=}58$ bags (§7.1). Bag-level CIs are presented as the headline
uncertainty. The median filter stays; the spatial confound is handled by a lag-stratified statistic
rather than an ablation (§6).

Also settled: λ is selected **per spectrum, label-free**, preserving the no-global-parameter rule
the rest of the pipeline already follows — which removes the comparability problem entirely (§2.1).
The spatial median stays and its justification is written into the methods (§6). The 1D-CNN is
fixed rather than dropped, since it doubles as the Tier-3 instance trunk (§7.2). Spectral despiking
is deferred (§9.1).

Also settled: λ by **per-spectrum GCV**, fixed before anything else, using the established
Craven–Wahba / Eilers formulation (§2.1). Augmentation and the CNN fix land **before** the MIL
models (Phase 3). `patient_cap` is retired in favour of explicit bag-size-aware pooling (§4).

**Empirical checkpoints — cannot be answered by argument, only by running them.** Each needs a
criterion fixed *now* so the result is reportable either way:

1. **Is $\sigma$ estimable at 58 bags?** Criterion: if the profile likelihood on $\sigma$ is flat
   within 2 log-likelihood units across an order of magnitude, declare it unidentified, fall back to
   a fixed moderately-wide prior, and say so. Do not report an unidentified hyperparameter as an
   estimate.
2. **Does the mixture model (M) identify?** In scope, and **reported either way** — if it works it
   is the stronger result; if it fails that is a publishable negative finding about what 58 bags can
   support, with (C) still carrying the claim. Criterion, fixed in advance: (M) succeeds if its
   components are stable across folds (mean pairwise cosine between fold-matched component means
   above a threshold set on (C)'s null components) **and** it does not reduce to (C) — i.e. some bag
   genuinely puts mass on a class other than its own label and $\arg\max_k \pi_{bk} = Y_b$ still
   holds. Anything else is a negative result; do not tune until it looks reasonable.
3. **Sub-bag size and count** for Tier 3 — sweep, centred on the $\pi_{\min}$-derived value from
   §4.1 rather than guessed. No longer entangled with `patient_cap`, which is gone; the sweep now
   interacts with true bag size $n_b$ and with $\hat\pi_b$.

**Still genuinely open.**

4. How much of the existing notebook corpus is worth migrating versus rewriting against the new
   terminology (§11).

---

## 11. Branch, terminology, config migration

This is a major revision touching preprocessing, data layer, models, evaluation, configs, notebooks
and the paper. Treat it as such.

### Branch

Work on a dedicated branch, e.g. `mil-reframe`. **Precondition**: the tree currently has uncommitted
changes across `Thesis/`, `paper/`, `configs/`, `notebooks/`, `pipeline/` and `src/`. Commit or
stash those on `thesis` first, so the dissertation state is frozen and recoverable, then branch. The
thesis is not to be touched by this work; the paper is.

Existing `results/` become non-comparable once the baseline selector changes (Phase 0). **Archive
rather than overwrite** — move to a versioned prefix so the submitted paper's numbers stay
reproducible while the reframed ones are generated alongside.

### Terminology

The rename is the point, not cosmetic — the old names encode the false assumption. Apply
consistently across code, configs, notebooks and paper.

| old | new | why |
|---|---|---|
| slide | **bag** | the unit of supervision |
| spectrum | **instance** | the unit of prediction |
| `y_class` (per-spectrum) | `y_bag` | names it as a bag label broadcast to instances, not an instance label |
| `slide_ids` | `bag_ids` | — |
| `y_tumor` | *retired* | misnomer; the mask is viability, and the field is constant 1 |
| "tumour mask" | **viable-signal mask** | what it actually is |
| `patient_cap` | *retired* | a preprocessing hack standing in for bag-size-aware pooling (§4) |
| `stratified_slide_accuracy` | `within_bag_instance_recall` | demoted to a diagnostic; `bag_accuracy` becomes the headline |

### Config migration

YAML structure is kept; the contents change.

**Retire** — `cv_strategy.patient_cap`, `cv_strategy.cap_strategy`, `preprocessing.y_tumor`,
`preprocessing.airpls_lambda` (now per-spectrum GCV; survives only as an optional fixed-λ ablation
arm). **Delete outright** — keys verified as read by nothing: `preprocessing.use_precomputed_masks`,
`preprocessing.spatial_median_filter*` (printed only), `models.cvae.num_classes` (the code uses
`models.cnn.num_classes`), `models.cvae.reconstruction_loss_mode`, `models.cvae.validation_split`.

**New blocks**:

```yaml
baseline:      {selector: gcv, lambda_grid: [...], fallback: lcurve, fixed_lambda_arm: 100}
mil:           {formulation: constrained|mixture, M_null: 3, pi_floor: 0.02,
                pooling: mean|topq|lse, topq_fraction: 0.1, sub_bag: {size: 256, n: ...}}
latent:        {store: results/{exp}/latent/latent_labels.h5, method: ..., variant: ...}
```

`ConfigManager.REQUIRED_FIELDS` (`src/utils/config_manager.py:79-87`) validates nothing under
`models.cvae`, so new keys pass validation silently — add explicit validation for the `mil` block
rather than inheriting that weakness.

### Notebooks

All change. The structurally important one: **notebook 01 §4 is currently the real dataset
producer** (`src/data/preprocessing.py` is non-functional). Phase 1 folds that export into the
module, after which notebook 01 becomes exploratory only and stops being a build dependency. That is
a prerequisite for the rest, not a cleanup.

### Suites

The existing ablation axes still make sense on top of MIL-derived labels — reconstruction loss
(201–206), latent dimension (302/303), TSRTR ratio (401–403). They get an added `mil:` block and are
re-run, rather than being redesigned.

## 12. Standing invariants — re-checked after every stage

These are step 3 of the per-stage workflow: the regression suite a later stage must not silently
break. Each becomes a test as soon as the stage that introduces it lands, and all of them run from
then on.

- **Reprojection**: assert `reproject(...)` round-trips against `tumor_maps.npy` for all 58 slides
  (verified by hand on 3; the convention is `coords[:,0]=col`, `coords[:,1]=row`).
- **Alignment**: assert `latent_labels.h5` checksums match the source HDF5 before any consumer reads
  a posterior.
- **Leakage**: assert that for every fold, posteriors used to score validation slides come from that
  fold's model; add a test that deliberately mismatches folds and expects a raised error.
- **Metric fixes**: unit-test `precision_macro`, `kappa`, `f1_macro` against hand-built single-class
  bags; each currently returns a known-degenerate value.
- **Baseline correction**: simulated-baseline recovery RMSE and peak fidelity at 1004/1440/1660/
  2850–2950 cm⁻¹ for the per-spectrum selector vs fixed λ=100 vs IARPLS, across the fluorescence
  range present in the data.
- **No-global-parameter assertion**: a test that preprocessing output for one spectrum is bit-identical
  whether it is processed alone or inside the full batch. This currently holds
  (`_preprocess_spectrum:141-153` is per-spectrum) and must keep holding after the λ change.
- **1D-CNN**: assert the early-stopping monitor is fed by an inner split drawn only from training
  slides, and that no test-fold sample id appears in it. Confirm `dropout_rate` reaches the model.
- **Degeneracy**: report fitted $\pi_b$ for every run; flag as collapsed if median $\pi_b > 0.9$, if
  $\pi_b < \pi_{\min} + \epsilon$, or if the across-bag spread of $\pi_b$ is indistinguishable from
  zero (the shrinkage check of §1).
- **Null components**: pairwise cosine between the $M$ null component means; report the effective
  component count so collapse is visible.
- **Sub-bag label fidelity** (§4.1): empirical fraction of sub-bags carrying zero diagnostic mass
  under the final model, against the $\varepsilon$ the sizing targeted. Materially above it means
  the Tier-3 numbers rest on mislabelled sub-bags. Also assert sub-bags are drawn uniformly, not as
  contiguous patches.
- **Spatial**: variogram / Moran's I **by lag band**. The claim rests on lag ≥ 3, where a 3×3 median
  cannot reach; lags 1–2 are shown but excluded. Report the variogram range — 1–2 px means the
  filter, 10–20 px means tissue.
- **Controls**: the shuffled-bag-label refit must collapse coherence at lag ≥ 3. If it does not, the
  spatial argument does not hold and §6 must be rewritten. A no-filter export is a one-line
  sensitivity check, not a gate.
- **$\pi_b$ external validity**: correlation between inferred $\pi_b$ and the `viable_legacy`
  fraction / NMF cluster proportions. This is the strongest external check available — report it
  even if it is negative.
- **End-to-end**: `python -m pipeline.run_suite --config exp_2xx_mil --stages mil_fit,mil_eval`
  produces a bag-level summary CSV plus per-slide posterior maps.
