# Stage 0 findings — baseline correction

**Status: complete, with one claim withdrawn after a follow-up experiment.** Branch
`mil-reframe`. Studies: `scripts/stage0_baseline_study.py` and
`scripts/stage0_generator_stress.py`; results in `results/stage0_baseline/`.

> **Reproducing this document.**
> [`notebooks/00_Stage0_Baseline_Correction.ipynb`](../notebooks/00_Stage0_Baseline_Correction.ipynb)
> reproduces every number, step by step, in about a minute. Regenerate the studies with
> `python scripts/stage0_baseline_study.py --n-spectra 40 --out-dir results/stage0_baseline`
> and `python scripts/stage0_generator_stress.py --out-dir results/stage0_baseline`.

## Summary of what may and may not be claimed

| claim | status |
|---|---|
| airPLS λ=100 differentially suppresses broad bands and lets peak heights drift with fluorescence | **Holds.** Confirmed on synthetic *and* real spectra. |
| IARPLS outperforms airPLS λ=100 | **Holds.** Wins 12/12 stress conditions by 2.4–11.5×, and on real spectra by invariance. |
| GCV cannot select λ here | **Holds.** Structural, explained in §6. |
| ~~λ is inert over 1e2–1e4~~ | **WITHDRAWN.** An artifact of our baseline generator. See §5. |
| λ = 1e4 is the right production constant | **Not established.** Deferred to Stage 1 with a sensitivity analysis. See §5.4. |

---

## 1. How the synthetic data was generated

Ground truth is required to score a baseline corrector, and real spectra do not come
with a known baseline. So the primary arm is fully synthetic — and its construction is
the main thing a reviewer should scrutinise, because **every conclusion is conditional
on the generator spanning the relevant space.**

### 1.1 The signal

14 bands at fixed Raman positions (`SYNTHETIC_BANDS` in `scripts/stage0_baseline_study.py`),
chosen to approximate brain-biopsy tissue — in particular a **dominant lipid CH-stretch
envelope**, which is the defining feature of this dataset:

| region | bands (cm⁻¹) | relative amplitude | FWHM (cm⁻¹) |
|---|---|---|---|
| protein / nucleic acid | 1004 (Phe), 1032, 1128, 1230, 1260, 1300, 1340 | 0.30–1.00 | 8–18 |
| lipid / protein | 1440 (CH₂ deformation), 1550, 1660 (amide I) | 0.35–1.20 | 14–20 |
| **lipid CH stretch** | **2850, 2885, 2930** | **1.45–1.80** | 18–22 |
| aromatic | 3060 | 0.30 | 20 |

Per spectrum, amplitudes are jittered ×U(0.75, 1.25) and widths ×U(0.85, 1.15).
Absolute scale `peak_scale = 400`, so peak amplitudes land in the hundreds, matching the
measured fingerprint intensities of real median-filtered tissue spectra.

### 1.2 The baseline

```
t        = (wn - wn.min()) / ptp(wn)                       # normalised abscissa
fluor    = exp(-0.5 * ((t - c) / w)^2)   c~U(-0.1,0.45), w~U(0.15,0.45)
rayleigh = exp(-t / U(0.01, 0.04))                         # notch-filter cutoff rise
poly     = U(-0.3,0.3)·t + U(-0.25,0.25)·t² + U(-0.15,0.15)·t³
baseline = normalise(fluor + 0.6·rayleigh + poly) × amplitude
```

`amplitude` is 2×, 8× or 25× the signal peak-to-peak ("low/medium/high fluorescence").

### 1.3 Noise and observation

Additive i.i.d. Gaussian, σ = signal_ptp / 235, where 235 is the median SNR measured
from real 3×3-median-filtered tissue spectra. Observation is `signal + baseline + noise`.
Wavenumber axis is the real one: 1738 points, 50–3400 cm⁻¹, 1.93 cm⁻¹/point.

### 1.4 The second arm avoids the problem entirely

Because any synthetic generator is arguable, the invariance arm uses **real** spectra and
**no ground truth**: correct a real spectrum with and without added fluorescence, and
compare peak heights. An invariant corrector gives the same answer either way. Nothing
about this metric depends on how a reference was constructed.

---

## 2. Headline: the inherited λ=100 differentially suppresses broad bands

Peak height recovered / true, median over all fluorescence levels (n=40 per cell):

| method | 1004 Phe (narrow) | 1440 CH₂ | 1660 amide I | 2930 CH (broad) |
|---|---|---|---|---|
| **legacy λ=100 (current)** | **0.900** | **0.691** | **0.574** | **0.584** |
| gcv | 0.126 | 0.025 | 0.017 | 0.011 |
| lcurve | 1.002 | 0.997 | 0.987 | 0.927 |
| **iarpls** | **1.003** | **1.003** | **1.004** | **1.003** |

The current pipeline keeps 90% of the narrow 1004 cm⁻¹ band but only 57% of amide I. A
λ=100 baseline is flexible enough to climb into broad bands but not narrow ones, so it
does not merely scale spectra — **it changes band-height ratios**, which is what the
classifiers key on.

Error in the modelled region (700–1800 cm⁻¹), median NRMSE:

| method | low | medium | high | all |
|---|---|---|---|---|
| gcv | 0.0476 | 0.1178 | 0.1134 | 0.1099 |
| legacy λ=100 | 0.0267 | 0.0512 | 0.0967 | 0.0502 |
| lcurve | 0.0110 | 0.0264 | 0.0245 | 0.0213 |
| **iarpls** | **0.0048** | **0.0047** | **0.0047** | **0.0047** |

---

## 3. The result that matters most for the paper

Peak-height drift on **real** spectra when fluorescence is added (0 = perfectly invariant):

| method | low | medium | high |
|---|---|---|---|
| gcv | 0.473 | 0.546 | 0.848 |
| lcurve | 0.467 | 0.545 | 0.632 |
| **legacy λ=100 (current)** | **0.161** | **0.274** | **0.553** |
| **iarpls** | **0.0004** | **0.0014** | **0.0030** |

Under the current pipeline a spectrum's measured peak heights change by **16–55% purely
as a function of how fluorescent the sample is**. Fluorescence is a property of specimen
and acquisition and varies by slide; the CV groups by slide; so the confound aligns with
the label. It gives a classifier a route to slide identity that has nothing to do with
tissue. This arm uses no synthetic signal at all, so it is not subject to §5.

---

## 4. IARPLS versus an oracle airPLS

Searching the whole grid for the λ minimising true error gives an oracle no selector
could beat. IARPLS still wins (fingerprint NRMSE, median of 5 spectra per level):

| fluorescence | airPLS λ=100 | airPLS **oracle** λ | oracle NRMSE | IARPLS | better by |
|---|---|---|---|---|---|
| low | 0.0219 | 2.15e4 | 0.0069 | 0.0048 | 1.5× |
| medium | 0.0430 | 4.64e4 | 0.0120 | 0.0048 | 2.5× |
| high | 0.0883 | 2.15e4 | 0.0141 | 0.0049 | 2.9× |

Note the oracle λ is ~2e4, **200× larger than the inherited 100**.

---

## 5. Withdrawn: "λ is inert over 1e2–1e4"

### 5.1 Why it was questioned

Kizilov et al. (*J. Raman Spectrosc.* 2026, [arXiv:2507.17917](https://arxiv.org/html/2507.17917))
report the opposite of insensitivity: *"IARPLS is more sensitive to the choice of its
parameters compared to traditional ALS"* and *"selecting optimal λ and p values is
crucial, as improper tuning can lead to under- or over-fitting."* They quote λ ∈ 10²–10⁹
as the typical range and use λ=10⁸ on their own synthetic data — a value that is
catastrophic on ours. Their samples are medical ointment, PTFE and polymeric aligners,
not tissue.

An insensitivity result that contradicts the literature is far more likely to be a
property of our generator than a property of the algorithm. That was the right suspicion.

*(One part of their claim does not transfer: `pybaselines`/`ramanspy` IARPLS exposes only
`lam`, `diff_order`, `max_iter`, `tol` — **there is no `p`**. The asymmetry weighting is
derived internally; `p` belongs to ALS. So only the λ half of their sensitivity claim
applies to our implementation. Their library default is `lam=1e5`.)*

### 5.2 The follow-up experiment

`scripts/stage0_generator_stress.py` hardens the generator along three axes and watches
where λ sensitivity appears:

- **baseline family** — `smooth` (§1.2), `structured` (3–5 fluorophore-like Gaussians),
  `sharp` (adds narrow, high-curvature components);
- **lineshape** — Gaussian vs **Lorentzian** (real Raman bands are Lorentzian/Voigt; the
  heavier tails are baseline-like and can be eaten);
- **noise** — i.i.d. Gaussian vs **Poisson** shot noise, which is signal-dependent.

Sensitivity is reported against *curvature energy* `Σ(Δ²b)² / ptp(b)²` — the exact
quantity the λ penalty acts on — so "harder" is measured, not asserted.

### 5.3 Result: the inert window collapses with curvature

IARPLS fingerprint NRMSE vs λ, by baseline family (median, n=12):

| λ | smooth | structured | sharp |
|---|---|---|---|
| 1e2 | 0.0169 | 0.0113 | 0.0309 |
| 3.2e2 | 0.0121 | 0.0108 | **0.0290** |
| 1e3 | 0.0115 | 0.0127 | 0.0400 |
| 3.2e3 | 0.0111 | 0.0114 | 0.4942 |
| **1e4** | **0.0107** | 0.0122 | **0.7950** |
| 1e5 | 0.0119 | 0.2313 | 1.2984 |
| 1e6 | 0.8861 | 1.7073 | 2.4629 |

Width of the λ window within 25% of that family's best:

| family | best λ | best NRMSE | inert window | decades |
|---|---|---|---|---|
| smooth | 1e4 | 0.0107 | 3.2e2 – 1e5 | **2.5** |
| structured | 3.2e2 | 0.0108 | 1e2 – 3.2e4 | **2.5** |
| **sharp** | 3.2e2 | 0.0290 | 1e2 – 3.2e2 | **0.5** |

**The inert window collapses from 2.5 decades to 0.5 as baseline curvature rises, and at
λ=1e4 the sharp family scores 0.795 — catastrophic.** Our original insensitivity finding
was a property of the smooth baseline family, exactly as suspected. This is consistent
with Kizilov et al. rather than contradicting them.

Lineshape and noise matter too (median NRMSE, λ≤1e4): Lorentzian bands cost ~5×
(0.0237 vs 0.0048 on the smooth family), and Poisson noise compounds it. The original
all-Gaussian, all-Gaussian-noise generator flattered every method.

### 5.4 Where real data actually sits — and why λ stays open

Curvature energy of estimated baselines from 60 real median-filtered tissue spectra,
against the synthetic families:

| source | curvature energy (median) |
|---|---|
| synthetic `smooth` | 1.6e-6 |
| synthetic `structured` | 1.2e-6 |
| synthetic `sharp` | 3.3e-6 |
| **real, morphological envelope** (least circular) | **2.5e-5** |
| real, IARPLS λ=1e3 derived | 4.8e-5 |
| real, IARPLS λ=1e5 derived | 8.8e-7 |

**Real baselines appear 7–15× more curved than even the `sharp` family.** The estimates
span 8.8e-7 to 4.8e-5 — a 28× range — because each depends on the estimator: a stiff λ
produces a smooth baseline by construction (circular), while a rolling-minimum envelope
tracks noise and band tails (inflated). **That uncertainty is itself the finding: we
cannot currently place real data in a synthetic regime, so we cannot justify a fixed λ on
inertness grounds.**

A real-data λ sweep using the ground-truth-free invariance criterion does not resolve it
either, because the criterion is degenerate on its own — drift is minimised by an
over-flexible corrector that eats the signal:

| IARPLS λ | drift (↓ better) | fingerprint retained (`ptp` ratio) |
|---|---|---|
| 1e2 | 0.0000 | **0.420** ← eats 58% of the signal |
| 1e3 | 0.0000 | 0.605 |
| 1e4 | 0.0009 | 0.786 |
| 1e5 | 0.0039 | 0.916 |
| 1e6 | 0.0177 | 0.928 |
| *airPLS λ=100* | *0.2822* | *0.446* |

Minimum drift picks λ=1e2, but retention there (0.420) shows the same over-correction
pathology as airPLS λ=100 (0.446). Retention alone is no better a criterion — it tends to
1 simply by doing nothing.

**Therefore λ is carried into Stage 1 as an explicit, documented choice with a sensitivity
analysis, not as a settled constant.** `IARPLS_LAMBDA = 1e4` is the current default
because it sits in the low-drift, reasonable-retention region and matches the synthetic
oracle, but it is a *provisional* value. Candidate resolutions for Stage 1: per-spectrum
L-curve selection applied to IARPLS; or a small set of manually curated reference
baselines to calibrate a combined drift-plus-retention criterion.

---

## 6. Why GCV failed

GCV pinned at the grid floor (λ=1) in **83.3%** of spectra, driving peak recovery to
0.01–0.13. The cause is structural. GCV assumes the residual is i.i.d. noise; here the
residual at the points airPLS keeps weighted is *structured Raman signal*, so GCV reads
that structure as "the fit is too stiff" and drives λ to the floor. No grid extension
fixes it. **L-curve never pinned (0% edge rate)**, confirming the diagnosis is specific to
GCV rather than to the grid.

---

## 7. Other caveats for Stage 1

- **IARPLS does not model the Rayleigh wing.** NRMSE by region: <200 cm⁻¹ **1.32**,
  200–300 **0.19**, 300–500 0.013, ≥500 **0.0047**. Below ~300 cm⁻¹ is unusable and must
  be cropped — recorded as `IARPLS_VALID_FROM_CM1 = 300.0`. The existing 700–1800 crop
  already satisfies this, but the export must not silently ship the bad region.
- **The endpoint "fix" is inert.** `legacy_lam100` and `ref_lam100` agree to four decimals
  on every metric. The reference rule is still correct and is now the default, but fixing
  it changes no published number.
- **ramanspy is already a pinned dependency** (`requirements.txt:27`, `ramanspy==0.2.10`),
  in both Modal images, so IARPLS adds nothing new. Stage 1 may still reimplement it
  natively against the banded solver to drop it from the hot path.
- **L-curve airPLS is retained** and tested — the fallback if IARPLS is rejected, and the
  arm that separates "λ choice" from "algorithm choice" in the methods narrative.

---

## 8. What changed in the code

| file | change |
|---|---|
| `src/preprocessing/baseline.py` | **new.** Banded Whittaker solver (`solveh_banded`, ~50× faster than the previous `spsolve`), airPLS with both endpoint rules, GCV and L-curve selectors, `correct_iarpls` as the production entry point. |
| `src/preprocessing/__init__.py` | **new.** Public surface. |
| `tests/test_baseline.py` | **new.** 24 tests: banded penalty vs dense, banded solve vs dense, Hutchinson trace vs exact, determinism, purity, peak-fidelity and λ-inertness regressions. |
| `scripts/stage0_baseline_study.py` | **new.** Primary study. |
| `scripts/stage0_generator_stress.py` | **new.** Generator-limitation follow-up (§5). |
| `notebooks/00_Stage0_Baseline_Correction.ipynb` | **new.** Reviewer-facing reproduction, committed with outputs. |
| `results/stage0_baseline/` | **new.** Raw CSVs, summaries, verdicts, figure. |

**Deviation from the plan:** the plan put "rewire `_airPLS`, delete
`_preprocess_batch_ramanspy`" in Stage 0. Both are deferred to Stage 1, which rewrites
`src/data/preprocessing.py` anyway; the verdict also reverses the planned deletion, since
IARPLS lives in that path.

⚠️ **`tests/test_baseline.py::test_iarpls_lambda_is_inert_over_the_documented_range` now
asserts something narrower than the code comment suggests.** It passes on a smooth toy
baseline. Per §5 it should be re-scoped in Stage 1 to a *regression* test on the smooth
family only, and must not be read as evidence of general inertness.

---

## 9. How to state this in the paper

The defensible framing, given §5:

> We designed a controlled experiment to compare baseline-correction algorithms, using
> synthetic spectra modelled on the band structure of our biopsies — in particular a
> dominant lipid CH-stretch envelope — with additive fluorescence backgrounds at 2×, 8×
> and 25× the signal amplitude. Under these conditions IARPLS recovered peak heights to
> within 0.3% and outperformed the previously used airPLS (λ=100) by 2.4–11.5× in the
> 700–1800 cm⁻¹ region across all tested baseline shapes, lineshapes and noise models.
> On real spectra, a ground-truth-free invariance test showed airPLS (λ=100) peak heights
> drifting 16–55% with fluorescence level, against ~0.1% for IARPLS.

What must **not** be claimed:

- ❌ "IARPLS is insensitive to λ in 1e2–1e4." It is, for smooth baselines; the window
  narrows to half a decade as curvature rises, and real baselines are more curved than
  any family we tested.
- ❌ "λ=1e4 is optimal." Not established on real data; provisional.
- ❌ Any general statement about IARPLS beyond our band structure and fluorescence range.
  Kizilov et al. tested ointment, PTFE and polymer; we tested tissue-like spectra. The
  agreement on *superiority* is worth citing; the disagreement on *sensitivity* is
  explained by baseline curvature, not by a dispute about the algorithm.

---

## 10. Consequences for the paper

1. **Methods change.** Baseline correction becomes IARPLS with a stated λ, a stated
   sensitivity analysis, and a stated valid-from wavenumber, replacing "airPLS λ=100,
   following Lita et al.".
2. **All numbers move.** Every downstream result was computed on spectra whose broad bands
   were suppressed by up to 43%. v1 is archived at `paper-v1` and `paper/versions/v1/`.
3. **A limitation becomes reportable.** §3 is a genuine confound in the submitted work —
   fluorescence-dependent peak heights under slide-grouped CV.
4. **Crop region needs a note.** Correction runs on the full 1738 bands; output is valid
   only above 300 cm⁻¹.

## Review of Stage 0
The study is informative, but poorly presented. The findings and notebook need to follow a structure such that the experiments setup and results are easy to read and understand.
Notebook should be redesigned using the following structure:
Title: Study on baseline correction methods for our dataset.
Description: In order to determine the best baseline correction method, we designed a suite of experiments using a synthetic data generator that approximates brain biopsy spectra. We decided to use this approach in order to not overfit the lambda parameter to our particular dataset, but to estimate it for this type of samples. 
Synthetic data generator: We generate the signal 
s = peaks + baseline + noise
where:
 peaks = sum of 14 bands at fixed position to extimate brain biopsy spectra (use math notation)
 baseline = fluorescence + rayleigh + poly
 noise = iid gaussian with sigma = signal_ptp/SNR, with SNR computed from data.
Experiment 1:
iarpls lambda fit. (Currently this experiment appears quite late, but it's actually needed early for an acurate comparison of algorithms, as GCV and L curve are algos that fit lambda for airpls. We first run this one to have a good iarpls to compare against.)
Experiment 2:
Synthetic data test: GCV, L curve, IARPLS. (Tests the 3 methods on the synthetic generator)
Experiment 3:
Invariance test on real data: use real data, perform correction, then add baseline and perform correction again, measure drift. This experiment exists as well.
Experiment 4: iarpls vs airpls optimal lambda 

We draw conclusions only after all experiments.

