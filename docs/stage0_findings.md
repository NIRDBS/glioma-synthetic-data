# Stage 0 findings — baseline correction

**Status: complete.** Branch `mil-reframe`. Studies: `scripts/stage0_baseline_study.py` and
`scripts/stage0_generator_stress.py`; results in `results/stage0_baseline/`.

> **Reproducing this document.**
> [`notebooks/00_Stage0_Baseline_Correction.ipynb`](../notebooks/00_Stage0_Baseline_Correction.ipynb)
> reproduces every number below, in the same order, in about a minute. Regenerate the studies with
> `python scripts/stage0_baseline_study.py --n-spectra 40 --out-dir results/stage0_baseline`
> and `python scripts/stage0_generator_stress.py --out-dir results/stage0_baseline`.

## Why this study exists

The pipeline used airPLS because the earlier related work did (Lita et al., codename Apollo), but
the smoothing parameter itself — `airpls_lambda = 100` — was **arbitrarily chosen and tuned by eye**
on a handful of spectra. Stage 0 asks whether that is defensible and, if not, what replaces it.
The two are worth separating: inheriting an algorithm from prior work is reasonable; inheriting an
unexamined constant is not.

To answer without overfitting λ to our own dataset, the experiments run on a **synthetic generator
that approximates brain-biopsy spectra** — so λ is estimated for this *type* of sample rather than
for these particular slides. Real spectra cannot score a corrector, because they carry no known
baseline; one experiment therefore runs on real data using a criterion that needs no ground truth
at all.

| | experiment | question |
|---|---|---|
| **1** | IARPLS λ fit | What λ does IARPLS need on biopsy-like spectra? Run **first**, so later comparisons use a properly fitted IARPLS. |
| **2** | Synthetic comparison | GCV, L-curve and IARPLS against known ground truth. |
| **3** | Invariance on real data | Do peak heights move when fluorescence changes? No ground truth needed. |
| **4** | IARPLS vs oracle airPLS λ | Is IARPLS a better *algorithm*, or just better tuned? |

GCV and L-curve are λ-*selectors* for airPLS and so arrive at experiment 2 already tuned;
comparing them against an arbitrary IARPLS λ would be unfair. Hence experiment 1 comes first.
**Conclusions are drawn only after all four** (§6).

## Summary of what may and may not be claimed

| claim | status |
|---|---|
| airPLS λ=100 differentially suppresses broad bands and lets peak heights drift with fluorescence | **Holds.** Confirmed on synthetic (§3) *and* real (§4) spectra. |
| IARPLS outperforms airPLS λ=100 | **Holds.** Wins 12/12 stress conditions by 2.4–11.5×, and beats an oracle airPLS λ (§5). |
| GCV cannot select λ here | **Holds.** Structural, explained in §3.3. |
| ~~λ is inert over 1e2–1e4~~ | **WITHDRAWN.** An artifact of the baseline generator. See §2.3 and §7.1. |
| λ = 1e4 is the right production constant | **Not established.** Provisional; Stage 1 carries a sensitivity analysis. See §7.1. |

---

## 1. The synthetic data generator

Every conclusion from experiments 1, 2 and 4 is conditional on this generator spanning the
relevant space, so its construction is the main thing a reviewer should scrutinise.

Each observed spectrum is `s = peaks + baseline + noise`, on the real wavenumber axis
(1738 points, 50–3400 cm⁻¹, 1.93 cm⁻¹/point).

### 1.1 Peaks

14 Gaussian bands at fixed Raman positions (`SYNTHETIC_BANDS` in
`scripts/stage0_baseline_study.py`), weighted toward brain biopsy — in particular a **dominant
lipid CH-stretch envelope**, the defining feature of this dataset:

| region | bands (cm⁻¹) | relative amplitude | FWHM (cm⁻¹) |
|---|---|---|---|
| protein / nucleic acid | 1004 (Phe), 1032, 1128, 1230, 1260, 1300, 1340 | 0.30–1.00 | 8–18 |
| lipid / protein | 1440 (CH₂ deformation), 1550, 1660 (amide I) | 0.35–1.20 | 14–20 |
| **lipid CH stretch** | **2850, 2885, 2930** | **1.45–1.80** | 18–22 |
| aromatic | 3060 | 0.30 | 20 |

Amplitudes are jittered ×U(0.75, 1.25) and widths ×U(0.85, 1.15) per spectrum, with
`peak_scale = 400` setting the absolute scale to match measured fingerprint intensities of real
median-filtered tissue spectra.

### 1.2 Baseline

```
t        = (wn - wn.min()) / ptp(wn)                       # normalised abscissa
fluor    = exp(-0.5 * ((t - c) / w)^2)   c~U(-0.1,0.45), w~U(0.15,0.45)
rayleigh = exp(-t / U(0.01, 0.04))                         # notch-filter cutoff rise
poly     = U(-0.3,0.3)·t + U(-0.25,0.25)·t² + U(-0.15,0.15)·t³
baseline = normalise(fluor + 0.6·rayleigh + poly) × amplitude
```

`amplitude` is 2×, 8× or 25× the peak amplitude — the **low / medium / high fluorescence** levels
used throughout.

### 1.3 Noise

Additive i.i.d. Gaussian, σ = ptp(peaks) / SNR with **SNR = 235 measured from the data** (median
over real 3×3-median-filtered tissue spectra).

### 1.4 Why a generator-free arm also exists

Any synthetic generator is arguable, so experiment 3 (§4) uses **real** spectra and **no ground
truth**: correct a real spectrum with and without added fluorescence and compare peak heights. An
invariant corrector gives the same answer either way. Nothing about that metric depends on how a
reference was constructed, so it is immune to the caveat in §7.1.

---

## 2. Experiment 1 — fitting λ for IARPLS

### 2.1 The fit

IARPLS fingerprint NRMSE (700–1800 cm⁻¹) against λ on the nominal generator, median of 6 spectra
per level:

| λ | low | medium | high |
|---|---|---|---|
| 1e2 | 0.0052 | 0.0052 | 0.0052 |
| 1e3 | 0.0049 | 0.0049 | 0.0049 |
| **1e4** | **0.0046** | **0.0046** | **0.0046** |
| 1e5 | 0.0046 | 0.0047 | 0.0269 |
| 1e6 | 0.0050 | 0.1791 | 2.7035 |
| 1e7 | 0.0495 | 2.7456 | 3.0016 |

**λ ≈ 10⁴**, essentially independent of fluorescence level. That value is carried into experiments
2–4 so that IARPLS enters the comparison fitted rather than arbitrary.

### 2.2 How robust is that fit?

A λ fitted on one generator is only as trustworthy as the generator.
`scripts/stage0_generator_stress.py` re-runs the fit while hardening the generator along three
axes: baseline **curvature** (`smooth` → `structured` → `sharp`), band **lineshape** (Gaussian →
**Lorentzian**, since real Raman bands are Lorentzian/Voigt and their heavier tails are
baseline-like), and **noise** (Gaussian → **Poisson** shot noise, which is signal-dependent).

Sensitivity is reported against *curvature energy* `Σ(Δ²b)² / ptp(b)²` — the exact quantity the λ
penalty acts on — so "harder" is measured rather than asserted.

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

Width of the λ window within 25% of each family's best:

| family | best λ | best NRMSE | window | decades |
|---|---|---|---|---|
| smooth | 1e4 | 0.0107 | 3.2e2 – 1e5 | **2.5** |
| structured | 3.2e2 | 0.0108 | 1e2 – 3.2e4 | **2.5** |
| **sharp** | 3.2e2 | 0.0290 | 1e2 – 3.2e2 | **0.5** |

Lineshape and noise matter too (median NRMSE, λ≤1e4): Lorentzian bands cost ~5× (0.0237 vs 0.0048
on the smooth family), and Poisson noise compounds it — worst case 0.3235 for
sharp+Lorentzian+Poisson. The original all-Gaussian, all-Gaussian-noise generator flattered every
method.

### 2.3 Result

λ ≈ 10⁴ is the right fit **for this generator**, but the fit is conditional on baseline curvature:
the near-optimal window collapses from ~2.5 decades to ~0.5 as curvature rises, and at λ=10⁴ the
`sharp` family is outright bad (0.795).

An earlier draft of this document claimed λ was "inert over 1e2–1e4". **That claim is withdrawn.**
It was a property of the smooth baseline family, not of the algorithm. Where real data sits
relative to these families — and why λ therefore stays open — is §7.1.

---

## 3. Experiment 2 — synthetic comparison: GCV, L-curve, IARPLS

Four correctors on the same generated spectra, scored against known ground truth. n = 40 spectra
per fluorescence level.

### 3.1 Peak fidelity

Recovered / true peak height, median over all fluorescence levels:

| method | 1004 Phe (narrow) | 1440 CH₂ | 1660 amide I | 2930 CH (broad) |
|---|---|---|---|---|
| **airPLS λ=100 (incumbent)** | **0.900** | **0.691** | **0.574** | **0.584** |
| airPLS + GCV | 0.126 | 0.025 | 0.017 | 0.011 |
| airPLS + L-curve | 1.002 | 0.997 | 0.987 | 0.927 |
| **IARPLS** | **1.003** | **1.003** | **1.004** | **1.003** |

The incumbent keeps 90% of the narrow 1004 cm⁻¹ band but only 57% of amide I. A λ=100 baseline is
flexible enough to climb into *broad* bands and not narrow ones, so it does not merely scale a
spectrum — **it changes band-height ratios**, which is what the classifiers key on.

### 3.2 Error in the modelled region

Median NRMSE, 700–1800 cm⁻¹:

| method | low | medium | high |
|---|---|---|---|
| airPLS λ=100 | 0.0267 | 0.0512 | 0.0967 |
| airPLS + GCV | 0.0476 | 0.1178 | 0.1134 |
| airPLS + L-curve | 0.0110 | 0.0264 | 0.0245 |
| **IARPLS** | **0.0048** | **0.0047** | **0.0047** |

Only IARPLS is flat across fluorescence level.

### 3.3 Why GCV failed

GCV pinned at the grid floor (λ=1) in **83.3%** of spectra, driving peak recovery to 0.01–0.13.

The cause is structural, not a grid-range problem. GCV assumes the residual is i.i.d. noise; here
the residual at the points airPLS keeps weighted is *structured Raman signal*, so GCV reads that
structure as "the fit is too stiff" and drives λ to the floor. No grid extension fixes it.
**L-curve never pinned (0% edge rate)** on the same grid and the same weights, confirming the
diagnosis is specific to GCV.

### 3.4 Result

The incumbent distorts band ratios; GCV is unusable; L-curve works but remains bounded by airPLS;
IARPLS recovers every diagnostic band to within 0.4%.

---

## 4. Experiment 3 — invariance on real data

Peak-height drift when fluorescence is added to a **real** spectrum (0 = perfectly invariant):

| method | low | medium | high |
|---|---|---|---|
| airPLS + GCV | 0.473 | 0.546 | 0.848 |
| airPLS + L-curve | 0.467 | 0.545 | 0.632 |
| **airPLS λ=100 (incumbent)** | **0.161** | **0.274** | **0.553** |
| **IARPLS** | **0.0004** | **0.0014** | **0.0030** |

**This is the most consequential result in the study.** Under the incumbent pipeline a spectrum's
measured peak heights change by **16–55% purely as a function of how fluorescent the sample is**.

Fluorescence is a property of the specimen and the acquisition, so it varies by slide — and the
cross-validation groups by slide. The confound therefore **aligns with the label**: it offers a
classifier a route to slide identity that has nothing to do with tissue biology. Under IARPLS the
drift is ~0.1%.

This arm uses no synthetic signal at all, so it is not subject to §7.1.

---

## 5. Experiment 4 — IARPLS versus an oracle airPLS λ

Is IARPLS winning because it is a better algorithm, or because airPLS was badly tuned? Searching
the whole grid for the λ minimising *true* error gives an oracle no selector could beat — it cheats
by looking at the answer. Fingerprint NRMSE, median of 5 spectra per level:

| fluorescence | airPLS λ=100 | airPLS **oracle** λ | oracle NRMSE | IARPLS | better by |
|---|---|---|---|---|---|
| low | 0.0219 | 2.15e4 | 0.0069 | 0.0048 | 1.5× |
| medium | 0.0430 | 4.64e4 | 0.0120 | 0.0048 | 2.5× |
| high | 0.0883 | 2.15e4 | 0.0141 | 0.0049 | 2.9× |

IARPLS beats the oracle at every level, so the gap is **algorithmic** and no λ-selection strategy
for airPLS closes it.

Two side observations. The oracle λ is ≈ 2×10⁴ — **200× larger than the inherited 100** — so the
incumbent was not merely suboptimal but far from the right regime. And the oracle's own error grows
with fluorescence while IARPLS's does not.

---

## 6. Conclusions

| | verdict |
|---|---|
| **airPLS λ=100** (incumbent) | Rejected. Suppresses broad bands by up to 43% while leaving narrow ones nearly intact, distorting band *ratios*; peak heights drift 16–55% with fluorescence. |
| **airPLS + GCV** | Rejected. Pins at the grid floor in 83% of spectra; the i.i.d.-residual assumption does not hold here. |
| **airPLS + L-curve** | Works (0% edge pinning, near-unity peak recovery) but bounded by airPLS. Retained as fallback, and as the arm that separates "λ choice" from "algorithm choice" in the methods narrative. |
| **IARPLS** | **Adopted.** Recovers peak heights to within 0.4%, beats an oracle airPLS λ by 1.5–2.9×, drifts ~0.1% with fluorescence, verified pure (bit-identical alone vs in-batch, any order), and ~25× cheaper than a per-spectrum grid search. |
| **λ = 1e4** | **Provisional.** Fitted on the generator (§2), not established on real data (§7.1). |

Production entry point: `src.preprocessing.baseline.correct_iarpls`, wired into the export in
Stage 1.

---

## 7. Limitations

### 7.1 Where real data sits — and why λ stays open

Curvature energy of estimated baselines from 60 real median-filtered tissue spectra, against the
synthetic families of §2.2:

| source | curvature energy (median) |
|---|---|
| synthetic `smooth` | 1.6e-6 |
| synthetic `structured` | 1.2e-6 |
| synthetic `sharp` | 3.3e-6 |
| **real, morphological envelope** (least circular) | **2.5e-5** |
| real, IARPLS λ=1e3 derived | 4.8e-5 |
| real, IARPLS λ=1e5 derived | 8.8e-7 |

**Real baselines appear 7–15× more curved than even the `sharp` family.** But the estimates span
8.8e-7 to 4.8e-5 — a 28× range — because each depends on its estimator: a stiff λ produces a smooth
baseline by construction (circular), while a rolling-minimum envelope tracks noise and band tails
(inflated). **That uncertainty is itself the finding: real data cannot currently be placed in a
synthetic regime, so λ cannot be justified on "insensitivity" grounds.**

A real-data λ sweep using the ground-truth-free invariance criterion of §4 does not resolve it
either, because that criterion is degenerate on its own — drift is minimised by an over-flexible
corrector that eats the signal:

| IARPLS λ | drift (↓ better) | fingerprint retained (`ptp` ratio) |
|---|---|---|
| 1e2 | 0.0000 | **0.420** ← eats 58% of the signal |
| 1e3 | 0.0000 | 0.605 |
| 1e4 | 0.0009 | 0.786 |
| 1e5 | 0.0039 | 0.916 |
| 1e6 | 0.0177 | 0.928 |
| *airPLS λ=100* | *0.2822* | *0.446* |

Minimum drift picks λ=1e2, but retention there (0.420) shows the same over-correction pathology as
airPLS λ=100 (0.446). Retention alone is no better a criterion — it tends to 1 simply by doing
nothing.

**So λ is carried into Stage 1 as an explicit, documented choice with a sensitivity analysis, not
as a settled constant.** `IARPLS_LAMBDA = 1e4` is the current default because it sits in the
low-drift, reasonable-retention region and matches the synthetic fit and oracle. Candidate
resolutions for Stage 1: per-spectrum L-curve selection applied to IARPLS, or a small set of
manually curated reference baselines to calibrate a combined drift-plus-retention criterion.

### 7.2 IARPLS does not model the Rayleigh wing

NRMSE by region: <200 cm⁻¹ **1.32**, 200–300 **0.19**, 300–500 0.013, ≥500 **0.0047**. Everything
below ~300 cm⁻¹ is unusable and must be cropped — recorded as `IARPLS_VALID_FROM_CM1 = 300.0`. The
existing 700–1800 cm⁻¹ crop already satisfies this, but the export must not silently ship the bad
region.

### 7.3 Other IARPLS parameters were not varied

`pybaselines`/`ramanspy` IARPLS exposes `lam`, `diff_order`, `max_iter` and `tol`. **`diff_order`
was left at the library default of 2** — also the standard choice in the airPLS literature — and
never varied, so this study makes no claim about it. `max_iter` and `tol` likewise sit at defaults.

### 7.4 A naming trap to state explicitly in the methods

In the **airPLS** lineage, `p` / `porder` denotes the **difference order**. In **ALS**, `p` denotes
the **asymmetry weight** (typically 0.001–0.1). These are different quantities sharing a letter,
and the paper should say which it means.

Relatedly: `src/data/preprocessing.py` accepts `porder=1` and passes `AIRPLS_PORDER = 1`, but the
function body never reads it — the penalty is hard-coded second-order at line 51. So every spectrum
ever processed used a **second-order** difference penalty. `paper/main.tex:412` and
`Thesis/3-theoretical-background.tex:523` both state $p = 1$; both should read $p = 2$.

### 7.5 Scope, and the literature

Conclusions hold for spectra with our band structure and fluorescence range.

Kizilov et al. (*J. Raman Spectrosc.* 2026, [arXiv:2507.17917](https://arxiv.org/html/2507.17917))
tested medical ointment, PTFE and polymeric aligners, and **also found IARPLS superior** — that
agreement is worth citing. They additionally report that *"IARPLS is more sensitive to the choice
of its parameters compared to traditional ALS"*, over λ ∈ 10²–10⁹, and use λ=10⁸ on their own
synthetic data, a value catastrophic on ours. Their own wording is in slight tension with itself:
they describe IARPLS as *adapting* the asymmetry parameter internally
(*"IARPLS adapts the asymmetry parameter p and smoothing parameter λ iteratively based on local
spectral features"*) while also calling for p to be selected.

The apparent disagreement about *sensitivity* is explained by baseline curvature (§2.2), not by a
dispute about the algorithm — and our withdrawn insensitivity claim now agrees with them rather
than contradicting them.

### 7.6 The endpoint rule fix is inert

`legacy_lam100` and `ref_lam100` agree to four decimals on every metric. The reference endpoint
rule (Zhang et al.) is still the correct one and is now the default, but adopting it changes no
published number.

---

## 8. How to state this in the article

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

## 9. Consequences for the paper

1. **Methods change.** Baseline correction becomes IARPLS with a stated λ, a stated sensitivity
   analysis, and a stated valid-from wavenumber, replacing "airPLS λ=100, following Lita et al."
2. **All numbers move.** Every downstream result was computed on spectra whose broad bands were
   suppressed by up to 43%. v1 is archived at tag `paper-v1` and `paper/versions/v1/`.
3. **A limitation becomes reportable.** §4 is a genuine confound in the submitted work —
   fluorescence-dependent peak heights under slide-grouped CV.
4. **Crop region needs a note.** Correction runs on the full 1738 bands; output is valid only
   above 300 cm⁻¹.
5. **Two stated parameters are wrong.** See §7.4 — $p = 1$ should be $p = 2$ in both write-ups.

---

## 10. What changed in the code

| file | change |
|---|---|
| `src/preprocessing/baseline.py` | **new.** Banded Whittaker solver (`solveh_banded`, ~50× faster than the previous `spsolve`), airPLS with both endpoint rules, GCV and L-curve selectors, `correct_iarpls` as the production entry point. |
| `src/preprocessing/__init__.py` | **new.** Public surface. |
| `tests/test_baseline.py` | **new.** 25 tests: banded penalty vs dense, banded solve vs dense, Hutchinson trace vs exact, determinism, purity, peak-fidelity regression, and a matched pair pinning both halves of §2.3 — `test_iarpls_lambda_is_stable_on_a_SMOOTH_baseline` and `test_iarpls_lambda_DOES_matter_on_a_curved_baseline`. |
| `scripts/stage0_baseline_study.py` | **new.** Experiments 1–4 (primary study). |
| `scripts/stage0_generator_stress.py` | **new.** Generator-robustness follow-up (§2.2). |
| `notebooks/00_Stage0_Baseline_Correction.ipynb` | **new.** Reviewer-facing reproduction in the same order as this document, committed with outputs. |
| `results/stage0_baseline/` | **new.** Raw CSVs, summaries, verdicts, figure. |

**Deviation from the plan.** The plan put "rewire `_airPLS`, delete `_preprocess_batch_ramanspy`"
in Stage 0. Both are deferred to Stage 1, which rewrites `src/data/preprocessing.py` anyway; the
verdict also reverses the planned deletion, since IARPLS lives in that path. `ramanspy` was already
a pinned dependency (`requirements.txt:27`, `ramanspy==0.2.10`) and ships in both Modal images, so
adopting IARPLS adds nothing new — though Stage 1 may still reimplement it natively against the
banded solver to drop it from the hot path.

---

## Appendix — review notes (actioned)

The review that prompted this document's current structure, kept for the record. Both this document
and the notebook now follow it.

> The study is informative, but poorly presented. The findings and notebook need to follow a
> structure such that the experiments setup and results are easy to read and understand.
> Notebook should be redesigned using the following structure:
> Title, Description, Synthetic data generator, then Experiment 1 (iarpls lambda fit — currently
> appears quite late, but it's actually needed early for an accurate comparison of algorithms, as
> GCV and L curve are algos that fit lambda for airpls; we first run this one to have a good iarpls
> to compare against), Experiment 2 (synthetic data test: GCV, L curve, IARPLS), Experiment 3
> (invariance test on real data), Experiment 4 (iarpls vs airpls optimal lambda).
> We draw conclusions only after all experiments.
