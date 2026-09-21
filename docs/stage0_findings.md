# Stage 0 findings — baseline correction

**Status: complete.** Branch `mil-reframe`. Study: `scripts/stage0_baseline_study.py`,
results in `results/stage0_baseline/`, figure `stage0_baseline_study.png`.

> **Reproducing this document.** Every number below is reproduced, with the reasoning
> shown step by step, in
> [`notebooks/00_Stage0_Baseline_Correction.ipynb`](../notebooks/00_Stage0_Baseline_Correction.ipynb).
> It runs top to bottom in about a minute (committed with outputs, so it can also just be
> read). Live cells use small samples; the headline tables in its §8 load the committed
> full study so they match this document exactly. To regenerate that study:
> `python scripts/stage0_baseline_study.py --n-spectra 40 --out-dir results/stage0_baseline`

> **The plan's recommendation did not survive contact with the data.** Stage 0 was
> specified as "per-spectrum λ by GCV, L-curve as fallback; IARPLS kept only as a
> comparator." GCV fails outright, L-curve works but is unnecessary, and the
> comparator wins on every metric. The recommendation is now **IARPLS at λ=1e4**.

---

## 1. What was measured

Two arms, because the question needs ground truth and real spectra do not have it.

| arm | construction | what it can show |
|---|---|---|
| **synthetic** | Raman-like signal (14 Gaussian bands at real positions) + known baseline + noise at the measured SNR (235) | exact recovery error and peak fidelity; zero circularity — the "true" signal was never produced by any corrector |
| **invariance** | real median-filtered tissue spectra, corrected with and without extra synthetic fluorescence | whether peak heights move when fluorescence changes; needs **no** ground truth, so it cannot be gamed by the choice of reference |

Fluorescence levels: 2×, 8×, 25× the signal peak-to-peak. 40 spectra per cell.
Baselines are broad Gaussian fluorescence + Rayleigh wing + low-order polynomial drift.

Arms compared: `legacy_lam100` (reproduces the current pipeline), `ref_lam100`
(endpoint rule fixed), `gcv`, `lcurve`, `iarpls`.

---

## 2. Headline: the inherited λ=100 differentially suppresses broad bands

Peak height recovered / true, median over all fluorescence levels:

| method | 1004 Phe (narrow) | 1440 CH₂ | 1660 amide I | 2930 CH (broad) |
|---|---|---|---|---|
| **legacy λ=100 (current)** | **0.900** | **0.691** | **0.574** | **0.584** |
| gcv | 0.126 | 0.025 | 0.017 | 0.011 |
| lcurve | 1.002 | 0.997 | 0.987 | 0.927 |
| **iarpls** | **1.003** | **1.003** | **1.004** | **1.003** |

The current pipeline loses **10% of the narrow 1004 cm⁻¹ band but 43% of amide I**.
The distortion is *differential*: a λ=100 baseline is flexible enough to climb into
broad bands but not narrow ones, so it does not merely scale spectra — **it changes
band-height ratios**, which is precisely what the classifiers key on.

Error in the modelled region (700–1800 cm⁻¹), median NRMSE:

| method | low | medium | high | all |
|---|---|---|---|---|
| gcv | 0.0476 | 0.1178 | 0.1134 | 0.1099 |
| legacy λ=100 | 0.0267 | 0.0512 | 0.0967 | 0.0502 |
| lcurve | 0.0110 | 0.0264 | 0.0245 | 0.0213 |
| **iarpls** | **0.0048** | **0.0047** | **0.0047** | **0.0047** |

IARPLS is **10.7× more accurate than the current pipeline**, and — unlike every other
arm — its error is *flat* across fluorescence level.

---

## 3. The invariance result is the one that matters for the paper

Peak-height drift on **real** spectra when fluorescence is added (0 = perfectly invariant):

| method | low | medium | high |
|---|---|---|---|
| gcv | 0.473 | 0.546 | 0.848 |
| lcurve | 0.467 | 0.545 | 0.632 |
| **legacy λ=100 (current)** | **0.161** | **0.274** | **0.553** |
| **iarpls** | **0.0004** | **0.0014** | **0.0030** |

Under the current pipeline, a spectrum's measured peak heights change by **16–55%
purely as a function of how fluorescent the sample is**. Fluorescence is a property of
the specimen and acquisition — it varies by slide and patient — so measured peak
heights partially encode *which slide a spectrum came from* rather than its biology.

That is a confound with teeth: it gives a classifier a route to slide identity that has
nothing to do with tissue, and the CV design groups by slide, so the confound aligns
exactly with the label. It does not invalidate the existing results, but it is a
plausible contributor to instance-level performance and it needs stating. Under IARPLS
the drift is ~0.1–0.3%, i.e. gone.

---

## 4. Why GCV failed

GCV pinned at the grid floor (λ=1) in **83.3%** of spectra, driving peak recovery to
0.01–0.13 — the baseline swallows the signal.

The cause is structural, not a tuning problem. GCV assumes the residual is iid noise.
Here the residual at the points airPLS keeps weighted is not noise, it is *structured
Raman signal*, so GCV reads that structure as "fit is too stiff" and drives λ to the
floor. No grid extension fixes this; the criterion is answering a different question
than the one being asked. **L-curve, by contrast, never pinned (0% edge rate)** and
selected sensible λ (median 4.6e3–1e5), confirming the diagnosis is specific to GCV.

---

## 5. Why IARPLS rather than a per-spectrum selector

Three independent reasons, in order of weight:

1. **It beats the best achievable fixed λ.** Searching the whole grid for the λ that
   minimises true error against known ground truth gives an *oracle* no selector could
   beat. IARPLS still wins (fingerprint NRMSE, median of 5 spectra per level):

   | fluorescence | airPLS λ=100 | airPLS **oracle** λ | oracle NRMSE | IARPLS | IARPLS better by |
   |---|---|---|---|---|---|
   | low | 0.0219 | 2.15e4 | 0.0069 | 0.0048 | 1.5× |
   | medium | 0.0430 | 4.64e4 | 0.0120 | 0.0048 | 2.5× |
   | high | 0.0883 | 2.15e4 | 0.0141 | 0.0049 | 2.9× |

   So this is a better *algorithm*, not a better-tuned one — no selector for airPLS can
   close that gap. Note also that the oracle λ is ~2e4, i.e. **200× larger than the
   inherited 100**, and that even the oracle degrades with fluorescence while IARPLS does
   not.
2. **λ is inert, which is stronger than selecting it well.** Fingerprint NRMSE across
   the sensitivity sweep:

   | λ | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 |
   |---|---|---|---|---|---|---|
   | low fluor. | 0.0052 | 0.0048 | 0.0046 | 0.0046 | 0.0047 | 0.0351 |
   | medium | 0.0052 | 0.0048 | 0.0046 | 0.0047 | 0.1433 | 2.187 |
   | high | 0.0052 | 0.0048 | 0.0046 | 0.0269 | 4.680 | 4.680 |

   Over **1e2–1e4 the result is identical to three decimal places at every fluorescence
   level**. The constant is therefore not "fitted" to anything — the no-global-parameter
   rule is satisfied because the parameter demonstrably carries no information, which is
   easier to defend in print than a per-spectrum selector.
3. **It is ~25× cheaper.** One solve per spectrum instead of a 25-point grid search,
   each point of which runs airPLS to convergence.

**Verified pure**: IARPLS output is bit-identical whether a spectrum is corrected alone
or inside a batch, in any order (`test_iarpls_is_pure_batch_vs_alone`). Preprocessing
stays fold-independent, so one export still suffices.

---

## 6. Caveats and things that must be carried into Stage 1

- **IARPLS does not model the Rayleigh wing.** NRMSE by region: <200 cm⁻¹ **1.32**,
  200–300 **0.19**, 300–500 0.013, ≥500 **0.0047**. Everything below ~300 cm⁻¹ is
  unusable and must be cropped. This is not a practical loss — that region is the notch
  filter cutoff and is routinely discarded — but the export must not silently ship it.
  Recorded as `IARPLS_VALID_FROM_CM1 = 300.0`.
- **The endpoint "fix" is inert.** `legacy_lam100` and `ref_lam100` agree to four
  decimals on every metric. The reference endpoint rule is still the correct one and is
  now the default, but fixing it changes no published number — worth knowing before
  attributing any change to it.
- **ramanspy is already a pinned dependency** (`requirements.txt:27`, `ramanspy==0.2.10`)
  and ships in both Modal images, so adopting IARPLS adds no new dependency. Stage 1
  should still consider reimplementing it natively (~30 lines) against the new banded
  solver, to drop the dependency from the hot path and gain the same ~50× speedup the
  airPLS rewrite got.
- **L-curve airPLS is retained**, tested, and usable. It is the fallback if IARPLS is
  ever rejected, and it is the arm that isolates "λ choice" from "algorithm choice" in
  the paper's methods narrative.

---

## 7. What changed in the code

| file | change |
|---|---|
| `src/preprocessing/baseline.py` | **new.** Banded Whittaker solver (`solveh_banded`, ~50× faster than the old `spsolve`), airPLS with both endpoint rules, GCV and L-curve selectors, and `correct_iarpls` as the recommended production entry point. |
| `src/preprocessing/__init__.py` | **new.** Public surface. |
| `tests/test_baseline.py` | **new.** 24 tests: banded penalty vs dense, banded solve vs dense, Hutchinson trace vs exact, determinism, purity, peak-fidelity regression, λ-inertness regression. |
| `scripts/stage0_baseline_study.py` | **new.** The study, reproducible via `--seed`. |
| `notebooks/00_Stage0_Baseline_Correction.ipynb` | **new.** Reviewer-facing reproduction of every number here, committed with outputs. |
| `results/stage0_baseline/` | **new.** Raw CSVs, summaries, `stage0_verdict.json`, figure. |

**Deviation from the plan:** the plan put "rewire `_airPLS`, delete
`_preprocess_batch_ramanspy`" in Stage 0. Both are deferred to Stage 1, because
(a) `src/data/preprocessing.py` is rewritten there anyway and touching a known-broken
file twice invites mistakes, and (b) the verdict reverses the deletion — IARPLS lives in
that path, so the correct move is to lift it into `src/preprocessing/baseline.py` (done)
and delete the SavGol/cropping wrapper during the Stage 1 rewrite, not before.

---

## 8. Consequences for the paper

1. **Methods change.** Baseline correction becomes IARPLS(λ=1e4) with a stated
   insensitivity range and a stated valid-from wavenumber, replacing "airPLS λ=100,
   following Lita et al.". The λ-sensitivity table is the justification and should be a
   supplementary figure.
2. **All numbers move.** Every downstream result was computed on spectra whose broad
   bands were suppressed by up to 43%. Existing results are archived at `paper-v1` and
   `paper/versions/v1/`; the reframed paper reports the corrected pipeline.
3. **A limitation becomes reportable.** §3 is a genuine confound in the submitted work —
   fluorescence-dependent peak heights under slide-grouped CV. Stating it plainly, with
   the fix, is stronger than hoping no reviewer computes it.
4. **Crop region needs a note.** Correction runs on the full 1738 bands (settled), but
   the output is only valid above 300 cm⁻¹. The existing 700–1800 crop already satisfies
   this; the constraint should be explicit rather than incidental.
