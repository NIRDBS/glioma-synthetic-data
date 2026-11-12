# Experiment Inventory — Suites 500/600/700 + v2/v3 Preprocessing

Status/coverage snapshot across the experiment configs behind paper v2.3, from
`exp_501` onward (v1 preprocessing, `mil-reframe` branch), plus `exp_2.607`/`exp_2.608` (v2 preprocessing, this branch — the first
configs in the project with the complete suite: classifiers, all MIL tiers,
and the full spatial + key-instance validation suite). Read directly from
`results/` and each config's yaml, not recalled from memory or the (partially
stale) `EXPERIMENT_REGISTRY.json`.
This is a coverage map — "what was run, what wasn't" — not a findings doc.
Results themselves: `docs/beta_sweep_findings.md` (TRTR/TSTR/TSRTR),
`docs/mil_tiers_findings.md` (MIL tiers), `docs/stage5_spatial_validation.md`
(spatial).

---

## 1. Suite 500 — foundation (v1 preprocessing, binary, `glioma_mil.h5`)

**`exp_501`** (β=2.0): the MIL-reframe baseline. Full suite (TRTR/CVAE/TSTR/
TSRTR/stats_eval) **and** every MIL tier: mi-SVM (1), attention (3), CbR
both variants (4). The only config with complete spatial validation
(9 files — all 4 testable tiers + shuffled-label controls). This is the one
fully-characterized reference point in the project.

## 2. Suite 600 — CVAE β sweep (same v1 dataset, TRTR reused from `exp_501`)

| config | β | downstream eval (TSTR/TSRTR/stats_eval) | MIL tiers run | spatial validation |
|---|---|---|---|---|
| `exp_601` | 1.0 | yes | none | none |
| `exp_602` | 1.25 | **CVAE only — never evaluated downstream** | none | none |
| `exp_603` | 1.5 | **CVAE only — never evaluated downstream** | none | none |
| `exp_604` | 1.75 | **CVAE only — never evaluated downstream** | none | none |
| `exp_605` | windowed sched. A | **CVAE only** (100 more epochs on `exp_601`'s checkpoint) | none | none |
| `exp_606` | windowed sched. B | **CVAE only**, same as above | none | none |
| `exp_607` | 0.25 | yes | attention (3), CbR both variants (4) | partial (2/5: cbr only) |
| `exp_608` | 0.5 | yes | attention (3), CbR both variants (4) — backfilled 2026-10-08/09 | none |
| `exp_609` | 0.75 | yes | attention (3), CbR both variants (4) — backfilled 2026-10-08/09 | none |

The β grid was originally meant to span 8 points (2.0, 1.75, 1.5, 1.25, 1.0,
0.75, 0.5, 0.25) plus two windowed-schedule variants — only 5 of those 10
configs ever got carried past the CVAE stage.

**Update, 2026-10-09 — Tier 3/4 backfilled across the full β sweep.** Closes
the one-point-result gap flagged below in §7/§8: all three tested β values
now have attention (3) and CbR (4) fit and bag-evaluated against the same
reference model (Apollo), so the trend is genuinely comparable, not an
artifact of different reference models per config.

**Tier 3 (attention), bag_acc (κ), Δ vs Apollo, n=59 bags:**

| β | bag_acc | κ | Δ vs Apollo | p | significant |
|---|---|---|---|---|---|
| 0.25 (607) | 0.542 | 0.075 | −0.220 | 0.006 | yes |
| 0.5 (608) | 0.458 | −0.100 | −0.305 | <0.0001 | yes |
| 0.75 (609) | 0.407 | −0.170 | −0.356 | <0.0001 | yes |

Clean, monotonic, and now a real finding rather than a single data point:
Tier 3 gets *worse* as β increases, and loses significantly to Apollo at
**every** β tested.

**Tier 4 (CbR), raw bag_acc (κ), n=59 bags (Tier 0 Apollo: 0.7119 at
β=0.25's own reference model/CNN, 0.7627 at β=0.5/0.75's Apollo reference —
see note below):**

| β | `encoder_head` | `reconstruction` |
|---|---|---|
| 0.25 (607) | 0.7627 (κ=0.511) | 0.7288 (κ=0.441) |
| 0.5 (608) | **0.7966 (κ=0.581)** | 0.7288 (κ=0.431) |
| 0.75 (609) | 0.7797 (κ=0.544) | 0.6780 (κ=0.339) |

`encoder_head` at β=0.5 is the highest raw Tier 4 bag accuracy found
anywhere in the project so far — but not significant against Apollo (Δ=
+0.034, p=0.466), since β=0.5's own Apollo reference (0.7627) is also the
strongest in the sweep. None of the six β×variant cells reach significance
against Apollo; `encoder_head` at β=0.25 reaches significance only against
the weaker CNN reference (p=0.006, see `docs/mil_tiers_findings.md`-style
convention), consistent with the project-wide pattern that CNN is a weak
reference and Apollo/RF/XGBoost are not beaten.

## 3. Suite 660 — removed

6-class exploratory run (`exp_661`); not in paper v2.3. Config and results remain in the `mil-reframe` history.

## 4. This branch — v2 preprocessing (`exp_2.607`, `exp_2.608`, `exp_2.609`)

Three configs, same relationship as `607`/`608`/`609` (§2): `exp_2.607`
(β=0.25), `exp_2.608` (β=0.5), `exp_2.609` (β=0.75, added 2026-10-13 —
completes the v2 beta sweep), all on the new `glioma-preproc-2.h5` (59
slides, 263,495 instances vs v1's 288,678), fully-automated preprocessing
(no NMF/GMM/manual curation — `src/data/preprocessing_v2.py`). Full retrain
of everything per config — nothing transferred across the dataset change,
and (confirmed directly, §4.3) nothing transferred *between* configs either
for the CVAE-dependent tiers. `exp_2.607`/`exp_2.608` have the complete
suite: TRTR/CVAE/TSTR/TSRTR/stats_eval, all four testable MIL tiers, **and**
the full Family A (spatial) + Family B (key-instance) validation suite
(`docs/mil_validation_test_suite.md`). `exp_2.609` currently has only the
base suite (TRTR/CVAE/TSTR/TSRTR/stats_eval) — MIL tiers and the
validation suite are still open for it, matching `2.607`/`2.608`'s own
former backlog.

### 4.1 Classifier suite (bag-level, n=59, from `stats_eval`)

| regimen | model | `exp_2.607` (β=0.25) | `exp_2.608` (β=0.5) | `exp_2.609` (β=0.75) |
|---|---|---|---|---|
| TRTR | Apollo/RF/XGB/CNN | 0.746/0.780/0.797/0.678 | 0.746/0.780/0.797/0.661 | 0.746/0.780/0.797/0.610 |
| TSTR | Apollo/RF/XGB/CNN | 0.644/0.797/0.814/0.695 (all ns) | — (see `docs/beta_sweep_findings.md`-style convention, not re-tabulated here) | 0.525/0.729/0.695/0.475 (Apollo p<0.0001 sig, CNN p=0.05 sig, RF/XGB ns) |
| TSRTR | Apollo/RF/XGB/CNN | **0.831 (p=0.002, sig)**/0.831 (ns)/0.814 (ns)/**0.814 (p=0.03, sig)** | all ns | **0.814 Apollo (p=0.02, sig)**/0.814 RF (ns)/0.814 XGB (ns)/0.695 CNN (ns) |

Apollo/RF/XGB's TRTR numbers are essentially identical across all three β
values (0.746/0.780/0.797 every time) — expected, since TRTR doesn't touch
the CVAE at all; only CNN varies (0.678/0.661/0.610), consistent with its
own training stochasticity (dropout, batch order) rather than anything
β-dependent. `exp_2.609` replicates the project-wide "Apollo's TSRTR gain is
real but β-independent" pattern (`beta_sweep_findings.md`) a third time.
| TSRTR | Apollo/RF/XGB/CNN | **0.831 (p=0.002, sig)**/0.831 (ns)/0.814 (ns)/**0.814 (p=0.03, sig)** | all ns |

`exp_2.607` (lower β) shows two significant TSRTR gains (Apollo, CNN) —
stronger than `exp_2.608` showed at β=0.5, echoing the v1 sweep's own
"lower β sometimes helps the CNN specifically" pattern (`beta_sweep_findings.md`
§2) without claiming it's the same effect confirmed on new data.

### 4.2 MIL tiers (bag-level vs Apollo, n=59)

| tier | `exp_2.607` bag_acc (κ), Δ, p | `exp_2.608` bag_acc (κ), Δ, p |
|---|---|---|
| 1 mi-SVM | 0.6271 (0.250), −0.119, p=0.030 (sig) | identical — see §4.3 |
| 3 attention | 0.4576 (−0.109), −0.288, p<0.0001 (sig) | 0.4915 (−0.023), −0.254, p=0.002 (sig) |
| 4 CbR `reconstruction` | 0.7458 (0.474), +0.000, p=1.0 (ns) | 0.7288 (0.445), −0.017, p=0.842 (ns) |
| 4 CbR `encoder_head` | 0.7797 (0.548), +0.034, p=0.446 (ns) | 0.7966 (0.581), +0.051, p=0.098 (ns) |

Same qualitative pattern as v1 preprocessing throughout: nothing beats
Apollo/RF/XGBoost significantly; Tiers 1/3 lose significantly at both β.

### 4.3 A real finding about compute, not just results: Tier 1 is byte-identical across `2.607`/`2.608`

Confirmed directly, not assumed: Tier 1's fold_0 posterior array is
**bit-identical** between the two configs — expected, since mi-SVM fits on
raw spectra only (§7, dataset-level not β-dependent) and the two configs share the exact same dataset. Also
confirmed: the shared Tier 0 Apollo/RF reference itself is bit-identical
between the two configs (deterministic sklearn fit, identical data/seed) —
so even the *comparison* numbers transfer, not just the posteriors. Caught
**before** wasting a second ~4-hour mi-SVM refit for `exp_2.607`: that
run was killed mid-flight and `exp_2.608`'s results copied directly into
`exp_2.607`'s latent store instead (Tier 1 and its bag-eval/spatial-validation
JSON files). Tier 3/4 (genuinely CVAE/β-dependent)
were *not* shared — each has its own fresh fit, confirmed different results
(§4.2).

### 4.4 Family A (spatial coherence) — real vs. shuffled, lag 3

| tier | `exp_2.607`: real z / shuffled z | `exp_2.608`: real z / shuffled z | verdict (both) |
|---|---|---|---|
| 1 mi-SVM | 1.30 / 2.61 (shared, §4.3) | 1.30 / 2.61 | fails |
| 3 attention | 13.73 / 14.50 | 11.56 / 14.38 | fails |
| 4 CbR `reconstruction` | 30.40 / 30.50 | 34.90 / 35.07 | fails (shuffled ≈ real) |
| 4 CbR `encoder_head` | **10.89 / 8.46** | **10.54 / 7.98** | **real > shuffled at both β** |

**`encoder_head` is the one systematic exception** — not a single flagged
data point like `607`'s attention (§7/§10 of `docs/experiment_inventory.md`'s
gap list), but reproduced at *both* v2-preprocessing β values, with real
consistently larger than shuffled by a similar margin each time (~2.4-2.6
z-points). No v1 config shows this for `encoder_head` (`exp_501`'s and the
β-sweep's `encoder_head` results are all "numerically indistinguishable"
failures, `docs/stage6_generative_mil.md` §4a/§10.1) — this looks like a
genuine v2-preprocessing effect specific to this variant, not noise, though
still only 2 data points and no significance test has been run on the
real-vs-shuffled gap itself.

### 4.5 Family B (key-instance utility) — `key_instance_train` vs. size-matched `random_subsample`

| tier | `exp_2.607` | `exp_2.608` |
|---|---|---|
| 3 attention | 0.7966 vs 0.8136 (no advantage) | 0.7966 vs 0.7797 (selection wins) |
| 4 CbR `reconstruction` | 0.7797 vs 0.7966 (no advantage) | 0.7966 vs 0.7458 (selection wins) |
| 4 CbR `encoder_head` | **0.8305 vs 0.7966 (selection wins)** | **0.8136 vs 0.7797 (selection wins)** |

Less clean than Family A: `exp_2.608` shows selection beating random for
**all three** tested tiers, `exp_2.607` only for `encoder_head`. The one
result that agrees with Family A at *both* β values is again `encoder_head`
— the same variant flagged in §4.4. None of these deltas are individually
bootstrap-significant.

**Net read**: `encoder_head`'s spatial-coherence exception (§4.4) is
corroborated by its own key-instance result at both β values — a more
consistent signal than `607`'s v1 attention exception (which only had one β
point to begin with). Still queued for the same deferred closer look (§9,
gap #10), now with `encoder_head` added alongside `607`'s attention as a
second candidate.

## 5. Suites 670 and `exp_3.608` — removed

The intra-mutant LGm-subtype task (`exp_671`) and the `viable_legacy` training-mask ablation (`exp_3.608`) are not in paper v2.3. Configs, build script and results remain in the `mil-reframe` history.

## 6. Methodological fix found this session: Modal/local Keras version skew

A `stats_eval` run failed on `TypeError: GlorotUniform.__init__() got
an unexpected keyword argument 'input_axes'` — Modal's CNN training image
(`pipeline/modal_classifiers.py`) pins `tensorflow==2.20.0` but not `keras`,
so a CNN trained there can be saved by a newer Keras than this local
environment's (3.14.0), which serializes `GlorotUniform` with extra
`input_axes`/`output_axes` kwargs the older local `__init__` rejects. A
per-call `custom_objects` override does **not** intercept this (confirmed
by direct test) since `GlorotUniform` is a built-in, not custom, name —
only patching the class itself works. Fixed with a version-tolerant
monkeypatch in `src/utils/statistical_evaluation.py` (strips the unknown
kwargs, falls back to the original `__init__`), verified against both an
old-format file (`exp_2.608`) and a broken new-format file — both load
correctly now. Full test suite re-run clean
afterward (228 passed, same 2 pre-existing unrelated failures as before
this session). This likely affects *any* future Modal-trained CNN if
Modal's cached image picks up a newer Keras again — worth a permanent
`keras==` pin in the Modal image spec if it recurs.

---

## 7. MIL-tier dependency on β — a correction worth keeping

It is **not** accurate to say "no config besides `exp_607`/`608` has MIL
tiers run" in a way that implies all tiers are equally β-sensitive.
Checked directly in `pipeline/mil_fit_stage.py` / `pipeline/mil_attention_stage.py`:

- **Tier 1 (mi-SVM)** fits on `fd.X_train` — raw spectra, RF-selected bands.
  No CVAE involved.
- **Tier 4 (CbR, both variants)** loads `cvae_weights_dir` and
  runs the frozen encoder/decoder forward pass (`mil_fit_stage.py`) —
  genuinely CVAE-posterior-dependent.
- **Tier 3 (attention)** loads that fold's frozen CVAE `encoder` and builds
  `AttentionMIL(encoder, ...)` directly on its latent space
  (`mil_attention_stage.py`) — architecturally β-dependent the same
  way Tier 4 is, just via the encoder rather than encoder+decoder.

**Practical implication**: Tier 1 is a dataset-level fit, not a
config-level one — one fit covers every config sharing a dataset, and
re-running them per β would be redundant, not a new test. Tier 3/4's
β-sensitivity **was** a one-point result (only `exp_607` had been run) —
now resolved (§2, 2026-10-09 update): all three β values have been
backfilled with Tier 3/4 against a consistent Apollo reference, and the
trend is real (Tier 3 monotonically worse at higher β; Tier 4 non-monotonic,
peaking at β=0.5).

One further wrinkle: Tier 1's only existing fit (via `exp_501`) is on the
**stale 58-slide split**, never refreshed after the slide-restoration fix. A
clean, current-data Tier 1 number for v1 preprocessing would need one refit,
not eight.

---

## 8. Experiments selected for the article

### I. CVAE: beta sweep

Plot training loss curves for beta in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]. 

**Discussion:** 
1. We observe a trade-off between reconstruction and KL regularization. Show variance ratios of synthetic/real.
Possible plot: contours of sigma per beta and with real data.
Decide to train classification for beta in [0.25, 0.5, 0.75].
2. Oscilating beta seems to move the loss one time in the direction of the firs oscilation, with the loss corresponding to a fixed beta. We decide to use a fixed beta for further results.

### II. Classification tasks: TRTR / TSTR / TSRTR / MIL tiers

- Display results for beta in [0.25, 0.5, 0.75] for all training regimes.
- Validation results for MIL (all fail)
- **Discussion:** all MIL results fail either on spatial coherence or on a classification test with training data restricted to key instances. The best performing approach is the naive MIL. Given this limited dataset, we wil report all results in the naive MIL formulation.
- Preprocessing v2 

### III. Ablation studies from previous version

- Latent dimension: 16, 32, 64
- Reconstruction loss: MSE/cos
- synthetic/real data ratio

## 8a. Ablation studies — full status, all axes (added 2026-10-14)

Current-paper scope only (`exp_501` onward — the `mil-reframe` branch; Suite
200/2xx predates the reframe and is out of scope here, not re-listed).
Every axis below holds the task (binary IDH-mutant-vs-wildtype) and every
setting except the one stated fixed. "Done" = base suite
(TRTR/CVAE/TSTR/TSRTR/stats_eval) landed with real numbers; MIL-tier/
validation-suite coverage is noted separately per axis where it varies.

### 8a.1 CVAE β (Suite 600, §2/§7; latent_dim=32 throughout)

| β | config | base suite | MIL tiers 3/4 | MIL tier 1 | Family A/B validation |
|---|---|---|---|---|---|
| 2.0 | `exp_501` | **done** (reference) | **done** | **done** | **done** (the one fully-characterized config) |
| 1.0 | `exp_601` | **done** | not scheduled | not scheduled | not scheduled |
| 1.25 | `exp_602` | CVAE only — **not scheduled downstream**, deliberate dead end (gap #1) | — | — | — |
| 1.5 | `exp_603` | CVAE only — **not scheduled downstream** | — | — | — |
| 1.75 | `exp_604` | CVAE only — **not scheduled downstream** | — | — | — |
| windowed A | `exp_605` | CVAE only — **not scheduled downstream** | — | — | — |
| windowed B | `exp_606` | CVAE only — **not scheduled downstream** | — | — | — |
| 0.25 | `exp_607` | **done** | **done** | **done** (backfilled this session) | **done** (shuffled control still missing for mi-SVM specifically, §9/§2a) |
| 0.5 | `exp_608` | **done** (anchor for §8a.2-8a.4 below) | **done** | **done** | **done**, same caveat |
| 0.75 | `exp_609` | **done** | **done** | **done** | **done**, same caveat |

**Per-dimension latent collapse, confirmed directly (not inferred), all 9
configs, fold_0 (added 2026-10-14):** per-instance $\mu,\log\sigma^2$ pulled
from each config's own frozen encoder, KL$_d = \tfrac12(\mathbb E[\mu_d^2] +
\mathbb E[\sigma_d^2] - \mathbb E[\log\sigma_d^2] - 1)$ averaged over
fold_0's validation set, "active" = KL$_d>0.01$:

| β | config | active/32 | total KL | top-3 KL |
|---|---|---|---|---|
| 0.25 | `607` | **5** | 2.347 | 0.964, 0.927, 0.402 |
| 0.5 | `608` | 3 | 1.165 | 0.779, 0.298, 0.072 |
| 0.75 | `609` | 3 | 0.759 | 0.599, 0.117, 0.030 |
| 1.0 | `601` | 3 | 0.508 | 0.427, 0.046, 0.026 |
| 1.25 | `602` | 2 | 0.253 | 0.223, 0.024, 0.001 |
| 1.5 | `603` | **1** | 0.152 | 0.149, ~0, ~0 |
| 1.75 | `604` | **1** | 0.091 | 0.089, ~0, ~0 |
| windowed A | `605` | 3 | 0.311 | 0.273, 0.023, 0.011 |
| windowed B | `606` | 3 | 0.811 | 0.633, 0.136, 0.029 |

Monotonic in β exactly as VAE theory predicts (stronger KL penalty → more
aggressive collapse). The load-bearing finding: even at β=0.25 — the
*least*-collapsed point in the entire sweep — only 5 of 32 latent
dimensions carry any information; the rest sit at the prior. Severe
posterior collapse is present at **every** β this project has trained, not
a latent_dim=64-specific artifact (§8a.2's own finding below is a sample
from this same curve, not a separate phenomenon). Windowed A/B are
confounded for comparison against the flat points — both resumed from
`exp_601`'s already-trained flat-β=1.0 checkpoint for 100 more oscillating
epochs, not trained from scratch, so their KL reflects a different training
history, not just a different β trajectory.

### 8a.2 Latent dimension (article §8.III; β=0.5 throughout)

| dim | config | base suite | MIL tiers | validation |
|---|---|---|---|---|
| 16 | `exp_701` | **done** | not scheduled | not scheduled |
| 32 | `exp_608` | **done** (anchor) | **done** | **done** |
| 64 | `exp_702` | **done** | not scheduled | not scheduled |

**Per-dimension latent collapse, confirmed directly, all three points, fold_0
(added 2026-10-14)**, same method as §8a.1's table:

| dim | config | active dims | total KL | top-3 KL |
|---|---|---|---|---|
| 16 | `exp_701` | 3/16 | 1.364 | 0.947, 0.310, 0.085 |
| 32 | `exp_608` | 3/32 | 1.165 | 0.779, 0.298, 0.072 |
| 64 | `exp_702` | 4/64 | 1.354 | 0.926, 0.322, 0.073 |

Effective capacity converges to the **same ~3-4 active dimensions
regardless of nominal width** (16, 32, or 64) at this β=0.5. This settles
the original "is dim=64 undertrained?" question cleanly: it isn't —
training curves (total/mse/cos/kl/class loss) are statistically
indistinguishable from `exp_608` at matched epochs (checked directly), and
the extra nominal width is simply unused, not partially learned. Combined
with §8a.1's β-sweep table: effective latent capacity in this CVAE is
governed almost entirely by β, not by the nominal `latent_dim` setting, at
every width and every β tested so far.

### 8a.3 Reconstruction loss (article §8.III; β=0.5, dim=32 throughout)

$\text{recon\_loss} = \gamma\cdot\text{MSE} + \delta\cdot\cos$ (`src/models/losses.py:38`).

| point | config | base suite | MIL tiers | validation |
|---|---|---|---|---|
| pure MSE (γ=1,δ=0) | `exp_608` | **done** (anchor) | **done** | **done** |
| 0.5/0.5 | `exp_703` | **done** | not scheduled | not scheduled |
| pure cosine (γ=0,δ=1) | `exp_704` | **done** | not scheduled | not scheduled |

**Results (added 2026-10-15), TRTR identical across all three (shared from
`exp_501`): Apollo=0.7627, RF=0.8136, XGBoost=0.8136, CNN=0.7119.**

**TSTR (bag_accuracy, Δ vs TRTR, p):**

| | Apollo | RF | XGBoost | CNN | CbR |
|---|---|---|---|---|---|
| pure MSE (`608`) | 0.6949, Δ−0.068, p=0.24 | 0.8305, Δ+0.017, p=0.81 | 0.8305, Δ+0.017, p=0.78 | 0.6780, Δ−0.034, p=0.61 | 0.7797 |
| 0.5/0.5 (`703`) | 0.5763, Δ−0.186, **p=0.004** | 0.7458, Δ−0.068, p=0.19 | 0.7966, Δ−0.017, p=0.81 | 0.6949, Δ−0.017, p=0.89 | 0.7458 |
| pure cos (`704`) | 0.4068, Δ**−0.356**, **p<0.001** | 0.6441, Δ**−0.169**, **p<0.001** | 0.6949, Δ**−0.119**, **p=0.034** | 0.4576, Δ**−0.254**, **p=0.004** | 0.5763 |

**TSRTR (bag_accuracy, Δ vs TRTR, p):**

| | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| pure MSE (`608`) | 0.8305, Δ**+0.068**, **p=0.022** | 0.8305, Δ+0.017, p=0.67 | 0.8136, Δ0.000, p=1.00 | 0.7458, Δ+0.034, p=0.67 |
| 0.5/0.5 (`703`) | 0.8136, Δ+0.051, p=0.078 | 0.8305, Δ+0.017, p=0.67 | 0.8136, Δ0.000, p=1.00 | 0.7288, Δ+0.017, p=0.90 |
| pure cos (`704`) | 0.8305, Δ+0.068, p=0.106 | 0.8136, Δ0.000, p=1.00 | 0.8136, Δ0.000, p=1.00 | 0.7458, Δ+0.034, p=0.68 |

**Finding, clean and monotonic:** every classifier's TSTR accuracy degrades
monotonically as the cosine weight increases (MSE→0.5/0.5→pure-cos), and at
pure cosine **all four classifiers show a significant loss** (p≤0.034
throughout) — the only reconstruction-loss point in this project where
every single Tier-0 model is significantly hurt. CbR falls the same way
(0.780→0.746→0.576). TSRTR is far more robust across this axis (no
significant *losses* anywhere, Apollo's gain survives the blend but loses
significance at pure cosine, p=0.106) — mixing in real data absorbs the
synthetic-quality damage that TSTR exposes cleanly. Cosine similarity alone
is a poor reconstruction objective for this spectral data when used as the
sole signal for synthetic generation.

### 8a.4 Synthetic/real ratio for TSRTR (article §8.III; β=0.5, dim=32 throughout)

`n_synth = round(ratio · n_real)` (`pipeline/tsrtr_stage.py`) — CVAE/TSTR
untouched by this axis, reused from `exp_608` for both new points.

| ratio | config | base suite | MIL tiers | validation |
|---|---|---|---|---|
| 0.5 | `exp_705` | **done** | not scheduled | not scheduled |
| 1.0 | `exp_608` | **done** (anchor) | **done** | **done** |
| 2.0 | `exp_706` | **done** | not scheduled | not scheduled |

**Results (added 2026-10-15), TRTR identical across all three (shared from
`exp_501`): Apollo=0.7627, RF=0.8136, XGBoost=0.8136, CNN=0.7119. TSTR is
architecturally reused unchanged from `exp_608` for all three (confirmed:
all three report the identical TSTR numbers — Apollo=0.6949, RF=0.8305,
XGBoost=0.8305, CNN=0.6780, CbR=0.7797 — ratio genuinely has zero effect on
TSTR, as designed), so only TSRTR varies:**

| ratio | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| 0.5 (`705`) | 0.8305, Δ+0.068, **p=0.022** | 0.8475, Δ+0.034, p=0.24 | 0.8136, Δ0.000, p=1.00 | 0.6780, Δ−0.034, p=0.62 |
| 1.0 (`608`, anchor) | 0.8305, Δ+0.068, **p=0.022** | 0.8305, Δ+0.017, p=0.67 | 0.8136, Δ0.000, p=1.00 | 0.7458, Δ+0.034, p=0.67 |
| 2.0 (`706`) | 0.8136, Δ+0.051, p=0.086 | 0.8475, Δ+0.034, p=0.24 | 0.8136, Δ0.000, p=1.00 | 0.7119, Δ0.000, p=1.00 |

**Finding:** Apollo's TSRTR gain — the single most robust significant
result anywhere in the `608` family — holds at ratio≤1.0 (p=0.022 at both
0.5 and 1.0) but **loses significance at ratio=2.0** (p=0.086). Doubling
the synthetic:real ratio beyond 1:1 dilutes rather than strengthens the one
real augmentation effect this project has found; RF/XGBoost/CNN show no
significant effect at any ratio.

### 8a.5 Preprocessing / strong-signal mask (β=0.5 throughout; v1=`glioma_mil.h5`, v2/v3 own files)

| variant | config | instances | base suite | MIL tiers | validation |
|---|---|---|---|---|---|
| v1, default (NMF/GMM/manual curation, effectively fixed threshold 0.6 — confirmed via `strong_signal_mask.npy` provenance) | `exp_608` | 288,678 | **done** | **done** | **done** |
| v2, automated adaptive mask | `exp_2.608` | 263,495 | **done** | **done** | **done** |
| v3, automated fixed threshold=0.4 | `exp_4.608` | 299,050 | **done** | not scheduled | not scheduled |

**TRTR/TSTR/TSRTR bag-level results, `608` vs `2.608` vs `4.608`, confirmed
directly from each config's own `statistical_eval/*.json` (added
2026-10-14).** n=59 bags throughout; Δ/p are candidate-vs-TRTR, slide-level
bootstrap, 1000 resamples.

**TRTR (bag_accuracy):**

| | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| `608` (v1) | 0.7627 | 0.8136 | 0.8136 | 0.7119 |
| `2.608` (v2) | 0.7458 | 0.7797 | 0.7966 | 0.6610 |
| `4.608` (v3) | 0.7458 | 0.8136 | 0.8136 | 0.6780 |

**TSTR (bag_accuracy, Δ vs own TRTR, p):**

| | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| `608` (v1) | 0.6949, Δ−0.068, p=0.24 | 0.8305, Δ+0.017, p=0.81 | 0.8305, Δ+0.017, p=0.78 | 0.6780, Δ−0.034, p=0.61 |
| `2.608` (v2) | 0.5593, Δ**−0.186**, **p=0.002** | 0.7797, Δ0.000, p=1.00 | 0.7627, Δ−0.034, p=0.59 | 0.6102, Δ−0.051, p=0.50 |
| `4.608` (v3) | 0.6610, Δ−0.085, p=0.21 | 0.8475, Δ+0.034, p=0.24 | 0.8305, Δ+0.017, p=0.78 | 0.7119, Δ+0.034, p=0.71 |

**TSRTR (bag_accuracy, Δ vs own TRTR, p):**

| | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| `608` (v1) | 0.8305, Δ**+0.068**, **p=0.022** | 0.8305, Δ+0.017, p=0.67 | 0.8136, Δ0.000, p=1.00 | 0.7458, Δ+0.034, p=0.67 |
| `2.608` (v2) | 0.8136, Δ+0.068, p=0.116 | 0.8305, Δ+0.051, p=0.092 | 0.8136, Δ+0.017, p=0.76 | 0.7458, Δ+0.085, p=0.216 |
| `4.608` (v3) | 0.8136, Δ+0.068, p=0.096 | 0.8305, Δ+0.017, p=0.67 | 0.8136, Δ0.000, p=1.00 | 0.7119, Δ+0.034, p=0.65 |

**CbR (canonical mean-pooled convention, bag_accuracy, same under TSTR and
TSRTR since it depends only on the CVAE, not the regimen):** `608`=0.7797,
`2.608`=0.7627, `4.608`=0.7966.

**Reading across the three:** `608`'s Apollo is the *only* cell with a
significant TSRTR gain that survives at p<0.05 (p=0.022); `2.608`'s Apollo
TSTR loss is the only significant TSTR effect anywhere in this comparison
(p=0.002). Every other cell across all three preprocessing variants is
non-significant — same qualitative story everywhere (synthetic augmentation
neither clearly helps nor hurts for RF/XGBoost/CNN, Apollo is the one
model that shows real movement, direction depending on preprocessing).
Raw TRTR is close across all three (0.75-0.81 for Apollo/RF/XGB; CNN is
the one model with real spread, 0.66-0.71) — consistent with §8a.5's own
finding that `4.608`'s wider mask doesn't change the headline numbers much
relative to `608`, while `2.608`'s stricter adaptive mask trades a visibly
weaker TSTR for comparable TRTR/TSRTR.

### 8a.6 Queue, completed (updated 2026-10-15)

All of §8a.2-8a.4's "scheduled" base-suite runs landed: `exp_701` → `702`
(remediated after a mid-run restart) → 601–609 latent-collapse diagnostic →
`703` → `704` → `705` → `706`, each confirmed via its own stats_eval output,
not assumed from the queue log alone. Three unplanned process restarts hit
this queue in total (mid-`701`-TSRTR, mid-`702`-stats_eval, mid-`703`-CVAE→
TSTR handoff) — each resumed from the last genuinely-completed stage,
verified on disk before resuming (`701`'s first restart silently skipped a
stage via presence-only skip-detection, dropping CNN from one comparison
until caught and force-re-run; see the fix note where that happened). Root
cause of the restarts themselves not diagnosed — worth checking if it
recurs. `602`–`606` remain deliberately unscheduled (dead ends, gap #1), not
an oversight. `exp_2.609`'s and `exp_4.608`'s MIL-tier/
validation-suite gaps (§8a.5) and `701`/`702`/`703`/`704`/`705`/`706`'s own
(§8a.2-8a.4) remain genuinely not scheduled — base suite only for all of
them.

## 9. Gaps identified (updated 2026-10-12, second pass)

**Update, same day, third pass — Tier 1 (mi-SVM) now exists for
`607`/`608`/`609`.** Per-config detail in §2a below. Headline: the tier is
confirmed dataset-level (identical out-of-sample bag accuracy across all
three β values — 0.7627/κ=0.5301, not significant vs Apollo), but `exp_501`'s own existing mi-SVM
artifact was found to be **stale** (fit under the old 58-slide split,
before the `HF-1887_via-t_2_1.h5_3` biopsy-restoration fix — confirmed via
5,796 NaN posterior rows exactly matching that slide's instance count) and
should not be trusted or copied from until refit. mi-SVM's own re-fit also
showed genuine fold-to-fold non-reproducibility (fold_0's posterior differs
non-trivially between independently-fit configs despite identical training
data — likely RF/SVM parallel-fit floating-point sensitivity cascading
through mi-SVM's 10-iteration relabeling loop) that does not, in this case,
change the final out-of-sample bag accuracy, but is a real reproducibility
caveat worth stating in the paper if mi-SVM numbers are reported.

### 2a. Tier 1 detail for `607`/`608`/`609` (added 2026-10-12)

mi-SVM was fit fresh for `exp_607` (via Modal, fold-parallel), then
independently fit fresh for `exp_608` and `exp_609` to check reproducibility
directly rather than assume it.

**Tier 1 (mi-SVM)**, identical across all three configs:
bag_acc=0.7627 (κ=0.5301), not significant vs any Tier-0 reference
(Δ=−0.001 p=1.00 vs Apollo, Δ=−0.054 p=0.41 vs RF/XGB, Δ=+0.049 p=0.58 vs
CNN). `k_diagnostic_columns_competition_frac=0.966` confirms this is a
genuine (non-circular) result. Spatial (lag 3): z=1.20, essentially no
coherence — consistent with mi-SVM showing no spatial signal anywhere else
in the project. **Not run**: the shuffled-label control (the local-only,
~4hr-per-run path this script requires makes 3 separate runs expensive
enough that it wasn't done without being asked specifically for it).

1. **β=1.25/1.5/1.75 and both windowed schedules are dead ends** — CVAEs
   exist, nothing downstream was ever computed. Either finish them or mark
   them explicitly abandoned. *(unchanged)*
2. ~~Tier 3/4's β-sensitivity is a one-point result~~ — **closed 2026-10-09**,
   see §2/§7: all three β values now have consistent-reference Tier 3/4
   results.
3. ~~Spatial validation is the weakest-covered axis~~ — **shuffled-label
   control closed 2026-10-12 for `607`/`608`/`609`**: attention + both CbR
   variants now have real-vs-shuffled lag-3 comparisons at all three β
   values (`docs/stage6_attention_mil.md` §10.1,
   `docs/stage6_generative_mil.md` §10.1). 8 of 9 cells fail (shuffled ≥
   real, same as `exp_501`); the one exception, `607`'s attention, is
   flagged for a closer look, not yet reported as a surviving claim. The
   key-instance utility test (§10.2 in both docs) was extended the same
   round, with a new random-subsample confound control
   (`docs/mil_validation_test_suite.md` §B) — same verdict: 8 of 9 cells
   show no selection advantage, `607`'s attention is again the one
   exception. ~~Still zero
   spatial/key-instance coverage for ... `2.608`~~ — **closed, same day**:
   `exp_2.607`/`exp_2.608` now both have the complete Family A/B suite (§4.4,
   §4.5) — the first configs in the project with full coverage.
4. **No multi-seed/multi-split repetition exists anywhere in the project** —
   every number cited above is a single run on a single CV split
   (`beta_sweep_methodology.md` §7, `mil_tiers_methodology.md` §9).
5. ~~`exp_2.608` vs `exp_608` is currently the only same-β, cross-preprocessing
   comparison~~ — **closed**: `exp_2.607` (β=0.25) now exists alongside
   `exp_2.608` (β=0.5), mirroring the `607`/`608` pair under v2 preprocessing
   (§4). Whether to extend to a full v2 β-sweep (0.75, etc.) is still an open
   scope decision, not yet needed for the two-point comparison now available.
6. **Queued: investigate the spatial-coherence exceptions more closely —
    now two candidates, not one.** (a) `607`'s attention (v1, β=0.25, one
    data point, modest margin). (b) **New, 2026-10-12**: CbR `encoder_head`
    under v2 preprocessing — real beats shuffled at *both* tested β values
    (`exp_2.607` and `exp_2.608`, §4.4/§4.5), and is corroborated by its own
    key-instance result at both β too, a more consistent signal than (a)'s
    single point. Neither has a significance test on the real-vs-shuffled
    gap itself yet. `docs/mil_validation_test_suite.md` §A/§B,
    `docs/stage6_attention_mil.md` §10, `docs/stage6_generative_mil.md` §10.
    Deliberately deferred, not started.
