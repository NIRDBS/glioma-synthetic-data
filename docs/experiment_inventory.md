# Experiment Inventory — Suites 500/600/660/670 + v2 Preprocessing

Status/coverage snapshot across every experiment config from `exp_501` through
`exp_661` and the new Suite 670 (`exp_671`) (v1 preprocessing, `mil-reframe`
branch) plus `exp_2.608` (v2 preprocessing, this branch). Read directly from
`results/` and each config's yaml, not recalled from memory or the (partially
stale) `EXPERIMENT_REGISTRY.json`.
This is a coverage map — "what was run, what wasn't" — not a findings doc.
Results themselves: `docs/beta_sweep_findings.md` (TRTR/TSTR/TSRTR),
`docs/mil_tiers_findings.md` (MIL tiers), `docs/stage5_spatial_validation.md`
(spatial).

---

## 1. Suite 500 — foundation (v1 preprocessing, binary, `glioma_mil.h5`)

**`exp_501`** (β=2.0): the MIL-reframe baseline. Full suite (TRTR/CVAE/TSTR/
TSRTR/stats_eval) **and** every MIL tier ever built for this dataset: mi-SVM
(1), EM (2 — later proven circular, excluded from all comparisons), attention
(3), CbR both variants (4). The only config with complete spatial validation
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
| `exp_608` | 0.5 | yes | attention (3), CbR both variants (4) — backfilled 2026-10-08/09; Bayesian (5) fit, blind-scored, spatial-Potts-extended | 1 file (bayes_mil), via the **circular** (non-blind) posterior |
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

## 3. Suite 660 — 6-class exploratory

**`exp_661`** (β=0.5, inherited from `exp_608`, not retuned for K=6): own full
TRTR/CVAE/TSTR/TSRTR/stats_eval (can't reuse `exp_501`'s binary TRTR). **Zero
MIL tiers run** — explicit scope decision recorded in the config itself.
Severe class imbalance (5/12/9/13/15/5 bags across LGm1–LGm6) is a hard power
floor, not something more modeling fixes.

## 4. This branch — `exp_2.608` (v2 preprocessing, fully-automated, no clustering)

Same settings as `exp_608` (β=0.5, 120 epochs) replicated on the new
`glioma-preproc-2.h5` (59 slides, 263,495 instances vs v1's 288,678) with a
full retrain of everything — nothing transferred across the dataset change.
Full TRTR/CVAE/TSTR/TSRTR/stats_eval, **and all four testable MIL tiers** (1
mi-SVM, 3 attention, 4 CbR both variants, 5 Bayesian via the genuinely-blind
scorer) — the first config in the project to get all four in one pass, now
fully bag-evaluated:

| tier | bag_acc (κ) | Δ vs Apollo (0.7458) | p |
|---|---|---|---|
| 1 mi-SVM | 0.6271 (0.250) | −0.119 | 0.030 (sig, worse) |
| 3 attention | 0.4915 (−0.023) | −0.254 | 0.002 (sig, worse) |
| 4 CbR `encoder_head` | 0.7966 (0.581) | +0.051 | 0.098 (ns) |
| 4 CbR `reconstruction` | 0.7288 (0.445) | −0.017 | 0.842 (ns) |
| 5 Bayesian (blind, non-circular) | 0.6825 pooled / 0.6898 mean-per-fold | — | (no vs-Tier0 test run; see `docs/tier5_bayesian_mil.md`) |

Same qualitative pattern as v1 preprocessing: nothing beats Apollo/RF/XGBoost
significantly; Tiers 1/3 lose significantly; `encoder_head`'s 0.7966 is
numerically identical to `exp_608`'s own `encoder_head` result (§2) and the
old pre-fix `exp_501` number — very likely just 47/59 being a common
achievable fraction at this n, not a deeper pattern, but noted since it
recurred a third time.

**Zero spatial validation run** — the clearest hole on this branch specifically.

## 5. Suite 670 — intra-mutant LGm subtype (new, 2026-10-08/09)

Clinical two-stage framing: stage 1 is the existing IDH-mutant-vs-wildtype
task; stage 2, run only when stage 1 calls mutant, asks which LGm subtype
(1/2/3) the slide is. Implemented as a standalone one-vs-rest binary
decomposition, **not** a shared-pipeline-code change — see
`scripts/build_glioma_mil_lgm_subtype.py`: filters `glioma_mil.h5` down to
the IDH-mutant population (LGm1/2/3, 26 of 59 slides) and bakes the binary
label directly into a new file's `y_bag`, chosen so the existing,
*unmodified* `nc==2` rule in `pipeline/_common.py` produces the correct
label with zero pipeline code changes.

**`exp_671`** (LGm1 vs {LGm2,LGm3}, β=0.5 inherited from `exp_608`, own
TRTR/CVAE — different population *and* label definition, not reusable):
full TRTR/CVAE/TSTR/TSRTR/stats_eval complete. MIL tiers and spatial
validation **not** run (out of scope for this config, same kind of explicit
decision as `exp_661`'s).

TRTR bag-level (n=26 bags: 5 LGm1, 21 rest) — full metrics in
`docs/mil_tiers_findings.md`-style detail, not just raw accuracy:

| model | bag_acc | balanced_acc | κ | sens. (LGm1) | spec. (rest) | AUROC |
|---|---|---|---|---|---|---|
| Apollo | 0.885 | 0.776 | 0.598 | 0.600 | 0.952 | 0.743 |
| RF | 0.885 | 0.700 | 0.519 | 0.400 | 1.000 | 0.705 |
| XGBoost | 0.885 | 0.700 | 0.519 | 0.400 | 1.000 | 0.695 |
| CNN | 0.731 | 0.529 | 0.062 | 0.200 | 0.857 | 0.467 |

TSTR fails significantly for Apollo/RF/XGBoost (p≤0.012); TSRTR is
statistically indistinguishable from TRTR for every model (RF/XGBoost
identical decisions, p=1.0) — plausibly because 26 bags leaves synthetic
augmentation too little real data to move either direction. Severe power
limitation stated up front in the config itself: every one of the 5 folds
has exactly 1 LGm1 bag in its held-out test set (never 0, so no fold is
degenerate, but per-fold recall is necessarily binary).

**`exp_672`/`exp_673`** (LGm2/LGm3 vs rest): not yet built. Trivial to
produce — `python scripts/build_glioma_mil_lgm_subtype.py --positive-class
{1,2} --out data/processed/glioma_mil_lgm{2,3}_vs_rest.h5`, then copy
`exp_671.yaml` changing only name/paths/processed_file/description.

## 6. Methodological fix found this session: Modal/local Keras version skew

`exp_671`'s `stats_eval` failed on `TypeError: GlorotUniform.__init__() got
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
old-format file (`exp_2.608`) and the broken new-format file
(`exp_671`) — both load correctly now. Full test suite re-run clean
afterward (228 passed, same 2 pre-existing unrelated failures as before
this session). This likely affects *any* future Modal-trained CNN if
Modal's cached image picks up a newer Keras again — worth a permanent
`keras==` pin in the Modal image spec if it recurs.

---

## 7. MIL-tier dependency on β — a correction worth keeping

It is **not** accurate to say "no config besides `exp_607`/`608` has MIL
tiers run" in a way that implies all four tiers are equally β-sensitive.
Checked directly in `pipeline/mil_fit_stage.py` / `pipeline/mil_attention_stage.py`:

- **Tier 1 (mi-SVM)** fits on `fd.X_train` — raw spectra, RF-selected bands.
  No CVAE involved.
- **Tier 5 (Bayesian)** — explicit comment at `mil_fit_stage.py:233`: "PCA
  deliberately, **NOT** the CVAE's latent." Also raw-spectra-only.
- **Tier 4 (CbR, both variants + `latent_rf`)** loads `cvae_weights_dir` and
  runs the frozen encoder/decoder forward pass (`mil_fit_stage.py:143-203`) —
  genuinely CVAE-posterior-dependent.
- **Tier 3 (attention)** loads that fold's frozen CVAE `encoder` and builds
  `AttentionMIL(encoder, ...)` directly on its latent space
  (`mil_attention_stage.py:104-109`) — architecturally β-dependent the same
  way Tier 4 is, just via the encoder rather than encoder+decoder.

**Practical implication**: Tier 1 and Tier 5 are dataset-level fits, not
config-level ones — one fit covers every config sharing a dataset, and
re-running them per β would be redundant, not a new test. Tier 3/4's
β-sensitivity **was** a one-point result (only `exp_607` had been run) —
now resolved (§2, 2026-10-09 update): all three β values have been
backfilled with Tier 3/4 against a consistent Apollo reference, and the
trend is real (Tier 3 monotonically worse at higher β; Tier 4 non-monotonic,
peaking at β=0.5).

One further wrinkle: Tier 1's only existing fit (via `exp_501`) is on the
**stale 58-slide split**, never refreshed after the slide-restoration fix,
while Tier 5's fit (via `exp_608`) is already on current 59-slide data. A
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

- Latent dimension: 16, 23, 64
- Reconstruction loss: MSE/cos
- synthetic/real data ratio

## 9. Gaps identified (updated 2026-10-09)

1. **β=1.25/1.5/1.75 and both windowed schedules are dead ends** — CVAEs
   exist, nothing downstream was ever computed. Either finish them or mark
   them explicitly abandoned. *(unchanged)*
2. ~~Tier 3/4's β-sensitivity is a one-point result~~ — **closed 2026-10-09**,
   see §2/§7: all three β values now have consistent-reference Tier 3/4
   results.
3. **Spatial validation is the weakest-covered axis**: full only for
   `exp_501`; partial for `607`/`608` (and `608`'s one file uses the
   circular, not blind, Bayesian posterior — worth rerunning cleanly); zero
   for `609`, `661`, `2.608`, and the new `671`.
4. **No multi-seed/multi-split repetition exists anywhere in the project** —
   every number cited above is a single run on a single CV split
   (`beta_sweep_methodology.md` §7, `mil_tiers_methodology.md` §9).
5. **`exp_2.608` vs `exp_608`** is currently the only same-β,
   cross-preprocessing comparison available — worth deciding if that's the
   one comparison this paper iteration needs, rather than re-opening the
   full 8-point sweep under v2 preprocessing.
6. **`exp_661` (6-class) has never had any MIL tier run against it** — an
   explicit scope decision, not an oversight, but still open if the paper
   ends up wanting a K=6 MIL result.
7. **`exp_672`/`exp_673` (LGm2/LGm3 vs rest) not yet built** — the two
   remaining one-vs-rest decompositions needed to complete Suite 670's
   stage-2 framing (§5); trivial to produce from the existing build script.
8. **`exp_671` has no MIL tiers or spatial validation** — same explicit
   out-of-scope decision as `exp_661`'s, not an oversight, but open if the
   paper wants MIL-level evidence for the subtype task too.
