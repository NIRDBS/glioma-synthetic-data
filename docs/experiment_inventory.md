# Experiment Inventory — Suites 500/600/660 + v2 Preprocessing

Status/coverage snapshot across every experiment config from `exp_501` through
`exp_661` (v1 preprocessing, `mil-reframe` branch) plus `exp_2.608` (v2
preprocessing, this branch). Read directly from `results/` and each config's
yaml, not recalled from memory or the (partially stale) `EXPERIMENT_REGISTRY.json`.
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
| `exp_608` | 0.5 | yes | Bayesian (5) only — fit, blind-scored, spatial-Potts-extended | 1 file, via the **circular** (non-blind) posterior |
| `exp_609` | 0.75 | yes | none | none |

The β grid was originally meant to span 8 points (2.0, 1.75, 1.5, 1.25, 1.0,
0.75, 0.5, 0.25) plus two windowed-schedule variants — only 5 of those 10
configs ever got carried past the CVAE stage. `docs/beta_sweep_findings.md`
§4 already flags part of this: "`exp_608`/`609` have not had Tier 3/4 run
against their CVAEs yet (only `exp_607` has)."

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
scorer) — the first config in the project to get all four in one pass.
**Zero spatial validation run** — the clearest hole on this branch specifically.

---

## 5. MIL-tier dependency on β — a correction worth keeping

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
re-running them per β would be redundant, not a new test. Only **Tier 3 and
Tier 4's β-sensitivity is actually open**, and only `exp_607` (β=0.25) has
ever had them run — `exp_608`/`609` have zero Tier 3/4 results, a real gap,
not a redundant rerun waiting to happen.

One further wrinkle: Tier 1's only existing fit (via `exp_501`) is on the
**stale 58-slide split**, never refreshed after the slide-restoration fix,
while Tier 5's fit (via `exp_608`) is already on current 59-slide data. A
clean, current-data Tier 1 number for v1 preprocessing would need one refit,
not eight.

---

## 6. Gaps identified (as of 2026-10-08)

1. **β=1.25/1.5/1.75 and both windowed schedules are dead ends** — CVAEs
   exist, nothing downstream was ever computed. Either finish them or mark
   them explicitly abandoned.
2. **Tier 3/4's β-sensitivity is a one-point result** (`exp_607` only) — can't
   say whether the β=0.25 CbR/`encoder_head` pattern holds at other β values.
3. **Spatial validation is the weakest-covered axis**: full only for
   `exp_501`; partial for `607`/`608` (and `608`'s one file uses the
   circular, not blind, Bayesian posterior — worth rerunning cleanly); zero
   for `609`, `661`, and `2.608`.
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
