# Experiment Inventory — Suites 500/600/660/670 + v2 Preprocessing

Status/coverage snapshot across every experiment config from `exp_501` through
`exp_661` and Suite 670 (`exp_671`) (v1 preprocessing, `mil-reframe` branch)
plus `exp_2.607`/`exp_2.608` (v2 preprocessing, this branch — the first
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
| 5 Bayesian (blind) | 0.7119 merged, −0.034, p=0.592 (ns) | identical — see §4.3 |

Same qualitative pattern as v1 preprocessing throughout: nothing beats
Apollo/RF/XGBoost significantly; Tiers 1/3 lose significantly at both β.

### 4.3 A real finding about compute, not just results: Tiers 1/5 are byte-identical across `2.607`/`2.608`

Confirmed directly, not assumed: Tier 5's chain log-likelihoods
(`[691936.9, 691969.5, 691962.9]` for fold_0, etc.) and Tier 1's fold_0
posterior array are **bit-identical** between the two configs — expected,
since both tiers fit on raw spectra/PCA only (§7, dataset-level not
β-dependent) and the two configs share the exact same dataset. Also
confirmed: the shared Tier 0 Apollo/RF reference itself is bit-identical
between the two configs (deterministic sklearn fit, identical data/seed) —
so even the *comparison* numbers transfer, not just the posteriors. Caught
**before** wasting a second ~4-hour mi-SVM refit for `exp_2.607`: that
run was killed mid-flight and `exp_2.608`'s results copied directly into
`exp_2.607`'s latent store instead (Tier 1, Tier 5, and their respective
bag-eval/spatial-validation JSON files). Tier 3/4 (genuinely CVAE/β-dependent)
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

## 5a. `exp_3.608` — viable_legacy ablation (new, 2026-10-12/13, exploratory)

Direct request: "same config as 608 except the viable_legacy flag
activated" — does restricting training to the old, narrower, manually-
curated tumour-cluster subset (`viable_legacy==True` in `glioma_mil.h5`,
NMF(100)→GMM(3)→human-picked cluster per slide, `src/data/preprocessing.py`)
change anything, before investing further in this direction. **Not** a new
preprocessing pipeline — same `glioma_mil.h5`, a stricter instance filter
applied at load time.

Implementation: new `preprocessing.viable_legacy_only` config key, read by
`GliomaDataLoader.from_config`/`_load_data` (`src/data/loader.py`) — filters
`X`/`y_class`/`groups`/`coords`/`tissue_cluster` to `viable_legacy==True`
immediately after load, before the CV split (which is keyed by slide name
and unaffected — every slide keeps a healthy pixel count post-filter, min
865/slide, confirmed before running anything). Config: `configs/exp_3.608.yaml`,
same β=0.5/120-epoch/hyperparameters as `exp_608`, own TRTR (exp_501's/
exp_608's TRTR are trained on the unfiltered set and not valid references
here). Filter keeps 227,344/288,678 instances (78.8%); binary class balance
is tighter than the unfiltered set (114,826/112,518 vs 162,742/125,936).

### Results: base suite only (TRTR/CVAE/TSTR/TSRTR/stats_eval), complete

| | Apollo | RF | XGBoost | CNN |
|---|---|---|---|---|
| TRTR, full mask (`exp_501`/`608` shared) | 0.763 | 0.814 | 0.814 | 0.712 |
| TRTR, viable_legacy-only (`exp_3.608`) | 0.680 | 0.711 | 0.747 | 0.640 |
| Δ | −0.083 | −0.103 | −0.067 | −0.072 |
| TSTR, full mask (`exp_608`, β=0.5) | 0.695 | 0.831 | 0.831 | 0.678 |
| TSTR, viable_legacy-only (`exp_3.608`) | 0.470 | 0.673 | 0.635 | 0.530 |
| Δ | −0.225 | −0.158 | −0.196 | −0.148 |
| TSRTR, full mask (`exp_608`, β=0.5) | 0.831 | 0.831 | 0.814 | 0.746 |
| TSRTR, viable_legacy-only (`exp_3.608`) | 0.720 | 0.740 | 0.752 | 0.732 |
| Δ | −0.111 | −0.091 | −0.062 | −0.014 |

**Verdict: don't pursue further.** Every cell, every regime, every
classifier is worse under the legacy mask — the widened v1 mask (already
the project default) strictly dominates on this evidence. Caveat for
precision: `exp_3.608`'s validation set is itself drawn from the filtered
subset (the loader filters before the CV split), so this is "best model
trained+evaluated on the narrow mask" vs "...on the wide mask," not a
fixed-test-set comparison — the direction/magnitude (7–22 points, zero
exceptions across 12 cells) is large enough not to need that control to be
convincing, but it's the honest caveat if a reviewer asks. Internally,
`exp_3.608` still replicates the project-wide pattern (Apollo and CNN gain
significantly from TSRTR, p=0.012 both; RF/XGB don't) — the ablation
doesn't break the underlying mechanism, just the ceiling.

### To do, only if a reviewer specifically asks for this as supplementary material

Nothing below has been run. Pattern to follow: `2.609`'s backlog
(§4), since it's the same shape of task.

1. **mi-SVM (Tier 1)** — `python -m pipeline.mil_fit_stage --config exp_3.608
   --methods misvm --backend modal` (fast, fold-parallel). Dataset-level, so
   should be checkable for consistency against `608`'s or `3.608`'s own
   repeat, not assumed — see §9/§2a's mi-SVM reproducibility caveat (fold-
   level non-determinism is possible even on identical data).
2. **Bayesian (Tier 5)** — `python -m pipeline.mil_fit_stage --config
   exp_3.608 --methods bayes_mil --backend local` (Modal not supported for
   this method yet). Also dataset-level, but confirm bit-identity against
   another config empirically before reusing anywhere, same discipline as
   §2a.
3. **Attention (Tier 3) + CbR (Tier 4, both variants)** — CVAE-dependent,
   must be fit fresh here (not transferable from `608`): `pipeline/mil_attention_stage.py`,
   `pipeline/mil_fit_stage.py --methods cbr`.
4. **Bag-eval for all four tiers** vs Apollo/RF/XGB/CNN
   (`scripts/stage6_bag_eval.py`, `scripts/stage6_attention_bag_eval.py`).
5. **Family A (spatial) + Family B (key-instance)** validation suite —
   `scripts/stage5_spatial_validation.py`, `scripts/stage6_shuffled_label_control.py`,
   `scripts/cvae_weighted_rf_vote.py` — mirrors `2.607`/`2.608`'s full
   coverage (§4.4/§4.5).
6. **Formal bootstrap of `exp_3.608` vs `exp_608` directly** (not just the
   raw deltas above) if a reviewer wants a p-value on the ablation itself,
   not only each config's own TRTR-vs-TSTR/TSRTR internal comparison.

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

| β | config | base suite | MIL tiers 3/4 | MIL tiers 1/5 | Family A/B validation |
|---|---|---|---|---|---|
| 2.0 | `exp_501` | **done** (reference) | **done** | **done** | **done** (the one fully-characterized config) |
| 1.0 | `exp_601` | **done** | not scheduled | not scheduled | not scheduled |
| 1.25 | `exp_602` | CVAE only — **not scheduled downstream**, deliberate dead end (gap #1) | — | — | — |
| 1.5 | `exp_603` | CVAE only — **not scheduled downstream** | — | — | — |
| 1.75 | `exp_604` | CVAE only — **not scheduled downstream** | — | — | — |
| windowed A | `exp_605` | CVAE only — **not scheduled downstream** | — | — | — |
| windowed B | `exp_606` | CVAE only — **not scheduled downstream** | — | — | — |
| 0.25 | `exp_607` | **done** | **done** | **done** (backfilled this session) | **done** (shuffled control still missing for mi-SVM/Bayesian specifically, §9/§2a) |
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
| 0.5/0.5 | `exp_703` | **scheduled**, queued behind `701`/`702` remediation | not scheduled | not scheduled |
| pure cosine (γ=0,δ=1) | `exp_704` | **scheduled**, queued behind `exp_703` | not scheduled | not scheduled |

### 8a.4 Synthetic/real ratio for TSRTR (article §8.III; β=0.5, dim=32 throughout)

`n_synth = round(ratio · n_real)` (`pipeline/tsrtr_stage.py`) — CVAE/TSTR
untouched by this axis, reused from `exp_608` for both new points.

| ratio | config | base suite | MIL tiers | validation |
|---|---|---|---|---|
| 0.5 | `exp_705` | **scheduled**, queued behind `exp_704` | not scheduled | not scheduled |
| 1.0 | `exp_608` | **done** (anchor) | **done** | **done** |
| 2.0 | `exp_706` | **scheduled**, queued behind `exp_705` | not scheduled | not scheduled |

### 8a.5 Preprocessing / strong-signal mask (β=0.5 throughout; v1=`glioma_mil.h5`, v2/v3 own files)

| variant | config | instances | base suite | MIL tiers | validation |
|---|---|---|---|---|---|
| v1, default (NMF/GMM/manual curation, effectively fixed threshold 0.6 — confirmed via `strong_signal_mask.npy` provenance) | `exp_608` | 288,678 | **done** | **done** | **done** |
| v1, restricted to legacy tumour-cluster subset (`viable_legacy` filter) | `exp_3.608` | 227,344 | **done** — verdict: uniformly worse, don't pursue (§5a) | not scheduled | not scheduled |
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

### 8a.6 Queue order, confirmed (2026-10-14)

Everything above tagged "scheduled"/"running" is genuinely in one live,
strictly-sequential queue (checked directly, not assumed): `exp_701`
(running) → `exp_702` stats_eval fix → 601–609 latent-collapse diagnostic →
`exp_703` → `exp_704` → `exp_705` → `exp_706`. Nothing is silently missing
from the queue; `602`–`606` are deliberately unscheduled (dead ends, gap #1),
not an oversight. `exp_2.609`'s and `exp_4.608`'s MIL-tier/validation gaps
mirror `exp_3.608`'s §5a to-do list and are not yet scheduled.

## 9. Gaps identified (updated 2026-10-12, second pass)

**Update, same day, third pass — Tier 1 (mi-SVM) and Tier 5 (Bayesian) now
exist for `607`/`608`/`609`.** Per-config detail in §2a below. Headline:
both tiers are confirmed dataset-level (identical out-of-sample bag
accuracy across all three β values — mi-SVM 0.7627/κ=0.5301, Bayesian blind
0.7288, neither significant vs Apollo), but `exp_501`'s own existing mi-SVM
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

### 2a. Tier 1 / Tier 5 detail for `607`/`608`/`609` (added 2026-10-12)

mi-SVM and Bayesian Tier 5 were fit fresh for `exp_607` (mi-SVM via Modal,
fold-parallel; Bayesian locally), then independently fit fresh for `exp_608`
and `exp_609` (mi-SVM) to check reproducibility directly rather than assume
it; Bayesian's `exp_607` fit was confirmed bit-identical to `exp_608`'s
existing fit (all 5 folds, posterior and `in_sample` arrays, plus the
chain-log-likelihoods/R-hat diagnostics to many decimal places) and copied
into `exp_609` rather than re-run a third time.

**Tier 1 (mi-SVM)**, identical across all three configs:
bag_acc=0.7627 (κ=0.5301), not significant vs any Tier-0 reference
(Δ=−0.001 p=1.00 vs Apollo, Δ=−0.054 p=0.41 vs RF/XGB, Δ=+0.049 p=0.58 vs
CNN). `k_diagnostic_columns_competition_frac=0.966` confirms this is a
genuine (non-circular) result. Spatial (lag 3): z=1.20, essentially no
coherence — consistent with mi-SVM showing no spatial signal anywhere else
in the project. **Not run**: the shuffled-label control (the local-only,
~4hr-per-run path this script requires makes 3 separate runs expensive
enough that it wasn't done without being asked specifically for it).

**Tier 5 (Bayesian)**, identical across all three configs: the plain
bag-eval (0.9153/κ=0.8315) is the method's own **known-circular** metric
(`competition_frac=0.0000` — correctly triggers the script's own
circularity warning; Model (C) makes this trivially near-1.0, not a
finding). The genuine number is the blind, merged bag classification:
**0.7288**, vs. Apollo's 0.7627 (Δ=−0.033, p=0.668, not significant).
Per-fold circular accuracy is 1.0 for 4/5 folds and **0.643 for fold_2** —
the one fold whose Gibbs chains didn't converge (R-hat=9.09; fold_3 also
fails, R-hat=11.8). Cross-fold π_b correlation is uneven (range −0.42 to
+0.99 pairwise) and weakest for the non-converged folds. Spatial (lag 3):
z=33.66, strong pooled coherence — but this reuses the same circular
posterior as the bag-eval and **has no shuffled-label control** (no such
control exists in code for `bayes_mil` yet, `scripts/stage6_shuffled_label_control.py`
only supports misvm/em/attention/cvae/cbr) — same uncontrolled caveat
`608` already carried alone, now shown to apply identically to `607`/`609`.

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
   exception. **`608`'s Bayesian spatial file still uses the circular, not
   blind, posterior** — worth rerunning cleanly, not yet done. ~~Still zero
   spatial/key-instance coverage for ... `2.608`~~ — **closed, same day**:
   `exp_2.607`/`exp_2.608` now both have the complete Family A/B suite (§4.4,
   §4.5) — the first configs in the project with full coverage. Still zero
   for `609`'s Bayesian tier (N/A — Tier 5 was never fit for 609), `661`,
   and `671`.
4. **No multi-seed/multi-split repetition exists anywhere in the project** —
   every number cited above is a single run on a single CV split
   (`beta_sweep_methodology.md` §7, `mil_tiers_methodology.md` §9).
5. ~~`exp_2.608` vs `exp_608` is currently the only same-β, cross-preprocessing
   comparison~~ — **closed**: `exp_2.607` (β=0.25) now exists alongside
   `exp_2.608` (β=0.5), mirroring the `607`/`608` pair under v2 preprocessing
   (§4). Whether to extend to a full v2 β-sweep (0.75, etc.) is still an open
   scope decision, not yet needed for the two-point comparison now available.
6. **`exp_661` (6-class) has never had any MIL tier run against it** — an
   explicit scope decision, not an oversight, but still open if the paper
   ends up wanting a K=6 MIL result.
7. **`exp_672`/`exp_673` (LGm2/LGm3 vs rest) not yet built** — the two
   remaining one-vs-rest decompositions needed to complete Suite 670's
   stage-2 framing (§5); trivial to produce from the existing build script.
8. **`exp_671` has no MIL tiers or spatial validation** — same explicit
   out-of-scope decision as `exp_661`'s, not an oversight, but open if the
   paper wants MIL-level evidence for the subtype task too.
9. **Planned, not started: a simplified 2-state Model (C) variant.**
   Current Model (C) gives each instance 4 possible latent states
   ($z_{bi}\in\{0_1,0_2,0_3,Y_b\}$ — 3 separate null sub-types [stroma/
   necrosis/normal] plus the bag's diagnostic class). Hypothesis to test:
   the 3-way null split may be *hurting* rather than helping, adding
   complexity/parameters the data doesn't support distinguishing — already
   hinted at by Tier 2/5's own `null_component_collapse_diagnostic`
   (pairwise cosine similarity near 1.0 would mean the 3 null components
   collapsed into duplicates anyway, §6 of `mil_tiers_methodology.md`).
   Plan: a 2-state variant, $z_{bi}\in\{0,Y_b\}$ — one combined null state,
   not three — for Tier 2 (EM) and Tier 5 (Bayesian) specifically, the only
   tiers with this explicit multi-null structure (Tiers 1/3/4 already
   report a single lifted non-diagnostic mass, so this change doesn't apply
   to them). Deliberately queued, not started — machine is at capacity with
   the Family A/B backfill (§2, §8 of `docs/mil_validation_test_suite.md`).
10. **Queued: investigate the spatial-coherence exceptions more closely —
    now two candidates, not one.** (a) `607`'s attention (v1, β=0.25, one
    data point, modest margin). (b) **New, 2026-10-12**: CbR `encoder_head`
    under v2 preprocessing — real beats shuffled at *both* tested β values
    (`exp_2.607` and `exp_2.608`, §4.4/§4.5), and is corroborated by its own
    key-instance result at both β too, a more consistent signal than (a)'s
    single point. Neither has a significance test on the real-vs-shuffled
    gap itself yet. `docs/mil_validation_test_suite.md` §A/§B,
    `docs/stage6_attention_mil.md` §10, `docs/stage6_generative_mil.md` §10.
    Deliberately deferred, not started.
