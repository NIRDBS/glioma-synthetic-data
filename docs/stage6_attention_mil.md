# Stage 6 (Tier 3) — Attention MIL

**Status: complete for the real fit, the bag-level evaluation against all
four Tier 0 references, and spatial validation. The shuffled-bag-label
control has not been run for this tier.** New `src/mil/attention.py`,
`pipeline/mil_attention_stage.py`, `scripts/stage6_attention_bag_eval.py`.
Full architecture and design rationale: docs/mil_tiers_methodology.md §6.3–6.4
(methodology only, no results — this file is the results counterpart,
mirroring docs/stage6_generative_mil.md's treatment of Tier 4).

The honest summary: a real, non-degenerate attention model that trains
cleanly on all 5 folds, recovers genuine (if modest) lag≥3 spatial
coherence — weaker than Tier 4's `reconstruction` variant, stronger than
Tier 1's mi-SVM — but is a **significantly worse bag-level classifier than
three of the four Tier 0 references** (Apollo, RF, XGBoost), and not
significantly better than the fourth (the CNN, the weakest Tier 0 reference
by a wide margin). This tier's own debugging process also surfaced three
real, sequentially-discovered bugs before any of the numbers below could be
trusted — recorded here because each is a property of attention MIL on this
data, not a one-off implementation slip.

> **Reproducing this.** `pytest tests/test_stage6_attention_mil.py
> tests/test_stage6_attention_bag_eval.py`. Real fit:
> `python -m pipeline.mil_attention_stage --config exp_501` (needs exp_501's
> `cvae` stage already trained — reuses its saved per-fold encoder weights,
> frozen). Spatial validation:
> `python scripts/stage5_spatial_validation.py --config exp_501 --method
> attention`. Bag-level eval:
> `python scripts/stage6_attention_bag_eval.py --config exp_501
> --reference-model {cnn,apollo,rf,xgb}`.

## 1. What this tier is, briefly

Gated attention (Ilse et al. 2018) over instance logits, computed on each
fold's frozen CVAE encoder trunk, trained on randomly-sampled 228-instance
sub-bags (sized from the π_min identifiability floor, per
docs/mil_tiers_methodology.md §6.3). Full rationale for every design choice
— why logits rather than embeddings, why the trunk is frozen, why
sub-bagging, the regularization, the two separate posterior methods — is in
the methodology doc; not repeated here.

## 2. Three real bugs, found and fixed before trusting any number

All three were caught by smoke-testing before the real 5-fold run, each
diagnosed from first principles and verified empirically rather than
guessed at:

1. **Bag-identity overfitting.** The first, unregularized smoke test showed
   sub-bag training loss falling steadily while held-out bag accuracy got
   *worse* with more training — the signature of the model memorizing which
   of the ~38 real training bags it was looking at, rather than learning
   diagnostic content, given thousands of sub-bag draws per epoch from so
   few real bags. Fixed with dropout (0.3), L2 (1e-3), a smaller attention
   dimension (32), and per-(sub-)bag feature centering.
2. **A fold-specific vanishing-gradient stall.** fold_1's smoke test froze
   at `loss ≈ ln(2)` (chance-level, for a 2-class cross-entropy) for 70+
   epochs. Traced — by direct inspection of gradients and activations, not
   guessed — to each fold's independently-trained CVAE encoder producing
   `z_mean` on a different absolute scale (fold_1's raw standard deviation
   measured ~7.5× smaller than fold_0's). Per-bag centering could not fix
   this by construction (it changes location, not scale — confirmed
   directly: the before/after-centering std ratio was 1.0000 exactly).
   Fixed with `LayerNormalization` applied unconditionally after centering.
3. **An evaluation-methodology mismatch, not a training bug.** With both
   fixes above, training looked excellent, but *naive* pooling of the
   attention-free `instance_posterior` (majority vote or mean, the same
   pooling Tier 0/1/2/4 use) gave chance-level bag accuracy on a model whose
   own attention-weighted training loss was excellent. Root cause: an
   instance attention learns to downweight during training receives
   near-zero training gradient regardless of what it individually predicts,
   so it can develop an arbitrary individual posterior — exactly the
   instances a naive vote lets dominate. Fixed by adding `bag_posterior`
   (attention-weighted, over a bag's true full instance count) as a
   separate scoring path, verified against `call()`'s own batched
   computation and a "resists dilution by noise instances" behavioral test.
   This also meant the shared latent-label store (which holds only the
   attention-free posterior, by design — Stage 5 needs one context-free
   value per instance) could not be the source for bag-level evaluation, so
   `mil_attention_stage.py` additionally persists each fold's true
   held-out test bags' attention-weighted decision directly, while the
   fold's model is still in memory, to
   `results/{exp}/latent/attention_bag_level.json`
   (`scripts/stage6_attention_bag_eval.py` reads this artifact, not the
   store — see that script's own docstring for why this is a deliberate,
   documented split rather than an inconsistency).

## 3. The real 5-fold fit

All 5 folds trained cleanly, no stalls, `sub_bag_null_rate=0.0000` in every
fold (target ε=0.01 — sub-bags essentially always contained at least one
diagnostic instance, well inside tolerance):

| fold | fit bags | inner-val bags | best epoch inner_val_loss | inner_val bag_acc | sub_bag_null_rate |
|---|---|---|---|---|---|
| fold_0 | 38 | 7 | 0.6031 | 0.8571 | 0.0000 |
| fold_1 | 39 | 8 | 0.1189 | 1.0000 | 0.0000 |
| fold_2 | 38 | 7 | 0.2766 | 0.8571 | 0.0000 |
| fold_3 | 40 | 8 | 0.4715 | 0.8750 | 0.0000 |
| fold_4 | 39 | 8 | 0.2042 | 0.8750 | 0.0000 |

**Read the inner-validation numbers as a training-health check, not as the
headline result.** Mean inner-val bag accuracy across folds is 0.893 — but
this is computed on 7–8 held-out-from-training bags per fold (still drawn
from the same slides the model's early stopping was tuned against,
group-aware but small-N), not the true CV test fold. §4's pooled n=59
true-test-bag number is substantially lower (0.644) — worth stating
plainly rather than letting the better-looking inner number stand as if it
were the real result. Some of that gap is expected (7–8-bag accuracy has a
wide sampling range — one bag is worth ~13 points), but its direction and
size are consistent with the model still not generalizing as well to
genuinely unseen bags as the inner-validation loop suggested during
training.

## 4. Bag-level accuracy vs Tier 0 — all four references

`scripts/stage6_attention_bag_eval.py`, exp_501, n=59 bags, 1000-resample
class-stratified bootstrap, seed 42. Tier 0 references reproduced from
docs/stage6_generative_mil.md §3 (same numbers, same script family):

| reference | Tier 0 bag_acc | attention bag_acc | Δ (attention − Tier0) | 95% CI | p | significant? |
|---|---|---|---|---|---|---|
| 1D-CNN | 0.5932 | 0.6441 | +0.0508 | [−0.102, +0.203] | 0.604 | no |
| Apollo (RF+SVM) | 0.7627 | 0.6441 | −0.1186 | [−0.220, −0.034] | 0.012 | **yes, worse** |
| Random Forest | 0.8136 | 0.6441 | −0.1695 | [−0.288, −0.068] | 0.002 | **yes, worse** |
| XGBoost | 0.8136 | 0.6441 | −0.1695 | [−0.288, −0.068] | 0.002 | **yes, worse** |

(attention: bag_F1=0.6228, bag_kappa=0.2567, n=59 throughout — one
`bag_posterior` per bag, duplicate cross-fold bags already averaged per
`_load_bag_level_across_folds`.)

**Read this the same way docs/stage6_generative_mil.md §3 reads Tier 4's
table — by every reference, not just the flattering one.** Against the
CNN, attention MIL shows a positive but non-significant delta, the same
qualitative read Tier 4's `reconstruction` variant got against the CNN
(+0.119, p=0.150) — a hint, not a result. Against the three stronger, tree-
based Tier 0 classifiers, attention MIL is **significantly worse**, not
merely tied — the one tier, so far, whose comparison table has no ✓ at all.
The tree ensembles (Apollo/RF/XGBoost) remain the strongest bag-level
classifiers in this project across every tier evaluated to date (Tier 0
themselves, Tier 1 mi-SVM, Tier 4 both variants, and now Tier 3).

## 5. Spatial validation

Full lag sweep, all 59/59 slides, 288,678/288,678 instances (the HF-1887
fix applies here — this is a fresh fit, not inherited from before that
fix), 200 within-slide permutations/slide, seed 42
(`results/exp_501/spatial_validation/stage5_attention_attention.json`):

| lag | Moran's I (mean ± std) | permutation p (mean ± std) | permutation z (mean ± std) | variogram γ |
|---|---|---|---|---|
| 1 [contaminated] | +0.152 ± 0.070 | 0.0004 ± 0.003 | +20.47 ± 9.61 | 0.00536 |
| 2 [contaminated] | +0.058 ± 0.039 | 0.0015 ± 0.007 | +11.20 ± 7.72 | 0.00599 |
| **3 [headline]** | **+0.0236 ± 0.028** | **0.1105 ± 0.210** | **+5.634 ± 7.255** | 0.00622 |
| 4 | +0.0176 ± 0.026 | 0.1749 ± 0.286 | +5.003 ± 7.802 | 0.00627 |
| 5 | +0.0148 ± 0.023 | 0.1873 ± 0.283 | +4.637 ± 7.313 | 0.00629 |
| 6–10 | declining, +0.0068 to +0.0122 | rising, 0.19–0.28 | declining, +3.2 to +4.2 | ≈0.0063, flat |

Variogram range: median 2.0px (n=59). External agreement vs
`viable_legacy`: **ARI=0.0000, AMI=0.0000** (n=288,678) — the identical
structural degeneracy docs/stage6_generative_mil.md §4 diagnoses for Tier
4's two variants: attention's instance head is a genuine, unconstrained
K=2 softmax (docs/mil_tiers_methodology.md §6.3 states this was a
deliberate design choice, made *before* this consequence was confirmed),
so `max_k p_k ≥ 0.5` by construction and the lifted non-diagnostic mass is
provably bounded to `[0, 1/3]` — never crossing the 0.5 hard-diagnostic
threshold used for ARI/AMI, regardless of what the posterior encodes. This
is the K=2-softmax convention bug §4 of that doc predicted would apply to
Tier 3 "if its attention head is a plain softmax too" — now confirmed
directly rather than left as a flag. As with Tier 4, the *continuous* non-
diagnostic mass driving the Moran's I/variogram numbers above is unaffected
by this — only the hard-threshold external-agreement check is degenerate.

**A real tension worth stating plainly: mean-p and mean-z disagree at the
headline lag.** At lag 3, the mean permutation p-value (0.1105) does not
clear the conventional 0.05 bar, while the mean z-score (+5.634) would
imply an extremely small p under a normal approximation. This is not a
contradiction in the underlying test — permutation p is bounded below by
`1/(n_perm+1) ≈ 0.005` per slide while z is not, and the large std on both
columns (0.210 on p, 7.255 on z, both close to or exceeding their own
means) signals a **highly heterogeneous per-slide effect**: some slides
show strong, low-p coherence and others show none, and naively averaging
p-values across 59 such slides is a conservative, easily-dominated-by-
null-slides summary compared to averaging z (which the large positive
slides pull up more). Neither the mean-p nor the mean-z framing alone is a
clean verdict; both are reported rather than picking the one that reads
better. The unambiguous parts: Moran's I is positive and monotonically
declining from lag 3 to lag 10 (not noise-like), and every lag from 3–10
has a positive mean z. Whether that adds up to "real spatial coherence"
survives a stricter bar than this file settles — see §7.

## 6. Cross-tier spatial comparison, lag 3

Consolidating every tier's headline lag-3 number in one place (mi-SVM and
Tier 4 numbers reproduced from docs/stage5_spatial_validation.md and
docs/stage6_generative_mil.md §4 respectively — not re-run here):

| method | Moran's I (lag 3) | mean p | mean z | variogram range |
|---|---|---|---|---|
| mi-SVM (Tier 1) | +0.006 | 0.361 | +1.5 | median 2px |
| EM (Tier 2) | — | — | — | collapsed (variogram ≡ 0) |
| **attention (Tier 3)** | **+0.024** | **0.111** | **+5.6** | median 2.0px |
| CbR reconstruction (Tier 4) | +0.122 | 0.028 | +29.4 | median 3.0px |
| CbR encoder_head (Tier 4) | +0.049 | 0.057 | +11.1 | median 3.0px |

By every column, Tier 3 sits between mi-SVM (weakest, not significant by
either summary) and Tier 4's two variants (strongest, `reconstruction`
clearing the conventional p<0.05 bar on the mean-p summary too). Tier 3's
own mean-z is respectable — ahead of `encoder_head`'s is not true (11.1 >
5.6), so more precisely: ahead of mi-SVM, behind both Tier 4 variants on
every column.

## 7. Honest overall read

Attention MIL is a genuine, working model — not degenerate, not
overfit (after the fixes in §2), trains reproducibly across all 5 folds.
It recovers spatial structure beyond what mi-SVM showed, though weaker
than either Tier 4 variant, and the p-vs-z tension in §5 means even that
should be read as suggestive rather than conclusively established without
the shuffled-label control (§8). But as a bag-level classifier — the
task's actual clinical objective — it is **the first tier in this project
whose comparison table shows a significant loss** against multiple Tier 0
references, not merely a non-significant tie. Two things are true at once
and neither should be softened to fit the other: the architecture and its
debugging were done correctly (§2, §3, verified in §9), and the resulting
model is, on the evidence gathered so far, not competitive with the
simpler Tier 0 tree ensembles at the one task that matters most.

## 8. Deviations from the plan / honest gaps

- **The shuffled-bag-label control has not been run for this tier.** Per
  the standing backlog, this is the definitive test of whether §5/§6's
  lag≥3 coherence is genuinely label-driven; without it, and given §5's
  own p-vs-z tension, the spatial finding here is suggestive, not
  definitive — weaker footing than mi-SVM's control-tested (and ultimately
  negative) result, or Tier 4's control-pending-but-cleaner p<0.05 result.
- **The inner-validation/true-test generalization gap (§3) is unexplained.**
  Mean inner-val bag accuracy (0.893) substantially exceeds the true
  pooled test accuracy (0.644). Plausible contributors — small inner-val
  N (7–8 bags), residual overfitting the current regularization doesn't
  fully remove, or a genuine distribution shift between training-fold
  bags and held-out bags — have not been distinguished from one another.
- **The K=2 softmax external-agreement bound (§5) is documented, not
  fixed** — same status as Tier 4's identical issue.
- **No hyperparameter sweep.** Dropout/L2/attn_dim/sub-bag size were fixed
  from the smoke-testing process (§2) and the π_min-derived formula
  respectively, not tuned against the true test folds (which would itself
  be a leakage risk given how few bags exist).

## 9. Verification

- **New `tests/test_stage6_attention_mil.py`, `tests/test_stage6_attention_bag_eval.py`,
  `tests/test_stage6_shuffle_control.py`** (the last covers the shuffle
  infrastructure that exists but hasn't been run for this tier yet).
- **Full suite re-run against `tests/` (this pass, 2026-09-29): 152
  passed, 2 failed** — the same two pre-existing `test_config_manager.py`
  failures present throughout this project's history, unrelated to Tier 3
  or to anything touched this stage.
- **Real data, full pipeline**: fit (5 folds) → store + `attention_bag_level.json`
  → spatial validation → bag-level eval against all four Tier 0 references
  ran end-to-end with no errors.
- **`bag_posterior` correctness** (§2, bug 3) verified two ways before
  being trusted for the real run: consistency against `call()`'s own
  batched computation, and a behavioral test that a bag's decision resists
  dilution by added noise instances (an attention model that had not
  actually learned to downweight noise would fail this).
