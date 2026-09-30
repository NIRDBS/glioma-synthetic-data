# CVAE Counterfactual Probe

**Status: complete for the core probe, all three reviewer validity checks, the
shuffled-label control, and the α=0 ablation.** Two results, both important: the
shuffled-label control is genuinely mixed (§3) — the first case in this project where a
shuffled control does NOT uniformly match or exceed the real one, though it only clears
the bar on one of four statistics. The α=0 ablation (§3a) is unambiguous — removing the
auxiliary class head's gradient pressure on z changes **nothing**, within noise, on every
statistic this probe reports. The counterfactual mechanism does not depend on that head.

This is a free-standing analysis, not a Stage-N MIL tier (see
docs/mil_tiers_methodology.md for the four tiers; this probe doesn't produce a
competing instance posterior and isn't subject to that machinery). New
`src/mil/counterfactual.py`, `scripts/counterfactual_probe.py`,
`tests/test_counterfactual_probe.py`. Approved plan:
`C:\Users\Andrei\.claude\plans\i-think-bootstrap-on-smooth-aho.md`.

> **Reproducing this.** `pytest tests/test_counterfactual_probe.py` (20 tests, synthetic
> fakes only). Real run: `python scripts/counterfactual_probe.py --config exp_501`.
> Shuffled run (reuses the already-existing `cvae_shuffled/` weights, zero new
> training): `python scripts/counterfactual_probe.py --config exp_501 --shuffled`.
> α=0 run (reuses the already-existing `cvae_alpha0/` weights, zero new training):
> `python scripts/counterfactual_probe.py --config exp_501 --store-suffix alpha0`.
> To retrain the α=0 CVAE from scratch (real GPU cost):
> `python -c "from pipeline.modal_cvae import run_on_modal; run_on_modal('exp_501_alpha0', store_suffix='alpha0')"`.
> Smoke test on one fold: add `--folds fold_0`.

## 1. What this is

Encode a real, held-out spectrum to `z_mean` (this project's established
z_mean-only convention), then decode that SAME `z_mean` under both class
conditions. `Δ(z) = D(z, mutant) − D(z, wildtype)` is a learned, per-instance
"IDH effect" spectrum — a genuinely different kind of claim from the four MIL
tiers, all of which failed their spatial-coherence shuffled-label controls this
session (docs/mil_tiers_findings.md §2). Full methodology, class-index
convention (0=mutant, 1=wildtype), and architecture citations: see the module
docstring in `src/mil/counterfactual.py` and the approved plan file.

## 2. The reviewer's three validity checks — results

A reviewer flagged that the CVAE's encoder never sees the label directly, but
its α-weighted auxiliary class head shares the exact same `features` tensor
that produces `z_mean` (confirmed directly, `src/models/cvae.py:78,79,91`) — so
class information could leak into `z_mean`, making the decoder partially
ignore the label and understating Δ's true size.

**Check 1 — how much class information is in z.** A fresh
`z_mean → class` logistic-regression probe, fit per fold on that fold's
training instances, scored out-of-sample:

| | instance-level accuracy (per fold) | bag-level accuracy |
|---|---|---|
| **real CVAE** (α=0.2) | 0.530, 0.726, 0.710, 0.714, 0.622 | **0.7966** |
| **α=0 CVAE** (§3a — class head untrained) | 0.620, 0.789, 0.718, 0.760, 0.603 | **0.8136** |
| chance (majority-class rate, n=59 bags: 26 mutant/33 wildtype) | — | 0.5593 |
| `encoder_head`'s own bag accuracy (reference point) | — | 0.7966 |

The real CVAE's z-probe bag accuracy (0.7966) is **exactly** `encoder_head`'s
own bag accuracy — verified this is not an artefact of accidentally reusing
`encoder_head`'s predictions (the z-probe is a completely independent
`sklearn.LogisticRegression` fit on `z_mean` alone; `class_pred` is never
read anywhere in `src/mil/counterfactual.py`, confirmed by direct grep before
trusting this). Per the reviewer's own stated reading, this lands **at**
`encoder_head`'s accuracy, not near chance — **Δ should be read as a lower
bound on the true counterfactual effect, not the full effect**: some amount
of class-discriminative signal likely exists in `z_mean` that the label swap
cannot remove, since the decoder can partially infer the true class from `z`
regardless of which label condition it's asked to decode under.

**A further, humbling wrinkle found via the shuffled control (§3)**: the
*shuffled*-trained CVAE's `z_mean`, scored against the **true** labels (never
seen during that CVAE's training), still reaches **0.7458** bag accuracy —
far above chance. This means a substantial share of `z_mean`'s apparent
class-predictiveness is not something the α-weighted class head specifically
injected; it is recoverable from the raw data's own clusterability
regardless of which labels (real or a random permutation) trained the
encoder. Check 1's result should therefore be read as an upper bound on how
much of the leakage is *architecturally caused* — the true figure is likely
smaller than 0.7966−0.5593 suggests.

**The α=0 ablation (§3a) confirms this directly rather than just bounding
it.** With the class head's gradient pressure on `features` completely
removed, `z_mean`'s bag accuracy against true labels doesn't drop — it rises
slightly, to **0.8136**. The shared-features architectural concern that
motivated this whole set of checks does not appear to be the mechanism:
whatever class information ends up in `z_mean` is apparently a side effect
of reconstructing well under a label-conditioned decoder, not something the
α-weighted auxiliary head specifically injects.

**Check 2 — how much the decoder uses the label** (Δ's size vs. the natural
within-class reconstruction spread, `delta_effect_size` = |Δ|/spread per
feature):

| band | Δ (real) | effect size (real) | fold sign agreement |
|---|---|---|---|
| Phe (1003) | −0.0205 | 13.4× | 5/5 |
| Phosphodiester (1087) | −0.0274 | 3.9× | 5/5 |
| Amide III (1250) | −0.0012 | 1.6× | 4/5 |
| CH2 twist / Lipid (1301) | +0.0294 | 12.7× | 5/5 |
| CH2/CH3 def. (1450) | +0.0043 | 1.5× | 4/5 |
| Amide II (1550) | +0.0078 | 8.8× | 5/5 |
| Amide I (1655) | +0.0020 | 2.9× | 4/5 |

Every named band's Δ is **larger than the natural within-class reconstruction
noise floor** (effect sizes 1.5×–13.4×), and 4–5 of 5 independently-trained
folds agree on its sign at every single band. Read alongside §3: this
establishes the decoder's swap has a real, reproducible-in-shape effect — it
does not establish that shape is *label-driven* (see §3's Δ-magnitude
result).

**Check 3 — counterfactual validity** (does swapping z's label flip an
independent, non-CVAE classifier's prediction to the intended opposite
class): **`flip_to_target_rate` = 0.9987 (RF), 0.9116 (XGBoost)**. Nearly
every counterfactual reconstruction is read by an independently-trained tree
ensemble as the class it was asked to become. This is the strongest, cleanest
result this probe produced — and, critically, it is the one statistic that
**does not** replicate under the shuffled-label control (§3).

## 3. The shuffled-label control — mixed, and precisely stated

Reuses `results/exp_501/models/cvae_shuffled/` (already trained this session,
seed=123, full 5-fold/100-epoch schedule on Modal GPU, for the encoder_head
spatial control — zero new training needed here). The shuffled decoder's
condition "0"/"1" is a **permuted** label from training, not true
mutant/wildtype; every check below is still scored against the **true**
labels.

| statistic | real | shuffled | verdict |
|---|---|---|---|
| mean ‖Δ‖ (bag-of-bags, bootstrap CI) | 0.3490 [0.3316, 0.3666] | 0.4285 [0.4097, 0.4502] | **shuffled larger — fails** |
| cross-fold stability (mean off-diag cosine, z vs. random-direction null) | 0.7848 (z=+59.4) | 0.8914 (z=+67.5) | **shuffled larger — fails** |
| per-bag consistency (cosine to grand mean) | 0.901 | 0.955 | **shuffled larger — fails** |
| z-class-probe bag accuracy (scored vs. TRUE labels) | 0.7966 | 0.7458 | real larger, but both ≫ chance (0.559) |
| flip-to-target rate, RF | **0.9987** | 0.4113 | **real decisively larger — passes** |
| flip-to-target rate, XGBoost | **0.9116** | 0.5284 | **real decisively larger — passes** |

**Three of four statistic families fail the pre-registered criterion** ("if
the shuffled statistic is comparable-or-larger than the real one, the real
finding cannot be attributed to the true label") — the same outcome every
other spatial/representational claim in this project has shown (mi-SVM,
attention, both Tier 4 CbR variants; docs/mil_tiers_findings.md §2). **The
flip-to-target rate is the first statistic in this entire project where the
shuffled control comes back clearly, decisively negative** — the shuffled
model's swapped reconstructions are read by independent classifiers as the
target class at a rate indistinguishable from chance (0.41, 0.53), while the
real model's are read that way almost universally (0.9987, 0.9116).

**Why these can diverge — a real, mechanistically sensible split, not a
contradiction.** ‖Δ‖ and cross-fold stability are *direction-agnostic
magnitude/consistency* statistics: they only ask "does swapping condition 0
vs. 1 produce some sizeable, reproducible change," and apparently *any* two
training partitions the decoder is asked to distinguish — even an arbitrary,
biologically meaningless permutation of 59 bags — is enough to produce
*some* systematic, cross-fold-reproducible reconstruction change, simply
because that is what a flexible conditional decoder does with any two-class
training signal. Flip-to-target asks a *direction-specific, semantically
grounded* question instead — does the swap move the reconstruction toward
what an independent, real-label-trained classifier recognizes as the
*true* opposite class — and only the real model, trained on the true label,
has any reason to do that consistently.

## 3a. The α=0 ablation — unambiguous, and it settles the reviewer's architectural concern

Per the approved plan, this ablation was contingent on Check 1 (§2) showing
strong class-predictiveness (near `encoder_head`'s 0.7966 rather than near
chance) — it landed at exactly 0.7966, so the trigger condition was met.
Run 2026-09-30: one config value changed from `configs/exp_501.yaml`
(`models.cvae.loss_weights.alpha: 0.2 → 0.0`, verified programmatically to be
the *only* functional difference before spending any GPU time), a full real
5-fold/100-epoch CVAE retrain on Modal GPU (`configs/exp_501_alpha0.yaml`,
`store_suffix='alpha0'`, landing at `results/exp_501/models/cvae_alpha0/` —
confirmed the real and shuffled weight directories were untouched by file
timestamp, same check applied to every Modal GPU run this session).

| statistic | real (α=0.2) | α=0 | shuffled (for reference) |
|---|---|---|---|
| mean ‖Δ‖ (bag-of-bags, bootstrap CI) | 0.3490 [0.332, 0.367] | 0.3526 [0.335, 0.372] | 0.4285 [0.410, 0.450] |
| cross-fold stability (mean off-diag cosine) | 0.7848 | 0.8218 | 0.8914 |
| z-class-probe bag accuracy (vs. TRUE labels) | 0.7966 | **0.8136** | 0.7458 |
| flip-to-target rate, RF | 0.9987 | 0.9988 | 0.4113 |
| flip-to-target rate, XGBoost | 0.9116 | 0.9117 | 0.5284 |

**Every statistic is unchanged, within noise, between the real (α=0.2) and
α=0 CVAEs** — the z-class-probe accuracy, if anything, is very slightly
*higher* at α=0. Removing the auxiliary class head's gradient pressure on
the shared `features` tensor that also produces `z_mean` — the exact
mechanism the reviewer's architectural concern named — changes **nothing**
about how much class information ends up in `z_mean`, how large Δ is, or
how reliably the counterfactual swap flips an independent classifier.

**This settles the reviewer's concern directly, rather than merely bounding
it as §2's Check 1 alone could.** The class-discriminative information in
`z_mean`, and the counterfactual mechanism's validity, do not depend on the
α-weighted class head at all — they are apparently a side effect of
reconstructing well under a label-conditioned decoder, present with or
without that head's explicit training signal. This *strengthens* confidence
in the flip-to-target finding specifically: it is not an artefact of the
auxiliary classifier, since removing that classifier's influence entirely
leaves the finding intact. It does **not** change §3's Δ-magnitude/
consistency result — those still fail their shuffled-label control
regardless of α, since α was never the variable that control was testing.

## 4. Honest overall verdict

**What survives**: the counterfactual *mechanism* itself is validated — the
CVAE's label-conditioning genuinely and specifically encodes true-class
information the decoder can act on, confirmed by an independent classifier at
a rate that collapses under label permutation (§3) and is unaffected by
removing the auxiliary class head entirely (§3a). This is a real, positive,
control-tested, ablation-tested finding — the first one in this project's
entire spatial/representational-claim programme to survive an adversarial
shuffled-label test on any statistic.

**What does not survive**: the *magnitude and cross-fold "consistency"* of
the average Δ spectrum — the headline deliverable this probe originally set
out to produce (docs/mil_tiers_findings.md-style "the average Δ spectrum,
with bands mapped to known assignments") — behaves the same way every prior
representational claim in this project has: a meaningless label permutation
produces a comparable-or-larger version of the same statistic. **The specific
per-band pattern reported in §2's Check 2 table should not be presented as
"the IDH effect" without this caveat stated alongside it** — its *existence
and reproducibility* is established; its *attribution to the true label* is
not, by the same statistics that would need to establish it.

**Net**: this experiment does not cleanly repeat the negative pattern of
Stages 0–6, nor does it cleanly deliver the positive, publication-ready
"here is the IDH effect spectrum" result the original proposal envisioned.
It delivers something more precise and more defensible than either: firm
evidence the CVAE's conditioning mechanism carries real, label-specific
information (flip-to-target), alongside firm evidence that the *aggregate
shape and size* of that information is not yet distinguishable from what an
arbitrary label partition would also produce. Both halves belong in the
paper; reporting only the flip-to-target result would be as selective as
reporting only the Δ-magnitude result.

## 5. The α=0 ablation — done, see §3a

## 6. Verification

- `pytest tests/test_counterfactual_probe.py`: 20/20 passing. Every function
  hand-checked against a deterministic fake encoder/decoder pair
  (`z_mean == X`, exact reconstruction on class match, `+5.0` off — making Δ
  an exact, hand-computable ±5 constant), including
  `test_class_index_constants_match_project_convention` (a loud guard, since
  every reported sign in this document depends on it),
  `test_mean_delta_per_bag_grand_mean_is_bag_of_bags_not_pooled_instances`
  (pinned against the naive alternative on a case where they visibly
  differ), and `test_independent_classifier_flip_rate_hand_computed` (a case
  where `flip_to_target_rate` and `flip_from_own_rate` are forced to differ,
  by construction).
- Full suite re-run: same 2 pre-existing `test_config_manager.py` failures,
  nothing else broken.
- The 0.7966 exact match (§2) was independently verified not to be a bug via
  direct code inspection and grep, before being reported as a finding (§2).
- The α=0 config (`configs/exp_501_alpha0.yaml`) was diffed programmatically
  against `configs/exp_501.yaml` before spending any GPU time on it —
  confirmed the only functional difference is `models.cvae.loss_weights.alpha`
  (name/description/notes also differ, nothing else).
- Both the α=0 smoke test (2 epochs) and the real run (100 epochs × 5 folds)
  had the real `cvae/` and shuffled `cvae_shuffled/` weight directories'
  file timestamps checked and confirmed unchanged afterward.
- All numbers in this document were read directly from
  `results/exp_501/counterfactual_probe/{report,report_shuffled,report_alpha0}.json`,
  not transcribed from console output.

## 7. Deviations from the plan / honest gaps

- **No formal bootstrap comparison between the real/shuffled/α=0 z-probe bag
  accuracies** (0.7966 / 0.7458 / 0.8136) — reported as point estimates. The
  other statistics in §3/§3a's tables do carry a bootstrap CI or a null-based
  z-score; this one does not yet.
- **Per-instance consistency statistics are confounded by within-slide
  spatial autocorrelation** (stated in the JSON report's own `note` field,
  and in §3's per-bag-vs-per-instance framing) — the per-bag statistic is
  primary throughout this document for that reason.
- **Cross-fold stability mixes two questions that are not separable with
  this project's single CV split**: whether independently-trained CVAEs
  agree, and whether the effect is homogeneous across the different bags
  each fold happens to hold out. Not decomposed here.
- **No window/tolerance averaging around each named peak** — a direct,
  single-nearest-index lookup, matching this codebase's only existing
  precedent (`src/evaluation/synthetic_realism.py`'s own idiom). A wider
  window was not attempted.
