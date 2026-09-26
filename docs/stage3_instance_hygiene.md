# Stage 3 — Instance-model hygiene

**Status: complete (code + tests); real-data retrain in progress.** Core:
`pipeline/_common.py`, `src/models/cnn.py`, `src/data/augmentation.py`,
`pipeline/cvae_stage.py`. New dataset-migrated config: `configs/exp_501.yaml`.

> **Reproducing this.** `pytest tests/test_stage3_hygiene.py` (16 tests, synthetic
> data only, no dependency on the real dataset). Real validation:
> `python -m pipeline.run_suite --configs exp_501 --stages trtr cvae tstr tsrtr
> stats_eval mil_eval --force` retrains everything on the new dataset with the
> fixes in place.

## 0. A prerequisite decision, made before this stage's own work started

Stage 4 needs `viable_legacy` (its own §4 test list: "pi_b external correlation
against `viable_legacy` fraction"), a field that exists only in Stage 1's
`data/processed/glioma_mil.h5` — every experiment config, including exp_201,
still pointed at the old per-experiment `glioma_dataset.h5`. Training this
stage's CNN fix on the old dataset while Stage 4's MIL models train on the new
one would confound "does the formulation help" with "did the data change" —
raised directly by the user before this stage began. Resolved by migrating
first: new `configs/exp_501.yaml`, exp_201's exact model/training
hyperparameters, only `paths.processed_file` repointed at
`data/processed/glioma_mil.h5`. Verified compatible via a direct probe-load
through the existing loader/config machinery before committing to any
training (all 5 folds load cleanly, 288,678 instances, correct label range,
`coords`/`groups` intact, zero code changes needed for the swap itself), then
smoke-tested via the `trtr` stage alone before the full run.

## 1. What changed and why

Four concrete defects, confirmed in the code before touching it (per
docs/MIL_REFRAME_PLAN.md §7.2), and directly observed in exp_501's own
pre-fix TRTR smoke test: the 1D-CNN early-stopped after 6–11 of 30 possible
epochs per fold, with wildly unstable validation accuracy (0.617 ± 0.162
across 5 folds) — a live demonstration of the bug this stage exists to fix.

- **Early-stopping monitored the true held-out test fold itself**
  (`pipeline/_common.py`'s old `train_four_models`: `validation_data=(X_val,
  y_val)` where `X_val`/`y_val` *is* the fold's test set). `restore_best_weights`
  was therefore selecting weights based on held-out data — leakage, not merely
  optimism. Fixed with a new `_inner_validation_split(y, bag_ids, val_fraction,
  seed)`: carves a held-out slice from *training* data only, group-aware when
  bag ids identify real slides (a slide goes entirely to one side, never
  straddling the split), falling back to a plain per-row stratified split for
  rows with no group (synthetic instances have no slide to leak across).
  Handles all three regimens correctly: TRTR (100% grouped), TSTR (bag_ids=None
  entirely, since synthetic training data has no slide structure), and TSRTR
  (a real mix — real rows keep their bag id, synthetic rows get an explicit
  `None` sentinel so the group-aware split only ever applies to rows where a
  group genuinely exists).
- **`dropout_rate` never forwarded** — the config's `0.6` was silently
  discarded; `build_1d_cnn`'s hardcoded default `0.4` was used regardless.
  Fixed by threading it through the one authoritative `build_1d_cnn(...)` call.
- **Compiled twice** — `build_1d_cnn` compiled once internally
  (`sparse_categorical_crossentropy`, its own default), then
  `pipeline/_common.py` compiled again with the config's `optimizer`/`loss`
  as a bare string, silently losing the `learning_rate` config value in the
  process (a bonus bug found while fixing this: `Adam(learning_rate=...)`
  from the first compile was overwritten by a bare `"adam"` string in the
  second, which resolves to Keras' own default LR — numerically harmless here
  only because that default also happens to be 0.001). Fixed by giving
  `build_1d_cnn` a `loss` parameter and making its call in `_common.py` the
  **only** compile call anywhere in the path.
- **No augmentation, 30 epochs, patience 5** — undertrained on ~22–24k capped
  instances. Fixed: augmentation now wired in (below); epochs raised to 100,
  patience to 10 in `exp_501.yaml` specifically (exp_201 is not being
  retrained under this stage, so its config is untouched).

### Augmentation

`src/data/augmentation.py` already held better implementations
(`spectral_shift`: fractional, cubic-interpolated; `within_class_mixup`) than
`pipeline/cvae_stage.py`'s inline duplicate (a single integer `tf.roll`
**shared across the whole batch** — one shift value applied identically to
every row). Two separate things changed:

1. **The CNN gets augmentation for the first time**, via
   `AugmentedDataGenerator` (unmodified interface, now actually wired into
   `train_four_models`'s CNN block instead of being dead code).
2. **The CVAE's inline duplicate is fixed, not literally replaced.** The plan
   says "use the good one and delete the duplicate"; literally reusing
   `spectral_shift`'s per-row `scipy.interpolate.interp1d` call would cost
   roughly 7.7M interpolator constructions across a full 70-epoch × 5-fold
   CVAE run (the single most expensive stage in the suite) — a real,
   quantifiable slowdown, not a stylistic objection. Instead, the *specific
   named defect* (shared-across-batch, not per-sample) is fixed with a
   vectorised per-row `tf.gather`-based shift, and a new vectorised baseline-
   wobble term (two random-amplitude/phase low-frequency sinusoids) stands in
   for `src/data/augmentation.py::baseline_wobble`'s per-row cubic-spline
   version, for the same reason. Documented as a deliberate deviation in the
   code itself, not silently done differently.

New augmentation primitives added to `src/data/augmentation.py`, all
independently switchable via config: `baseline_wobble` (default-on — "the
most valuable of these" per §7.2, directly motivated by the Stage 0 baseline
study), `intensity_jitter`, `additive_noise`, `band_dropout` — plus the
existing `spectral_shift`/`within_class_mixup`, all wired into
`AugmentedDataGenerator.augment_sample`.

## 2. A real bug caught by actually running it, not just unit tests

The first attempt at wiring augmentation into the CNN passed every unit test
(1-epoch smoke test, augmentation disabled or enabled) but failed on the very
first real multi-epoch training run: `tf.data.Dataset.from_generator(...,
epochs=1)` does not reliably get re-invoked fresh by Keras'
`model.fit(dataset, epochs=N)` — the training run hit "input ran out of data"
after epoch 1. Fixed with an explicit `.repeat()` plus `steps_per_epoch`,
rather than relying on `.fit()`'s undocumented dataset-restart behaviour. A
new regression test (`test_train_four_models_cnn_survives_multiple_epochs_
with_augmentation`, 3 real epochs, augmentation enabled) pins this — the
1-epoch tests that existed before this did not and could not have caught it.

This is also why the exp_501 full retrain was **launched twice**: the first
launch started running before this fix landed (a long-running Python process
does not pick up file edits mid-execution), so it was stopped and restarted
once the fix was confirmed working, rather than left to produce results from
stale code.

## 3. Verification

- **New `tests/test_stage3_hygiene.py`, 16/16 passing.** Covers the plan's
  three Stage 3 test bullets directly: an end-to-end leak-fix test
  (monkeypatches `tf.keras.Model.fit` to capture the actual `validation_data`
  Keras receives, and asserts it is *not* the true test fold, using a
  deliberately distinct numeric fingerprint on the real held-out set to make
  the check unambiguous) plus four `_inner_validation_split` unit tests
  (disjoint/covering, group-awareness, the ungrouped fallback, the mixed
  real+synthetic case); `dropout_rate`/`loss` forwarding; and per-augmentation
  shape/finiteness/range checks plus the multi-epoch regression test above.
- **Full suite re-run**: no regressions in Stage 0–2 (unrelated pre-existing
  `test_config_manager.py` failures aside, as in every prior stage).
- **Loader change** (`tissue_cluster` now threaded through, mirroring
  `coords`'s existing 8 touch points exactly, for Stage 4's use): Stage 1's
  own `tests/test_stage1_export.py` re-run clean, 15/15.
- **Real data, not just synthetic**: `exp_501`'s `trtr` stage alone (pre-fix
  code) surfaced the exact leak symptom the fix targets (6–11/30 epochs,
  0.617 ± 0.162 accuracy) before any of this stage's changes landed — see §0.
  The full retrain with the fix in place was in progress when this document
  was written; §5 will be updated with the resulting numbers.

## 4. Deviations from the plan

- **CVAE augmentation fix is a vectorised stand-in, not a literal reuse of
  `src/data/augmentation.py`'s functions** — performance, not disagreement;
  see §1.
- **`tissue_cluster` loader wiring was not in this stage's own scope**, but
  was done here anyway (mirroring `coords`'s exact pattern, 8 touch points)
  because Stage 4's EM formulation needs it and Stage 3 is the last hygiene
  pass before any MIL model is built — doing it as a separate, later, ad hoc
  addition would have meant touching the well-tested loader again outside any
  stage's planned scope. In the event, Stage 4's actual EM cold-start did not
  end up using `tissue_cluster` (see docs/stage4_mil.md §2) — the wiring is in
  place and inert, a documented, available refinement rather than a load-
  bearing dependency.

## 5. Real-data validation (to be filled in once the retrain completes)

`exp_501`'s full retrain (`trtr, cvae, tstr, tsrtr, stats_eval, mil_eval,
mil_fit`, all with the Stage 3 fixes) was launched and in progress when this
document was written. Once complete: 1D-CNN's per-fold epoch counts and
accuracy variance vs. the pre-fix numbers in §0, and whether it becomes
competitive with Apollo/RF/XGBoost per §7.2's "fair fight" framing.

## 6. What Stage 4 inherits

- A CNN trunk that is actually trained under a leak-free protocol, ready to be
  the Tier-3 attention trunk (Stage 6) once it exists.
- `tissue_cluster` available from the loader for any future MIL formulation
  that wants it (not used by Stage 4's own EM — see docs/stage4_mil.md).
- The augmentation primitives (`baseline_wobble`, `intensity_jitter`,
  `additive_noise`, `band_dropout`) are available to any future instance-level
  model, not just the CNN.
