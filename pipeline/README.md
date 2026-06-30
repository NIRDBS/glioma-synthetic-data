# `pipeline/` — Headless experiment runner

Runs the full glioma synthetic-data experiments **without notebooks**, for
reproducible and reviewable execution. Every stage is config-driven: behaviour is
read from `configs/exp_*.yaml`, and outputs are written under
`results/{experiment}/` so artifacts can be reused across experiments.

The notebooks and this package share the same core helpers (`pipeline/_common.py`),
so a notebook run and a headless run produce identical training behaviour.

---

## Stages

The end-to-end flow is a sequence of stages. Each experiment's config lists the
stages it needs under `pipeline.stages`.

| Stage         | Module                       | Trains on / does                                  | Notebook |
|---------------|------------------------------|---------------------------------------------------|----------|
| `preprocessing` | `preprocessing_stage.py`   | Raw cubes → cropped/normalised H5 dataset         | 01 / 00  |
| `trtr`        | `trtr_stage.py`              | Real (slide-capped) → 4 models, test real         | 02       |
| `cvae`        | `cvae_stage.py`              | Train CVAE, generate synthetic pool per fold      | 03       |
| `tstr`        | `tstr_stage.py`              | Synthetic pool → 4 models, test real              | 02 / 04  |
| `tsrtr`       | `tsrtr_stage.py`             | Real + on-demand synthetic (ratio) → 4 models     | 02       |
| `stats_eval`  | `evaluation_stage.py`        | Bootstrap CIs: TRTR vs TSTR / TSRTR (+ CbR)       | 09       |

The four benchmark classifiers (Apollo RF+SVM, Random Forest, XGBoost, 1D-CNN)
and per-fold preparation (0-indexing, binary mapping, train-only slide cap) live
in `_common.py` and are reused by every classifier stage.

### Slide cap semantics (`patient_cap`)

`cv_strategy.patient_cap` is the max number of spectra kept **per slide** in the
**training** split (validation is never capped). It exists so slides that
contribute many highly-correlated spectra don't dominate training. The *same cap
size* is applied differently by the two kinds of stage:

- **Classifier stages** (`trtr`/`tstr`/`tsrtr`, via `prepare_fold`): a **single
  fixed subsample** is drawn once per fold.
- **CVAE stage** (`cvae_stage`): the cap is **re-sampled fresh every epoch** — a
  different random subset each epoch, acting as implicit regularisation
  ([`cvae_stage.py`](cvae_stage.py) epoch loop).

The `cap_strategy: "dynamic_per_epoch"` key in the configs documents the *CVAE*
behaviour; it is **not read by any code** (the cap behaviour is hardcoded per
stage, keyed only on `patient_cap`).

Default stage order (`run_suite.DEFAULT_STAGES`):
`trtr → cvae → tstr → tsrtr → stats_eval`.

---

## Quick start — run a suite

```bash
# Full run for exp_201, then the three reuse-based ratio ablations
python -m pipeline.run_suite --configs exp_201 exp_401 exp_402 exp_403

# Only the ratio ablations (exp_201 already done)
python -m pipeline.run_suite --configs exp_401 exp_402 exp_403

# Smoke test: tiny CVAE, force rerun even if outputs exist
python -m pipeline.run_suite --configs exp_201 --stages cvae --epochs 2 --force
```

`run_suite` reads each config's `pipeline.stages` (override with `--stages`),
runs them in order, and **skips stages whose outputs already exist** (disable
with `--force`). A per-experiment `results/{exp}/suite_log.json` records each
stage's status and duration. Reuse is intentional: e.g. exp_401/402/403 reuse
exp_201's CVAE decoder weights and TRTR baseline.

### Key flags

| Flag                       | Default | Meaning                                                        |
|----------------------------|---------|----------------------------------------------------------------|
| `--configs` (required)     | —       | One or more experiment names, e.g. `exp_201 exp_401`           |
| `--stages`                 | config  | Override the stage list for all configs                        |
| `--cvae-backend`           | `local` | `local` or `modal` (GPU offload for CVAE training)             |
| `--classifiers-backend`    | `local` | `modal` dispatches each stage across all configs in parallel   |
| `--epochs`                 | config  | Override CVAE epochs (smoke tests)                             |
| `--continue-on-error`      | off     | Keep going to the next stage/experiment on failure             |
| `--force`                  | off     | Rerun stages even if their outputs already exist               |

---

## Run a single stage

Each stage module is also a standalone CLI (handy for debugging one step):

```bash
python -m pipeline.trtr_stage   --config exp_201
python -m pipeline.cvae_stage   --config exp_201 --backend local --epochs 70
python -m pipeline.tstr_stage   --config exp_201
python -m pipeline.tsrtr_stage  --config exp_401          # uses tsrtr.synth_ratio
python -m pipeline.evaluation_stage --config exp_401 \
        --baseline-dir results/exp_201/models/trtr \
        --candidate-dir results/exp_401/models/tsrtr \
        --baseline-label TRTR --candidate-label TSRTR
```

(`preprocessing_stage` has no standalone CLI — run it via `run_suite --stages preprocessing`.)

---

## Backends: local vs Modal

- **`local`** (default): everything runs on this machine. TF uses GPU memory
  growth so it doesn't grab all VRAM up front.
- **`--cvae-backend modal`**: CVAE training is offloaded to a Modal GPU; weights
  and the synthetic pool are downloaded back to `results/{exp}/...`.
- **`--classifiers-backend modal`**: switches to **stage-first** execution — all
  configs' instances of the same classifier stage are batched and dispatched in
  parallel on Modal. Stage order across configs is still respected (e.g. exp_201's
  `cvae` finishes before any config's `tsrtr`, since the ratio ablations reuse its
  decoder). CVAE and `stats_eval` still run locally.

See `pipeline/modal_cvae.py` and `pipeline/modal_classifiers.py` for the Modal apps.

---

## Outputs layout

Per experiment, under `results/{experiment}/` (paths come from the config's
`paths:` block):

```
results/exp_201/
├── data/glioma_dataset.h5          # processed_file (from preprocessing)
├── models/
│   ├── trtr/   apollo_fold_{i}.joblib, rf_*.joblib, xgb_*.json, cnn_*.keras
│   ├── tstr/   …same filename scheme
│   ├── tsrtr/  …
│   └── cvae/   decoder_fold_{i}.weights.h5, encoder_*.weights.h5
├── synthetic/  fold_{i}_X_synth.npy, fold_{i}_y_synth.npy
├── statistical_eval/ statistical_evaluation_*.json
└── suite_log.json                  # per-stage status + timing
```

The canonical model filename scheme (`{model}_{fold}.{ext}`) is shared by all
classifier stages, so `stats_eval` and the notebooks can locate any model by
`(experiment, regimen, fold)`.

---

## Config keys that drive the pipeline

| Key                              | Used by            | Purpose                                         |
|----------------------------------|--------------------|-------------------------------------------------|
| `pipeline.stages`                | `run_suite`        | Which stages to run, in order                   |
| `cv_strategy.n_folds`            | all                | Number of CV folds                              |
| `cv_strategy.patient_cap`        | `prepare_fold`, `cvae_stage` | Max spectra per **slide** in the train split (see note) |
| `cv_strategy.cap_strategy`       | —                  | Descriptive annotation only; **not read by code** |
| `models.cnn.num_classes`         | `prepare_fold`     | 2 → binary mapping; else multiclass             |
| `tsrtr.synth_ratio`              | `tsrtr_stage`      | Synthetic-to-real ratio (e.g. 0.5, 1.0, 2.0)    |
| `tsrtr.cvae_weights_dir`         | `tsrtr_stage`      | Decoder weights to reuse (cross-experiment)     |
| `paths.*`                        | all                | Input/output locations (see layout above)       |
| `training.random_seed`           | `set_seeds`        | numpy / random / TF seed                        |
| `statistics.*`                   | `evaluation_stage` | Bootstrap samples, alpha, primary metric        |
