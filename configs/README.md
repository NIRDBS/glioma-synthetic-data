# Experiment configurations

One YAML file per experiment, `configs/<name>.yaml`. Run it with
`python -m pipeline.run_suite --configs <name>` (see `pipeline/README.md`). `exp_608` is the default
configuration; every other experiment changes one factor of it, or one dataset-level choice.

| Experiment | $\beta$ | $d$ | reconstruction | synthetic:real | spectra per class | epochs | intensity mask | CVAE weights from | Paper |
|---|---|---|---|---|---|---|---|---|---|
| `exp_501` | 2.0 | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own | controlled-experiments appendix |
| `exp_601` | 1 | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own | $\beta$-sensitivity and controlled-experiments appendices |
| `exp_602` | 1.25 | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own | $\beta$-sensitivity and controlled-experiments appendices |
| `exp_603` | 1.5 | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own | $\beta$-sensitivity and controlled-experiments appendices |
| `exp_604` | 1.75 | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own | $\beta$-sensitivity and controlled-experiments appendices |
| `exp_605` | windowed | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own (resumed from exp_601) | $\beta$-sensitivity appendix (windowed $\beta$) |
| `exp_606` | windowed | 32 | MSE | 1 | 20,000 | 100 | 0.60 | own (resumed from exp_601) | $\beta$-sensitivity appendix (windowed $\beta$) |
| `exp_607` | 0.25 | 32 | MSE | 1 | 20,000 | 120 | 0.60 | own | $\beta$ analysis |
| `exp_608` | 0.5 | 32 | MSE | 1 | 20,000 | 120 | 0.60 | own | default; every analysis |
| `exp_609` | 0.75 | 32 | MSE | 1 | 20,000 | 120 | 0.60 | own | $\beta$ analysis |
| `exp_701` | 0.5 | 16 | MSE | 1 | 20,000 | 120 | 0.60 | own | d analysis |
| `exp_702` | 0.5 | 64 | MSE | 1 | 20,000 | 120 | 0.60 | own | d analysis |
| `exp_703` | 0.5 | 32 | MSE + cosine | 1 | 20,000 | 120 | 0.60 | own | loss analysis |
| `exp_704` | 0.5 | 32 | cosine | 1 | 20,000 | 120 | 0.60 | own | loss analysis |
| `exp_705` | 0.5 | 32 | MSE | 0.5 | 20,000 | 120 | 0.60 | exp_608 | ratio analysis |
| `exp_706` | 0.5 | 32 | MSE | 2 | 20,000 | 120 | 0.60 | exp_608 | ratio analysis |
| `exp_707` | 0.5 | 32 | MSE | 3 | 20,000 | 120 | 0.60 | exp_608 | ratio analysis |
| `exp_708` | 0.5 | 32 | MSE | - | 40,000 | 120 | 0.60 | exp_608 | ratio analysis (pool size) |
| `exp_2.607` | 0.25 | 32 | MSE | 1 | 20,000 | 120 | adaptive | own | threshold analysis; controlled-experiments appendix |
| `exp_2.608` | 0.5 | 32 | MSE | 1 | 20,000 | 120 | adaptive | own | threshold analysis |
| `exp_2.609` | 0.75 | 32 | MSE | 1 | 20,000 | 120 | adaptive | own | controlled-experiments appendix |
| `exp_4.608` | 0.5 | 32 | MSE | 1 | 20,000 | 120 | 0.40 | own | threshold analysis |

- **Intensity mask:** a number is the fixed threshold on a spectrum's maximum intensity
  (`glioma_mil.h5` is built with 0.60, `glioma-preproc-3.h5` with 0.40); `adaptive` is the per-slide rule of
  `glioma-preproc-2.h5` (README, step 1).
- **TR/TR models:** the experiments on `glioma_mil.h5` read them from `exp_501` (`paths.models_baseline_dir`) and
  skip their own `trtr` stage; `exp_2.6xx` and `exp_4.608` train their own, since their data differ.
- **Reuse:** `exp_705`-`707` train only their TSR/TR models, on synthetic spectra drawn from the CVAE of
  `exp_608`; `exp_708` generates a pool of 40,000 spectra per class from the same decoders and trains only
  TS/TR models (`synth_gen` replaces `cvae`).
- **`exp_605`, `exp_606`:** `epochs` is the number of *additional* epochs. They continue the `exp_601` weights
  under a $\beta$ schedule (`beta_window_schedule`: one value per window of 10 epochs). Their CVAE stage is
  started with `--resume-from` (README, step 2).
- **Experiment names:** the digit after `exp_` groups the experiments: `5xx` TR/TR baseline, `6xx` $\beta$,
  `7xx` the other controlled factors, `2.6xx` and `4.608` the intensity mask.

## Keys that the code reads

| Block | Keys |
|---|---|
| `paths` | `processed_file`, `splits_file`, the model, synthetic-data and statistics directories, `calib_file` |
| `pipeline` | `stages`: any of `trtr`, `cvae`, `synth_gen`, `tstr`, `tsrtr`, `stats_eval`, `mil_fit` |
| `preprocessing` | `crop_region` (cm$^{-1}$), `signal_length` |
| `augmentation` | `enabled`, `max_shift`, `mixup_alpha` (also `baseline_wobble`, `intensity_jitter`, `additive_noise`, `band_dropout`) |
| `cv_strategy` | `split_source`, `n_folds`, `patient_cap` (maximum spectra per slide in a training partition) |
| `training` | `random_seed` |
| `models` | `apollo` (RF+SVM), `random_forest`, `xgboost`, `cnn`, `cvae` (architecture, `loss_weights`: `alpha` classifier, `beta` KL, `gamma` MSE, `delta` cosine) |
| `tsrtr` | `synth_ratio`, `cvae_weights_dir` |
| `statistics` | `bootstrap_samples`, `alpha`, `random_seed` |
| `mil` | `pi_floor` (minimum selection fraction $\pi_{\min}$), `svm_subsample_per_bag`, `cbr_temperature` |

The preprocessing of the data (median filter, IARPLS with $\lambda=10^4$, the intensity mask) is fixed when the
dataset is built, not by these configs.
