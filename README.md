# Glioma IDH Classification via CVAE: Synthetic Augmentation, Generative Classification, and MIL for Raman Biopsies

Code and experiments for paper v2 of the same title. The manuscript is not part of this repository;
`scripts/paper/` holds the generator of its tables.

A β-regularized conditional variational autoencoder (β-CVAE) is used in three roles on a small Raman
cohort, for binary IDH status (mutant vs wildtype):

1. **Generator** of synthetic spectra, evaluated under **TS/TR** (train on synthetic, test on real) and
   **TSR/TR** (train on synthetic + real, test on real) against the **TR/TR** baseline.
2. **Generative classifier**: **Classification by Reconstruction (CbR)**.
3. **Multiple-instance learning (MIL)** model of slides. CbR-based CVAE MIL is compared with naive MIL
   (majority vote over spectra), mi-SVM and attention MIL; each MIL model is also tested for spatial
   coherence (real vs shuffled-label refit) and key-instance utility.

Cross-validation is patient-grouped (every slide of a patient lies in one partition); confidence
intervals come from a class-stratified bootstrap over patients.

## Repository layout

```text
src/            Library: data loading and preprocessing, models (CVAE, CNN, RF+SVM), MIL models
                (src/mil/), spatial statistics, bootstrap and metrics (src/utils/statistical_evaluation.py)
pipeline/       Config-driven experiment runner (stages, Modal backends): see pipeline/README.md
scripts/        Dataset builders, MIL tests, bootstraps, analyses and figures
scripts/paper/  Generator of the paper's tables
configs/        One YAML file per experiment: see configs/README.md for the list
data/           splits/manual_splits.json (the five folds) and the wavenumber table; the spectra are downloaded
tests/          Unit and regression tests (pytest)
```

`results/` and the datasets are generated or downloaded; nothing under them is versioned.

## Reproducing the paper

All commands run from the repository root. Use Python 3.13 with `pip install -r requirements.txt`.
Each experiment is a config (`configs/README.md` lists them and the paper analysis they belong to);
results go to `results/<experiment>/`. Steps skip work whose outputs already exist.

**0. Data.** The raw cubes and the processed datasets are in
<https://drive.google.com/drive/folders/1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd>. Place the cubes under
`data/raw/LGm-*/` and the HDF5 datasets under `data/processed/`; with the datasets in place, skip step 1.
`python scripts/dataset_summary.py` prints the class composition of the paper's Table 1.

**1. Datasets** (only to rebuild them from the raw cubes). Each builder applies a 3×3 spatial median filter,
normalises the slide, keeps the spectra whose maximum intensity passes the mask, then applies IARPLS
baseline correction (λ = 10⁴) and per-spectrum min-max normalisation.

| Dataset | Intensity mask | Used by | Command |
|---|---|---|---|
| `glioma_mil.h5` | maximum ≥ 0.60 | all experiments except those below | `python scripts/build_glioma_preproc_3.py --threshold 0.6 --out data/processed/glioma_mil.h5` |
| `glioma-preproc-2.h5` | adaptive, per slide | `exp_2.607`–`exp_2.609` | `python scripts/build_glioma_preproc_2.py` |
| `glioma-preproc-3.h5` | maximum ≥ 0.40 | `exp_4.608` | `python scripts/build_glioma_preproc_3.py --threshold 0.4` |

**2. Classifiers, CVAE and slide-level statistics.** `pipeline/README.md` describes the stages and the Modal
backends. The 1D-CNN trains on a Modal GPU by default (`models.cnn.backend: modal` in the configs); set it
to `local` to train it on this machine.
```bash
python -m pipeline.run_suite --configs exp_501 exp_608       # TR/TR models (shared); default CVAE (β = 0.5)
python -m pipeline.run_suite --configs exp_601 exp_602 exp_603 exp_604 exp_607 exp_609 \
    exp_701 exp_702 exp_703 exp_704 exp_705 exp_706 exp_707 exp_708 \
    exp_2.607 exp_2.608 exp_2.609 exp_4.608
# exp_605 and exp_606 continue the exp_601 weights for 100 more epochs under a windowed β schedule:
for e in exp_605 exp_606; do
    python -m pipeline.cvae_stage --config $e --resume-from results/exp_601/models/cvae --epoch-offset 100
done
python -m pipeline.run_suite --configs exp_605 exp_606       # their remaining stages; the CVAE stage is skipped
```
Run `exp_501` and `exp_608` first: the other experiments on `glioma_mil.h5` reuse the TR/TR models of
`exp_501`, and `exp_705`–`exp_708` reuse the CVAE decoders of `exp_608`.

**3. MIL models and their tests**, per configuration. The mi-SVM fits are slow on CPU (`MISVM_BACKEND=modal`
runs the five folds in parallel containers); the CVAE control retrains the CVAE on a Modal GPU (skip it with
`SKIP_CVAE_CONTROL=1`):
```bash
for e in exp_607 exp_608 exp_609 exp_2.607 exp_2.608 exp_4.608; do bash scripts/run_mil_tests.sh $e; done
```
The script fits mi-SVM, attention MIL and CVAE MIL (`pipeline/mil_fit_stage.py`,
`pipeline/mil_attention_stage.py`), scores them at bag level (`scripts/bag_level_eval.py`,
`scripts/attention_bag_level_eval.py`), refits them under shuffled bag labels
(`scripts/shuffled_label_control.py`), tests spatial coherence of real vs shuffled fits
(`scripts/spatial_paired_real_vs_shuffled.py`) and key-instance utility
(`scripts/cvae_weighted_rf_vote.py`, `scripts/key_instance_paired_bootstrap.py`).

**4. Pairwise and patient-level bootstraps** (the intervals in the paper):
```bash
python scripts/latent_dim_pairwise_bootstrap.py     # d = 16 vs 32 vs 64, slide level
python scripts/synth_pool_pairwise_bootstrap.py     # synthetic pool of exp_608 vs exp_708, slide level
python scripts/patient_bootstrap.py --stage all     # writes results/patient_bootstrap/
```
The bootstraps of steps 2–3 and the two pairwise scripts resample slides. `patient_bootstrap.py` repeats every
paper comparison resampling patients, after checking that it reproduces the stored slide-level intervals and
that point estimates agree.

**5. Analyses and figures**
```bash
python scripts/baseline_correction_study.py --out-dir results/stage0_baseline     # baseline-correction appendix (n = 40 per level)
python scripts/baseline_generator_stress.py --out-dir results/stage0_baseline
python scripts/latent_kl_per_dim.py --check         # active latent dimensions per fold (β appendix)
python scripts/plot_beta_sensitivity_combined.py    # β-sensitivity figure
python scripts/plot_max_intensity_histogram.py      # maximum-intensity histogram (threshold analysis)
```
The figure scripts write into `$PAPER_DIR/img/` (default `./paper/img/`).

**6. Paper tables**
```bash
PAPER_DIR=<manuscript dir> python scripts/paper/gen_appendix_controlled_experiments.py
```
Writes the controlled-experiment, results and MIL tables. Intervals are the patient-level ones of step 4
(`--unit slide` selects the slide-level intervals of steps 2–3); point estimates are identical. The tables of
the baseline-correction appendix are assembled from the CSV files of step 5, and the β appendix tables are
written by `scripts/latent_kl_per_dim.py` to `results/latent_kl_per_dim/kl_per_dim_tables.tex`.

## Where the paper's elements are implemented

| Paper | Code |
|---|---|
| Preprocessing, intensity threshold | `src/data/preprocessing_v2.py`, `preprocessing_v3.py`; `src/preprocessing/baseline.py` (IARPLS) |
| β-CVAE architecture and loss | `src/models/cvae.py`, `src/models/losses.py`; training loop `pipeline/cvae_stage.py` |
| Classification by reconstruction | `pipeline/evaluation_stage.py::_cbr_predictions` |
| RF+SVM, RF, XGBoost, 1D-CNN | `src/models/baselines.py`, `src/models/cnn.py`, `pipeline/_common.py`; the code and result files call RF+SVM `Apollo` (the method of Lita et al.) |
| TR/TR, TS/TR, TSR/TR | `pipeline/trtr_stage.py`, `tstr_stage.py`, `tsrtr_stage.py` |
| Bag-level metrics, bootstraps | `src/utils/statistical_evaluation.py`, `scripts/patient_bootstrap.py` |
| mi-SVM, attention MIL, CVAE MIL | `src/mil/misvm.py`, `src/mil/attention.py`, `src/mil/generative.py` |
| Spatial coherence, key-instance utility | `src/evaluation/spatial.py`, `scripts/spatial_paired_real_vs_shuffled.py`, `scripts/cvae_weighted_rf_vote.py` |
| Folds | `data/splits/manual_splits.json` |

Utilities: `scripts/train_cnn_modal_parallel.py` trains the 1D-CNNs of several folds concurrently on Modal
(`GLIOMA_DEFER_CNN=1` defers them in `run_suite`).

## Dataset

300,506 Raman spectra from 59 glioma biopsies of 46 patients (Lita et al.), each a `(H, W, 1738)`
hyperspectral cube; 288,678 spectra are retained at the 0.60 threshold. Labels LGm1–LGm3 are IDH-mutant,
LGm4–LGm6 IDH-wildtype. The dataset has no tissue annotations: the mask selects spectra by signal strength,
not by tumour content.

## Acknowledgements

This work was supported by the project PNRR-I8 no. 842027778, contract no. 760096.
