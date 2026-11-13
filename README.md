# Glioma IDH Classification via CVAE: Synthetic Augmentation, Generative Classification, and MIL for Raman Biopsies

Code and experiments for paper v2 of the same title. The manuscript sources are not
part of this repository; `scripts/paper/` holds the generators of its tables.

A β-regularized conditional variational autoencoder (β-CVAE) is used in three
roles on a small Raman cohort, for binary IDH status (mutant vs wildtype):

1. **Generator** of synthetic spectra, evaluated under **TS/TR** (train on
   synthetic, test on real) and **TSR/TR** (train on synthetic + real, test on
   real) against the **TR/TR** baseline.
2. **Generative classifier**: **Classification by Reconstruction (CbR)**.
3. **Multiple-instance learning (MIL)** classifier of slides. CbR is compared with
   naive MIL (majority vote over spectra), mi-SVM and attention MIL, and each MIL
   posterior is tested for spatial coherence (real vs shuffled-label refit) and
   key-instance utility.

Cross-validation is patient-grouped (every slide of a patient lies in one
partition), and confidence intervals come from a class-stratified bootstrap over
patients.

## Repository layout

```text
src/            Core library: data loading and preprocessing, models (CVAE, CNN,
                Apollo baseline), MIL formulations (src/mil/), spatial evaluation,
                statistics (src/utils/statistical_evaluation.py)
pipeline/       Headless, config-driven experiment runner — see pipeline/README.md
scripts/        Dataset builders, MIL evaluation tests, bootstraps, figures
scripts/paper/  Generators of the paper tables (write into $PAPER_DIR)
configs/        Experiment definitions (exp_*.yaml) + EXPERIMENT_REGISTRY.json
tests/          Unit and regression tests
requirements.txt
```

## Reproducing the paper

All commands run from the repository root. Each step reads `configs/<exp>.yaml` and
writes under `results/<exp>/` (or the directory named in the script's docstring).

**0. Environment and data**
```bash
pip install -r requirements.txt        # tested with Python 3.13
```
The raw spectral cubes and the processed datasets are available at
<https://drive.google.com/drive/folders/1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd>.
Place the cubes under `data/raw/LGm-*/` and the HDF5 datasets under
`data/processed/`; with the HDF5 datasets in place, skip step 1.

**1. Datasets** (only to rebuild from the raw cubes)

| Dataset | Used by | Build |
|---------|---------|-------|
| `glioma_mil.h5` (fixed mask, 0.60) | `exp_501`, `exp_6xx`, `exp_7xx` | `python scripts/build_glioma_preproc_3.py --threshold 0.6 --out data/processed/glioma_mil.h5` |
| `glioma-preproc-2.h5` (adaptive mask) | `exp_2.607`–`exp_2.609` | `python scripts/build_glioma_preproc_2.py` |
| `glioma-preproc-3.h5` (fixed mask, 0.40) | `exp_4.608` | `python scripts/build_glioma_preproc_3.py` |

Each builder applies a 3×3 spatial median filter, normalises the slide, keeps the
spectra whose maximum intensity passes the mask, and applies IARPLS baseline
correction (λ = 10⁴) per spectrum. The `--threshold 0.6` build reproduces the
distributed `glioma_mil.h5` exactly (spectra, coordinates, slide and patient IDs,
labels); it omits only that file's `tissue_cluster` and `viable_legacy` columns,
which come from the v1 NMF/GMM pipeline and are not used by any v2 analysis.
`scripts/build_glioma_mil.py` is the v1 builder of that file and needs the v1
per-slide clustering artifacts.

**2. Classifiers, CVAE and slide-level statistics** — see
[`pipeline/README.md`](pipeline/README.md) for stages, flags and the Modal backends.
```bash
python -m pipeline.run_suite --configs exp_501 exp_608      # TR/TR baseline, default CVAE (β = 0.5)
python -m pipeline.run_suite --configs exp_601 exp_602 exp_603 exp_604 exp_605 exp_606 \
    exp_607 exp_609 exp_701 exp_702 exp_703 exp_704 exp_705 exp_706 exp_707 exp_708 \
    exp_2.607 exp_2.608 exp_2.609 exp_4.608
python scripts/patch_cbr_auroc.py --experiments exp_501 exp_608 ...   # CbR bag-level AUROC
```
Run `exp_501` and `exp_608` first: the ratio and pool-size experiments
(`exp_705`–`exp_708`) reuse their TR/TR models and CVAE decoders.

**3. MIL** (per configuration; `exp_608` shown)
```bash
python -m pipeline.mil_eval_stage --config exp_608                      # naive-MIL baselines
python -m pipeline.mil_fit_stage --config exp_608 --methods misvm cbr   # mi-SVM fit, CbR posteriors
python -m pipeline.mil_attention_stage --config exp_608                 # attention MIL
python scripts/stage6_bag_eval.py --config exp_608 --method misvm --variant mi_svm   # likewise cbr
python scripts/stage6_attention_bag_eval.py --config exp_608
# spatial coherence: real fit, shuffled-label refit, paired test (--method misvm | attention | cbr)
python scripts/stage5_spatial_validation.py --config exp_608 --method misvm
python scripts/stage6_shuffled_label_control.py --config exp_608 --method misvm
python scripts/spatial_paired_real_vs_shuffled.py --config exp_608
python scripts/standardized_variogram.py --config exp_608
# key-instance utility for all appendix configurations and tiers, with its paired test
bash scripts/run_key_instance_paired.sh               # PY=<python> to choose the interpreter
```
For the configurations other than `exp_608`, `python scripts/run_missing_mil_tests.py
--list` lists the remaining tests in dependency order and `--run` executes them
(`--include-modal` adds the shuffled-label refits on Modal).

**4. Pairwise and patient-level bootstraps** (the intervals reported in the paper)
```bash
python scripts/latent_dim_pairwise_bootstrap.py     # d = 16 vs 32 vs 64
python scripts/synth_pool_pairwise_bootstrap.py     # pool size: exp_608 vs exp_708
python scripts/patient_bootstrap.py --stage all     # writes results/patient_bootstrap/
```
The bootstraps of steps 2–3 and the two pairwise scripts resample slides;
`patient_bootstrap.py` re-runs every
paper comparison resampling patients, after checking that it reproduces the stored
slide-level intervals.

**5. Analyses and figures**
```bash
python scripts/latent_kl_per_dim.py --check         # active latent dimensions per fold (App. B)
python scripts/plot_beta_sensitivity_combined.py    # β-sensitivity figure
python scripts/plot_max_intensity_histogram.py      # max-intensity distribution
python scripts/stage0_baseline_study.py             # baseline-correction study (App. A)
python scripts/stage0_generator_stress.py
python scripts/stage1_lambda_sensitivity.py         # IARPLS λ sensitivity
```
The figure scripts write into `$PAPER_DIR/img/` (default `./paper/img/`).

**6. Paper tables**
```bash
PAPER_DIR=<manuscript dir> python scripts/paper/gen_appendix_controlled_experiments.py
```
This regenerates the controlled-experiment, results and MIL tables from the
slide-level results. The published tables carry the patient-level intervals of step
4 and later hand edits, so regenerating overwrites them. The other scripts in
`scripts/paper/` are one-off migrations of earlier manuscript versions (v2.2.1,
v2.3), kept for provenance.

Utilities: `scripts/train_cnn_modal_parallel.py` trains the 1D-CNNs of several
folds concurrently on Modal; `scripts/concat_cvae_history.py` merges the training
history of a resumed CVAE run.

## Dataset

300,506 Raman spectra from 59 glioma biopsies of 46 patients (Lita et al.), each a
`(H, W, 1738)` hyperspectral cube; 288,678 spectra are retained at the 0.60
intensity threshold. Labels LGm1–LGm3 are IDH-mutant, LGm4–LGm6 IDH-wildtype. The
dataset contains no tissue annotations: the mask selects spectra by signal strength,
not by tumour content. See Table 1 in the paper for the class composition.
