# Glioma IDH Classification via CVAE: Synthetic Augmentation, Generative Classification, and MIL for Raman Biopsies

Code and experiments for the paper of the same title (v2.4). The manuscript sources are
not tracked in this repository; `paper/` keeps only the table generators.

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
scripts/        Dataset builders, MIL evaluation tests, bootstraps and paper figures
configs/        Experiment definitions (exp_*.yaml) + EXPERIMENT_REGISTRY.json
notebooks/      The six review notebooks (one per analysis stage, below)
paper/          Generators of the paper tables (paper/gen_*.py); manuscript sources are not tracked
docs/           Design notes and findings per stage
tests/          Unit and regression tests
requirements.txt
```

## Analysis notebooks

Each notebook is a thin, runnable driver over the shared `pipeline/` (or
`src/`) code, so a notebook run and a headless run produce identical results.
Set the experiment via `CONFIG_NAME` near the top of each notebook.

| Notebook | Paper stage | What it does |
|----------|-------------|--------------|
| `00_Stage0_Baseline_Correction` | Baseline correction | Baseline-correction method and IARPLS λ selection (Appendix: Baseline Correction) |
| `01_NMF_GMM_Tissue_Classification` | Spectrum selection | Per-slide strong-signal mask (3×3 median filter, normalisation, maximum intensity ≥ 0.60), the input of `scripts/build_glioma_mil.py`; also the legacy NMF + GMM clustering, kept only as the unsupervised `viable_legacy` reference |
| `02_Train_Classifiers` | TR/TR · TS/TR · TSR/TR | Trains the 4 benchmark classifiers (Apollo RF+SVM, RandomForest, XGBoost, 1D-CNN) under a selectable regimen |
| `03_CVAE` | Generation | Trains the β-CVAE per fold and generates the synthetic spectrum pool |
| `05_Classification_reconstruction` | CbR | Visual walkthrough of Classification by Reconstruction (illustrative; metrics live in nb 09) |
| `09_Statistical_evaluation` | Results | Slide-level bootstrap comparison of regimens; the paper's patient-level intervals come from `scripts/patient_bootstrap.py` (below) |

The dataset contains no tissue annotations: the mask selects spectra by signal
strength, not by tumour content.

Preprocessing (normalisation to [0, 1], then IARPLS baseline correction with
λ = 10⁴; patient-grouped splits) is handled by `pipeline/preprocessing_stage.py`
and `src/data/`.

## Reproducing the experiments

**1. Environment**
```bash
pip install -r requirements.txt        # tested with Python 3.13; TensorFlow, scikit-learn, xgboost, h5py …
```

**2. Data.** The raw spectral cubes and processed dataset are available here:
<https://drive.google.com/drive/folders/1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd>
Place them under `data/` as referenced by the configs (`data/raw/`, `data/splits/`).

**3. Run end-to-end (headless).** The full pipeline runs from the command line —
no notebooks required. See **[`pipeline/README.md`](pipeline/README.md)** for the
stage list, CLI flags, backends (local/Modal GPU), and output layout.
```bash
# TR/TR baseline, the default configuration (beta=0.5), then the ratio
# ablations that reuse its CVAE
python -m pipeline.run_suite --configs exp_501 exp_608 exp_705 exp_706 exp_707

# A single stage for one experiment
python -m pipeline.tstr_stage --config exp_608
```
Each stage reads its configuration from `configs/<name>.yaml` and writes models,
synthetic pools, and statistics under `results/<name>/`.

**4. Patient-level bootstrap** (the intervals reported in the paper):
```bash
python scripts/patient_bootstrap.py --stage all    # writes results/patient_bootstrap/
```
The stage bootstraps above are slide-level. `paper/gen_appendix_controlled_experiments.py`
regenerates the paper tables from those slide-level results; the v2.4 tables carry the
patient-level values and later hand edits, so running the generator overwrites them.

**5. Or step through the notebooks** in the order in the table above.

## Dataset

300,506 Raman spectra from 59 glioma biopsies of 46 patients (Lita et al.), each a
`(H, W, 1738)` hyperspectral cube; 288,678 spectra are retained at the 0.60
intensity threshold. Labels LGm1–LGm3 are IDH-mutant, LGm4–LGm6 IDH-wildtype. See
Table 1 in the paper for the class composition.
