# Generative Augmentation of Raman Spectra for Glioma Classification

Code and experiments for the paper *Generative Augmentation of Raman Spectra for
Glioma Classification* (see [`paper/main.tex`](paper/main.tex)).

We study whether a conditional variational autoencoder (β-CVAE) can generate
useful synthetic Raman spectra in a small-cohort glioma setting. Synthetic data
is evaluated under **TS/TR** (train on synthetic, test on real), **TSR/TR**
(train on synthetic + real, test on real), and a reconstruction-based inference
strategy, **Classification by Reconstruction (CbR)** — all under strict
**slide-isolated** cross-validation to prevent leakage. Task: binary IDH status
(mutant vs wildtype). The code also covers multiple-instance learning (MIL)
formulations evaluated at slide level.

## Repository layout

```text
src/            Core library: data loading and preprocessing, models (CVAE, CNN,
                Apollo baseline), MIL formulations (src/mil/), spatial evaluation,
                statistics (src/utils/statistical_evaluation.py)
pipeline/       Headless, config-driven experiment runner — see pipeline/README.md
scripts/        Dataset builders, MIL evaluation tests, bootstraps and paper figures
configs/        Experiment definitions (exp_*.yaml) + EXPERIMENT_REGISTRY.json
notebooks/      The six review notebooks (one per analysis stage, below)
paper/          Manuscript sources and the generators of its tables (paper/gen_*.py)
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
| `01_NMF_GMM_Tissue_Classification` | Tumour selection | Spatial filtering + NMF + GMM to isolate pure tumour spectra and build the HDF5 dataset |
| `02_Train_Classifiers` | TR/TR · TS/TR · TSR/TR | Trains the 4 benchmark classifiers (Apollo RF+SVM, RandomForest, XGBoost, 1D-CNN) under a selectable regimen |
| `03_CVAE` | Generation | Trains the β-CVAE per fold and generates the synthetic spectrum pool |
| `05_Classification_reconstruction` | CbR | Visual walkthrough of Classification by Reconstruction (illustrative; metrics live in nb 09) |
| `09_Statistical_evaluation` | Results | Slide-level cluster-bootstrap comparison of regimens (the reported numbers) |

Preprocessing (stage i: airPLS baseline correction, normalisation, slide-isolated
splits) is handled by `pipeline/preprocessing_stage.py` and `src/data/`.

## Reproducing the experiments

**1. Environment**
```bash
pip install -r requirements.txt        # Python 3.11+, TensorFlow, scikit-learn, xgboost, h5py …
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

**4. Or step through the notebooks** in the order in the table above.

## Dataset

~270,000 Raman spectra from 59 glioma biopsies of 46 patients (Lita et al.), each a
`(H, W, 1738)` hyperspectral cube. Labels LGm1–LGm3 are IDH-mutant, LGm4–LGm6
IDH-wildtype. See Table 1 in the paper for the class composition.
