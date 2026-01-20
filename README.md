# glioma-synthetic-data

# Synthetic Raman Spectra for Glioma Detection

**Project Status:** 🚧 Phase 1: Data Engineering & Baseline (In Progress)

This project aims to develop generative machine learning models (CVAE, RBM) to synthesize high-fidelity Raman spectra of glioma brain tissue. The core objective is to validate whether synthetic data can replace or augment real data in clinical classification pipelines ("Train on Synthetic, Test on Real").

## 📂 Project Structure

```text
glioma-synthetic-data/
├── data/
│   ├── raw/                  # Source .npy spectral cubes (LGm-1 to LGm-6)
│   ├── processed/            # Optimized HDF5 dataset (glioma_dataset.h5)
│   ├── splits/               # Frozen JSON files defining Train/Test folds
│   └── tumor_maps.npy        # Ground truth tumor masks for raw files
│
├── src/                      # Core Python Library
│   ├── data/
│   │   ├── preprocessing.py  # The "Builder": AirPLS + Normalization + H5 Writer
│   │   └── loader.py         # The "Loader": Lazy loading & Leakage-free splitting
│   ├── models/
│   │   └── baselines.py      # Apollo pipeline (RF Selector + SVM Classifier)
│
├── notebooks/
│   ├── 00_Preprocessing.ipynb  # Executable notebook to build the dataset
│   └── 01_Apollo_Baseline.ipynb # Benchmarking real data accuracy
│
├── tests/                    # Data integrity & alignment tests
└── README.md

```

## ⚙️ Data Pipeline & Preprocessing

Raman spectroscopy data is high-dimensional and noisy. We implement a rigorous **"Builder-Loader" pattern** to handle ~300,000 spectra efficiently.

### 1. The Builder (`src/data/preprocessing.py`)

This module converts raw 2D spectral scans into a machine-learning-ready format.

* **Streaming Architecture:** Processes samples one-by-one to keep RAM usage low (<500MB), writing directly to disk.
* **Alignment:** Strictly aligns raw files with `tumor_maps.npy` by sorting via full file paths, ensuring strict 1-to-1 correspondence.
* **Signal Processing:**
* **Baseline Correction:** Uses **AirPLS** (, order=1) to remove non-biological fluorescence background.
* **Normalization:** Min-Max scaling to  per spectrum.
* **Filtering:** Excludes specific low-quality samples (e.g., `HF-1887`) as per established literature.


* **Output:** Generates `data/processed/glioma_dataset.h5`, a hierarchical file containing:
* `/spectra`: The cleaned signals.
* `/y_class`: Glioma subtype labels (1-6).
* `/sample_ids`: Unique patient/sample identifiers (critical for splitting).



### 2. The Loader (`src/data/loader.py`)

A fast, memory-efficient data manager for training loops.

* **Lazy Loading:** Reads data from HDF5 on-demand (supports memory mapping).
* **Leakage Prevention:** Implements **Group-Aware Splitting**. It ensures that if a tissue sample (e.g., `HF-448`) is in the Training Set, *no pixel* from that sample appears in the Test Set.
* **Reproducibility:** Loads frozen split definitions from `data/splits/manual_splits.json` to guarantee that every model (SVM, CVAE, RBM) is tested on the exact same data folds.

## 🔬 Methodology: The "Apollo" Baseline

To evaluate the quality of our synthetic data, we first reproduce the "Gold Standard" methodology from prior work (Code Name: Apollo).

* **Feature Selection:** A Random Forest classifier identifies the **Top 20** most discriminative spectral bands from the real data.
* **Classification:** A Support Vector Machine (SVM) with an RBF kernel is trained on these 20 features.
* **Goal:** This establishes the "Target Accuracy." Our generative models will be judged by their ability to train *this same SVM architecture* to achieve comparable performance on real test data.

## 🚀 Usage

### 1. Build the Dataset

Run the preprocessing pipeline. This takes ~1-2 hours but only needs to be run once.

```bash
# Or open notebooks/00_Preprocessing.ipynb
python -m src.data.preprocessing

```

### 2. Run Tests

Verify that the raw files and tumor maps are perfectly aligned.

```bash
python -m tests.test_data_integrity

```

### 3. Run Baseline

Train the reference SVM on real data to get the benchmark score.

```bash
# Or open notebooks/01_Apollo_Baseline.ipynb
python -m src.models.baselines

```

## 📅 Roadmap

* [x] **Phase 1:** Data Pipeline & Integrity Checks
* [ ] **Phase 2:** Baseline Reproduction (RF + SVM)
* [ ] **Phase 3:** Generative Model Development (CVAE & RBM)
* [ ] **Phase 4:** TSTR Evaluation (Train Synthetic, Test Real)