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

## Detailed Roadmap:

## 📊 Project Status Dashboard

**Current Phase:** Phase 1 (Foundation & Benchmarking)
**Status:** 🟡 **In Progress** (Data Processing)

| Component | Status | Notes |
| --- | --- | --- |
| **Data Pipeline** | 🟢 **Ready** | `preprocessing.py` fixed (alignment logic corrected). Running/Ready to run. |
| **Data Integrity** | 🟢 **Verified** | `test_data_integrity.py` passed. Files and maps align perfectly. |
| **Data Loader** | 🟢 **Ready** | `loader.py` implements Group-Aware splitting to prevent leakage. |
| **Baseline Model** | 🔴 **Pending** | `src/models/baselines.py` needs creation. |
| **Benchmark** | 🔴 **Pending** | `01_Apollo_Baseline.ipynb` needs to be run to get target accuracy. |

---


### **Phase 1: Foundation & Benchmarking (We are here)**

*Goal: Build the data engine and establish the "Gold Standard" accuracy to beat.*

* **1.1 Data Engineering:**
* **Action:** Run `preprocessing.py` to generate `glioma_dataset.h5`.
* **Action:** Run `splitter.py` to freeze the 5-Fold Cross-Validation splits (Manual Split from Apollo paper).


* **1.2 The Apollo Baseline Reproduction:**
* **Action:** Implement `ApolloPipeline` in `src/models/baselines.py`.
* *Logic:* Random Forest (300 trees) selects Top 20 bands  SVM (RBF) classifies.


* **Action:** Train/Test on Real Data (TRTR).
* **Artifact:** `assets/apollo_features.npy` (The indices of the 20 most discriminative bands).
* **Output:** The **Target Accuracy** (e.g., 72%). Our synthetic data must enable an SVM to reach this number.



### **Phase 2: Generative Model Development**

*Goal: Develop models that can generate high-fidelity spectra AND classify biological states.*

* **2.1 CVAE Development (The "Dual-Purpose" Model):**
* **Architecture:** 1D-CNN Encoder  Latent Space ()  1D-CNN Decoder.
* **Task A (Generation):** Train Encoder+Decoder to minimize Reconstruction Loss + KL Divergence.
* **Task B (Classification):** Freeze the trained Encoder, attach a Softmax Head, and train *only* the head on Real Data.
* **Hypothesis:** The CVAE Encoder (Deep Learning) should outperform the Apollo SVM (Shallow Learning).


* **2.2 RBM Development:**
* **Architecture:** Gaussian-Bernoulli RBM.
* **Implementation:** Custom TensorFlow layer/loop using Contrastive Divergence (CD-k).
* **Task:** Train joint distribution . Use "Clamping" to generate spectra.



### **Phase 3: The "TSTR" Evaluation (The Turing Test)**

*Goal: Prove synthetic data utility by training the Apollo SVM on it.*

* **3.1 Synthetic Data Generation:**
* Generate 50,000 synthetic spectra per class using the best CVAE.
* Generate 50,000 synthetic spectra per class using the best RBM.


* **3.2 The Comparative Experiment (Critical Step):**
* **Constraint:** All models must use the *exact same* Top 20 features found in Phase 1.2.
* **Model A (Control):** Apollo SVM trained on **Real Data**.
* **Model B (Experiment):** Apollo SVM trained on **CVAE Synthetic Data**.
* **Model C (Experiment):** Apollo SVM trained on **RBM Synthetic Data**.


* **3.3 Success Criteria:** If Model B/C accuracy is within ~5% of Model A, the synthetic data is "High Fidelity."

### **Phase 4: Interpretation & Thesis**

* **4.1 Spectral Analysis:** Compare mean spectra and standard deviations (Real vs. Synthetic).
* **4.2 Latent Space Analysis:** t-SNE visualization of Real vs. Synthetic spectra.
* **4.3 Spectrometer Fingerprinting:** Check if synthetic data accidentally learned the "noise" of specific machines (a common failure mode).

Data can be found at the following link:
https://drive.google.com/drive/folders/1CNAwZa3H22FQtprSRXwHXsJ8FdazIUBd