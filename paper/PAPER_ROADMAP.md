# Paper Status & Roadmap

> **Paper:** *Synthetic Data Augmentation in Glioma Raman Spectroscopy*  
> **Authors:** Andrei Iușan, Iulian Vasile  
> **Last Updated:** 2026-02-28

---

## Current Status Overview

| Section | Status | Notes |
|:--------|:-------|:------|
| **Abstract** |  Complete | Well-written, may need minor revisions once TSTR numbers are in |
| **1. Introduction** |  Complete | Clinical context, data scarcity, research objectives, paper outline |
| **2. Background** |  Complete | Raman fundamentals, VAE/CVAE theory, synthetic data lit review |
| **3. Methods** |  Complete | Preprocessing, annotation, architecture, training, experimental design |
| **4. Results** | 🟡 Framework Ready | TRTR table filled (29.5%); TSTR, Wasserstein, t-SNE need data |
| **5. Discussion** |  Draft Ready | Generalization gap, annotation role, TSTR interpretation, limitations |
| **6. Conclusion** |  Draft Ready | Summary of contributions; needs TSTR numbers for final claims |
| **Bibliography** |  Created | `references.bib` with 15 key references; expand as needed |
| **Figures** |  Pending | Notebook `05_Paper_Figures.ipynb` created to generate all figures |

---

## What's Done

### Code & Experiments
- [x] Data preprocessing pipeline (airPLS + Min-Max normalization → HDF5)
- [x] Spatio-spectral GMM annotator (per-slide tumor separation → ~88k tumor spectra)
- [x] Data splitting (5-fold patient-level CV, frozen to `manual_splits.json`)
- [x] Apollo baseline (RF Top-20 + SVM): **29.5% ± 5.1%** on 6-class grading
- [x] 1D-CNN baseline architecture implemented (`src/models/cnn.py`)
- [x] β-CVAE architecture (1D-Conv encoder/decoder, auxiliary classifier)
- [x] CVAE training (single fold, weights saved in `experiments/`)
- [x] Presentation visuals notebook (t-SNE, reconstruction, class centroids)

### Paper Sections Written
- [x] Abstract
- [x] Introduction (3 subsections + paper outline)
- [x] Background: Raman spectroscopy fundamentals with citations
- [x] Background: VAE theory (ELBO, reparameterization trick)
- [x] Background: Conditional VAEs
- [x] Background: Synthetic data in biomedicine (TSTR, fidelity/diversity/privacy)
- [x] Methods: Data & preprocessing
- [x] Methods: Spatio-spectral tumor annotation (MRF + GMM, NEW section)
- [x] Methods: Model architecture (with TikZ diagram)
- [x] Methods: Training protocol (loss function, hyperparameters)
- [x] Methods: Experimental design (TRTR, TSTR, statistical fidelity)
- [x] Results: TRTR baseline table (actual fold-level numbers)
- [x] Results: Section scaffolding for TSTR, t-SNE, reconstruction, privacy
- [x] Discussion: Generalization gap interpretation
- [x] Discussion: Role of spatio-spectral annotation
- [x] Discussion: TSTR interpretation framework
- [x] Discussion: Privacy & data sharing
- [x] Discussion: Limitations & future work
- [x] Conclusion: Summary of contributions

---

## What's Left to Do

###  Priority 1: Experiments (Blocking)

These generate the numbers needed to fill the paper tables and figures.

| Task | Effort | Dependency |
|:-----|:-------|:-----------|
| **1a. Train CVAE in all 5 CV folds** | ~2-4 hours (GPU) | Notebook 03 needs to loop over folds |
| **1b. Generate synthetic data per fold** | ~30 min | Trained CVAE per fold |
| **1c. Run TSTR evaluation** | ~1 hour | Synthetic data + Apollo pipeline |
| **1d. Run 1D-CNN baseline (TRTR)** | ~1 hour | Compare shallow (SVM) vs deep (CNN) |

**Action:** Run `notebooks/05_Paper_Figures.ipynb` end-to-end. It handles TSTR + all figure generation. If the CVAE has only been trained on 1 fold, you need to either:
- Option A: Train the CVAE inside a 5-fold loop (modify notebook 03)
- Option B: Use the single-fold CVAE for all figures and note the limitation

### 🟡 Priority 2: Fill Paper Sections (After experiments)

| Section | What's Needed |
|:--------|:-------------|
| **Table 2 (TSTR)** | Fill the `---` entries in `\ref{tab:tstr}` with actual TSTR fold accuracies |
| **§4.2 Statistical Fidelity** | Report Wasserstein-1 distances (per class means from notebook 05) |
| **§4.3 Latent Space** | Uncomment `\includegraphics` for t-SNE figure, add interpretation |
| **§4.4 Reconstruction** | Uncomment `\includegraphics` for reconstruction figure |
| **§4.5 Privacy Metrics** | Report authenticity (NN distances), α-precision, β-recall |
| **Abstract** | Update with actual TSTR accuracy numbers |
| **Conclusion** | Update claims based on actual TSTR vs TRTR delta |

### 🟢 Priority 3: Polish (Final pass)

| Task | Notes |
|:-----|:------|
| Expand bibliography | Add 5-10 more references (especially batch effects, domain adaptation) |
| Architecture diagram | The TikZ diagram is inline; wrap in `\begin{figure}` with caption/label |
| Table formatting | Verify table widths compile correctly in LaTeX |
| Remove TODO comments | Search for `% TODO` in main.tex |
| Proofread | Grammar, consistency, notation |
| Compile LaTeX | Verify no errors with `pdflatex` + `bibtex` |

---

## File Map

| File | Purpose |
|:-----|:--------|
| `paper/main.tex` | The paper (all sections) |
| `paper/references.bib` | Bibliography entries |
| `paper/figures/` | Output directory for generated figures (PDF/PNG) |
| `notebooks/05_Paper_Figures.ipynb` | Generates all figures + runs TSTR evaluation |
| `results/baseline_metrics.csv` | TRTR fold-level accuracies |
| `results/tstr_metrics.csv` | TSTR fold-level accuracies (generated by notebook 05) |

---

## Recommended Workflow

```
1. Activate environment:       conda activate glioma-apollo
2. Run CVAE training:          Execute notebook 03 (all 5 folds)
3. Generate figures + TSTR:    Execute notebook 05 end-to-end
4. Fill paper tables:          Update main.tex with numbers from step 3
5. Compile paper:              cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
6. Review & polish:            Proofread, verify figures, remove TODOs
```
