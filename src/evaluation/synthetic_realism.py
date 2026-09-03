"""
Synthetic realism evaluation for glioma Raman spectra.

Analyses:
  A. Mean ± std spectra per class (real vs synthetic)
  B. Representative overlay plots per class
  C. PCA / t-SNE / UMAP scatter plots coloured by source and class
  D. Biologically relevant Raman peak comparison
  E. Distributional distances: MMD + sliced Wasserstein
  F. Classifier two-sample test (C2ST) — logistic regression
  G. Nearest-neighbour memorisation check
  H. Intra-class diversity metrics

CLI usage:
  python -m src.evaluation.synthetic_realism --config exp_201 --fold 0

Module usage:
  from src.evaluation.synthetic_realism import run_all
  results = run_all(config, figures_dir='figures/synth_eval')
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn(
        "umap-learn not installed — UMAP plot will be skipped. "
        "Install with: pip install umap-learn"
    )

# ── Biologically relevant peaks within 700–1800 cm⁻¹ (paper §1 & §3) ────────
RAMAN_PEAKS: Dict[str, int] = {
    "Phe (1003)":               1003,
    "Phosphodiester (1087)":    1087,
    "Amide III (1250)":         1250,
    "CH2 twist / Lipid (1301)": 1301,
    "CH2/CH3 def. (1450)":      1450,
    "Amide II (1550)":          1550,
    "Amide I (1655)":           1655,
}

CLASS_NAMES: Dict[int, str] = {0: "IDH-Mutant", 1: "IDH-Wildtype"}

REAL_COLOR   = "#1f77b4"   # real spectra, in the two-group figures
REAL_TRAIN_COLOR = "#1f77b4"   # slides the generator was fitted to
REAL_TEST_COLOR  = "#17becf"   # slides held out in this fold
SYNTH_COLOR  = "#ff7f0e"
CLASS_COLORS = ["#2ca02c", "#d62728"]
ALPHA_FILL   = 0.20
ALPHA_LINE   = 0.55
FIG_DPI      = 150


# =============================================================================
# Data loading
# =============================================================================

def load_data(
    config: dict,
    fold_idx: int = 0,
    seed: int = 42,
    split: str = 'all',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real (cropped, renormalised) and synthetic spectra for one fold.

    Parameters
    ----------
    config    : experiment config dict
    fold_idx  : which fold's synthetic pool to load (0-4)
    seed      : unused here; kept for signature symmetry with the plot functions
    split     : which real spectra to return.
                'all'   — every spectrum in the dataset (default; the historical
                          behaviour, kept so existing callers are unaffected)
                'train' — only the slides in fold `fold_idx`'s TRAINING split,
                          i.e. exactly the spectra this fold's CVAE was fitted to
                'test'  — only the slides held out in fold `fold_idx`

                'all' pools train and test slides of every fold together, which
                is the wrong comparison for a per-fold figure: it mixes the data
                the generator saw with data it never saw. Prefer 'train'/'test'
                when the figure is about one fold's generator.

    Returns
    -------
    X_real      : (N, 571) float32
    y_real      : (N,)     int   — 0=IDH-Mutant, 1=IDH-Wildtype
    X_synth     : (M, 571) float32
    y_synth     : (M,)     int
    wavenumbers : (571,)   float64  cm⁻¹
    """
    import sys
    import os

    if split not in ('all', 'train', 'test'):
        raise ValueError(f"split must be 'all', 'train' or 'test'; got {split!r}")

    project_root = _find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.data.loader import GliomaDataLoader

    h5_path = project_root / config['paths']['processed_file']

    # Disable dynamic patient cap — we want all spectra for evaluation
    cfg_eval = {
        **config,
        'cv_strategy': {
            **config['cv_strategy'],
            'patient_cap': None,
            'cap_strategy': 'fixed',
        },
    }

    print(f"  Loading real spectra from {h5_path.name} ...")
    loader = GliomaDataLoader.from_config(str(h5_path), cfg_eval, memory_mapped=False)

    if split == 'all':
        X_real = loader.X.astype(np.float32)
        y_raw  = loader.y_class.astype(int).flatten()
    else:
        # Reuse the loader's own fold iteration, so the slide assignment here is
        # by construction the same one the experiments trained against.
        # Note the naming: the loader calls the held-out member 'val'; the split
        # JSON calls it 'test'.
        member = 'train' if split == 'train' else 'val'
        fold = None
        for f in loader:
            if f['fold'] == fold_idx:
                fold = f
                break
        if fold is None:
            raise ValueError(f"fold {fold_idx} not found in {loader.split_file}")
        Xf, yf = fold[member]
        n_slides = len(fold['train_sample_ids' if split == 'train' else 'val_sample_ids'])
        print(f"  Fold {fold_idx} '{split}' split: {len(Xf):,} spectra from {n_slides} slides")
        X_real = np.asarray(Xf).astype(np.float32)
        y_raw  = np.asarray(yf).astype(int).flatten()

    # Binary: Mutant=0 (LGm1-3), Wildtype=1 (LGm4-6)
    y_real = np.where(y_raw < 3, 0, 1) if np.max(y_raw) > 1 else y_raw

    # Wavenumbers — updated in-place by apply_spectral_crop
    wn = loader.wavenumbers
    if wn is not None and len(wn) == X_real.shape[1]:
        wavenumbers = wn.astype(np.float64)
    else:
        # Fallback: read calibration file
        calib = pd.read_csv(project_root / config['paths']['calib_file'])
        all_wn = calib['freq'].values
        wavenumbers = all_wn[(all_wn >= 700) & (all_wn <= 1800)].astype(np.float64)

    # Synthetic spectra for the requested fold
    synth_dir = project_root / config['paths']['synthetic_dir']
    print(f"  Loading synthetic fold {fold_idx} from {synth_dir.name}/ ...")
    X_synth     = np.load(synth_dir / f'fold_{fold_idx}_X_synth.npy').astype(np.float32)
    y_synth_raw = np.load(synth_dir / f'fold_{fold_idx}_y_synth.npy').astype(int).flatten()
    y_synth     = np.where(y_synth_raw < 3, 0, 1) if np.max(y_synth_raw) > 1 else y_synth_raw

    # Guarantee canonical shapes: X must be 2-D (N, features), wn must be 1-D
    X_real      = np.asarray(X_real).squeeze()
    X_synth     = np.asarray(X_synth).squeeze()
    wavenumbers = np.asarray(wavenumbers).ravel()

    return X_real, y_real, X_synth, y_synth, wavenumbers


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'configs').exists() and (parent / 'src').exists():
            return parent
    return Path.cwd()


def _subsample(
    X: np.ndarray, y: np.ndarray,
    max_per_class: int, rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, min(len(ci), max_per_class), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _save_fig(fig: plt.Figure, base: Path) -> List[Path]:
    saved = []
    for ext in ('.pdf', '.png'):
        p = base.with_suffix(ext)
        fig.savefig(p, bbox_inches='tight', dpi=FIG_DPI)
        saved.append(p)
        print(f"  Saved: {p.name}")
    plt.close(fig)
    return saved


# =============================================================================
# A. Mean ± std spectra
# =============================================================================

def plot_mean_std_spectra(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    wavenumbers: np.ndarray, output_dir: Path,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> List[Path]:
    """Mean ± 1σ per class, real vs synthetic overlaid."""
    classes = sorted(np.unique(y_real).tolist())
    fig, axes = plt.subplots(1, len(classes),
                              figsize=(7 * len(classes), 4), sharey=False)
    if len(classes) == 1:
        axes = [axes]

    wn = np.asarray(wavenumbers).ravel()
    for ax, c in zip(axes, classes):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        mr = np.asarray(Xr.mean(0)).ravel()
        sr = np.asarray(Xr.std(0)).ravel()
        ms = np.asarray(Xs.mean(0)).ravel()
        ss = np.asarray(Xs.std(0)).ravel()

        ax.plot(wn, mr, color=REAL_COLOR,  lw=1.5,
                label=f"Real  (n={len(Xr):,})")
        ax.fill_between(wn, mr - sr, mr + sr,
                         color=REAL_COLOR, alpha=ALPHA_FILL)
        ax.plot(wn, ms, color=SYNTH_COLOR, lw=1.5,
                label=f"Synth (n={len(Xs):,})")
        ax.fill_between(wn, ms - ss, ms + ss,
                         color=SYNTH_COLOR, alpha=ALPHA_FILL)

        for shift in RAMAN_PEAKS.values():
            ax.axvline(shift, color='gray', lw=0.5, ls='--', alpha=0.45)

        ax.set_title(f"{class_names.get(c, f'Class {c}')} — Mean ± 1σ",
                     fontsize=11)
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalised intensity")
        ax.legend(fontsize=9)

    fig.suptitle("Mean ± 1σ Spectra: Real vs Synthetic", fontsize=13, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, output_dir / "A_mean_std_spectra")


# =============================================================================
# B. Representative overlay plots
# =============================================================================

def plot_representative_overlay(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    wavenumbers: np.ndarray, output_dir: Path,
    n_samples: int = 5, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> List[Path]:
    """n_samples real + n_samples synthetic spectra overlaid per class."""
    rng = np.random.RandomState(seed)
    wn  = np.asarray(wavenumbers).ravel()
    saved: List[Path] = []

    for c in sorted(np.unique(y_real).tolist()):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        ir  = rng.choice(len(Xr), min(n_samples, len(Xr)), replace=False)
        is_ = rng.choice(len(Xs), min(n_samples, len(Xs)), replace=False)

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, xi in enumerate(Xr[ir]):
            ax.plot(wn, np.asarray(xi).ravel(), color=REAL_COLOR,
                    alpha=ALPHA_LINE, lw=0.9,
                    label="Real" if i == 0 else "_")
        for i, xi in enumerate(Xs[is_]):
            ax.plot(wn, np.asarray(xi).ravel(), color=SYNTH_COLOR,
                    alpha=ALPHA_LINE, lw=0.9,
                    label="Synthetic" if i == 0 else "_")
        for shift in RAMAN_PEAKS.values():
            ax.axvline(shift, color='gray', lw=0.5, ls='--', alpha=0.35)

        ax.set_title(
            f"{class_names.get(c, f'Class {c}')} — {n_samples} representative spectra",
            fontsize=11,
        )
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalised intensity")
        ax.legend(fontsize=9)
        fig.tight_layout()
        saved.extend(_save_fig(fig, output_dir / f"B_overlay_class{c}"))

    return saved


# =============================================================================
# C. Dimensionality reduction  (PCA / t-SNE / UMAP)
# =============================================================================

def plot_dimensionality_reduction(
    sources: List[Tuple[str, np.ndarray, np.ndarray, str]],
    output_dir: Path,
    max_per_class: int = 2000, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> List[Path]:
    """
    Scatter plots coloured by source and by class, for PCA and t-SNE (and UMAP).

    Parameters
    ----------
    sources : list of (label, X, y, colour), one entry per group to plot.
        All groups share a single embedding: the reducer is fitted on the pooled,
        z-scored data so that positions are directly comparable across groups.
        Two groups reproduce the original real-vs-synthetic figure; three allow
        real-train / real-test / synthetic, which separates the distribution the
        generator was fitted to from the one it is evaluated against.

    Each group is subsampled to `max_per_class` spectra per class before pooling.
    """
    rng   = np.random.RandomState(seed)
    saved: List[Path] = []

    Xs, ys, src_idx, labels, colours = [], [], [], [], []
    for i, (label, X, y, colour) in enumerate(sources):
        Xi, yi = _subsample(X, y, max_per_class, rng)
        Xs.append(Xi)
        ys.append(yi)
        src_idx.append(np.full(len(Xi), i))
        labels.append(label)
        colours.append(colour)
        print(f"    {label}: {len(Xi):,} spectra after subsampling")

    X_all  = np.vstack(Xs)
    y_all  = np.concatenate(ys)
    source = np.concatenate(src_idx)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_all)

    # PCA
    print("    PCA ...")
    pca = PCA(n_components=2, random_state=seed)
    Z   = pca.fit_transform(X_sc)
    ev  = pca.explained_variance_ratio_
    saved.extend(_scatter_pair(
        Z, source, y_all, labels, colours, class_names,
        f"PCA  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%",
        output_dir / "C_pca",
    ))

    # t-SNE (reduce to 50 PCs first for speed)
    print("    t-SNE (may take a few minutes) ...")
    Z50  = PCA(n_components=min(50, X_sc.shape[1]),
               random_state=seed).fit_transform(X_sc)
    tsne = TSNE(n_components=2, random_state=seed,
                perplexity=50, max_iter=1000, verbose=0)
    Z    = tsne.fit_transform(Z50)
    saved.extend(_scatter_pair(Z, source, y_all, labels, colours, class_names,
                               "t-SNE", output_dir / "C_tsne"))

    # UMAP
    if HAS_UMAP:
        print("    UMAP ...")
        reducer = umap_lib.UMAP(n_components=2, random_state=seed, verbose=False)
        Z       = reducer.fit_transform(X_sc)
        saved.extend(_scatter_pair(Z, source, y_all, labels, colours, class_names,
                                   "UMAP", output_dir / "C_umap"))
    else:
        print("    UMAP skipped (umap-learn not installed).")

    return saved


def _scatter_pair(
    Z: np.ndarray, source: np.ndarray, y: np.ndarray,
    source_labels: List[str], source_colours: List[str],
    class_names: Dict[int, str], title: str, base: Path,
) -> List[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the largest group first so smaller ones stay visible on top.
    order = sorted(range(len(source_labels)),
                   key=lambda s: -int(np.sum(source == s)))
    for s in order:
        m = source == s
        axes[0].scatter(Z[m, 0], Z[m, 1], c=source_colours[s], alpha=0.35, s=3,
                        label=source_labels[s], rasterized=True)
    axes[0].set_title(f"{title} — by Source")
    axes[0].legend(markerscale=4, fontsize=9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for i, c in enumerate(sorted(np.unique(y).tolist())):
        m = y == c
        axes[1].scatter(
            Z[m, 0], Z[m, 1],
            c=CLASS_COLORS[i % len(CLASS_COLORS)],
            alpha=0.35, s=3,
            label=class_names.get(c, f"Class {c}"),
            rasterized=True,
        )
    axes[1].set_title(f"{title} — by Class")
    axes[1].legend(markerscale=4, fontsize=9)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    return _save_fig(fig, base)


# =============================================================================
# D. Raman peak comparison
# =============================================================================

def plot_raman_peak_comparison(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    wavenumbers: np.ndarray, output_dir: Path,
    peaks: Dict[str, int] = RAMAN_PEAKS,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> Tuple[List[Path], dict]:
    """Grouped bar chart of mean intensity at key Raman peaks."""
    wn    = np.asarray(wavenumbers)
    saved: List[Path] = []
    results: dict = {}

    for c in sorted(np.unique(y_real).tolist()):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        cname  = class_names.get(c, f"Class {c}")
        results[cname] = {}

        labels, vr, vs = [], [], []
        for lbl, shift in peaks.items():
            idx = int(np.argmin(np.abs(wn - shift)))
            mr, ms = float(Xr[:, idx].mean()), float(Xs[:, idx].mean())
            labels.append(lbl)
            vr.append(mr)
            vs.append(ms)
            results[cname][lbl] = {
                "wavenumber_cm-1":  shift,
                "mean_real":        round(mr, 6),
                "mean_synth":       round(ms, 6),
                "ratio_synth_real": round(ms / (mr + 1e-10), 4),
            }

        x, w = np.arange(len(labels)), 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 4))
        ax.bar(x - w / 2, vr, w, color=REAL_COLOR,  label="Real")
        ax.bar(x + w / 2, vs, w, color=SYNTH_COLOR, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel("Mean normalised intensity")
        ax.set_title(f"{cname} — Raman peak intensities: Real vs Synthetic")
        ax.legend()
        fig.tight_layout()
        saved.extend(_save_fig(fig, output_dir / f"D_peaks_class{c}"))

    return saved, results


# =============================================================================
# E. Distributional distances  (MMD + sliced Wasserstein)
# =============================================================================

def _mmd_rbf(
    Xr: np.ndarray, Xs: np.ndarray,
    max_n: int = 1000, seed: int = 42,
) -> float:
    """Unbiased sqrt(MMD²) with RBF kernel (median-heuristic bandwidth)."""
    from sklearn.metrics import pairwise_distances
    rng = np.random.RandomState(seed)
    r   = Xr[rng.choice(len(Xr), min(max_n, len(Xr)), replace=False)]
    s   = Xs[rng.choice(len(Xs), min(max_n, len(Xs)), replace=False)]

    d     = pairwise_distances(r[:200], metric='euclidean').ravel()
    gamma = 1.0 / (2.0 * (np.median(d[d > 0]) ** 2) + 1e-10)

    K_rr  = rbf_kernel(r, r, gamma=gamma)
    K_ss  = rbf_kernel(s, s, gamma=gamma)
    K_rs  = rbf_kernel(r, s, gamma=gamma)
    mmd2  = K_rr.mean() + K_ss.mean() - 2.0 * K_rs.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def _sliced_wasserstein(
    Xr: np.ndarray, Xs: np.ndarray,
    n_proj: int = 100, seed: int = 42,
) -> float:
    """Sliced Wasserstein distance — no external library required."""
    rng  = np.random.RandomState(seed)
    proj = rng.randn(n_proj, Xr.shape[1])
    proj /= np.linalg.norm(proj, axis=1, keepdims=True)
    return float(np.mean([wasserstein_distance(Xr @ p, Xs @ p) for p in proj]))


def compute_distributional_distances(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    max_samples: int = 1000, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> dict:
    """Per-class MMD and sliced Wasserstein distance."""
    results: dict = {}
    for c in sorted(np.unique(y_real).tolist()):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        cname  = class_names.get(c, f"Class {c}")
        print(f"    {cname}  (n_real={len(Xr):,}, n_synth={len(Xs):,}) ...")
        mmd = _mmd_rbf(Xr, Xs, max_n=max_samples, seed=seed)
        swd = _sliced_wasserstein(Xr, Xs, seed=seed)
        results[cname] = {
            "mmd":                round(mmd, 6),
            "sliced_wasserstein": round(swd, 6),
        }
        print(f"      MMD={mmd:.6f}  Sliced-WD={swd:.6f}")
    return results


# =============================================================================
# F. Classifier two-sample test  (C2ST)
# =============================================================================

def compute_c2st(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    max_per_class: int = 5000, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> dict:
    """
    Within-class logistic-regression real(0) vs synthetic(1) classifier.
    Chance = 0.50; lower accuracy => more realistic synthetic spectra.
    """
    rng     = np.random.RandomState(seed)
    results: dict = {}

    for c in sorted(np.unique(y_real).tolist()):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        cname  = class_names.get(c, f"Class {c}")
        n      = min(len(Xr), len(Xs), max_per_class)
        ir     = rng.choice(len(Xr), n, replace=False)
        is_    = rng.choice(len(Xs), n, replace=False)
        X_c    = np.vstack([Xr[ir], Xs[is_]])
        y_c    = np.array([0] * n + [1] * n)

        X_sc  = StandardScaler().fit_transform(X_c)
        clf   = LogisticRegression(max_iter=500, C=1.0,
                                    random_state=seed, solver='lbfgs')
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        proba = cross_val_predict(clf, X_sc, y_c, cv=skf, method='predict_proba')
        acc   = float(accuracy_score(y_c, proba.argmax(1)))
        auroc = float(roc_auc_score(y_c, proba[:, 1]))

        results[cname] = {
            "c2st_accuracy": round(acc,   4),
            "c2st_auroc":    round(auroc, 4),
            "n_per_source":  n,
        }
        print(f"    {cname}: Acc={acc:.4f}  AUROC={auroc:.4f}  (chance=0.50)")

    return results


# =============================================================================
# G. Nearest-neighbour memorisation check
# =============================================================================

def compute_nn_memorisation(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    output_dir: Path,
    max_real: int = 5000, max_synth: int = 2000, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> Tuple[List[Path], dict]:
    """
    Histogram of synth->real NN distances vs real->real NN distances.
    ratio < 0.70 flags potential memorisation.
    """
    rng     = np.random.RandomState(seed)
    results: dict = {}
    classes = sorted(np.unique(y_real).tolist())
    fig, axes = plt.subplots(1, len(classes),
                              figsize=(6 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, c in zip(axes, classes):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        cname  = class_names.get(c, f"Class {c}")
        ir  = rng.choice(len(Xr), min(max_real,  len(Xr)), replace=False)
        is_ = rng.choice(len(Xs), min(max_synth, len(Xs)), replace=False)
        Xr_s, Xs_s = Xr[ir], Xs[is_]

        # synth -> real
        d_sr = (NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
                .fit(Xr_s).kneighbors(Xs_s)[0].ravel())

        # real -> real (2nd NN as leave-one-out proxy)
        d_rr = (NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
                .fit(Xr_s).kneighbors(Xr_s)[0][:, 1].ravel())

        ax.hist(d_rr, bins=60, density=True, alpha=0.55,
                color=REAL_COLOR,  label="Real->Real (baseline)")
        ax.hist(d_sr, bins=60, density=True, alpha=0.55,
                color=SYNTH_COLOR, label="Synth->Real")
        ax.set_title(f"{cname} — NN distances")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        ratio = float(d_sr.mean()) / (float(d_rr.mean()) + 1e-10)
        flag  = "POTENTIAL MEMORISATION" if ratio < 0.70 else "OK"
        results[cname] = {
            "mean_synth_to_real_nn": round(float(d_sr.mean()), 4),
            "mean_real_to_real_nn":  round(float(d_rr.mean()), 4),
            "ratio_sr_rr":           round(ratio, 4),
            "memorisation_flag":     flag,
        }
        print(f"    {cname}: synth->real={d_sr.mean():.4f}  "
              f"real->real={d_rr.mean():.4f}  ratio={ratio:.4f}  [{flag}]")

    fig.suptitle("Nearest-Neighbour Distances: Memorisation Check",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, output_dir / "G_nn_memorisation"), results


# =============================================================================
# H. Intra-class diversity
# =============================================================================

def compute_diversity(
    X_real: np.ndarray, y_real: np.ndarray,
    X_synth: np.ndarray, y_synth: np.ndarray,
    output_dir: Path,
    max_samples: int = 2000, n_pairs: int = 10_000, seed: int = 42,
    class_names: Dict[int, str] = CLASS_NAMES,
) -> Tuple[List[Path], dict]:
    """Pairwise Euclidean distance distributions within each class (random pairs)."""
    rng     = np.random.RandomState(seed)
    results: dict = {}
    classes = sorted(np.unique(y_real).tolist())
    fig, axes = plt.subplots(1, len(classes),
                              figsize=(6 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    def _rand_pair_dists(X: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
        n_s = len(X)
        i1  = rng.randint(0, n_s, n)
        i2  = rng.randint(0, n_s, n)
        i2[i1 == i2] = (i2[i1 == i2] + 1) % n_s
        return np.linalg.norm(X[i1] - X[i2], axis=1)

    for ax, c in zip(axes, classes):
        Xr, Xs = X_real[y_real == c], X_synth[y_synth == c]
        cname  = class_names.get(c, f"Class {c}")
        ir  = rng.choice(len(Xr), min(max_samples, len(Xr)), replace=False)
        is_ = rng.choice(len(Xs), min(max_samples, len(Xs)), replace=False)
        dr  = _rand_pair_dists(Xr[ir], n_pairs, rng)
        ds  = _rand_pair_dists(Xs[is_], n_pairs, rng)

        ax.hist(dr, bins=60, density=True, alpha=0.55,
                color=REAL_COLOR,  label="Real")
        ax.hist(ds, bins=60, density=True, alpha=0.55,
                color=SYNTH_COLOR, label="Synthetic")
        ax.set_title(f"{cname} — Pairwise distances (diversity)")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

        results[cname] = {
            "real_mean_pairwise":         round(float(dr.mean()), 4),
            "real_std_pairwise":          round(float(dr.std()),  4),
            "synth_mean_pairwise":        round(float(ds.mean()), 4),
            "synth_std_pairwise":         round(float(ds.std()),  4),
            "diversity_ratio_synth_real": round(
                float(ds.mean()) / (float(dr.mean()) + 1e-10), 4
            ),
        }
        print(f"    {cname}: real={dr.mean():.4f}  synth={ds.mean():.4f}  "
              f"ratio={results[cname]['diversity_ratio_synth_real']:.4f}")

    fig.suptitle("Intra-class Diversity: Pairwise Distances",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, output_dir / "H_diversity"), results


# =============================================================================
# Main orchestrator
# =============================================================================

def run_all(
    config: dict,
    figures_dir: str = "figures/synth_eval",
    metrics_path: str = "results/synth_eval_metrics.json",
    fold_idx: int = 0,
    seed: int = 42,
    real_split: str = 'train',
) -> dict:
    """
    Run the complete synthetic realism evaluation pipeline for one experiment.

    `real_split` selects which real spectra the per-class analyses (A, B, D-H)
    compare the synthetic pool against:
      'train' — the slides this fold's CVAE was fitted to (default)
      'test'  — the slides held out in this fold
      'all'   — every slide, pooled (the historical behaviour; mixes seen and
                unseen data and is not recommended for a per-fold figure)

    The embedding plots (C) always show real-train, real-test and synthetic as
    three separate groups on one shared embedding, regardless of `real_split`.
    """
    exp_name = config.get('name', 'unknown')
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SYNTHETIC REALISM EVALUATION ({exp_name})")
    print(f"{'='*70}")
    print(f"Fold: {fold_idx}  |  Real split: {real_split}")
    print(f"Figures -> {out_dir}  |  Metrics -> {metrics_path}\n")

    print("Loading data ...")
    X_real, y_real, X_synth, y_synth, wavenumbers = load_data(
        config, fold_idx=fold_idx, seed=seed, split=real_split
    )
    print(f"  Real ({real_split}): {X_real.shape}  classes={np.unique(y_real)}")
    print(f"  Synth: {X_synth.shape}  classes={np.unique(y_synth)}\n")

    all_metrics: dict = {}

    print("[A] Mean +/- std spectra ...")
    plot_mean_std_spectra(X_real, y_real, X_synth, y_synth, wavenumbers, out_dir)

    print("\n[B] Representative overlay ...")
    plot_representative_overlay(X_real, y_real, X_synth, y_synth, wavenumbers, out_dir)

    print("\n[C] Dimensionality reduction (PCA / t-SNE / UMAP) ...")
    # Three groups on one shared embedding: the data the generator was fitted to,
    # the data it is evaluated against, and the data it produced.
    # Reuse the already-loaded split rather than re-reading the HDF5 for it.
    if real_split == 'train':
        X_tr, y_tr = X_real, y_real
    else:
        X_tr, y_tr, _, _, _ = load_data(config, fold_idx=fold_idx, seed=seed, split='train')
    if real_split == 'test':
        X_te, y_te = X_real, y_real
    else:
        X_te, y_te, _, _, _ = load_data(config, fold_idx=fold_idx, seed=seed, split='test')
    plot_dimensionality_reduction(
        [
            ("Real (train slides)", X_tr,    y_tr,     REAL_TRAIN_COLOR),
            ("Real (held-out slides)", X_te, y_te,     REAL_TEST_COLOR),
            ("Synthetic",           X_synth, y_synth,  SYNTH_COLOR),
        ],
        out_dir, max_per_class=2000, seed=seed,
    )

    print("\n[D] Raman peak comparison ...")
    _, all_metrics["raman_peaks"] = plot_raman_peak_comparison(
        X_real, y_real, X_synth, y_synth, wavenumbers, out_dir
    )

    print("\n[E] Distributional distances (MMD + sliced Wasserstein) ...")
    all_metrics["distributional_distances"] = compute_distributional_distances(
        X_real, y_real, X_synth, y_synth, max_samples=1000, seed=seed
    )

    print("\n[F] C2ST (classifier two-sample test) ...")
    all_metrics["c2st"] = compute_c2st(
        X_real, y_real, X_synth, y_synth, seed=seed
    )

    print("\n[G] Nearest-neighbour memorisation check ...")
    _, all_metrics["nn_memorisation"] = compute_nn_memorisation(
        X_real, y_real, X_synth, y_synth, out_dir, seed=seed
    )

    print("\n[H] Intra-class diversity ...")
    _, all_metrics["diversity"] = compute_diversity(
        X_real, y_real, X_synth, y_synth, out_dir, seed=seed
    )

    # Save JSON
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics JSON saved: {metrics_path}")

    # Save flat CSV for sections with per-class scalar metrics
    csv_rows = []
    for section in ("distributional_distances", "c2st", "nn_memorisation", "diversity"):
        for cls_name, cls_data in all_metrics.get(section, {}).items():
            row = {"section": section, "class": cls_name}
            row.update(cls_data)
            csv_rows.append(row)
    if csv_rows:
        csv_path = Path(metrics_path).with_suffix('.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  Metrics CSV saved:  {csv_path}")

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    return all_metrics


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == '__main__':
    import sys
    import os

    project_root = _find_project_root()
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    from src.utils.config_manager import ConfigManager

    parser = argparse.ArgumentParser(description="Synthetic Realism Evaluation")
    parser.add_argument("--config", default="exp_201",
                        help="Experiment config name (default: exp_201, the "
                             "experiment reported in the thesis results table)")
    parser.add_argument("--fold",   type=int, default=0,
                        help="Fold index for synthetic data (default: 0)")
    parser.add_argument("--out",    default=None,
                        help="Output directory for figures "
                             "(default: <experiment_root>/figures/synth_eval)")
    parser.add_argument("--real-split", default="train",
                        choices=["train", "test", "all"], dest="real_split",
                        help="Which real spectra the per-class analyses compare "
                             "against (default: train, i.e. the slides this fold's "
                             "CVAE was fitted to)")
    args = parser.parse_args()

    # load_config returns a ConfigManager, not a mapping; run_all wants a dict
    cfg = ConfigManager.load_config(args.config).to_dict()
    exp_root = cfg['paths']['experiment_root']
    out_dir  = args.out or f'{exp_root}/figures/synth_eval'
    run_all(cfg, figures_dir=out_dir,
            metrics_path=f'{exp_root}/synth_eval_metrics.json',
            fold_idx=args.fold, real_split=args.real_split)
