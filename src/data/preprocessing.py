"""Raman spectrum preprocessing: raw cube -> modelling HDF5.

This is the **single** dataset producer (Stage 1 of the MIL reframe). It replaces
the previous ``RamanPreprocessor``, which could not run at all -- ``gaussian_filter``
was used without being imported, ``_build_file_to_map_correspondence`` was called
but never defined, and the tumour-map key lookup appended ``.npy`` to a key that
never had it, so the mask was always ``None`` and every slide was skipped. The
actual producer for every dataset on disk was notebook 01 section 4 (mirrored in
``_archive/dev_scripts/export_hdf5_section4.py``); that logic is folded in here. The
notebook has since been removed. ``scripts/build_glioma_preproc_3.py --threshold 0.6``
builds the same spectra without the clustering artifacts.

What changed from that notebook logic, deliberately:

* **Mask widened to all strong-signal pixels**, not just the manually-selected
  "tumour" clusters. That selection was never a tumour label -- it is a
  biologically-viable-signal mask (NMF(100) -> GMM(3) per slide, then a human
  picked which of the 3 clusters to call "tumour" in ``file_names.csv``). See
  ``docs/stage1_data_layer.md`` for the full argument. The old, narrower selection
  survives as the boolean ``viable_legacy`` column, so anything that depended on
  the exact old subset still can.
* **Baseline correction is IARPLS**, not airPLS(lambda=100) -- the Stage 0 finding
  (``docs/stage0_findings.md``). Still strictly per-spectrum: no statistic is ever
  computed across spectra, preserving the project's no-global-parameter rule and
  keeping the export fold-independent.
* **Clustering is not re-run.** NMF(100)/GMM(3) per slide already exists on disk
  (``data/processed/<slide>/{hard_labels,strong_signal_mask,tumor_mask}.npy``,
  produced by the v1 NMF/GMM pipeline) and is expensive (NMF alone is
  ~500 iterations per slide). This module reads those artifacts; it does not
  recompute them. Regenerate them first if a slide's directory is missing.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.preprocessing.baseline import IARPLS_LAMBDA, correct_iarpls

__all__ = ["RamanPreprocessor", "SlideResult"]

_FILENAME_RE = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")
_MEDIAN_KERNEL = 3  # matches notebook 01 MEDIAN_KERNEL and the paper's methods text


class SlideResult:
    """Per-slide export payload. Plain container so it survives a process-pool hop."""

    __slots__ = (
        "sample_name", "patient_id", "class_id", "shape",
        "spectra", "coords", "tissue_cluster", "viable_legacy",
    )

    def __init__(self, sample_name, patient_id, class_id, shape,
                 spectra, coords, tissue_cluster, viable_legacy):
        self.sample_name = sample_name
        self.patient_id = patient_id
        self.class_id = class_id
        self.shape = shape
        self.spectra = spectra
        self.coords = coords
        self.tissue_cluster = tissue_cluster
        self.viable_legacy = viable_legacy


def _extract_ids(path: Path) -> tuple[int, str]:
    """(0-indexed class id, 'HF-<N>' patient id string) from a raw file's path."""
    match = _FILENAME_RE.search(str(path))
    if not match:
        raise ValueError(f"path does not match LGm-<k>/HF-<n>_...: {path}")
    class_id = int(match.group("class_id")) - 1
    patient_id = f"HF-{match.group('patient_id')}"
    return class_id, patient_id


def process_slide(
    raw_path: str,
    processed_dir: str,
    wavenumbers: np.ndarray,
    baseline_lambda: float = IARPLS_LAMBDA,
    median_kernel: int = _MEDIAN_KERNEL,
) -> SlideResult | None:
    """Filter, mask, baseline-correct and normalise one slide. Pure function -- safe
    to run in a worker process; touches no shared state and writes nothing to disk.

    Returns ``None`` if the slide has no precomputed artifacts or no strong-signal
    pixels (both skip conditions are logged by the caller, not here, since a worker
    process's prints are not reliably ordered).
    """
    raw_path = Path(raw_path)
    sample_name = raw_path.stem
    slide_dir = Path(processed_dir) / sample_name

    mask_path = slide_dir / "strong_signal_mask.npy"
    if not mask_path.exists():
        return None
    strong_mask = np.load(mask_path).astype(bool)
    if strong_mask.sum() == 0:
        return None

    hard_labels = np.load(slide_dir / "hard_labels.npy")
    tumor_mask = np.load(slide_dir / "tumor_mask.npy").astype(bool)

    class_id, patient_id = _extract_ids(raw_path)

    cube = np.load(raw_path).astype(np.float32)
    if cube.shape[:2] != strong_mask.shape:
        raise ValueError(
            f"{sample_name}: raw cube {cube.shape[:2]} != mask shape {strong_mask.shape}"
        )
    filtered = np.empty_like(cube)
    for i in range(cube.shape[2]):
        filtered[:, :, i] = median_filter(cube[:, :, i], size=median_kernel, mode="reflect")

    rows, cols = np.where(strong_mask)  # boolean-index order == np.where order (C-order)
    raw_spectra = filtered[strong_mask]

    corrected = np.empty_like(raw_spectra, dtype=np.float32)
    for i, spectrum in enumerate(raw_spectra):
        res = correct_iarpls(spectrum.astype(np.float64), wavenumbers, lam=baseline_lambda)
        c = res["corrected"]
        lo, hi = float(c.min()), float(c.max())
        corrected[i] = np.zeros_like(c, dtype=np.float32) if hi - lo == 0 else ((c - lo) / (hi - lo)).astype(np.float32)

    return SlideResult(
        sample_name=sample_name,
        patient_id=patient_id,
        class_id=class_id,
        shape=cube.shape[:2],
        spectra=corrected,
        coords=np.column_stack((cols, rows)).astype(np.int32),  # (col, row) convention
        tissue_cluster=hard_labels[strong_mask].astype(np.int32),
        viable_legacy=tumor_mask[strong_mask],
    )


class RamanPreprocessor:
    """Builds the modelling HDF5 from raw cubes plus the precomputed per-slide
    clustering artifacts in ``processed_dir``.

    Unlike the class this replaces, there is no ``mask_strategy`` branch (strong-
    signal is now the only export mask -- narrower selections are a load-time
    filter on ``viable_legacy``, not a separate export), no ``use_ramanspy`` branch
    (IARPLS is simply the corrector), and no ``config=`` kwarg (the four callers
    that need config-driven paths resolve them before constructing this class).
    """

    def __init__(
        self,
        raw_data_dir,
        processed_dir,
        output_file,
        calibration_path,
        baseline_lambda: float = IARPLS_LAMBDA,
        median_kernel: int = _MEDIAN_KERNEL,
        n_workers: int | None = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.output_file = Path(output_file)
        self.calibration_path = Path(calibration_path)
        self.baseline_lambda = baseline_lambda
        self.median_kernel = median_kernel
        self.n_workers = n_workers

        self.wavenumbers = pd.read_csv(self.calibration_path).iloc[:, 1].values.astype(np.float64)

    def _discover_slides(self) -> list[Path]:
        return sorted(self.raw_data_dir.rglob("*.npy"))

    def run(self) -> Path:
        slide_paths = self._discover_slides()
        print(f"Found {len(slide_paths)} raw slides under {self.raw_data_dir}")

        n_features = len(self.wavenumbers)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        results: list[SlideResult] = []
        skipped: list[str] = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(
                    process_slide, str(p), str(self.processed_dir), self.wavenumbers,
                    self.baseline_lambda, self.median_kernel,
                ): p
                for p in slide_paths
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Exporting slides"):
                p = futures[fut]
                r = fut.result()
                if r is None:
                    skipped.append(p.stem)
                else:
                    results.append(r)

        if skipped:
            print(f"  [skip] {len(skipped)} slide(s) with no strong-signal pixels or no "
                  f"precomputed artifacts: {skipped}")

        # Deterministic write order regardless of pool completion order.
        results.sort(key=lambda r: r.sample_name)

        n_total = sum(r.spectra.shape[0] for r in results)
        print(f"Writing {n_total:,} instances from {len(results)} slides to {self.output_file}")

        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("wavenumbers", data=self.wavenumbers.astype("float32"))

            dset_spectra = h5f.create_dataset(
                "spectra", shape=(n_total, n_features), dtype="float32", compression="gzip"
            )
            dset_coords = h5f.create_dataset("coords", shape=(n_total, 2), dtype="int32")
            dset_y_bag = h5f.create_dataset("y_bag", shape=(n_total,), dtype="int32")
            dset_tissue = h5f.create_dataset("tissue_cluster", shape=(n_total,), dtype="int32")
            dset_legacy = h5f.create_dataset("viable_legacy", shape=(n_total,), dtype="bool")
            dset_patient = h5f.create_dataset(
                "patient_ids", shape=(n_total,), dtype=h5py.string_dtype(encoding="utf-8")
            )
            dset_samples = h5f.create_dataset(
                "sample_ids", shape=(n_total,), dtype=h5py.string_dtype(encoding="utf-8")
            )

            slide_names = [r.sample_name for r in results]
            slide_shapes = np.array([r.shape for r in results], dtype="int32")
            h5f.create_dataset(
                "slide_names", data=np.array(slide_names, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            h5f.create_dataset("slide_shapes", data=slide_shapes)

            cursor = 0
            for r in results:
                n = r.spectra.shape[0]
                sl = slice(cursor, cursor + n)
                dset_spectra[sl] = r.spectra
                dset_coords[sl] = r.coords
                dset_y_bag[sl] = np.full(n, r.class_id, dtype="int32")
                dset_tissue[sl] = r.tissue_cluster
                dset_legacy[sl] = r.viable_legacy
                dset_patient[sl] = np.full(n, r.patient_id)
                dset_samples[sl] = np.full(n, r.sample_name)
                cursor += n

            h5f.attrs["baseline_method"] = "iarpls"
            h5f.attrs["baseline_lambda"] = float(self.baseline_lambda)
            h5f.attrs["baseline_lambda_status"] = "provisional"  # see docs/stage0_findings.md 7.1
            h5f.attrs["median_kernel"] = self.median_kernel
            h5f.attrs["n_slides"] = len(results)
            h5f.attrs["n_instances"] = n_total
            h5f.attrs["schema_version"] = "mil-v1"

        n_legacy = sum(int(r.viable_legacy.sum()) for r in results)
        print(f"Done. {n_total:,} instances, {len(results)} slides, "
              f"{n_legacy:,} flagged viable_legacy (old tumour-cluster subset).")
        return self.output_file
