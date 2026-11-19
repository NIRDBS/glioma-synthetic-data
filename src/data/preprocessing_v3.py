"""Raman preprocessing with a fixed maximum-intensity threshold: raw cube -> modelling HDF5.

Identical to ``preprocessing_v2`` except for step 4:
    1. Load the raw cube.
    2. 3x3 spatial median filter.
    3. ``preprocessing_v2.cube_normalize`` (used only to compute the mask).
    4. Keep the spectra whose maximum normalised intensity is >= the threshold
       (``compute_fixed_threshold_mask``).
    5. IARPLS baseline correction of the kept spectra on the raw, median-filtered scale, then
       per-spectrum min-max normalisation.
    6. Optionally persist each slide's mask, and write the HDF5 (``RamanPreprocessorV3.run``).

Threshold 0.60 gives the paper's main dataset (``glioma_mil.h5``, 288,678 spectra); threshold 0.40
gives ``glioma-preproc-3.h5`` (exp_4.608). The default is 0.40.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.data.preprocessing_v2 import _extract_ids, cube_normalize
from src.preprocessing.baseline import IARPLS_LAMBDA, correct_iarpls

__all__ = [
    "RamanPreprocessorV3", "SlideResultV3", "compute_fixed_threshold_mask",
]

_MEDIAN_KERNEL = 3  # matches v1/v2
DEFAULT_THRESHOLD = 0.4


class SlideResultV3:
    """Per-slide export payload. Plain container so it survives a process-pool hop."""

    __slots__ = ("sample_name", "patient_id", "class_id", "shape", "spectra", "coords")

    def __init__(self, sample_name, patient_id, class_id, shape, spectra, coords):
        self.sample_name = sample_name
        self.patient_id = patient_id
        self.class_id = class_id
        self.shape = shape
        self.spectra = spectra
        self.coords = coords


def compute_fixed_threshold_mask(cube_normalized: np.ndarray, threshold: float = DEFAULT_THRESHOLD
                                 ) -> tuple[np.ndarray, dict]:
    """Fixed max-intensity threshold -- the mask is simply ``max_intensity > threshold``,
    no per-slide adaptation at all (unlike v2's ``compute_strong_signal_mask``).

    ``cube_normalized``: (rows, cols, wavelengths), already through
    :func:`src.data.preprocessing_v2.cube_normalize`.

    Returns ``(mask, info)``, ``info`` recording the threshold and the
    resulting keep-fraction, for inspectability (same shape as v2's info dict,
    minus the branch/percentile fields that don't apply here).
    """
    max_intensity = cube_normalized.max(axis=-1)
    mask = max_intensity > threshold
    info = {
        "threshold": threshold,
        "n_kept": int(mask.sum()),
        "n_total": int(mask.size),
        "frac_kept": float(mask.sum() / mask.size),
    }
    return mask, info


def process_slide_v3(
    raw_path: str,
    wavenumbers: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    baseline_lambda: float = IARPLS_LAMBDA,
    median_kernel: int = _MEDIAN_KERNEL,
    processed_dir: str | None = None,
) -> SlideResultV3 | None:
    """Filter, mask (fixed threshold), baseline-correct and normalise one
    slide. Pure function (aside from the optional mask persistence below) --
    safe to run in a worker process. Mirrors
    ``src.data.preprocessing_v2.process_slide_v2`` exactly except for the
    masking step.

    Returns ``None`` if the mask keeps zero pixels (logged by the caller).
    """
    raw_path = Path(raw_path)
    sample_name = raw_path.stem
    class_id, patient_id = _extract_ids(raw_path)

    cube = np.load(raw_path).astype(np.float32)
    filtered = np.empty_like(cube)
    for i in range(cube.shape[2]):
        filtered[:, :, i] = median_filter(cube[:, :, i], size=median_kernel, mode="reflect")

    normalized = cube_normalize(filtered)
    mask, mask_info = compute_fixed_threshold_mask(normalized, threshold=threshold)

    if processed_dir is not None:
        slide_dir = Path(processed_dir) / sample_name
        slide_dir.mkdir(parents=True, exist_ok=True)
        np.save(slide_dir / "strong_signal_mask_v3.npy", mask)

    if mask.sum() == 0:
        return None

    rows, cols = np.where(mask)  # boolean-index order == np.where order (C-order)
    raw_spectra = filtered[mask]  # IARPLS runs on the RAW (median-filtered) scale

    corrected = np.empty_like(raw_spectra, dtype=np.float32)
    for i, spectrum in enumerate(raw_spectra):
        res = correct_iarpls(spectrum.astype(np.float64), wavenumbers, lam=baseline_lambda)
        c = res["corrected"]
        lo, hi = float(c.min()), float(c.max())
        corrected[i] = np.zeros_like(c, dtype=np.float32) if hi - lo == 0 else ((c - lo) / (hi - lo)).astype(np.float32)

    return SlideResultV3(
        sample_name=sample_name,
        patient_id=patient_id,
        class_id=class_id,
        shape=cube.shape[:2],
        spectra=corrected,
        coords=np.column_stack((cols, rows)).astype(np.int32),  # (col, row), matches v1/v2
    )


class RamanPreprocessorV3:
    """Builds the v3 modelling HDF5 directly from raw cubes, fixed-threshold
    mask. Mirrors ``RamanPreprocessorV2`` exactly except for the threshold
    parameter and the mask function it passes to ``process_slide_v3``.
    """

    def __init__(
        self,
        raw_data_dir,
        output_file,
        calibration_path,
        threshold: float = DEFAULT_THRESHOLD,
        baseline_lambda: float = IARPLS_LAMBDA,
        median_kernel: int = _MEDIAN_KERNEL,
        processed_dir=None,
        n_workers: int | None = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.calibration_path = Path(calibration_path)
        self.threshold = threshold
        self.baseline_lambda = baseline_lambda
        self.median_kernel = median_kernel
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.n_workers = n_workers

        self.wavenumbers = pd.read_csv(self.calibration_path).iloc[:, 1].values.astype(np.float64)

    def _discover_slides(self) -> list[Path]:
        return sorted(self.raw_data_dir.rglob("*.npy"))

    def run(self) -> Path:
        slide_paths = self._discover_slides()
        print(f"Found {len(slide_paths)} raw slides under {self.raw_data_dir} "
             f"(fixed threshold={self.threshold})")

        n_features = len(self.wavenumbers)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        results: list[SlideResultV3] = []
        skipped: list[str] = []
        processed_dir_str = str(self.processed_dir) if self.processed_dir is not None else None

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(
                    process_slide_v3, str(p), self.wavenumbers, self.threshold,
                    self.baseline_lambda, self.median_kernel, processed_dir_str,
                ): p
                for p in slide_paths
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Exporting slides (v3)"):
                p = futures[fut]
                r = fut.result()
                if r is None:
                    skipped.append(p.stem)
                else:
                    results.append(r)

        if skipped:
            print(f"  [skip] {len(skipped)} slide(s) with zero strong-signal pixels: {skipped}")

        results.sort(key=lambda r: r.sample_name)  # deterministic write order

        n_total = sum(r.spectra.shape[0] for r in results)
        print(f"Writing {n_total:,} instances from {len(results)} slides to {self.output_file}")

        with h5py.File(self.output_file, "w") as h5f:
            h5f.create_dataset("wavenumbers", data=self.wavenumbers.astype("float32"))

            dset_spectra = h5f.create_dataset(
                "spectra", shape=(n_total, n_features), dtype="float32", compression="gzip"
            )
            dset_coords = h5f.create_dataset("coords", shape=(n_total, 2), dtype="int32")
            dset_y_bag = h5f.create_dataset("y_bag", shape=(n_total,), dtype="int32")
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
                dset_patient[sl] = np.full(n, r.patient_id)
                dset_samples[sl] = np.full(n, r.sample_name)
                cursor += n

            h5f.attrs["baseline_method"] = "iarpls"
            h5f.attrs["baseline_lambda"] = float(self.baseline_lambda)
            h5f.attrs["baseline_lambda_status"] = "provisional"  # see the paper, Appendix: Baseline Correction
            h5f.attrs["median_kernel"] = self.median_kernel
            h5f.attrs["fixed_threshold"] = float(self.threshold)
            h5f.attrs["n_slides"] = len(results)
            h5f.attrs["n_instances"] = n_total
            h5f.attrs["schema_version"] = "mil-v3-fixed-threshold"

        print(f"Done. {n_total:,} instances, {len(results)} slides. "
             f"No tissue_cluster/viable_legacy columns in this schema (v3 has no clustering stage).")
        return self.output_file
