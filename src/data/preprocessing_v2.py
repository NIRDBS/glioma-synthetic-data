"""Raman preprocessing with a per-slide adaptive mask: raw cube -> modelling HDF5 (exp_2.607-2.609).

    1. Load the raw cube.
    2. 3x3 spatial median filter.
    3. Normalise the filtered cube (shift to non-negative, scale by the 99.9th percentile, clip to
       [0, 1]). The result only feeds the strong-signal mask; it never reaches baseline correction
       (see ``cube_normalize``).
    4. Strong-signal mask: an adaptive, per-slide percentile rule on the per-pixel maximum
       intensity (``compute_strong_signal_mask``).
    5. IARPLS baseline correction of the kept spectra on the raw, median-filtered scale, then
       per-spectrum min-max normalisation.
    6. Optionally persist each slide's mask, and write the HDF5 (``RamanPreprocessorV2.run``).

No manual step and no clustering. The output (conventionally ``glioma-preproc-2.h5``) has the
columns of ``preprocessing_v3`` and none of the v1 clustering columns.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import re

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from tqdm import tqdm

from src.preprocessing.baseline import IARPLS_LAMBDA, correct_iarpls

_FILENAME_RE = re.compile(r"LGm-(?P<class_id>\d)[/\\]HF-(?P<patient_id>\d+)_")


def _extract_ids(path) -> tuple[int, str]:
    """(0-indexed class id, 'HF-<N>' patient id string) from a raw file's path."""
    match = _FILENAME_RE.search(str(path))
    if not match:
        raise ValueError(f"path does not match LGm-<k>/HF-<n>_...: {path}")
    return int(match.group("class_id")) - 1, f"HF-{match.group('patient_id')}"


__all__ = [
    "RamanPreprocessorV2", "SlideResultV2", "cube_normalize",
    "compute_strong_signal_mask",
]

_MEDIAN_KERNEL = 3  # matches v1 and the paper's methods text


class SlideResultV2:
    """Per-slide export payload. Plain container so it survives a process-pool hop."""

    __slots__ = ("sample_name", "patient_id", "class_id", "shape", "spectra", "coords")

    def __init__(self, sample_name, patient_id, class_id, shape, spectra, coords):
        self.sample_name = sample_name
        self.patient_id = patient_id
        self.class_id = class_id
        self.shape = shape
        self.spectra = spectra
        self.coords = coords


def cube_normalize(cube: np.ndarray) -> np.ndarray:
    """Shift to non-negative, scale by the 99.9th percentile, clip to [0, 1].

    Identical formula to v1's NMF-input normalisation (a single global
    scalar min/percentile over every element, so it does not matter whether
    ``cube`` is 3-D (rows, cols, wavelengths) or already flattened). Used here
    ONLY to compute the strong-signal mask threshold -- never fed to baseline
    correction, which stays on the raw (median-filtered) intensity scale IARPLS's
    validated lambda=1e4 was tuned against.
    Feeding a [0, 1]-clipped spectrum into IARPLS instead would silently
    invalidate that tuning, since the penalty strength is scale-dependent.
    """
    shifted = cube - cube.min() + 1e-8
    p999 = np.percentile(shifted, 99.9)
    return np.clip(shifted / p999, 0.0, 1.0)


def compute_strong_signal_mask(cube_normalized: np.ndarray) -> tuple[np.ndarray, dict]:
    """Adaptive, per-slide strong-signal mask -- replaces NMF+GMM+manual curation.

    ``cube_normalized``: (rows, cols, wavelengths), already through
    :func:`cube_normalize`.

    Confirmed directly against 15 real slides before settling on this (an
    earlier, simpler version checked ``p90 >= 1.0`` as the saturation trigger,
    which turned out to fire on every single real slide tested -- 62-90% of
    pixels sit exactly at the clip ceiling on this data, not a rare <=10%
    sliver, so that check was measuring the wrong thing). Three rules, applied
    as two MUTUALLY EXCLUSIVE branches selected by ``saturation_rate`` (the
    fraction of pixels exactly AT the [0,1] ceiling after :func:`cube_normalize`):

      Rule 1 (rare): ``saturation_rate > 0.9`` -- keep pixels with
      ``max_intensity > 0.95`` (strict), even if that keeps 100% of the slide.

      Rule 2 (main, otherwise): keep exactly the top 90% of pixels BY RANK --
      threshold = the slide's own 10th percentile of max intensity,
      ``max_intensity >= threshold`` (inclusive) -- an adaptive per-slide
      threshold, replacing the old fixed ``WEAK_SIGNAL_THRESH`` constant.

      Rule 3 (applied only within Rule 2's selection, never Rule 1's): an
      additional absolute floor, ``max_intensity > 0.8`` (strict) -- most
      slides' Rule 2 selection already clears this with no effect, but on a
      slide whose top-90%-by-rank group still has a long weak tail (confirmed
      on real data: one slide's own 10th-percentile value was 0.544, well
      below 0.8), this trims the genuinely weak remainder Rule 2 alone would
      have kept.

    Returns ``(mask, info)`` where ``info`` records which branch was taken and
    every threshold/percentile value used, for inspectability.
    """
    max_intensity = cube_normalized.max(axis=-1)
    saturation_rate = float((max_intensity == 1.0).mean())

    if saturation_rate > 0.9:
        threshold = 0.95
        mask = max_intensity > threshold
        info = {
            "saturation_rate": saturation_rate, "branch": "rule1 (saturated)",
            "threshold": threshold, "p10_rank_threshold": None,
        }
    else:
        p10 = float(np.percentile(max_intensity, 10))
        mask_rank = max_intensity >= p10
        mask_floor = max_intensity > 0.8
        mask = mask_rank & mask_floor
        info = {
            "saturation_rate": saturation_rate, "branch": "rule2+3 (rank90 + floor0.8)",
            "threshold": 0.8, "p10_rank_threshold": p10,
            "n_kept_by_rank_alone": int(mask_rank.sum()),
        }

    info.update({
        "n_kept": int(mask.sum()),
        "n_total": int(mask.size),
        "frac_kept": float(mask.sum() / mask.size),
    })
    return mask, info


def process_slide_v2(
    raw_path: str,
    wavenumbers: np.ndarray,
    baseline_lambda: float = IARPLS_LAMBDA,
    median_kernel: int = _MEDIAN_KERNEL,
    processed_dir: str | None = None,
) -> SlideResultV2 | None:
    """Filter, mask, baseline-correct and normalise one slide -- the whole
    pipeline in one pass, no precomputed artifacts required. Pure function
    (aside from the optional mask persistence below) -- safe to run in a
    worker process.

    ``processed_dir``, if given, persists ``strong_signal_mask_v2.npy`` under
    ``processed_dir/<slide_name>/`` for inspectability (a DIFFERENT filename
    from v1's ``strong_signal_mask.npy``, so this never touches or depends on
    the v1 artifacts). Mask computation is cheap now (no NMF/GMM), so this is
    not required for correctness the way v1's precomputed artifacts were --
    purely a debugging/reuse convenience.

    Returns ``None`` if the mask keeps zero pixels (logged by the caller, not
    here, since a worker process's prints are not reliably ordered).
    """
    raw_path = Path(raw_path)
    sample_name = raw_path.stem
    class_id, patient_id = _extract_ids(raw_path)

    cube = np.load(raw_path).astype(np.float32)
    filtered = np.empty_like(cube)
    for i in range(cube.shape[2]):
        filtered[:, :, i] = median_filter(cube[:, :, i], size=median_kernel, mode="reflect")

    normalized = cube_normalize(filtered)
    mask, mask_info = compute_strong_signal_mask(normalized)

    if processed_dir is not None:
        slide_dir = Path(processed_dir) / sample_name
        slide_dir.mkdir(parents=True, exist_ok=True)
        np.save(slide_dir / "strong_signal_mask_v2.npy", mask)

    if mask.sum() == 0:
        return None

    rows, cols = np.where(mask)  # boolean-index order == np.where order (C-order)
    raw_spectra = filtered[mask]  # IARPLS runs on the RAW (median-filtered) scale -- see cube_normalize's docstring

    corrected = np.empty_like(raw_spectra, dtype=np.float32)
    for i, spectrum in enumerate(raw_spectra):
        res = correct_iarpls(spectrum.astype(np.float64), wavenumbers, lam=baseline_lambda)
        c = res["corrected"]
        lo, hi = float(c.min()), float(c.max())
        corrected[i] = np.zeros_like(c, dtype=np.float32) if hi - lo == 0 else ((c - lo) / (hi - lo)).astype(np.float32)

    return SlideResultV2(
        sample_name=sample_name,
        patient_id=patient_id,
        class_id=class_id,
        shape=cube.shape[:2],
        spectra=corrected,
        coords=np.column_stack((cols, rows)).astype(np.int32),  # (col, row) convention, matches v1
    )


class RamanPreprocessorV2:
    """Builds the v2 modelling HDF5 directly from raw cubes -- no precomputed
    clustering artifacts, no manual curation step. See module docstring for the
    full pipeline.
    """

    def __init__(
        self,
        raw_data_dir,
        output_file,
        calibration_path,
        baseline_lambda: float = IARPLS_LAMBDA,
        median_kernel: int = _MEDIAN_KERNEL,
        processed_dir=None,
        n_workers: int | None = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file = Path(output_file)
        self.calibration_path = Path(calibration_path)
        self.baseline_lambda = baseline_lambda
        self.median_kernel = median_kernel
        self.processed_dir = Path(processed_dir) if processed_dir is not None else None
        self.n_workers = n_workers

        self.wavenumbers = pd.read_csv(self.calibration_path).iloc[:, 1].values.astype(np.float64)

    def _discover_slides(self) -> list[Path]:
        return sorted(self.raw_data_dir.rglob("*.npy"))

    def run(self) -> Path:
        slide_paths = self._discover_slides()
        print(f"Found {len(slide_paths)} raw slides under {self.raw_data_dir}")

        n_features = len(self.wavenumbers)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        results: list[SlideResultV2] = []
        skipped: list[str] = []
        processed_dir_str = str(self.processed_dir) if self.processed_dir is not None else None

        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {
                pool.submit(
                    process_slide_v2, str(p), self.wavenumbers,
                    self.baseline_lambda, self.median_kernel, processed_dir_str,
                ): p
                for p in slide_paths
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Exporting slides (v2)"):
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
            h5f.attrs["n_slides"] = len(results)
            h5f.attrs["n_instances"] = n_total
            h5f.attrs["schema_version"] = "mil-v2-no-clustering"

        print(f"Done. {n_total:,} instances, {len(results)} slides. "
              f"No tissue_cluster/viable_legacy columns in this schema (v2 has no clustering stage).")
        return self.output_file
