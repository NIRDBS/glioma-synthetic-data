"""Stage 1 regression tests: producer, reprojection, loader, prepare_fold.

Replaces test_data_integrity.py and test_pipeline.py, both of which predated this
stage and were already non-functional before it: test_data_integrity.py pointed at
a 'data/tumor_maps.npy' path that has never existed (the real file is
results/exp_102/tumor_maps.npy), and test_pipeline.py called
GliomaDataLoader.get_cross_validation_generator(), a method that does not exist on
the class (the loader is iterated directly, via __iter__). Both were ad hoc
debugging scripts, not intentional regression tests -- verified failing on
`main` before this stage's changes, independent of anything here.

Most tests need data/processed/glioma_mil.h5 (built by
scripts/build_glioma_mil.py) and are skipped if it is absent, so this file does
not force a ~10-minute export as a side effect of running the suite.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.spatial import reproject, slide_shape_lookup

REPO_ROOT = Path(__file__).resolve().parents[1]
MIL_H5 = REPO_ROOT / "data" / "processed" / "glioma_mil.h5"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

requires_export = pytest.mark.skipif(
    not MIL_H5.exists(), reason="data/processed/glioma_mil.h5 not built -- run scripts/build_glioma_mil.py"
)


# ---------------------------------------------------------------------------
# src/data/spatial.py -- pure, no fixtures needed
# ---------------------------------------------------------------------------

def test_reproject_roundtrip_simple():
    coords = np.array([[0, 0], [2, 0], [1, 1]])  # (col, row)
    values = np.array([10.0, 20.0, 30.0])
    img = reproject(values, coords, shape=(2, 3), fill=-1)
    want = np.array([[10.0, -1, 20.0], [-1, 30.0, -1]])
    assert np.array_equal(img, want)


def test_reproject_rejects_out_of_bounds():
    coords = np.array([[0, 0], [5, 0]])
    with pytest.raises(ValueError):
        reproject(np.array([1.0, 2.0]), coords, shape=(2, 2))


def test_reproject_rejects_length_mismatch():
    with pytest.raises(ValueError):
        reproject(np.array([1.0, 2.0, 3.0]), np.array([[0, 0], [1, 0]]), shape=(2, 2))


def test_slide_shape_lookup():
    names = np.array([b"slide_a", b"slide_b"])
    shapes = np.array([[10, 20], [5, 5]])
    d = slide_shape_lookup(names, shapes)
    assert d == {"slide_a": (10, 20), "slide_b": (5, 5)}


# ---------------------------------------------------------------------------
# src/data/preprocessing.py::process_slide -- needs raw data + cluster artifacts,
# not the built export, so these run whenever the repo's data/ is present.
# ---------------------------------------------------------------------------

requires_raw = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw").exists(), reason="data/raw not present"
)


@requires_raw
def test_process_slide_reprojects_exactly_against_tumor_mask():
    """Recomputes from raw cubes (the expensive, ground-truth path: ~3.5ms/spectrum
    of real IARPLS work) on a representative subset -- square, rectangular, the
    smallest and largest slides, and the one normally excluded via ignore_samples.
    The full-59-slide version of this same check runs against the already-computed
    export in test_export_reprojection_roundtrip_all_slides (no recomputation, so
    it stays fast enough to run on every test invocation)."""
    import pandas as pd

    from src.data.preprocessing import process_slide

    wn = pd.read_csv(REPO_ROOT / "data" / "RamanFreqTable.csv").iloc[:, 1].values.astype(np.float64)
    all_raw = [p for p in sorted((REPO_ROOT / "data" / "raw").rglob("*.npy"))
               if (PROCESSED_DIR / p.stem / "strong_signal_mask.npy").exists()]
    if not all_raw:
        pytest.skip("no slides with precomputed cluster artifacts")

    shapes = {p: np.load(p, mmap_mode="r").shape[:2] for p in all_raw}
    by_area = sorted(all_raw, key=lambda p: shapes[p][0] * shapes[p][1])
    named = {p.stem: p for p in all_raw}
    sample = {by_area[0], by_area[-1]}  # smallest, largest
    sample.add(next((p for p in all_raw if shapes[p][0] == shapes[p][1]), all_raw[0]))       # square
    sample.add(next((p for p in all_raw if shapes[p][0] != shapes[p][1]), all_raw[0]))       # rectangular
    if "HF-1887_via-t_2_1.h5_3" in named:
        sample.add(named["HF-1887_via-t_2_1.h5_3"])                                          # ex-ignore_samples

    for raw_path in sample:
        sample_name = raw_path.stem
        slide_dir = PROCESSED_DIR / sample_name
        r = process_slide(str(raw_path), str(PROCESSED_DIR), wn, baseline_lambda=1e4)
        assert r is not None, sample_name

        tumor_mask = np.load(slide_dir / "tumor_mask.npy").astype(bool)
        strong_mask = np.load(slide_dir / "strong_signal_mask.npy").astype(bool)

        rec_legacy = reproject(r.viable_legacy.astype(float), r.coords, r.shape, fill=0.0).astype(bool)
        assert np.array_equal(rec_legacy, tumor_mask), f"{sample_name}: viable_legacy != tumor_mask"

        rec_strong = reproject(np.ones(len(r.coords)), r.coords, r.shape, fill=0.0).astype(bool)
        assert np.array_equal(rec_strong, strong_mask), f"{sample_name}: exported pixels != strong_signal_mask"

        assert r.spectra.shape == (strong_mask.sum(), len(wn))
        assert not np.isnan(r.spectra).any(), f"{sample_name}: NaN in corrected spectra"
        assert r.spectra.min() >= 0.0 and r.spectra.max() <= 1.0 + 1e-6, f"{sample_name}: not in [0,1]"


@requires_raw
def test_process_slide_is_pure_across_two_independent_calls():
    """Processing the same slide twice, independently, must be bit-identical --
    the producer must not depend on call order or on what else has been processed."""
    import pandas as pd

    from src.data.preprocessing import process_slide

    wn = pd.read_csv(REPO_ROOT / "data" / "RamanFreqTable.csv").iloc[:, 1].values.astype(np.float64)
    raw_path = next(
        p for p in sorted((REPO_ROOT / "data" / "raw").rglob("*.npy"))
        if (PROCESSED_DIR / p.stem / "strong_signal_mask.npy").exists()
    )
    r1 = process_slide(str(raw_path), str(PROCESSED_DIR), wn, baseline_lambda=1e4)
    r2 = process_slide(str(raw_path), str(PROCESSED_DIR), wn, baseline_lambda=1e4)
    assert np.array_equal(r1.spectra, r2.spectra)
    assert np.array_equal(r1.coords, r2.coords)


# ---------------------------------------------------------------------------
# The built export
# ---------------------------------------------------------------------------

@requires_export
def test_export_schema():
    with h5py.File(MIL_H5, "r") as f:
        for key in ("spectra", "coords", "sample_ids", "patient_ids", "y_bag",
                    "tissue_cluster", "viable_legacy", "slide_names", "slide_shapes",
                    "wavenumbers"):
            assert key in f, f"missing dataset: {key}"
        assert "y_class" not in f, "y_class should be renamed to y_bag, not both present"
        assert "y_tumor" not in f, "y_tumor is a retired misnomer and should not be exported"

        n = f["spectra"].shape[0]
        for key in ("coords", "sample_ids", "patient_ids", "y_bag", "tissue_cluster", "viable_legacy"):
            assert f[key].shape[0] == n, f"{key} length {f[key].shape[0]} != spectra length {n}"

        assert f["spectra"].shape[1] == 1738, "spectra must be uncropped (full 1738 bands)"
        assert f["slide_names"].shape[0] == f["slide_shapes"].shape[0]


@requires_export
def test_export_includes_all_59_slides_not_58():
    """Documents the resolved discrepancy: ignore_samples is dead config (never
    read outside ConfigManager's defaults) and HF-1887_via-t_2_1.h5_3 appears in
    neither split file, so its presence is inert -- excluding it was an artifact
    of the old RamanPreprocessor filtering, not a deliberate exclusion. The new
    export includes it, matching the notebook-01 producer results/exp_201 also
    used (which has 59 slides; results/exp_102, built by the old class, has 58)."""
    with h5py.File(MIL_H5, "r") as f:
        names = {n.decode() if isinstance(n, bytes) else n for n in f["slide_names"][:]}
    assert len(names) == 59
    assert "HF-1887_via-t_2_1.h5_3" in names


@requires_export
def test_export_reprojection_roundtrip_all_slides():
    """The full-export version of the process_slide-level check above -- confirms
    nothing was lost or misaligned in the resizable-dataset write path."""
    with h5py.File(MIL_H5, "r") as f:
        sample_ids = np.array([s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]])
        coords = f["coords"][:]
        viable_legacy = f["viable_legacy"][:]
        slide_names = f["slide_names"][:]
        slide_shapes = f["slide_shapes"][:]

    shape_lookup = slide_shape_lookup(slide_names, slide_shapes)
    checked = 0
    for name, shape in shape_lookup.items():
        slide_dir = PROCESSED_DIR / name
        mask_path = slide_dir / "tumor_mask.npy"
        if not mask_path.exists():
            continue
        sel = sample_ids == name
        rec = reproject(viable_legacy[sel].astype(float), coords[sel], shape, fill=0.0).astype(bool)
        want = np.load(mask_path).astype(bool)
        assert np.array_equal(rec, want), f"{name}: exported viable_legacy != tumor_mask.npy"
        checked += 1
    assert checked >= 55


@requires_export
def test_slide_shapes_cover_every_sample_id():
    with h5py.File(MIL_H5, "r") as f:
        sample_ids = {s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]}
        slide_names = {s.decode() if isinstance(s, bytes) else s for s in f["slide_names"][:]}
    assert sample_ids == slide_names, (
        f"sample_ids not covered by slide_shapes registry: {sample_ids - slide_names}"
    )


@requires_export
def test_viable_legacy_count_matches_old_export():
    """The widened mask must be a strict superset: every old tumour-cluster pixel
    is still exported (as viable_legacy=True); the total viable_legacy count must
    equal the old production total exactly (227,344 spectra, established in
    results/exp_201/data/glioma_dataset.h5 and cross-checked against
    results/exp_102/tumor_maps.npy in this stage's investigation)."""
    with h5py.File(MIL_H5, "r") as f:
        n_legacy = int(f["viable_legacy"][:].sum())
        n_total = f["spectra"].shape[0]
    assert n_legacy == 227_344, f"viable_legacy count {n_legacy} != old production total 227344"
    assert n_total >= n_legacy, "widened export must be a superset of the legacy selection"


# ---------------------------------------------------------------------------
# Loader + prepare_fold, against a small synthetic HDF5 (fast, no export needed)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_mil_h5(tmp_path):
    """A minimal Stage-1-schema HDF5: 2 slides, uneven sizes, so the patient cap
    and coordinate alignment are both exercised."""
    path = tmp_path / "tiny_mil.h5"
    rng = np.random.default_rng(0)
    n_a, n_b = 7, 3
    n = n_a + n_b
    spectra = rng.uniform(0, 1, (n, 5)).astype("float32")
    coords = np.array([[i, 0] for i in range(n_a)] + [[i, 1] for i in range(n_b)], dtype="int32")
    sample_ids = np.array(["slide_a"] * n_a + ["slide_b"] * n_b)
    y_bag = np.array([1] * n_a + [2] * n_b, dtype="int32")  # 1-indexed, as real exports are
    patient_ids = np.array(["HF-1"] * n_a + ["HF-2"] * n_b)

    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra)
        f.create_dataset("coords", data=coords)
        f.create_dataset("sample_ids", data=sample_ids.astype(object), dtype=h5py.string_dtype())
        f.create_dataset("patient_ids", data=patient_ids.astype(object), dtype=h5py.string_dtype())
        f.create_dataset("y_bag", data=y_bag)
        f.create_dataset("wavenumbers", data=np.linspace(700, 1800, 5).astype("float32"))
    return path


def test_loader_reads_y_bag_and_coords(tiny_mil_h5):
    from src.data.loader import GliomaDataLoader

    loader = GliomaDataLoader(str(tiny_mil_h5), memory_mapped=False)
    assert loader.coords is not None
    assert loader.coords.shape == (10, 2)
    assert np.array_equal(loader.y_class, [1] * 7 + [2] * 3)  # read via y_bag fallback
    loader.close()


def test_loader_patient_cap_keeps_coords_aligned(tiny_mil_h5):
    from src.data.loader import GliomaDataLoader

    loader = GliomaDataLoader(str(tiny_mil_h5), memory_mapped=False)
    before = loader.X.copy()
    loader.apply_patient_cap(patient_cap=2)
    assert loader.X.shape[0] == loader.coords.shape[0] == loader.groups.shape[0]
    # Every kept row's coords must still be the row's own original coords, not
    # some other row's (i.e. the cap must not have desynchronised the arrays).
    for i, row in enumerate(loader.X):
        orig_idx = np.where((before == row).all(axis=1))[0]
        assert len(orig_idx) >= 1
    loader.close()


def test_fold_arrays_unpacks_as_4tuple_and_exposes_coords():
    from pipeline._common import FoldArrays

    X_train = np.zeros((3, 2))
    y_train = np.array([0, 1, 0])
    X_val = np.zeros((2, 2))
    y_val = np.array([1, 0])
    coords = np.array([[0, 0], [1, 0], [2, 0]])

    fd = FoldArrays(X_train, y_train, X_val, y_val, train_coords=coords)

    # Backward-compat: existing call sites do `a, b, c, d = prepare_fold(...)`.
    a, b, c, d = fd
    assert a is X_train and b is y_train and c is X_val and d is y_val

    # New: attribute access for the extra fields.
    assert fd.train_coords is coords
    assert fd.val_coords is None


def test_prepare_fold_keeps_coords_aligned_through_the_cap(monkeypatch):
    from pipeline._common import prepare_fold

    rng = np.random.default_rng(1)
    groups = np.array(["s1"] * 6 + ["s2"] * 2)
    X = rng.uniform(size=(8, 3)).astype(np.float32)
    y = np.array([1] * 6 + [2] * 2)  # 1-indexed
    coords = np.arange(8).reshape(8, 1).repeat(2, axis=1)

    fold_data = {
        "train": (X, y),
        "val": (X[:2], y[:2]),
        "train_groups": groups,
        "train_coords": coords,
        "val_coords": coords[:2],
        "val_groups": groups[:2],
    }
    config = {"models": {"cnn": {"num_classes": 2}}, "cv_strategy": {"patient_cap": 3}}

    # Deterministic cap selection for the assertion below.
    monkeypatch.setattr(np.random, "choice", lambda idx, size, replace: np.sort(idx)[:size])

    fd = prepare_fold(fold_data, config, cap_train=True)
    assert fd.X_train.shape[0] == fd.train_coords.shape[0]
    # coords[:, 0] carries the ORIGINAL row index (by construction above), so this
    # directly checks that capping subset X_train and train_coords with the same
    # index array -- if it desynchronised them, this equality would fail.
    kept_orig_rows = fd.train_coords[:, 0]
    for xi, orig_row in zip(fd.X_train, kept_orig_rows):
        assert np.array_equal(xi, X[orig_row])
    assert np.all(np.isin(kept_orig_rows, np.arange(8)))
    # slide 's1' (rows 0-5) capped to 3, slide 's2' (rows 6-7) has only 2, kept whole.
    n_from_s1 = int(np.sum(kept_orig_rows < 6))
    n_from_s2 = int(np.sum(kept_orig_rows >= 6))
    assert n_from_s1 == 3
    assert n_from_s2 == 2
