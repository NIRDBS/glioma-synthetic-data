"""Data layer: reprojection onto the pixel grid, the built dataset, the split files, the loader and prepare_fold.

Tests that read data/processed/glioma_mil.h5 are skipped when it is absent (see README, step 1).
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.spatial import reproject, slide_shape_lookup

REPO_ROOT = Path(__file__).resolve().parents[1]
MIL_H5 = REPO_ROOT / "data" / "processed" / "glioma_mil.h5"

requires_export = pytest.mark.skipif(
    not MIL_H5.exists(), reason="data/processed/glioma_mil.h5 not built -- see README, step 1"
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
# The built export
# ---------------------------------------------------------------------------

@requires_export
def test_export_schema():
    with h5py.File(MIL_H5, "r") as f:
        for key in ("spectra", "coords", "sample_ids", "patient_ids", "y_bag", "slide_names",
                    "slide_shapes", "wavenumbers"):
            assert key in f, f"missing dataset: {key}"
        assert "y_class" not in f, "y_class should be renamed to y_bag, not both present"
        assert "y_tumor" not in f, "y_tumor is a retired misnomer and should not be exported"

        n = f["spectra"].shape[0]
        for key in ("coords", "sample_ids", "patient_ids", "y_bag"):
            assert f[key].shape[0] == n, f"{key} length {f[key].shape[0]} != spectra length {n}"

        assert f["spectra"].shape[1] == 1738, "spectra must be uncropped (full 1738 bands)"
        assert f["slide_names"].shape[0] == f["slide_shapes"].shape[0]


@requires_export
def test_export_includes_all_59_slides():
    """Includes HF-1887_via-t_2_1.h5_3, which an earlier export dropped."""
    with h5py.File(MIL_H5, "r") as f:
        names = {n.decode() if isinstance(n, bytes) else n for n in f["slide_names"][:]}
    assert len(names) == 59
    assert "HF-1887_via-t_2_1.h5_3" in names


def test_hf1887_is_in_exactly_one_test_fold():
    """HF-1887_via-t_2_1.h5_3 is in the test partition of one fold and the training partition of the
    others; an earlier split file omitted it from every fold."""
    slide = "HF-1887_via-t_2_1.h5_3"
    splits = json.loads((REPO_ROOT / "data" / "splits" / "manual_splits.json").read_text())
    test_folds = [fk for fk, v in splits.items() if slide in v["test"]]
    train_folds = [fk for fk, v in splits.items() if slide in v["train"]]
    assert len(test_folds) == 1, f"expected exactly 1 test-fold membership, got {test_folds}"
    assert set(train_folds) == set(splits.keys()) - set(test_folds)


@requires_export
def test_slide_shapes_cover_every_sample_id():
    with h5py.File(MIL_H5, "r") as f:
        sample_ids = {s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]}
        slide_names = {s.decode() if isinstance(s, bytes) else s for s in f["slide_names"][:]}
    assert sample_ids == slide_names, (
        f"sample_ids not covered by slide_shapes registry: {sample_ids - slide_names}"
    )


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
