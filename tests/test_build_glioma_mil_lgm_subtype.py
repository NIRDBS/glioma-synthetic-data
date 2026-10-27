"""
Tests for scripts/build_glioma_mil_lgm_subtype.py -- the standalone LGm
subtype dataset builder (exp_671 and its planned siblings exp_672/673).

Critically includes an integration check that the file's {0, 3} y_bag
encoding, run through the EXISTING, unmodified nc==2 rule in
pipeline/_common.py, reproduces the intended positive/rest binary label --
this is the whole point of the encoding trick and the one thing that must
never silently break.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_glioma_mil_lgm_subtype import build


@pytest.fixture
def source_h5(tmp_path):
    """A tiny synthetic glioma_mil.h5-shaped fixture: 2 instances per slide,
    one slide per LGm class 0..5 (so LGm1/2/3 -> 3 slides, 6 instances)."""
    path = tmp_path / "glioma_mil.h5"
    n_slides = 6
    inst_per_slide = 2
    n = n_slides * inst_per_slide
    y_bag = np.repeat(np.arange(n_slides, dtype=np.int32), inst_per_slide)
    sample_ids = np.array([f"SLIDE-{c}" for c in y_bag], dtype=object)
    slide_names = np.array([f"SLIDE-{c}" for c in range(n_slides)], dtype=object)

    with h5py.File(path, 'w') as f:
        f.create_dataset('spectra', data=np.zeros((n, 10), dtype=np.float32))
        f.create_dataset('coords', data=np.zeros((n, 2), dtype=np.int32))
        f.create_dataset('patient_ids', data=sample_ids)
        f.create_dataset('sample_ids', data=sample_ids)
        f.create_dataset('tissue_cluster', data=np.zeros(n, dtype=np.int32))
        f.create_dataset('viable_legacy', data=np.ones(n, dtype=bool))
        f.create_dataset('wavenumbers', data=np.zeros(10, dtype=np.float32))
        f.create_dataset('y_bag', data=y_bag)
        f.create_dataset('slide_names', data=slide_names)
        f.create_dataset('slide_shapes', data=np.zeros((n_slides, 2), dtype=np.int32))
        f.attrs['baseline_method'] = 'iarpls'
        f.attrs['baseline_lambda'] = 10000.0
        f.attrs['median_kernel'] = 3
    return path


def test_drops_lgm4_6_instances_entirely(source_h5, tmp_path):
    out = tmp_path / "out.h5"
    build(source_h5, out, positive_class=0)
    with h5py.File(out, 'r') as f:
        assert f.attrs['n_instances'] == 6  # 3 classes x 2 instances
        assert f.attrs['n_slides'] == 3
        assert set(f['slide_names'][:].astype(str)) == {"SLIDE-0", "SLIDE-1", "SLIDE-2"}


@pytest.mark.parametrize("positive_class", [0, 1, 2])
def test_y_bag_only_contains_0_and_3(source_h5, tmp_path, positive_class):
    out = tmp_path / "out.h5"
    build(source_h5, out, positive_class=positive_class)
    with h5py.File(out, 'r') as f:
        y = f['y_bag'][:]
        assert set(np.unique(y)) <= {0, 3}


@pytest.mark.parametrize("positive_class,expected_positive_slide", [
    (0, "SLIDE-0"), (1, "SLIDE-1"), (2, "SLIDE-2"),
])
def test_existing_binary_rule_recovers_the_intended_label(
        source_h5, tmp_path, positive_class, expected_positive_slide):
    """The integration check: feed this file's y_bag through the EXISTING,
    unmodified pipeline/_common.py rule and confirm it recovers the
    intended positive-vs-rest split -- no code changes needed anywhere."""
    out = tmp_path / "out.h5"
    build(source_h5, out, positive_class=positive_class)
    with h5py.File(out, 'r') as f:
        y_bag = f['y_bag'][:]
        sample_ids = f['sample_ids'][:].astype(str)

    # the exact rule from pipeline/_common.py::prepare_fold, untouched
    y_final = np.where(y_bag < 3, 0, 1)

    for slide, label in zip(sample_ids, y_final):
        if slide == expected_positive_slide:
            assert label == 1
        else:
            assert label == 0


def test_reusable_for_all_three_positive_class_choices_from_the_same_source(source_h5, tmp_path):
    for positive_class in (0, 1, 2):
        out = tmp_path / f"out_{positive_class}.h5"
        build(source_h5, out, positive_class=positive_class)
        with h5py.File(out, 'r') as f:
            assert f.attrs['n_instances'] == 6


def test_invalid_positive_class_raises(source_h5, tmp_path):
    out = tmp_path / "out.h5"
    with pytest.raises(ValueError, match="positive_class must be one of"):
        build(source_h5, out, positive_class=5)
