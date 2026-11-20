"""Preprocessing primitives that are independent of the dataset layout.

Kept separate from the dataset builders (``src/data/preprocessing_v2.py``, ``_v3.py``)
so these routines can be unit-tested and reused without touching HDF5 or raw cubes.
"""

from src.preprocessing.baseline import (
    DEFAULT_LAMBDA_GRID,
    RAMAN_REFERENCE_BANDS,
    airpls,
    correct,
    difference_penalty_banded,
    select_lambda,
    whittaker,
)

__all__ = [
    "DEFAULT_LAMBDA_GRID",
    "RAMAN_REFERENCE_BANDS",
    "airpls",
    "correct",
    "difference_penalty_banded",
    "select_lambda",
    "whittaker",
]
