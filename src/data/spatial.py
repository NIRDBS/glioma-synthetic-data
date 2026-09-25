"""Pixel <-> slide-grid reprojection.

The modelling HDF5 stores each instance's pixel position (``coords``) but not the
slide's full raster extent, so reconstructing a 2-D map needs the per-slide shape
registry alongside it. This module is the single, tested implementation of that
round-trip -- do not re-derive the indexing elsewhere (a transposed version exists
in ``_archive/dev_scripts/consolidate_tumor_masks.py`` and is wrong for non-square
slides; this one is verified against ``tumor_maps.npy`` for every slide in
``tests/test_stage1_export.py``).

Convention, fixed by ``src/data/preprocessing.py`` at export time and unchanged
since Stage 0: ``coords[:, 0]`` is the column (raster axis 1), ``coords[:, 1]`` is
the row (raster axis 0). A slide's shape is ``(n_rows, n_cols)``, matching
``raw_cube.shape[:2]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["reproject", "slide_shape_lookup"]


def reproject(
    values: np.ndarray,
    coords: np.ndarray,
    shape: tuple[int, int],
    fill: float = np.nan,
) -> np.ndarray:
    """Scatter per-instance ``values`` back onto a slide's ``(n_rows, n_cols)`` grid.

    ``coords`` must already be restricted to the one slide being reprojected (e.g.
    ``coords[sample_ids == slide_name]``) and aligned 1:1 with ``values``. Pixels the
    slide has but ``coords`` does not cover keep ``fill`` -- the maps this project
    exports are sparse (only viable-signal pixels), so holes are expected, not a bug.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (n, 2), got shape {coords.shape}")
    if len(values) != len(coords):
        raise ValueError(f"values ({len(values)}) and coords ({len(coords)}) length mismatch")

    img = np.full(shape, fill, dtype=float)
    cols, rows = coords[:, 0], coords[:, 1]
    if len(rows) and (rows.max() >= shape[0] or cols.max() >= shape[1] or rows.min() < 0 or cols.min() < 0):
        raise ValueError(
            f"coords out of bounds for shape {shape}: "
            f"rows in [{rows.min()},{rows.max()}], cols in [{cols.min()},{cols.max()}]"
        )
    img[rows, cols] = values
    return img


def slide_shape_lookup(slide_names: np.ndarray, slide_shapes: np.ndarray) -> dict:
    """Build a ``{slide_name: (n_rows, n_cols)}`` dict from the HDF5 side tables."""
    names = np.asarray(slide_names)
    if names.dtype.kind in ("S", "O") or (names.dtype.kind == "U"):
        names = np.array([n.decode() if isinstance(n, bytes) else str(n) for n in names])
    shapes = np.asarray(slide_shapes)
    return {name: (int(shapes[i, 0]), int(shapes[i, 1])) for i, name in enumerate(names)}
