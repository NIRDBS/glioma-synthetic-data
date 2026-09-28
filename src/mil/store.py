"""
Latent-label store (docs/MIL_REFRAME_PLAN.md §3): every MIL formulation's
per-instance posterior, written to a sidecar HDF5 file next to (not inside)
the main dataset -- results/{exp}/latent/latent_labels.h5. The main HDF5 is
shared across experiments and must stay immutable; this file is the
experiment-specific, formulation-specific output layered on top of it.

Leakage rule -- this is the load-bearing part of §3. Index by fold: a bag in
fold f's *validation* set gets its posterior from the fold-f model, which
never saw it during fitting (in_sample=False for those rows). A bag in fold
f's *training* set is in-sample by necessity (in_sample=True) -- legitimate
for T3's CVAE conditioning, since that is train-side use. Enforced here with
an assertion at read time (assert_no_fold_leakage), not left as a convention
callers might forget.
"""
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


def alignment_checksum(sample_ids, coords):
    """Hash of (sample_ids, coords) from the source HDF5 -- guards against a
    silent re-export invalidating stored posteriors for a mismatched row
    order (§3)."""
    h = hashlib.sha256()
    h.update(np.asarray(sample_ids).astype('U').astype('S').tobytes())
    if coords is not None:
        h.update(np.ascontiguousarray(coords).tobytes())
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def write_posteriors(store_path, method, variant, fold_name, posterior, in_sample,
                     sample_ids, coords, config_hash='', overwrite=True):
    """
    Write one (method, variant, fold)'s posteriors to the store.

    posterior: (N, K+1) float32, row-aligned 1:1 to ``sample_ids``/``coords``'
        own order (the caller is responsible for scattering fold-local rows
        back to that full order first -- see pipeline/mil_fit_stage.py).
    in_sample: (N,) bool, same alignment -- True where this fold's own bags
        were in its TRAINING set, False for its validation bags.
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    posterior = np.asarray(posterior)
    K1 = posterior.shape[1]

    with h5py.File(store_path, 'a') as f:
        grp_path = f"{method}/{variant}/{fold_name}"
        if grp_path in f:
            if not overwrite:
                raise FileExistsError(f"{grp_path} already exists in {store_path}")
            del f[grp_path]
        grp = f.create_group(grp_path)
        grp.create_dataset('posterior', data=posterior.astype(np.float32))
        grp.create_dataset('in_sample', data=np.asarray(in_sample, dtype=bool))
        grp.attrs['method'] = method
        grp.attrs['variant'] = variant
        grp.attrs['K'] = K1 - 1
        grp.attrs['constrained'] = True
        grp.attrs['config_hash'] = config_hash
        grp.attrs['git_sha'] = _git_sha()
        grp.attrs['timestamp'] = datetime.now().isoformat()
        grp.attrs['alignment_checksum'] = alignment_checksum(sample_ids, coords)


def read_posteriors(store_path, method, variant, fold_name, expected_checksum=None):
    """
    Read one (method, variant, fold)'s posteriors back. If ``expected_checksum``
    is given, raises AssertionError on mismatch -- the guard against a silent
    re-export invalidating stored posteriors (§3's alignment_checksum).
    """
    with h5py.File(store_path, 'r') as f:
        grp_path = f"{method}/{variant}/{fold_name}"
        if grp_path not in f:
            raise KeyError(f"{grp_path} not found in {store_path}")
        grp = f[grp_path]
        posterior = grp['posterior'][:]
        in_sample = grp['in_sample'][:]
        checksum = grp.attrs['alignment_checksum']
        attrs = dict(grp.attrs)

    if expected_checksum is not None:
        assert checksum == expected_checksum, (
            f"alignment_checksum mismatch for {grp_path}: stored={checksum!r} "
            f"expected={expected_checksum!r} -- the source HDF5 was likely "
            "re-exported since these posteriors were written; do not use them."
        )
    return {'posterior': posterior, 'in_sample': in_sample, 'attrs': attrs}


def assert_no_fold_leakage(store_path, method, variant, fold_name, val_bag_ids_this_fold,
                          bag_ids_all):
    """
    Fold-leakage assertion (§3, and §12's standing invariant): every row
    belonging to this fold's OWN validation bags must be marked
    in_sample=False (scored by a model that did not train on them). A
    deliberately mismatched fold (passing another fold's validation bags)
    must raise -- see tests/test_stage4_mil.py.
    """
    data = read_posteriors(store_path, method, variant, fold_name)
    in_sample = data['in_sample']
    bag_ids_all = np.asarray(bag_ids_all)
    val_mask = np.isin(bag_ids_all, list(val_bag_ids_this_fold))
    if val_mask.any() and in_sample[val_mask].any():
        raise AssertionError(
            f"{method}/{variant}/{fold_name}: {int(in_sample[val_mask].sum())} of the "
            "given validation-bag rows are marked in_sample=True in the store -- a "
            "held-out bag's posterior must come from a model that never saw it."
        )


def merge_out_of_sample_posterior(store_path, method, variant, fold_names):
    """
    Combine every fold's out-of-sample (in_sample=False) rows into one
    full-dataset (N, K+1) posterior -- the single leak-free estimate for each
    instance, for consumers (Stage 5's spatial validation, T3's CVAE
    conditioning) that want one posterior per instance rather than
    per-(fold, instance).

    Each instance is out-of-sample in exactly one fold, EXCEPT the 4 bags
    docs/MIL_REFRAME_PLAN.md §7.1 documents as spanning two CV test folds --
    those get an out-of-sample estimate from both, which are averaged and
    renormalised here (consistent with Stage 2's own duplicate-bag handling),
    not arbitrarily picked from whichever fold happened to load first.

    Returns (posterior, n_folds_covering): posterior is (N, K+1) float32,
    NaN for any row no fold ever marked out-of-sample (should not happen for
    a correctly-run Stage 4 fit, but not silently assumed away);
    n_folds_covering is (N,) int, the number of folds that contributed to
    each row (1 for ordinary bags, 2 for the duplicated ones, 0 if missing).
    """
    sums = None
    counts = None
    for fold_name in fold_names:
        data = read_posteriors(store_path, method, variant, fold_name)
        posterior = data['posterior']
        in_sample = data['in_sample']
        if sums is None:
            sums = np.zeros_like(posterior, dtype=np.float64)
            counts = np.zeros(len(posterior), dtype=np.int64)
        oos = ~in_sample
        sums[oos] += posterior[oos]
        counts[oos] += 1

    merged = np.full_like(sums, np.nan)
    covered = counts > 0
    merged[covered] = sums[covered] / counts[covered, None]
    row_sums = merged[covered].sum(axis=1, keepdims=True)
    merged[covered] = merged[covered] / np.where(row_sums > 0, row_sums, 1.0)
    return merged.astype(np.float32), counts
