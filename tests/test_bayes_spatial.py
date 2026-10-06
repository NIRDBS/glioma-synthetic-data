"""
Tests for src/mil/bayes_spatial.py, in the order correctness actually
requires: pure graph-theory properties first (no sampling, no randomness
to confound a bug), then whether J=0 reproduces the non-spatial model,
then the two behavioural tests the whole spatial extension's justification
rests on -- a planted contiguous blob the i.i.d. model cannot find but the
spatial one should, and the predicted collapse-at-high-J failure mode,
demonstrated directly rather than assumed from the parameter-count
argument in docs/tier5_bayesian_mil.md.
"""
import numpy as np
import pytest

from src.mil.bayes import fit_bayes_mil
from src.mil.bayes_spatial import (
    build_spatial_adjacency, fit_bayes_mil_spatial, four_color_grid,
    score_bayes_mil_spatial_blind,
)
from src.mil.common import N_NULL_COMPONENTS


def _grid_coords(size):
    """(size*size, 2) int (col, row) coords for a full size x size grid."""
    cols, rows = np.meshgrid(np.arange(size), np.arange(size))
    return np.stack([cols.ravel(), rows.ravel()], axis=1)


def _planted_grid_bags(grid_size=16, blob_frac=0.25, class_offset=10.0, n_bags=4,
                       d=4, seed=0, n_classes=2, missing_frac=0.0):
    """
    One square grid per bag, a contiguous square blob at its centre is
    diagnostic, everything outside is null (split uniformly at random
    across the 3 null components). ``class_offset`` controls how far the
    diagnostic class's component mean sits from the nearest null mean --
    the knob that makes per-instance recovery easy (large offset) or hard
    enough that spatial smoothing could plausibly help (small offset).
    ``missing_frac``: fraction of grid cells dropped at random per bag,
    simulating the real pipeline's sparse (filtered) instance set.

    Returns X, bag_ids, y_bag, coords, z_true (coords in (col,row) --
    src/data/spatial.py's convention).
    """
    rng = np.random.default_rng(seed)
    null_means = [np.full(d, -4.0), np.zeros(d), np.full(d, 4.0)][:N_NULL_COMPONENTS]
    class_means = [null_means[0] + class_offset, null_means[-1] - class_offset][:n_classes]

    blob_half = int(round((grid_size * np.sqrt(blob_frac)) / 2))
    center = grid_size // 2

    X, bag_ids, y_bag, coords, z_true = [], [], [], [], []
    for b in range(n_bags):
        label = b % n_classes
        full_coords = _grid_coords(grid_size)
        if missing_frac > 0:
            keep = rng.random(len(full_coords)) > missing_frac
            full_coords = full_coords[keep]
        for col, row in full_coords:
            in_blob = (abs(col - center) <= blob_half) and (abs(row - center) <= blob_half)
            if in_blob:
                z = N_NULL_COMPONENTS + label
                x = rng.normal(class_means[label], 1.0)
            else:
                k = rng.integers(0, N_NULL_COMPONENTS)
                z = k
                x = rng.normal(null_means[k], 1.0)
            X.append(x)
            bag_ids.append(f"bag_{b}")
            y_bag.append(label)
            coords.append([col, row])
            z_true.append(z)

    return (np.array(X), np.array(bag_ids), np.array(y_bag),
           np.array(coords, dtype=int), np.array(z_true))


# ---------------------------------------------------------------------------
# 1. Pure graph-theory correctness -- no sampling, no randomness
# ---------------------------------------------------------------------------

def test_four_color_grid_never_colors_adjacent_cells_the_same():
    coords = _grid_coords(10)
    color = four_color_grid(coords)
    bag_ids = np.zeros(len(coords), dtype=int)
    adjacency = build_spatial_adjacency(coords, bag_ids)
    adjacency = adjacency.tocoo()
    for i, j in zip(adjacency.row, adjacency.col):
        assert color[i] != color[j], f"cells {i} and {j} are adjacent but share colour {color[i]}"


def test_four_color_grid_uses_all_four_colours_on_a_real_grid():
    # Guards against a degenerate "always returns 0" implementation passing
    # the no-same-colour-adjacent check above vacuously.
    coords = _grid_coords(10)
    color = four_color_grid(coords)
    assert set(np.unique(color).tolist()) == {0, 1, 2, 3}


def test_build_spatial_adjacency_matches_hand_built_expectations_on_3x3():
    coords = _grid_coords(3)  # 9 cells, (col,row) in {0,1,2}x{0,1,2}
    bag_ids = np.zeros(9, dtype=int)
    adjacency = build_spatial_adjacency(coords, bag_ids)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()

    # center cell (col=1,row=1) -> 8 neighbours; corners -> 3; edges -> 5.
    idx_of = {(int(c), int(r)): i for i, (c, r) in enumerate(coords)}
    assert degree[idx_of[(1, 1)]] == 8
    for corner in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        assert degree[idx_of[corner]] == 3
    for edge in [(1, 0), (0, 1), (2, 1), (1, 2)]:
        assert degree[idx_of[edge]] == 5
    assert adjacency.sum() == 8 + 4 * 3 + 4 * 5  # symmetric, directed-edge count


def test_build_spatial_adjacency_has_no_cross_bag_edges():
    # Two bags occupying the SAME coordinates (two different slides can
    # absolutely have overlapping pixel grids) must never be connected.
    coords = np.vstack([_grid_coords(4), _grid_coords(4)])
    bag_ids = np.array(['A'] * 16 + ['B'] * 16)
    adjacency = build_spatial_adjacency(coords, bag_ids)
    adjacency = adjacency.tocoo()
    for i, j in zip(adjacency.row, adjacency.col):
        assert bag_ids[i] == bag_ids[j]


def test_build_spatial_adjacency_has_no_phantom_edges_to_missing_cells():
    coords = _grid_coords(3)
    # drop the centre cell (col=1,row=1) entirely
    keep = ~((coords[:, 0] == 1) & (coords[:, 1] == 1))
    coords = coords[keep]
    bag_ids = np.zeros(len(coords), dtype=int)
    adjacency = build_spatial_adjacency(coords, bag_ids)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    idx_of = {(int(c), int(r)): i for i, (c, r) in enumerate(coords)}
    # every remaining cell loses exactly the edge it would have had to the
    # now-missing centre -- corners keep 3-1=2, edges keep 5-1=4
    for corner in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        assert degree[idx_of[corner]] == 2
    for edge in [(1, 0), (0, 1), (2, 1), (1, 2)]:
        assert degree[idx_of[edge]] == 4


# ---------------------------------------------------------------------------
# 2. J=0 must reduce to the non-spatial model
# ---------------------------------------------------------------------------

def test_J_equals_zero_recovers_as_well_as_the_non_spatial_model():
    """Not bit-identical (checkerboard-vs-one-shot Gumbel-max consumes
    randomness differently even at J=0), but at J=0 the coupling term is
    IDENTICALLY zero, so the full conditional at every site reduces
    exactly to the non-spatial model's -- recovery quality must match."""
    X, bag_ids, y_bag, coords, z_true = _planted_grid_bags(
        grid_size=14, class_offset=10.0, n_bags=6, d=4, seed=5)

    fit_iid = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=200,
                            burn_in=60, thin=5, seed=5, verbose=False)
    fit_spatial = fit_bayes_mil_spatial(X, bag_ids, y_bag, coords, n_classes=2, J=0.0,
                                       n_iter=200, burn_in=60, thin=5, seed=5,
                                       verbose=False)

    pred_iid = fit_iid['posterior'][:, 1:].sum(axis=1) > fit_iid['posterior'][:, 0]
    pred_spatial = fit_spatial['posterior'][:, 1:].sum(axis=1) > fit_spatial['posterior'][:, 0]
    truth = z_true >= N_NULL_COMPONENTS

    acc_iid = (pred_iid == truth).mean()
    acc_spatial = (pred_spatial == truth).mean()
    assert acc_iid > 0.9, f"sanity: non-spatial baseline itself should recover well, got {acc_iid:.3f}"
    assert acc_spatial > 0.9, f"J=0 should recover as well as the non-spatial model, got {acc_spatial:.3f}"
    assert abs(acc_iid - acc_spatial) < 0.1, (
        f"J=0 should behave like the non-spatial model: iid={acc_iid:.3f} spatial(J=0)={acc_spatial:.3f}")


# ---------------------------------------------------------------------------
# 3. The decisive positive control: a contiguous blob the i.i.d. model
#    cannot cleanly find but spatial smoothing should recover better
# ---------------------------------------------------------------------------

def test_spatial_coupling_beats_iid_when_signal_is_weak_but_spatially_contiguous():
    """class_offset is deliberately small -- per-instance signal alone is
    noisy/overlapping -- but the true diagnostic region is a clean
    contiguous blob. A model that can borrow strength from neighbours
    should recover the blob more accurately than one that can't.

    J=1.5, not a smaller value, because sweeping J on this exact planted
    setup (seed=7) found the relationship is NOT monotonic at this
    iteration budget: J=0.5 (0.849) and J=1.0 (0.796) both score BELOW
    the i.i.d. baseline (0.897) -- only J>=1.5 clearly beats it (0.968).
    That dip is itself a real finding (docs/tier5_bayesian_mil.md), not
    noise to average away by picking a flattering J -- recorded here
    rather than hidden by only testing the J that happens to work."""
    X, bag_ids, y_bag, coords, z_true = _planted_grid_bags(
        grid_size=16, blob_frac=0.25, class_offset=0.9, n_bags=6, d=4, seed=7)
    truth = z_true >= N_NULL_COMPONENTS

    fit_iid = fit_bayes_mil(X, bag_ids, y_bag, n_classes=2, n_iter=300,
                            burn_in=100, thin=5, seed=7, verbose=False)
    pred_iid = fit_iid['posterior'][:, 1:].sum(axis=1) > fit_iid['posterior'][:, 0]
    acc_iid = (pred_iid == truth).mean()

    fit_spatial = fit_bayes_mil_spatial(X, bag_ids, y_bag, coords, n_classes=2, J=1.5,
                                       n_iter=300, burn_in=100, thin=5, seed=7,
                                       verbose=False)
    pred_spatial = fit_spatial['posterior'][:, 1:].sum(axis=1) > fit_spatial['posterior'][:, 0]
    acc_spatial = (pred_spatial == truth).mean()

    assert acc_iid < 0.95, (
        f"sanity: class_offset should be weak enough that the i.i.d. model is imperfect, got {acc_iid:.3f}")
    assert acc_spatial > acc_iid + 0.03, (
        f"spatial coupling (J=1.5) should beat the i.i.d. model on a contiguous blob: "
        f"iid={acc_iid:.3f} spatial={acc_spatial:.3f}")


# ---------------------------------------------------------------------------
# 4. The decisive negative control: collapse at high J on PURE NOISE
#    (no true spatial structure at all) -- the failure mode
#    docs/tier5_bayesian_mil.md predicted from the parameter-count argument,
#    demonstrated directly rather than assumed.
# ---------------------------------------------------------------------------

def test_high_J_collapses_toward_a_single_state_on_pure_noise():
    """No planted contiguity at all -- z assigned independently at random,
    uncorrelated with the (real) grid adjacency. A well-behaved model
    should show close to the i.i.d. model's (noisy, non-degenerate)
    behaviour; a too-strongly-coupled one should collapse toward a single
    dominant state regardless of the data, exactly the risk flagged
    before building this module."""
    rng = np.random.default_rng(3)
    grid_size = 14
    d = 4
    n_bags = 4
    null_means = [np.full(d, -4.0), np.zeros(d), np.full(d, 4.0)]
    class_means = [null_means[0] + 10.0, null_means[-1] - 10.0]

    X, bag_ids, y_bag, coords = [], [], [], []
    for b in range(n_bags):
        label = b % 2
        for col, row in _grid_coords(grid_size):
            if rng.random() < 0.3:
                x = rng.normal(class_means[label], 1.0)
            else:
                k = rng.integers(0, 3)
                x = rng.normal(null_means[k], 1.0)
            X.append(x)
            bag_ids.append(f"bag_{b}")
            y_bag.append(label)
            coords.append([col, row])
    X, bag_ids, y_bag = np.array(X), np.array(bag_ids), np.array(y_bag)
    coords = np.array(coords, dtype=int)

    def frac_in_most_common_state(J):
        fit = fit_bayes_mil_spatial(X, bag_ids, y_bag, coords, n_classes=2, J=J,
                                   n_iter=150, burn_in=50, thin=5, seed=9, verbose=False)
        z = fit['z_final']
        _, counts = np.unique(z, return_counts=True)
        return counts.max() / len(z)

    frac_low = frac_in_most_common_state(J=0.0)
    frac_high = frac_in_most_common_state(J=4.0)
    assert frac_high > frac_low + 0.15, (
        f"expected high J to collapse toward a single dominant state: "
        f"frac_in_most_common_state low-J={frac_low:.3f} high-J={frac_high:.3f}")


# ---------------------------------------------------------------------------
# 5. Blind spatial scoring, end to end
# ---------------------------------------------------------------------------

def test_score_bayes_mil_spatial_blind_recovers_true_class_on_held_out_bags():
    X, bag_ids, y_bag, coords, _ = _planted_grid_bags(
        grid_size=14, blob_frac=0.3, class_offset=10.0, n_bags=4, d=4, seed=1)
    fit = fit_bayes_mil_spatial(X, bag_ids, y_bag, coords, n_classes=2, J=1.0,
                               n_iter=200, burn_in=60, thin=5, seed=1, verbose=False)

    X_new, bag_new, y_new, coords_new, _ = _planted_grid_bags(
        grid_size=14, blob_frac=0.3, class_offset=10.0, n_bags=4, d=4, seed=2)
    ll_by_bag = score_bayes_mil_spatial_blind(fit, X_new, bag_new, coords_new,
                                              n_classes=2, J=1.0, n_mini_iter=60,
                                              n_mini_burn=20, seed=0)
    true_label = {b: y_new[bag_new == b][0] for b in np.unique(bag_new)}
    n_correct = sum(int(np.argmax(ll) == true_label[b]) for b, ll in ll_by_bag.items())
    acc = n_correct / len(ll_by_bag)
    assert acc >= 0.75, f"blind spatial classification recovered only {acc:.2f} on a well-separated planted blob"
