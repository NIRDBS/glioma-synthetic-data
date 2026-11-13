"""Shuffled-bag-label control regression tests. Pure numpy, no real data needed.
"""
import numpy as np
import pytest

from src.mil.shuffle_control import apply_permuted_labels, permuted_bag_label_map


def test_permuted_bag_label_map_preserves_class_balance_exactly():
    """An EXACT permutation of the true label multiset -- not independent
    resampling -- so the shuffled run's class balance is identical to the
    real one, isolating the label-bag link as the only thing that changed."""
    bag_ids = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'])
    y_bag = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # bags a,b=class0; c,d=class1
    label_map = permuted_bag_label_map(bag_ids, y_bag, seed=1)

    assert set(label_map.keys()) == {'a', 'b', 'c', 'd'}
    true_multiset = sorted([0, 0, 1, 1])
    permuted_multiset = sorted(label_map.values())
    assert permuted_multiset == true_multiset


def test_permuted_bag_label_map_actually_changes_the_assignment():
    """Not a proof for every seed (a permutation can be the identity by
    chance), but across several seeds at least one must genuinely move a
    bag's label, or the 'permutation' isn't doing anything."""
    bag_ids = np.array(['a'] * 3 + ['b'] * 3 + ['c'] * 3 + ['d'] * 3)
    y_bag = np.array([0] * 3 + [0] * 3 + [1] * 3 + [1] * 3)
    true_map = {'a': 0, 'b': 0, 'c': 1, 'd': 1}

    changed = any(
        permuted_bag_label_map(bag_ids, y_bag, seed=s) != true_map
        for s in range(20)
    )
    assert changed


def test_permuted_bag_label_map_is_deterministic_given_a_seed():
    bag_ids = np.array(['a', 'a', 'b', 'b', 'c', 'c'])
    y_bag = np.array([0, 0, 1, 1, 2, 2])
    m1 = permuted_bag_label_map(bag_ids, y_bag, seed=7)
    m2 = permuted_bag_label_map(bag_ids, y_bag, seed=7)
    assert m1 == m2


def test_permuted_bag_label_map_same_label_for_a_bag_spanning_two_folds():
    """The 4 duplicated slides appear with the SAME
    bag id across two folds' instance rows -- the map must give exactly one
    permuted label per bag id regardless of how many rows/folds it spans."""
    bag_ids = np.array(['dup', 'dup', 'dup', 'other'])  # 'dup' appears 3x (2 folds worth)
    y_bag = np.array([0, 0, 0, 1])
    label_map = permuted_bag_label_map(bag_ids, y_bag, seed=2)
    assert 'dup' in label_map and 'other' in label_map
    assert len(label_map) == 2  # one entry per UNIQUE bag, not per row


def test_apply_permuted_labels_broadcasts_correctly():
    bag_ids = np.array(['a', 'b', 'a', 'c', 'b'])
    label_map = {'a': 1, 'b': 0, 'c': 1}
    out = apply_permuted_labels(bag_ids, label_map)
    np.testing.assert_array_equal(out, [1, 0, 1, 1, 0])


def test_apply_permuted_labels_raises_on_unknown_bag():
    bag_ids = np.array(['a', 'unmapped'])
    label_map = {'a': 0}
    with pytest.raises(KeyError):
        apply_permuted_labels(bag_ids, label_map)
