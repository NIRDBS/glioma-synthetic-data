"""Regression tests for scripts/attention_bag_level_eval.py's duplicate-bag
merging -- the one piece of non-trivial new logic in that script (everything
else reuses already-tested machinery: bag_level_bootstrap,
summarize_bootstrap_diffs, bag_level_eval._tier0_reference)."""
import json

import numpy as np

from scripts.attention_bag_level_eval import _load_bag_level_across_folds


def _write_bag_level_json(path, by_fold):
    with open(path, 'w') as f:
        json.dump(by_fold, f)


def test_ordinary_bag_appears_in_exactly_one_fold(tmp_path):
    path = tmp_path / 'attention_bag_level.json'
    _write_bag_level_json(path, {
        'fold_0': {'bag_ids': ['a', 'b'], 'posterior': [[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]],
                  'y_true': [0, 1]},
        'fold_1': {'bag_ids': ['c'], 'posterior': [[0.05, 0.05, 0.9]], 'y_true': [1]},
    })
    bag_ids, posterior, y_true = _load_bag_level_across_folds(path)
    assert list(bag_ids) == ['a', 'b', 'c']
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_array_equal(y_true, [0, 1, 1])


def test_duplicated_bag_across_two_folds_is_averaged_not_double_counted():
    """The 4 duplicated slides appear as an
    out-of-sample test bag in TWO folds -- must be averaged (Stage 2's own
    duplicate-bag convention), never appear twice in the returned arrays."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'attention_bag_level.json'
        _write_bag_level_json(path, {
            'fold_0': {'bag_ids': ['dup'], 'posterior': [[0.2, 0.8, 0.0]], 'y_true': [0]},
            'fold_1': {'bag_ids': ['dup'], 'posterior': [[0.4, 0.6, 0.0]], 'y_true': [0]},
        })
        bag_ids, posterior, y_true = _load_bag_level_across_folds(path)
        assert list(bag_ids) == ['dup']
        np.testing.assert_allclose(posterior[0], [0.3, 0.7, 0.0], atol=1e-8)
        assert y_true[0] == 0


def test_inconsistent_label_across_folds_raises():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'attention_bag_level.json'
        _write_bag_level_json(path, {
            'fold_0': {'bag_ids': ['x'], 'posterior': [[0.1, 0.9, 0.0]], 'y_true': [0]},
            'fold_1': {'bag_ids': ['x'], 'posterior': [[0.1, 0.9, 0.0]], 'y_true': [1]},
        })
        try:
            _load_bag_level_across_folds(path)
            assert False, "expected an AssertionError for inconsistent bag labels"
        except AssertionError as e:
            assert 'inconsistent labels' in str(e)
