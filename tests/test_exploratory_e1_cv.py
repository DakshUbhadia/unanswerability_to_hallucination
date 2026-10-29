import numpy as np

from src.exploratory.e1_cv import (conditional_bootstrap, nested_probe_oof, outer_splits,
                                   select_layer_C)


def _paired_data(seed=4, n_pairs=60, d=20, signal_layer=2):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_pairs)
    groups = np.repeat(np.arange(n_pairs), 2)
    X = rng.normal(size=(2 * n_pairs, 4, d)).astype(np.float32)
    X[y == 1, signal_layer, 0] += 2.0
    return X, y, groups


def test_group_integrity_and_deterministic_nested_scores():
    X, y, groups = _paired_data()
    for _, _, tr, te in outer_splits(y, groups, folds=3, repeats=2, seed=11):
        assert not set(groups[tr]) & set(groups[te])
    a, sel_a = nested_probe_oof(X, y, groups, folds=3, repeats=2, seed=11, C_grid=(0.1,), inner_folds=3)
    b, sel_b = nested_probe_oof(X, y, groups, folds=3, repeats=2, seed=11, C_grid=(0.1,), inner_folds=3)
    np.testing.assert_array_equal(a, b)
    assert sel_a == sel_b


def test_nested_layer_selection_finds_planted_layer():
    X, y, groups = _paired_data()
    selected = []
    for _, _, tr, _ in outer_splits(y, groups, folds=3, repeats=1, seed=9):
        layer, _ = select_layer_C(X[tr], y[tr], groups[tr], C_grid=(0.1,), inner_folds=3, seed=15)
        selected.append(layer)
    assert selected.count(2) >= 2


def test_conditional_bootstrap_returns_ordered_ci():
    X, y, groups = _paired_data()
    pred = X[:, 2, 0]
    result = conditional_bootstrap(pred, np.zeros_like(pred), y, groups, n=100, seed=2)
    assert result["ci_95"][0] <= result["ci_95"][1]
    assert result["ci_conditional_on_oof_predictions"] is True
