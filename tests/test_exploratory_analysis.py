import numpy as np

from src.exploratory.analysis import auc_summary, nested_group_probe_oof


def test_nested_probe_oof_keeps_pairs_together_and_returns_complete_scores():
    rng = np.random.default_rng(19)
    n_pairs = 30
    y = np.tile([0, 1], n_pairs)
    groups = np.repeat([f"pair_{i}" for i in range(n_pairs)], 2)
    X = rng.normal(size=(len(y), 6))
    X[:, 0] += y * 0.8

    predictions, chosen_c = nested_group_probe_oof(X, y, groups, folds=5, repeats=2, seed=23)

    assert predictions.shape == (2, len(y))
    assert np.isfinite(predictions).all()
    assert set(np.unique(chosen_c)) <= {0.01, 0.1, 1.0}


def test_auc_summary_returns_cluster_bootstrap_interval():
    y = np.tile([0, 1], 20)
    score = y.astype(float)
    groups = np.repeat([f"pair_{i}" for i in range(20)], 2)

    result = auc_summary(y, score, groups, n_boot=100, seed=5)

    assert result["auroc"] == 1.0
    assert result["ci95"]["low"] == 1.0
    assert result["ci95"]["high"] == 1.0
