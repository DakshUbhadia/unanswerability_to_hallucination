import numpy as np
import pandas as pd

from src.exploratory.e1_cv import outer_splits
from src.exploratory.e4_transfer import (
    Source,
    _auc,
    _stack_oof_fixed,
    fit_direction,
    label_cell,
    score,
    verify_dom_parity,
    cell,
)


def _source(name, signal=True, n_groups=160, seed=12, text_signal=False):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_groups // 2)
    rng.shuffle(y)
    x = rng.normal(size=(len(y), 5, 3, 6)).astype(np.float32)
    if signal:
        x[:, :, :, 0] += (3 * (2 * y - 1))[:, None, None]
    log_odds = (2 * y - 1).astype(float) * 3 if signal else rng.normal(size=len(y))
    meta = pd.DataFrame({"y": y, "group": [f"g{i}" for i in range(len(y))],
                         "split": "test", "log_odds": log_odds,
                         "maxp": 1 / (1 + np.exp(log_odds)), "entropy": rng.normal(size=len(y))})
    texts = (["token_yes common" if label else "token_no common" for label in y]
             if text_signal else ["common unrelated text" for _ in y])
    return Source(name, x, meta, "contextual", texts)


def test_off_diagonal_does_not_leak_test_labels():
    a, b = _source("a", True, seed=1), _source("b", False, seed=2)
    probe = fit_direction("dom", a.X[:, 1, 1], a.y)
    heldout = score(probe, b.X[:, 1, 1])
    assert 0.40 <= _auc(b.y, heldout) <= 0.60


def test_shared_direction_transfers_in_both_directions():
    a, b = _source("a", True, seed=3), _source("b", True, seed=4)
    for train, test in ((a, b), (b, a)):
        probe = fit_direction("dom", train.X[:, 1, 1], train.y)
        assert _auc(test.y, score(probe, test.X[:, 1, 1])) > 0.90


def test_logit_only_stack_has_no_incremental_signal():
    rng = np.random.default_rng(5)
    y = np.tile([0, 1], 100)
    rng.shuffle(y)
    logits = 4 * y + rng.normal(0, 0.2, len(y))
    groups = np.asarray([f"g{i}" for i in range(len(y))])
    base, full = _stack_oof_fixed(y, groups, np.column_stack((logits, -logits, np.abs(logits))),
                                  logits + rng.normal(0, 0.02, len(y)), repeats=2)
    assert _auc(y, full) - _auc(y, base) < 0.02
    result = {"auroc_probe": _auc(y, logits), "auroc_logit": _auc(y, logits),
              "incremental_delta": _auc(y, full) - _auc(y, base), "ci95": {
                  "auroc_probe": {"low": 0.9, "high": 1.0},
                  "incremental_delta": {"low": -0.01, "high": 0.01}, "auroc_logit": {"low": .9, "high": 1.0},
                  "auroc_text": None}}
    assert "transfers_but_explained_by_logits" in label_cell(result)


def test_surface_flag_fires_for_prompt_token_signal():
    src = _source("surface", False, seed=9, text_signal=True)
    from src.exploratory.e4_transfer import _diagonal_text_oof
    pred = _diagonal_text_oof(src)
    text_auc = _auc(src.y, pred)
    assert text_auc > 0.99
    result = {"auroc_probe": .51, "auroc_logit": .5, "incremental_delta": 0,
              "ci95": {"auroc_probe": {"low": .48, "high": .54},
                       "incremental_delta": {"low": -.02, "high": .01},
                       "auroc_logit": {"low": .45, "high": .55},
                       "auroc_text": {"low": text_auc, "high": 1.0}}}
    assert "surface_explainable" in label_cell(result)


def test_reversed_dom_is_not_sign_flipped():
    rng = np.random.default_rng(21)
    y = np.tile([0, 1], 100)
    x = rng.normal(size=(len(y), 3))
    x[:, 0] += 2 * (2 * y - 1)
    probe = fit_direction("dom", x, y)
    reversed_x = rng.normal(size=(len(y), 3))
    reversed_x[:, 0] += 2 * (1 - 2 * y)
    value = _auc(y, score(probe, reversed_x))
    assert value < 0.05
    result = {"auroc_probe": value, "auroc_logit": .5, "incremental_delta": 0,
              "ci95": {"auroc_probe": {"low": .01, "high": .40},
                       "incremental_delta": {"low": -.1, "high": .01},
                       "auroc_logit": {"low": .4, "high": .6}, "auroc_text": None}}
    assert "reversed_transfer" in label_cell(result)


def test_nested_fold_indices_are_group_disjoint():
    y = np.tile([0, 1], 20)
    groups = np.asarray([f"pair{i // 2}" for i in range(len(y))])
    for _, _, train, test in outer_splits(y, groups, folds=5, repeats=1):
        assert not set(train) & set(test)
        assert not set(groups[train]) & set(groups[test])


def test_dom_matches_supplied_diff_in_means():
    rng = np.random.default_rng(31)
    y = np.tile([0, 1], 30)
    x = rng.normal(size=(len(y), 8)) + y[:, None] * np.arange(8)
    expected = x[y == 1].mean(0) - x[y == 0].mean(0)
    result = verify_dom_parity(x, y, expected)
    assert result["passed"]
    assert result["cosine"] > 0.9999


def test_diagonal_cell_returns_all_preregistered_metrics_and_group_bootstrap():
    src = _source("squad", True, n_groups=40, seed=44)
    result = cell(src, src, est="dom", pos="suf-4", layer_mode="fixed", layer=1,
                  bootstrap=30, seed=3)
    assert {"auroc_probe", "auroc_logit", "delta", "incremental_delta", "auroc_text"} <= set(result)
    assert set(result["ci95"]) == {"auroc_probe", "auroc_logit", "delta", "incremental_delta", "auroc_text"}
    assert result["n_test"] == len(src.y)


def test_label_rules_cover_each_primary_outcome():
    def item(probe, inc, logit=.7, text=None):
        return {"auroc_probe": sum(probe) / 2, "auroc_logit": logit,
                "incremental_delta": sum(inc) / 2,
                "ci95": {"auroc_probe": {"low": probe[0], "high": probe[1]},
                         "incremental_delta": {"low": inc[0], "high": inc[1]},
                         "auroc_logit": {"low": logit, "high": logit}, "auroc_text": text}}
    assert "transfers_beyond_logits" in label_cell(item((.7, .9), (.03, .2)))
    assert "transfers_but_explained_by_logits" in label_cell(item((.7, .9), (-.1, .01)))
    assert "no_transfer" in label_cell(item((.48, .52), (-.1, .01)))
    assert "transfers_but_explained_by_logits" in label_cell(item((.7, .9), (-.1, .01), .99))
    assert "ceiling_limited" in label_cell(item((.6, .9), (-.1, .01), .99))
    assert "inconclusive" in label_cell(item((.6, .7), (-.1, .01)))
