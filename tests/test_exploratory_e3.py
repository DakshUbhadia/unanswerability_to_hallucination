import itertools

import numpy as np
import pandas as pd
import pytest

from src.exploratory import e3_analysis


def test_incremental_column_mapping_matches_registered_schema(monkeypatch):
    frame = pd.DataFrame({"maxp": [0.1, 0.9], "entropy": [2.0, 1.0], "log_odds_set": [0.2, -0.2],
                          "proj_own": [0.3, -0.3], "y": [1, 0], "matched_pair_id": ["a", "a"]})
    expected = {"same": True}
    def fake(df, **kwargs):
        assert list(df["log_odds"]) == list(frame["log_odds_set"])
        assert list(df["proj"]) == list(frame["proj_own"])
        assert kwargs["full_features"] == ["maxp", "entropy", "log_odds", "proj"]
        return expected
    monkeypatch.setattr(e3_analysis, "incremental_value_test", fake)
    assert e3_analysis.incremental_value_for_projection(frame, "proj_own") is expected


def test_instruction_contrast_bootstraps_conditions_independently():
    result = e3_analysis.independent_condition_bootstrap([0, 1, 0, 1], [0, 1, 1, 0],
                                                         [0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0], n_boot=50)
    assert result["resampling"] == "independent_by_condition"
    assert result["n_boot"] > 0


def test_pair_direction_and_in_context_auroc_are_fit_on_train_and_scored_on_test():
    rows = []
    acts = []
    for split in ("train", "test"):
        for pair in range(4):
            for part, sign in (("ans", -1), ("unans", 1)):
                rows.append({"pair_id": f"{split}{pair}", "split": split, "part": part})
                acts.append(np.full((2, 1, 3), sign, dtype=np.float32))
    diff, unit, pair_ids = e3_analysis.direction_from_pairs(np.asarray(acts), rows)
    assert pair_ids == ["train0", "train1", "train2", "train3"]
    assert diff.shape == unit.shape == (1, 2, 3)
    assert np.allclose(unit, 1 / np.sqrt(3))
    scores = e3_analysis.in_context_auroc(np.asarray(acts), rows, unit)
    assert scores.shape == (1, 2) and np.all(scores == 1.0)


def test_geometry_alignment_bootstrap_resamples_train_pairs():
    rng = np.random.default_rng(18)
    rows, acts = [], []
    for i in range(8):
        base = rng.normal(size=(5, 12, 4)).astype(np.float32)
        ans, unans = base.copy(), base.copy()
        ans[1, 11, 0] -= 1
        unans[1, 11, 0] += 1
        for part, value in (("ans", ans), ("unans", unans)):
            rows.append({"pair_id": str(i), "split": "train", "part": part})
            acts.append(value)
    reference = np.array([1., 0., 0., 0.])
    result = e3_analysis.bootstrap_train_direction_alignment(np.asarray(acts), rows, reference, n_boot=40)
    assert result["resampling"] == "train_pairs_with_replacement"
    assert result["n_boot"] == 40
    assert result["whitened_ci95"][0] > .99


def test_concordant_subset_keeps_only_identical_correctness_labels():
    instr = pd.DataFrame({"popqa_id": ["a", "b", "c"], "label": ["correct", "wrong", "abstain"]})
    neut = pd.DataFrame({"popqa_id": ["a", "b", "c"], "label": ["correct", "correct", "abstain"]})
    a, b = e3_analysis.label_concordant_subset(instr, neut)
    assert a.popqa_id.tolist() == b.popqa_id.tolist() == ["a"]


@pytest.mark.parametrize("survives,aligned,separation,residual", itertools.product(
    [False, True], [None, False, True], ["present", "absent", "inconclusive"],
    ["residual_null_equivalent", "residual_signal_present", "inconclusive"]))
def test_verdict_logic_covers_all_label_combinations(survives, aligned, separation, residual):
    verdict = e3_analysis.e3_verdict(in_context_survives=survives, aligned_with_instructed=aligned,
                                    separation=separation, residual=residual)
    if not survives:
        assert verdict == "Direction is instruction-dependent"
    elif separation == "absent":
        assert verdict == "Instruction-dependent transfer"
    elif separation == "present" and residual == "residual_signal_present":
        assert verdict == "Knowledge-state component under the neutral prompt"
    elif aligned is True and separation == "present" and residual == "residual_null_equivalent":
        assert verdict == "Instruction-robust confidence/abstention readout"
    else:
        assert verdict.startswith("mixed/inconclusive:")
