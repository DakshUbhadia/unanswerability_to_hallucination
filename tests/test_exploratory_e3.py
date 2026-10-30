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
