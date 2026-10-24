import json

import numpy as np
import pytest

from src.config import CFG
from src.popqa_steering import (confidence_threshold_curve, plot_figure2b_selective_steering,
                                run_stage10_popqa_steering_sweep, save_stage10_artifacts, selective_metrics,
                                verify_stage10_outputs)
from tests._tiny_model import StubTokenizer, tiny_contract, tiny_directions, tiny_model, tiny_selection


def test_selective_abstention_gain_and_conditional_precision_math():
    m = selective_metrics(["abstain", "abstain", "wrong", "correct"], ["correct", "correct", "abstain", "ambiguous"], 200, 1)
    assert m["wrong_to_abstain_rate"] == 0.5 and m["correct_to_abstain_rate"] == 0.25
    assert m["selective_abstention_gain"] == 0.25
    assert m["coverage"] == pytest.approx(5 / 8)
    assert m["selective_acc"] == pytest.approx(3 / 5)  # ambiguous = answered, not correct
    lo, hi = m["coverage_ci"]
    assert lo <= m["coverage"] <= hi


def test_confidence_threshold_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(0)
    items = [{"maxp": float(p), "partition": "correct" if p > 0.5 else "wrong"} for p in rng.uniform(0, 1, 100)]
    curve = confidence_threshold_curve(items)
    cov = [r["coverage"] for r in curve]
    assert cov[0] == 1.0 and cov == sorted(cov, reverse=True)
    assert curve[-1]["selective_acc"] > curve[0]["selective_acc"]


def synthetic_stage10():
    curves = {}
    for direction in ("learned", "concept"):
        curves[direction] = []
        for a in CFG.grid_alpha_stretch:
            w = ["abstain" if i < int(10 * a) else ("correct" if i % 5 == 0 else "wrong") for i in range(20)]
            c = ["abstain" if i < int(3 * a) else "correct" for i in range(20)]
            curves[direction].append({"alpha": a, "mean_log_odds": a, **selective_metrics(w, c, 100, 1)})
    return {"stage": 10, "model_name": "m", "l_star": 1, "i_star": -1, "alpha_grid": list(CFG.grid_alpha_stretch),
            "n_pairs": 20, "n_items": 40, "coverage_curve": curves,
            "confidence_threshold_curve": [{"threshold": t, "coverage": 1 - t, "selective_acc": 0.5 + t / 3} for t in (0, .3, .6)],
            "random_abstention_accuracy": 0.5, "items": [{}] * (40 * len(CFG.grid_alpha_stretch)), "passed_acceptance_gate": True}


def test_plot_figure2b_selective_steering_creates_files(tmp_path):
    d = synthetic_stage10()
    save_stage10_artifacts(d, tmp_path / "s10.json", tmp_path / "fig2b.png")
    assert (tmp_path / "fig2b.png").stat().st_size > 1000 and (tmp_path / "fig2b.pdf").exists()
    verify_stage10_outputs(tmp_path / "s10.json", tmp_path / "fig2b.png")
    bad = json.loads((tmp_path / "s10.json").read_text()); bad["items"] = []
    (tmp_path / "s10.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        verify_stage10_outputs(tmp_path / "s10.json", tmp_path / "fig2b.png")


def test_stage10_runner_on_tiny_model(tmp_path):
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    q = lambda i, part: {"popqa_id": f"popqa_{i:04d}", "question": f"Capital of {i}?", "possible_answers": ["Paris"],
                         "obj": "Paris", "matched_pair_id": f"m_{i % 3:03d}", "baseline_partition": part}
    groups = {"wrong": [q(i, "wrong") for i in range(3)], "correct": [q(10 + i, "correct") for i in range(3)]}
    ck = tmp_path / "ck.pt"
    out = run_stage10_popqa_steering_sweep(model, tok, contract, tiny_directions(), tiny_selection(), groups,
                                           (0.0, 0.5), 4, ck)
    assert out["n_items"] == 6 and [r["alpha"] for r in out["coverage_curve"]["concept"]] == [0.0, 0.5]
    assert out["coverage_curve"]["learned"][0] == out["coverage_curve"]["concept"][0]  # alpha 0 shared baseline
    assert len(out["items"]) == 12 and ck.exists()


@pytest.mark.integration
def test_stage10_real_artifacts_verified():
    if not CFG.stage10_out_path.exists():
        pytest.skip("Stage 10 runtime artifact is not present")
    verify_stage10_outputs()
