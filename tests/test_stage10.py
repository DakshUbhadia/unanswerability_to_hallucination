import json

import numpy as np
import pytest

from src.config import CFG
from src.tokenization_contract import CONTRACT_VERSION, canonical_input_ids, chat_template_sha256, input_ids_sha256
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
    items = [{"popqa_id": f"id-{i}", "maxp": float(p), "partition": "correct" if p > 0.5 else "wrong"}
             for i, p in enumerate(rng.uniform(0, 1, 100))]
    curve = confidence_threshold_curve(items)
    cov = [r["coverage"] for r in curve]
    assert cov[0] == 1.0 and cov == sorted(cov, reverse=True)
    assert curve[-1]["selective_acc"] > curve[0]["selective_acc"]



def test_confidence_curve_rejects_different_item_order():
    with pytest.raises(ValueError, match="item IDs/order"):
        confidence_threshold_curve(["correct","wrong"],[.7,.4],["a","b"],["b","a"])


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
            "random_abstention_accuracy": 0.5, "alpha0_label_identity_passed": True, "classifier_final": True,
            "items": [{}] * (80 * len(CFG.grid_alpha_stretch)), "passed_acceptance_gate": True}


def test_plot_figure2b_selective_steering_creates_files(tmp_path):
    d = synthetic_stage10()
    save_stage10_artifacts(d, tmp_path / "s10.json", tmp_path / "fig2b.png")
    assert (tmp_path / "fig2b.png").stat().st_size > 1000 and (tmp_path / "fig2b.pdf").exists()
    verify_stage10_outputs(tmp_path / "s10.json", tmp_path / "fig2b.png")
    bad = json.loads((tmp_path / "s10.json").read_text()); bad["items"] = []
    (tmp_path / "s10.json").write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        verify_stage10_outputs(tmp_path / "s10.json", tmp_path / "fig2b.png")


def test_stage10_runner_on_tiny_model(tmp_path, monkeypatch):
    import src.popqa_steering as steering
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    template_hash = chat_template_sha256(tok)
    def q(i, part):
        answers = ["Paris"] if part == "correct" else ["London"]
        obj = "Paris" if part == "correct" else ""
        prompt = f"Capital of {i}?"
        ids = canonical_input_ids(tok, prompt, contract["system_prompt_used"], contract["suffix_token_ids"])
        return {"popqa_id": f"popqa_{i:04d}", "question": prompt, "possible_answers": answers,
                "obj": obj, "matched_pair_id": f"m_{i % 3:03d}", "baseline_partition": part,
                "input_ids_sha256": input_ids_sha256(ids, contract["model_name"], template_hash),
                "tokenization_contract_version": CONTRACT_VERSION}
    groups = {"wrong": [q(i, "wrong") for i in range(3)], "correct": [q(10 + i, "correct") for i in range(3)]}
    def generate(_model, _tokenizer, batches, _contract, _mode, _vec, _layer, alpha, _max_new_tokens):
        return ["Paris" if alpha == 0 else "Unknown" for _ in range(sum(b["input_ids"].shape[0] for b in batches))]
    def classify(items, offline=False):
        records=[]
        for item in items:
            final = "ANSWER" if item["response"] == "Paris" else "ABSTAIN"
            records.append({**item, "regex_label": final, "judge_label": final, "final_label": final,
                            "agree": True, "classifier_final": True, "classifier_backend": "mock",
                            "judge_model_id": "mock", "judge_prompt_version": "v1",
                            "judge_prompt_sha256": "a" * 64, "cache_key": "mock", "parse_attempts": 1})
        return records
    monkeypatch.setattr(steering, "generate_with_hooks", generate)
    monkeypatch.setattr(steering, "classify_texts", classify)
    ck = tmp_path / "ck.pt"
    out = steering.run_stage10_popqa_steering_sweep(model, tok, contract, tiny_directions(), tiny_selection(), groups,
                                                    (0.0, 0.5), 4, ck)
    assert out["n_items"] == 6 and [r["alpha"] for r in out["coverage_curve"]["concept"]] == [0.0, 0.5]
    assert out["coverage_curve"]["learned"][0] == out["coverage_curve"]["concept"][0]
    assert len(out["items"]) == 24 and ck.exists()
    assert all(r["generation_batch_size"] == 1 for r in out["items"] if r["alpha"] == 0.0)



def test_stage10_alpha0_label_mismatch_writes_diagnostic(tmp_path, monkeypatch):
    from dataclasses import replace
    import src.popqa_steering as steering
    tok=StubTokenizer(); model=tiny_model(); contract=tiny_contract(tok)
    template_hash=chat_template_sha256(tok)
    def q(i,part):
        prompt=f"Capital of {i}?"
        ids=canonical_input_ids(tok,prompt,contract["system_prompt_used"],contract["suffix_token_ids"])
        return {"popqa_id":f"x{i}","question":prompt,"possible_answers":["Paris"],"obj":"Paris",
                "matched_pair_id":f"m{i}","baseline_partition":part,
                "input_ids_sha256":input_ids_sha256(ids,contract["model_name"],template_hash),
                "tokenization_contract_version":CONTRACT_VERSION}
    groups={"wrong":[q(1,"wrong")],"correct":[q(2,"correct")]}
    monkeypatch.setattr(steering,"CFG",replace(CFG,results_dir=tmp_path))
    monkeypatch.setattr(steering,"generate_with_hooks",lambda *a,**k:["Paris","Paris"])
    def classify(items,offline=False):
        return [{**x,"regex_label":"ANSWER","judge_label":"ABSTAIN" if x["uid"]=="x1" else "ANSWER",
                 "final_label":"ABSTAIN" if x["uid"]=="x1" else "ANSWER","agree":False,"classifier_final":True,
                 "classifier_backend":"mock","judge_model_id":"mock","judge_prompt_version":"v1",
                 "judge_prompt_sha256":"a"*64} for x in items]
    monkeypatch.setattr(steering,"classify_texts",classify)
    with pytest.raises(AssertionError,match="alpha=0"):
        steering.run_stage10_popqa_steering_sweep(model,tok,contract,tiny_directions(),tiny_selection(),groups,
                                                   (0.0,),2,tmp_path/"ck.pt")
    mismatch=tmp_path/"stage10_alpha0_mismatches.json"
    assert mismatch.exists() and json.loads(mismatch.read_text())[0]["popqa_id"]=="x1"


@pytest.mark.integration
def test_stage10_real_artifacts_verified():
    if not CFG.stage10_out_path.exists():
        pytest.skip("Stage 10 runtime artifact is not present")
    verify_stage10_outputs()
