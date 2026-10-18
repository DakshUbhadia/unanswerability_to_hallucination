import json
import math
from pathlib import Path

import pytest

from src.config import CFG
from src.pilot_check import (build_calibrated_system_prompt, classify_abstention_response,
    compute_log10_s_pop, determine_pilot_gate_status, match_popqa_by_popularity_caliper,
    normalize_and_grade_popqa_response, normalize_popqa_text)
from scripts.stage2_pilot_gate import verify_stage2_outputs
from src.stage_contract import resolve_downstream_model_name


def test_classify_abstention_strict_vs_regex():
    for text in ("Unknown", "Unknown.", "  unknown\n", "Unknown: The context does not state this."):
        assert classify_abstention_response(text) == (True, True)
    for text in ("I don't know.", "Cannot be determined from the text.", "There is insufficient information.", "Not mentioned in the profile.", "Unspecified"):
        assert classify_abstention_response(text) == (False, True)
    for text in ("Dr. Aris Thorne", "1984", "Unknown Soldier Memorial", "Unknown Mortal Orchestra"):
        assert classify_abstention_response(text) == (False, False)
    assert classify_abstention_response("") == (False, False)


def test_build_calibrated_system_prompt_all_modes():
    assert build_calibrated_system_prompt("base", "none") == "base"
    assert build_calibrated_system_prompt("base", "under_abstention") == "base" + CFG.under_abstention_clause
    assert build_calibrated_system_prompt("base", "over_abstention") == "base" + CFG.over_abstention_clause
    assert build_calibrated_system_prompt("base", "narrow_gap") == "base" + CFG.under_abstention_clause + " " + CFG.over_abstention_clause
    with pytest.raises(ValueError): build_calibrated_system_prompt("base", "nope")


def test_determine_pilot_gate_status_boundaries():
    assert determine_pilot_gate_status(.25, 0) == (True, .25, "none")
    assert determine_pilot_gate_status(.95, .15) == (True, .8, "none")
    assert determine_pilot_gate_status(.35, .12)[::2] == (False, "narrow_gap")
    assert determine_pilot_gate_status(.20, .05)[::2] == (False, "under_abstention")
    assert determine_pilot_gate_status(.60, .20)[::2] == (False, "over_abstention")
    assert determine_pilot_gate_status(.15, .25)[::2] == (False, "narrow_gap")


def test_popqa_normalization_and_word_boundary_grading():
    assert normalize_popqa_text("The United Kingdom!") == "united kingdom"
    assert normalize_and_grade_popqa_response("The company earned a million dollars", ["on", "io", "ll"]) == "wrong"
    assert normalize_and_grade_popqa_response("The United Kingdom!", ["united-kingdom", "UK"]) == "correct"
    assert normalize_and_grade_popqa_response("Unknown", ["unknown"]) == "abstain"
    assert normalize_and_grade_popqa_response("Horror", ["horror film"]) == "correct"
    assert normalize_and_grade_popqa_response("Hockey", ["ice hockey"]) == "ambiguous"


def test_compute_log10_s_pop_scale_invariance():
    assert compute_log10_s_pop(99) == 2.0
    assert compute_log10_s_pop(9999) == 4.0
    assert compute_log10_s_pop(-10) == 0.0
    rows=[]
    for i, value in enumerate((10, 100, 1_000, 10_000, 100_000, 1_000_000)):
        rows.append({"popqa_id":f"popqa_{i+1:04d}", "s_pop":value,
            "baseline_partition":"wrong" if i % 2 == 0 else "correct"})
    _, _, width=match_popqa_by_popularity_caliper(rows)
    assert 0.0 < width < 5.0


def test_downstream_model_contract_uses_stage2_model(tmp_path):
    stage2_path=tmp_path/"stage2.json"
    stage2_path.write_text(json.dumps({"model_name":"meta-llama/Llama-3.2-3B-Instruct",
        "system_prompt_used":"locked prompt", "passed_pilot_and_power_gate":True}),encoding="utf-8")
    assert resolve_downstream_model_name(stage2_path=stage2_path)=="meta-llama/Llama-3.2-3B-Instruct"
    assert resolve_downstream_model_name("meta-llama/Llama-3.2-3B-Instruct",stage2_path)=="meta-llama/Llama-3.2-3B-Instruct"
    with pytest.raises(RuntimeError,match="differs from Stage 2 locked model"):
        resolve_downstream_model_name(CFG.primary_model_name,stage2_path)


def test_caliper_matching_global_optimality_and_determinism():
    rows=[]
    # Dense overlapping locations create a large one-to-one matching.
    for i in range(120): rows.append({"popqa_id":f"popqa_{i+1:04d}","s_pop":float(i),"baseline_partition":"wrong","is_popularity_matched":False,"matched_pair_id":None})
    for i in range(120): rows.append({"popqa_id":f"popqa_{i+121:04d}","s_pop":float(i)+.1,"baseline_partition":"correct","is_popularity_matched":False,"matched_pair_id":None})
    for i in range(10): rows.append({"popqa_id":f"popqa_{i+241:04d}","s_pop":float(i),"baseline_partition":"abstain","is_popularity_matched":False,"matched_pair_id":None})
    a,n,width=match_popqa_by_popularity_caliper([dict(r) for r in rows],seed=CFG.seed_popqa)
    b,n2,_=match_popqa_by_popularity_caliper([dict(r) for r in rows],seed=CFG.seed_popqa)
    assert a==b and n==n2 and n>=100
    assert sum(r["is_popularity_matched"] for r in a)==2*n
    assert not any(r["is_popularity_matched"] for r in a if r["baseline_partition"]=="abstain")
    for pair_id in {r["matched_pair_id"] for r in a if r["matched_pair_id"]}:
        pair=[r for r in a if r["matched_pair_id"]==pair_id]
        assert len(pair)==2 and abs(compute_log10_s_pop(pair[0]["s_pop"])-compute_log10_s_pop(pair[1]["s_pop"]))<=width+1e-6


def test_caliper_widening_fallback_logic():
    rows=[]
    for i in range(100): rows.append({"popqa_id":f"w{i:03}","s_pop":float(i),"baseline_partition":"wrong"})
    for i in range(100): rows.append({"popqa_id":f"c{i:03}","s_pop":float(i)+5,"baseline_partition":"correct"})
    _,n1,w1=match_popqa_by_popularity_caliper([dict(r) for r in rows],.25)
    widened,n2,w2=match_popqa_by_popularity_caliper([dict(r) for r in rows],.5)
    assert n1 < 100 and n2 >= n1 and w2 == 2*w1
    assert all(abs(compute_log10_s_pop(a["s_pop"])-compute_log10_s_pop(b["s_pop"]))<=w2+1e-6 for a in widened if a["is_popularity_matched"] for b in widened if b["matched_pair_id"]==a["matched_pair_id"])


def _mock_artifacts(tmp_path):
    stage={"attempt_number":1,"model_name":CFG.primary_model_name,"system_prompt_used":CFG.system_prompt,
        "val_unans_abstention_rate":.5,"val_ans_abstention_rate":.1,"val_rate_gap":.4,
        "val_unans_abstention_rate_strict":.4,"val_ans_abstention_rate_strict":.05,
        "val_unans_mean_log_odds":1.,"val_ans_mean_log_odds":0.,"attempt_history":[],
        "popqa_evaluated_count":1000,"popqa_correct_count":500,"popqa_wrong_count":499,"popqa_abstain_count":1,"popqa_ambiguous_count":0,
        "popqa_matched_pair_count":100,"caliper_widened":False,"caliper_width_s_pop":4.99,"passed_pilot_and_power_gate":True}
    rows=[]
    for i in range(1000): rows.append({"popqa_id":f"popqa_{i+1:04d}","hf_row_idx":i,"baseline_response":"answer","baseline_partition":"correct" if i<500 else "wrong" if i<999 else "abstain","is_popularity_matched":False,"matched_pair_id":None,"s_pop":float(i)})
    for i in range(1,101):
        rows[i-1]["baseline_partition"]="wrong"; rows[i-1]["is_popularity_matched"]=True; rows[i-1]["matched_pair_id"]=f"m_{i:03d}"
        rows[i+499]["baseline_partition"]="correct"; rows[i+499]["is_popularity_matched"]=True; rows[i+499]["matched_pair_id"]=f"m_{i:03d}"
    sp=tmp_path/"stage2.json"; pp=tmp_path/"popqa.jsonl"
    sp.write_text(json.dumps(stage),encoding="utf-8"); pp.write_text("".join(json.dumps(r)+"\n" for r in rows),encoding="utf-8")
    return sp,pp,stage,rows


def test_verify_stage2_outputs_schema_and_invariants(tmp_path):
    sp,pp,stage,rows=_mock_artifacts(tmp_path)
    assert verify_stage2_outputs(sp,pp)
    stage.pop("model_name"); sp.write_text(json.dumps(stage),encoding="utf-8")
    with pytest.raises(AssertionError): verify_stage2_outputs(sp,pp)
    stage["model_name"]=CFG.primary_model_name; stage["popqa_correct_count"]-=1; sp.write_text(json.dumps(stage),encoding="utf-8")
    with pytest.raises(AssertionError): verify_stage2_outputs(sp,pp)


@pytest.mark.integration
def test_stage2_real_artifacts_on_repo():
    path=Path("results/stage2_pilot.json")
    if not path.exists(): pytest.skip("Stage 2 has not run on Kaggle")
    assert verify_stage2_outputs(path,Path("data/popqa_sample.jsonl"))
