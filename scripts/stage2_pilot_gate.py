#!/usr/bin/env python3
"""Run Stage 2 pilot and PopQA power gates."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.pilot_check import (
    compute_log10_s_pop,
    match_popqa_by_popularity_caliper,
    normalize_and_grade_popqa_response,
    run_popqa_grading_and_power_gate,
    run_validation_pilot_gate_state_machine,
)


def verify_stage2_outputs(stage2_json_path: Path, popqa_jsonl_path: Path) -> bool:
    with Path(stage2_json_path).open(encoding="utf-8") as f: summary=json.load(f)
    required={"attempt_number","model_name","system_prompt_used","val_unans_abstention_rate","val_ans_abstention_rate","val_rate_gap","val_unans_abstention_rate_strict","val_ans_abstention_rate_strict","val_unans_mean_log_odds","val_ans_mean_log_odds","attempt_history","popqa_evaluated_count","popqa_correct_count","popqa_wrong_count","popqa_abstain_count","popqa_ambiguous_count","popqa_matched_pair_count","caliper_widened","caliper_width_s_pop","passed_pilot_and_power_gate"}
    assert required <= summary.keys(), f"Missing Stage 2 keys: {required-summary.keys()}"
    assert summary["attempt_number"] in (1,2,3)
    assert 0.25 <= summary["val_unans_abstention_rate"] <= .95
    assert 0 <= summary["val_ans_abstention_rate"] <= .15
    assert math.isclose(summary["val_rate_gap"],summary["val_unans_abstention_rate"]-summary["val_ans_abstention_rate"],abs_tol=1e-6)
    assert summary["val_rate_gap"] >= .25
    assert 0 <= summary["val_unans_abstention_rate_strict"] <= summary["val_unans_abstention_rate"]
    assert 0 <= summary["val_ans_abstention_rate_strict"] <= summary["val_ans_abstention_rate"]
    assert math.isfinite(summary["val_unans_mean_log_odds"]) and math.isfinite(summary["val_ans_mean_log_odds"])
    assert summary["val_ans_mean_log_odds"] < summary["val_unans_mean_log_odds"]
    assert 0.0 < float(summary["caliper_width_s_pop"]) < 5.0
    with Path(popqa_jsonl_path).open(encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
    assert len(rows)==summary["popqa_evaluated_count"] and len(rows) in (1000,1250,1500,1750,2000)
    assert len({r["hf_row_idx"] for r in rows})==len(rows)
    counts={k:sum(r.get("baseline_partition")==k for r in rows) for k in ("correct","wrong","abstain","ambiguous")}
    assert all(r.get("baseline_response") and r["baseline_partition"] in counts for r in rows)
    assert counts["correct"]==summary["popqa_correct_count"] and counts["wrong"]==summary["popqa_wrong_count"] and counts["abstain"]==summary["popqa_abstain_count"]
    assert counts["ambiguous"]==summary["popqa_ambiguous_count"]
    assert sum(counts.values()) == len(rows)
    pairs={}
    for r in rows:
        if r.get("is_popularity_matched"):
            assert r.get("matched_pair_id") is not None
            pairs.setdefault(r["matched_pair_id"],[]).append(r)
        else: assert r.get("matched_pair_id") is None
    assert len(pairs)==summary["popqa_matched_pair_count"] and all(len(v)==2 for v in pairs.values())
    assert sum(bool(r.get("is_popularity_matched")) for r in rows)==2*len(pairs)
    for pair_rows in pairs.values():
        assert {r["baseline_partition"] for r in pair_rows} == {"wrong", "correct"}
        distance = abs(compute_log10_s_pop(pair_rows[0]["s_pop"]) - compute_log10_s_pop(pair_rows[1]["s_pop"]))
        assert distance <= float(summary["caliper_width_s_pop"]) + 1e-5
    assert summary["passed_pilot_and_power_gate"] is True
    assert summary["popqa_matched_pair_count"]>=100 or (len(rows)==2000 and summary["caliper_widened"] and summary["popqa_matched_pair_count"]>0)
    return True


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f: json.dump(payload,f,indent=2,ensure_ascii=False); f.write("\n")
    tmp.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--recompute-matching-only", action="store_true", help="Re-run log10 caliper matching on existing graded PopQA rows on CPU without loading the LLM.")
    parser.add_argument("--regrade-saved-popqa-only", action="store_true", help="Re-grade saved PopQA responses and re-match on CPU without generating new responses.")
    parser.add_argument("--device",default=None)
    parser.add_argument("--dtype",default="auto")
    parser.add_argument("--pairs-path",default=str(CFG.data_dir/"pairs.jsonl"))
    parser.add_argument("--popqa-path",default=str(CFG.data_dir/"popqa_sample.jsonl"))
    parser.add_argument("--stage0-path",default=str(CFG.stage0_out_path))
    parser.add_argument("--output-path",default=str(CFG.stage2_out_path))
    args=parser.parse_args(); out=Path(args.output_path); poppath=Path(args.popqa_path)
    if args.recompute_matching_only and args.regrade_saved_popqa_only:
        parser.error("Choose only one of --recompute-matching-only and --regrade-saved-popqa-only")
    if not args.force and not args.recompute_matching_only and not args.regrade_saved_popqa_only and out.exists() and poppath.exists():
        try:
            verify_stage2_outputs(out,poppath)
            print(f"[SKIP] Stage 2 already verified in {out}")
            return 0
        except (AssertionError,RuntimeError,ValueError,KeyError):
            # Schema upgrades such as the saved-response grader correction can
            # be applied without loading a model or regenerating responses.
            old_summary=json.loads(out.read_text(encoding="utf-8"))
            with poppath.open(encoding="utf-8") as f:
                old_rows=[json.loads(line) for line in f if line.strip()]
            if old_summary.get("passed_pilot_gate") is True and old_rows and all(r.get("baseline_response") for r in old_rows):
                args.regrade_saved_popqa_only=True
    if args.regrade_saved_popqa_only:
        if not out.exists() or not poppath.exists():
            raise FileNotFoundError("Existing Stage 2 summary and graded PopQA rows are required for CPU-only re-grading")
        summary=json.loads(out.read_text(encoding="utf-8"))
        with poppath.open(encoding="utf-8") as f:
            rows=[json.loads(line) for line in f if line.strip()]
        before={key:sum(r.get("baseline_partition")==key for r in rows) for key in ("correct","wrong","abstain","ambiguous")}
        for row in rows:
            response=row.get("baseline_response", "")
            if not response:
                raise RuntimeError(f"Missing saved baseline_response for {row.get('popqa_id', '<unknown row>')}")
            row["baseline_partition"]=normalize_and_grade_popqa_response(response,row.get("possible_answers",[]),row.get("obj",""))
        rows,matched,width=match_popqa_by_popularity_caliper(rows,CFG.caliper_sd_default,CFG.seed_popqa)
        after={key:sum(r["baseline_partition"]==key for r in rows) for key in ("correct","wrong","abstain","ambiguous")}
        power_passed=matched>=CFG.n_popqa_matched_min
        summary.update(popqa_correct_count=after["correct"],popqa_wrong_count=after["wrong"],popqa_abstain_count=after["abstain"],
            popqa_ambiguous_count=after["ambiguous"],popqa_matched_pair_count=matched,caliper_widened=False,
            caliper_width_s_pop=width,power_gate_passed=power_passed,
            passed_pilot_and_power_gate=bool(summary["passed_pilot_gate"] and power_passed),
            grading_revision="generic-tail and ambiguous partial-alias handling")
        _atomic_jsonl(poppath,rows)
        _atomic_json(out,summary)
        verify_stage2_outputs(out,poppath)
        print(f"Saved-response regrade counts: before={before}; after={after}; matched_pairs={matched}; caliper={width:.6f}")
        return 0 if summary["passed_pilot_and_power_gate"] else 2
    if args.recompute_matching_only:
        if not out.exists():
            raise FileNotFoundError(f"Missing {out}; Stage 2 pilot results must exist before recomputing matches")
        if not poppath.exists():
            raise FileNotFoundError(f"Missing {poppath}; graded PopQA rows must exist before recomputing matches")
        summary=json.loads(out.read_text(encoding="utf-8"))
        with poppath.open(encoding="utf-8") as f:
            rows=[json.loads(line) for line in f if line.strip()]
        if any(r.get("baseline_partition") not in ("correct", "wrong", "abstain", "ambiguous") or not r.get("baseline_response") for r in rows):
            raise RuntimeError("PopQA rows must all be graded before matching-only recomputation")
        rows,matched,width=match_popqa_by_popularity_caliper(rows, caliper_sd_mult=CFG.caliper_sd_default, seed=CFG.seed_popqa)
        counts={key:sum(r["baseline_partition"]==key for r in rows) for key in ("correct","wrong","abstain","ambiguous")}
        power_passed=matched >= CFG.n_popqa_matched_min
        summary.update(popqa_matched_pair_count=matched, caliper_widened=False,
            caliper_width_s_pop=width, power_gate_passed=power_passed,
            popqa_correct_count=counts["correct"],popqa_wrong_count=counts["wrong"],
            popqa_abstain_count=counts["abstain"],popqa_ambiguous_count=counts["ambiguous"],
            passed_pilot_and_power_gate=bool(summary["passed_pilot_gate"] and power_passed))
        _atomic_jsonl(poppath,rows)
        _atomic_json(out,summary)
        verify_stage2_outputs(out,poppath)
        print(f"Recomputed log10 popularity matching: {matched} pairs; caliper width={width:.6f}")
        return 0
    import torch
    if args.device is None:
        args.device="cuda:0" if torch.cuda.is_available() else "cpu"
    stage0,pairs_path=Path(args.stage0_path),Path(args.pairs_path)
    if not stage0.exists(): raise FileNotFoundError(f"Missing {stage0}; run Stage 0 first")
    if not pairs_path.exists(): raise FileNotFoundError(f"Missing {pairs_path}; run Stage 1 first")
    env=json.loads(stage0.read_text(encoding="utf-8"))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model_utils import resolve_runtime_dtype
    dtype,_,_=resolve_runtime_dtype(args.dtype,args.device)
    tokenizer=AutoTokenizer.from_pretrained(env["model_name"]); tokenizer.padding_side="left"
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(env["model_name"],torch_dtype=dtype,device_map=args.device,low_cpu_mem_usage=True).eval()
    with pairs_path.open(encoding="utf-8") as f:
        all_pairs=[json.loads(line) for line in f if line.strip()]
    val=[pair for pair in all_pairs if pair.get("split")=="val"]
    if len(val)!=96: raise RuntimeError(f"Expected 96 validation pairs, found {len(val)}")
    model,tokenizer,env,pilot=run_validation_pilot_gate_state_machine(model,tokenizer,env,val,args.device,args.dtype)
    if env["model_name"]!=json.loads(stage0.read_text(encoding="utf-8"))["model_name"]: _atomic_json(stage0,env)
    if not pilot["passed_pilot_gate"]:
        pilot.update(popqa_evaluated_count=0,popqa_correct_count=0,popqa_wrong_count=0,popqa_abstain_count=0,popqa_ambiguous_count=0,popqa_matched_pair_count=0,caliper_widened=False,caliper_width_s_pop=0.0,passed_pilot_and_power_gate=False)
        _atomic_json(out,pilot)
        if pilot.get("fallback_access_error"):
            print(f"[BLOCKED] {pilot['fallback_access_error']}", file=sys.stderr)
        else:
            print("[FAIL] Pilot gate did not pass after all configured attempts.", file=sys.stderr)
        return 2
    power=run_popqa_grading_and_power_gate(model,tokenizer,pilot["system_prompt_used"],[env["unknown_token_id_nospace"],env["unknown_token_id_space"]],poppath,args.force)
    summary={**pilot,**power,"passed_pilot_and_power_gate":bool(pilot["passed_pilot_gate"] and power["power_gate_passed"])}
    _atomic_json(out,summary)
    verify_stage2_outputs(out,poppath)
    print(f"Pilot attempt: {summary['attempt_number']} | PopQA: {summary['popqa_evaluated_count']} | matched: {summary['popqa_matched_pair_count']} | passed: {summary['passed_pilot_and_power_gate']}")
    return 0 if summary["passed_pilot_and_power_gate"] else 2


if __name__=="__main__": raise SystemExit(main())
