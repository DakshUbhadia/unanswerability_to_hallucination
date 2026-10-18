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
from src.pilot_check import run_popqa_grading_and_power_gate, run_validation_pilot_gate_state_machine


def verify_stage2_outputs(stage2_json_path: Path, popqa_jsonl_path: Path) -> bool:
    with Path(stage2_json_path).open(encoding="utf-8") as f: summary=json.load(f)
    required={"attempt_number","model_name","system_prompt_used","val_unans_abstention_rate","val_ans_abstention_rate","val_rate_gap","val_unans_abstention_rate_strict","val_ans_abstention_rate_strict","val_unans_mean_log_odds","val_ans_mean_log_odds","attempt_history","popqa_evaluated_count","popqa_correct_count","popqa_wrong_count","popqa_abstain_count","popqa_matched_pair_count","caliper_widened","caliper_width_s_pop","passed_pilot_and_power_gate"}
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
    with Path(popqa_jsonl_path).open(encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
    assert len(rows)==summary["popqa_evaluated_count"] and len(rows) in (1000,1250,1500,1750,2000)
    assert len({r["hf_row_idx"] for r in rows})==len(rows)
    counts={k:sum(r.get("baseline_partition")==k for r in rows) for k in ("correct","wrong","abstain")}
    assert all(r.get("baseline_response") and r["baseline_partition"] in counts for r in rows)
    assert counts["correct"]==summary["popqa_correct_count"] and counts["wrong"]==summary["popqa_wrong_count"] and counts["abstain"]==summary["popqa_abstain_count"]
    pairs={}
    for r in rows:
        if r.get("is_popularity_matched"):
            assert r.get("matched_pair_id") is not None
            pairs.setdefault(r["matched_pair_id"],[]).append(r)
        else: assert r.get("matched_pair_id") is None
    assert len(pairs)==summary["popqa_matched_pair_count"] and all(len(v)==2 for v in pairs.values())
    assert sum(bool(r.get("is_popularity_matched")) for r in rows)==2*len(pairs)
    assert summary["passed_pilot_and_power_gate"] is True
    assert summary["popqa_matched_pair_count"]>=100 or (len(rows)==2000 and summary["caliper_widened"] and summary["popqa_matched_pair_count"]>0)
    return True


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f: json.dump(payload,f,indent=2,ensure_ascii=False); f.write("\n")
    tmp.replace(path)


def main() -> int:
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",default="auto")
    parser.add_argument("--pairs-path",default=str(CFG.data_dir/"pairs.jsonl"))
    parser.add_argument("--popqa-path",default=str(CFG.data_dir/"popqa_sample.jsonl"))
    parser.add_argument("--stage0-path",default=str(CFG.stage0_out_path))
    parser.add_argument("--output-path",default=str(CFG.stage2_out_path))
    args=parser.parse_args(); out=Path(args.output_path); poppath=Path(args.popqa_path)
    if not args.force and out.exists() and poppath.exists():
        try:
            verify_stage2_outputs(out,poppath); print(f"[SKIP] Stage 2 already verified in {out}"); return 0
        except (AssertionError,RuntimeError,ValueError,KeyError): pass
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
        pilot.update(popqa_evaluated_count=0,popqa_correct_count=0,popqa_wrong_count=0,popqa_abstain_count=0,popqa_matched_pair_count=0,caliper_widened=False,caliper_width_s_pop=0.0,passed_pilot_and_power_gate=False)
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
