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
    compute_log10_s_pop, grade_popqa, match_popqa_by_popularity_caliper,
    run_popqa_grading_and_power_gate, run_validation_pilot_gate_state_machine,
    evaluate_prompts_log_odds_and_generations,
)
from src.abstention_classifier import classify_texts, require_judge_validation, require_low_margin_review
from src.tokenization_contract import CONTRACT_VERSION, canonical_input_ids, chat_template_sha256, input_ids_sha256


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
    if summary.get("label_source") == "canonical_v1_batch1_judged":
        from src.abstention_classifier import is_final_classifier_record
        bad_hash = [r.get("popqa_id") for r in rows if not r.get("input_ids_sha256") or r.get("tokenization_contract_version") != CONTRACT_VERSION]
        if bad_hash:
            raise AssertionError(f"Canonical Stage 2 hash/contract missing for PopQA IDs: {bad_hash}")
        if any(r.get("_generation_batch_size") != 1 for r in rows):
            raise AssertionError("Canonical Stage 2 generations must use batch size 1")
        if any(not r.get("_v4_baseline_response") or not r.get("_v4_baseline_partition") for r in rows):
            raise AssertionError("Canonical Stage 2 rows must preserve v4 response and partition")
        if any(not is_final_classifier_record(r.get("classifier", {})) for r in rows):
            raise AssertionError("Stage 2 verifier rejects non-final classifier records")
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


def _generate_canonical_popqa(model, tokenizer, rows, model_name, system_prompt, suffix_token_ids=None, offline=False):
    import torch
    template_hash = chat_template_sha256(tokenizer)
    for row in rows:
        if not row.get("baseline_response"):
            raise RuntimeError(f"Missing v4 baseline response for {row.get('popqa_id')}")
        row.setdefault("_v4_baseline_response", row["baseline_response"])
        row.setdefault("_v4_baseline_partition", row.get("baseline_partition"))
        row.setdefault("_v4_is_popularity_matched", bool(row.get("is_popularity_matched")))
        row.setdefault("_v4_matched_pair_id", row.get("matched_pair_id"))
        ids = canonical_input_ids(tokenizer, row["question"], system_prompt, suffix_token_ids)
        digest = input_ids_sha256(ids, model_name, template_hash)
        device = next(model.parameters()).device
        encoded = {"input_ids": torch.tensor([ids], dtype=torch.long, device=device),
                   "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
                   "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=16, do_sample=False, num_beams=1,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
        row["baseline_response"] = response
        row["input_ids_sha256"] = digest
        row["chat_template_sha256"] = template_hash
        row["tokenization_contract_version"] = CONTRACT_VERSION
        row["_generation_batch_size"] = 1
        row["label_source"] = "canonical_v1_batch1_unjudged"
        row["_stage2_model_name"] = model_name
        row["_stage2_system_prompt"] = system_prompt
        row["is_popularity_matched"], row["matched_pair_id"] = False, None
        row["classifier"] = None
        row["final_label"] = None
        row["grade"], row["grade_reason"], row["baseline_partition"] = None, None, None
    return rows


def _load_locked_model(stage0_path, stage2_summary, device, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.model_utils import resolve_runtime_dtype
    env = json.loads(Path(stage0_path).read_text(encoding="utf-8"))
    model_name = stage2_summary["model_name"]
    dtype, _, _ = resolve_runtime_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device,
                                                 low_cpu_mem_usage=True).eval()
    return model, tokenizer, env


def _rematch_saved(rows, summary, output_path, popqa_path, offline=False):
    require_judge_validation()
    for row in rows:
        record = row.get("classifier")
        if not record or record.get("response") != row.get("baseline_response"):
            record = classify_texts([{"uid": row["popqa_id"], "question": row["question"],
                                      "response": row["baseline_response"]}], offline=offline)[0]
        row["classifier"] = record
        grade = grade_popqa(row["baseline_response"], row.get("possible_answers", []), row.get("obj", ""), record)
        row["grade"], row["grade_reason"], row["baseline_partition"] = grade["grade"], grade["reason"], grade["grade"]
    rows, matched, width = match_popqa_by_popularity_caliper(rows, CFG.caliper_sd_default, CFG.seed_popqa)
    if matched < CFG.n_popqa_matched_min:
        raise RuntimeError(f"Branch Rule 5 requires top-up generation: only {matched} matched pairs. Run on GPU with the canonical top-up path.")
    counts = {k: sum(r["baseline_partition"] == k for r in rows) for k in ("correct", "wrong", "abstain", "ambiguous")}
    for key in ("popqa_evaluated_count", "popqa_correct_count", "popqa_wrong_count", "popqa_abstain_count",
                "popqa_ambiguous_count", "popqa_matched_pair_count", "caliper_width_s_pop", "caliper_widened"):
        summary.setdefault("v4_" + key, summary.get(key))
    summary.update(popqa_evaluated_count=len(rows), popqa_correct_count=counts["correct"],
                   popqa_wrong_count=counts["wrong"], popqa_abstain_count=counts["abstain"],
                   popqa_ambiguous_count=counts["ambiguous"], popqa_matched_pair_count=matched,
                   caliper_width_s_pop=width, caliper_widened=False, label_source="canonical_v1_batch1_judged",
                   passed_pilot_and_power_gate=bool(summary.get("passed_pilot_gate") and matched >= CFG.n_popqa_matched_min))
    for row in rows:
        row["label_source"] = "canonical_v1_batch1_judged"
    _atomic_jsonl(popqa_path, rows)
    _atomic_json(output_path, summary)
    return matched, counts, width


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--recompute-matching-only", action="store_true", help="Re-run log10 caliper matching on existing graded PopQA rows on CPU without loading the LLM.")
    parser.add_argument("--regrade-saved-popqa-only", action="store_true", help="Re-grade saved PopQA responses and re-match on CPU without generating new responses.")
    parser.add_argument("--regenerate-popqa-canonical", action="store_true", help="Regenerate existing PopQA rows from canonical inputs at batch size 1; requires a GPU; judge labels are added later in the offline judge pass.")
    parser.add_argument("--rematch-only", action="store_true", help="Regrade canonical saved responses and rematch on CPU.")
    parser.add_argument("--diagnostic-val-pilot", action="store_true", help="Re-measure the locked validation pilot without changing Stage 2 or model choice.")
    parser.add_argument("--offline", action="store_true", help="Require complete judge cache; never call the judge backend.")
    parser.add_argument("--device",default=None)
    parser.add_argument("--dtype",default="auto")
    parser.add_argument("--pairs-path",default=str(CFG.data_dir/"pairs.jsonl"))
    parser.add_argument("--popqa-path",default=str(CFG.data_dir/"popqa_sample.jsonl"))
    parser.add_argument("--stage0-path",default=str(CFG.stage0_out_path))
    parser.add_argument("--output-path",default=str(CFG.stage2_out_path))
    args=parser.parse_args(); out=Path(args.output_path); poppath=Path(args.popqa_path)
    stage0_path=Path(args.stage0_path);pairs_path=Path(args.pairs_path)
    special_modes = [args.recompute_matching_only, args.regrade_saved_popqa_only, args.regenerate_popqa_canonical,
                     args.rematch_only, args.diagnostic_val_pilot]
    if sum(bool(x) for x in special_modes) > 1:
        parser.error("Choose only one Stage 2 special mode")
    # Explicit modes must never be skipped merely because legacy artifacts pass
    # the ordinary Stage 2 verifier. Canonical regeneration/rematching are repair
    # actions and are specifically used to upgrade old rows in place.
    if not any(special_modes) and not args.force and out.exists() and poppath.exists():
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
    if args.rematch_only:
        if not args.offline:
            raise RuntimeError("--rematch-only must use --offline after scripts/judge_pass.py has populated the local cache")
        require_judge_validation()
        grader_validation = CFG.results_dir / "grader_validation.json"
        if not grader_validation.exists() or json.loads(grader_validation.read_text(encoding="utf-8")).get("gate_passed") is not True:
            raise RuntimeError("Final PopQA rematching requires a passing held-out grader validation in results/grader_validation.json")
        if not out.exists() or not poppath.exists():
            raise FileNotFoundError("Stage 2 summary and canonical PopQA rows are required")
        summary=json.loads(out.read_text(encoding="utf-8"))
        with poppath.open(encoding="utf-8") as f: rows=[json.loads(line) for line in f if line.strip()]
        matched, counts, width = _rematch_saved(rows, summary, out, poppath, args.offline)
        print(f"Rematched {matched} pairs; counts={counts}; caliper={width:.6f}")
        return 0
    if args.regenerate_popqa_canonical or args.diagnostic_val_pilot:
        if not out.exists() or not stage0_path.exists():
            raise FileNotFoundError("Locked Stage 2 and Stage 0 outputs are required")
        summary=json.loads(out.read_text(encoding="utf-8"))
        if args.device in (None, "cuda", "cuda:0"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("This Stage 2 mode requires a CUDA GPU; no GPU is available. No files were changed.")
        model, tokenizer, env = _load_locked_model(stage0_path, summary, args.device or "cuda:0", args.dtype)
        if args.regenerate_popqa_canonical:
            with poppath.open(encoding="utf-8") as f: rows=[json.loads(line) for line in f if line.strip()]
            _generate_canonical_popqa(model, tokenizer, rows, summary["model_name"], summary["system_prompt_used"],
                                      env.get("suffix_token_ids"), args.offline)
            _atomic_jsonl(poppath, rows)
            print(f"Regenerated {len(rows)} PopQA rows canonically at batch size 1; no matching performed")
            return 0
        with Path(args.pairs_path).open(encoding="utf-8") as f: pairs=[json.loads(line) for line in f if line.strip() and json.loads(line).get("split")=="val"]
        ans = evaluate_prompts_log_odds_and_generations(model, tokenizer, [p["ans_prompt"] for p in pairs], summary["system_prompt_used"], [], run_log_odds=False, offline=args.offline,
                                                    suffix_token_ids=env.get("suffix_token_ids"))
        unans = evaluate_prompts_log_odds_and_generations(model, tokenizer, [p["unans_prompt"] for p in pairs], summary["system_prompt_used"], [], run_log_odds=False, offline=args.offline,
                                                    suffix_token_ids=env.get("suffix_token_ids"))
        record = {"model_name":summary["model_name"],"system_prompt_used":summary["system_prompt_used"],
                  "n":len(pairs),"ans_abstention_rate_judged":sum(ans["abstained_judged"])/len(pairs),
                  "unans_abstention_rate_judged":sum(unans["abstained_judged"])/len(pairs),
                  "ans_degenerate_rate":sum(ans["degenerate"])/len(pairs),"unans_degenerate_rate":sum(unans["degenerate"])/len(pairs),
                  "ans_abstention_rate_regex_v4":sum(ans["abstained_regex"])/len(pairs),
                  "unans_abstention_rate_regex_v4":sum(unans["abstained_regex"])/len(pairs),"classifier_final":all(r["classifier_final"] for r in ans["classifier_records"]+unans["classifier_records"])}
        _atomic_json(CFG.results_dir / "stage2_val_pilot_diagnostic_v42.json", record)
        print(f"Wrote locked-model diagnostic to {CFG.results_dir / 'stage2_val_pilot_diagnostic_v42.json'}")
        return 0
    if args.regrade_saved_popqa_only:
        if not args.offline:
            raise RuntimeError("--regrade-saved-popqa-only must use --offline after scripts/judge_pass.py")
        if not out.exists() or not poppath.exists():
            raise FileNotFoundError("Existing Stage 2 summary and PopQA rows are required")
        summary=json.loads(out.read_text(encoding="utf-8"))
        with poppath.open(encoding="utf-8") as f: rows=[json.loads(line) for line in f if line.strip()]
        matched, counts, width = _rematch_saved(rows, summary, out, poppath, args.offline)
        print(f"Saved-response regrade counts={counts}; matched_pairs={matched}; caliper={width:.6f}")
        return 0
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
    stage0_path,pairs_path=Path(args.stage0_path),Path(args.pairs_path)
    if not stage0_path.exists(): raise FileNotFoundError(f"Missing {stage0_path}; run Stage 0 first")
    if not pairs_path.exists(): raise FileNotFoundError(f"Missing {pairs_path}; run Stage 1 first")
    env=json.loads(stage0_path.read_text(encoding="utf-8"))
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
    if env["model_name"]!=json.loads(stage0_path.read_text(encoding="utf-8"))["model_name"]: _atomic_json(stage0_path,env)
    if not pilot["passed_pilot_gate"]:
        pilot.update(popqa_evaluated_count=0,popqa_correct_count=0,popqa_wrong_count=0,popqa_abstain_count=0,popqa_ambiguous_count=0,popqa_matched_pair_count=0,caliper_widened=False,caliper_width_s_pop=0.0,passed_pilot_and_power_gate=False)
        _atomic_json(out,pilot)
        if pilot.get("fallback_access_error"):
            print(f"[BLOCKED] {pilot['fallback_access_error']}", file=sys.stderr)
        else:
            print("[FAIL] Pilot gate did not pass after all configured attempts.", file=sys.stderr)
        return 2
    power=run_popqa_grading_and_power_gate(model,tokenizer,pilot["system_prompt_used"],[env["unknown_token_id_nospace"],env["unknown_token_id_space"]],poppath,args.force,args.offline,env.get("suffix_token_ids"))
    summary={**pilot,**power,"passed_pilot_and_power_gate":bool(pilot["passed_pilot_gate"] and power["power_gate_passed"])}
    _atomic_json(out,summary)
    verify_stage2_outputs(out,poppath)
    print(f"Pilot attempt: {summary['attempt_number']} | PopQA: {summary['popqa_evaluated_count']} | matched: {summary['popqa_matched_pair_count']} | passed: {summary['passed_pilot_and_power_gate']}")
    return 0 if summary["passed_pilot_and_power_gate"] else 2


if __name__=="__main__": raise SystemExit(main())
