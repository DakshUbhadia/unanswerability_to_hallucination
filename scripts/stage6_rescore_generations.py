#!/usr/bin/env python3
"""CPU-only judged re-score of the committed Stage 6 greedy generations."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.abstention_classifier import classify_texts, require_low_margin_review
from src.config import CFG
from src.eval_common import atomic_write_json, atomic_write_jsonl, read_json, read_jsonl, write_csv_and_md
from src.pilot_check import legacy_v4_abstention_flags
from src.stats_utils import holm_bonferroni
from src.test_evaluation import TABLE4_COLUMNS, cluster_bootstrap, generation_conditions, verify_stage6_outputs


def _log_odds_view(data):
    """Fields that the rescore is forbidden to change."""
    return {
        "per_prompt": data.get("per_prompt"),
        "curves": [{k: row.get(k) for k in ("mode", "alpha", "mean_log_odds", "log_odds_ci95",
                                               "mean_delta_log_odds", "delta_ci95", "first_token_abstention_rate")}
                   for curve_name in ("necessity_curve", "sufficiency_curve") for row in data[curve_name]],
        "audit": [{k: row.get(k) for k in ("condition", "mean_log_odds", "log_odds_ci_95_low",
                                             "log_odds_ci_95_high", "mean_p_unknown")}
                  for row in data["generation_audit"]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Fail on any judge-cache miss")
    parser.add_argument("--generation-path", type=Path, default=CFG.stage6_gen_path)
    parser.add_argument("--stage6-path", type=Path, default=CFG.stage6_out_path)
    parser.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    parser.add_argument("--b-ctrl-path", type=Path, default=CFG.data_dir / "b_ctrl.jsonl")
    parser.add_argument("--table4-csv-path", type=Path, default=CFG.table4_csv_path)
    parser.add_argument("--table4-md-path", type=Path, default=CFG.table4_md_path)
    a = parser.parse_args(argv)
    judge_validation = CFG.results_dir / "judge_validation.json"
    if not judge_validation.exists() or json.loads(judge_validation.read_text(encoding="utf-8")).get("gate_passed") is not True:
        raise RuntimeError("Stage 6 rescore requires a passing held-out judge validation in results/judge_validation.json")

    rows = read_jsonl(a.generation_path)
    test_conditions = {name for name, part, _ in generation_conditions() if part != "bctrl"}
    assert len(test_conditions) == 15, f"Expected 15 test conditions, found {len(test_conditions)}"
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    assert set(by_condition) == test_conditions | {"b_ctrl_baseline"}, "Unexpected/incomplete Stage 6 conditions"
    for name in test_conditions:
        assert len(by_condition[name]) == CFG.n_pairs_test, f"{name}: expected 96 prompts, found {len(by_condition[name])}"
    assert len(by_condition["b_ctrl_baseline"]) == CFG.n_b_ctrl
    assert len(rows) == 15 * CFG.n_pairs_test + CFG.n_b_ctrl == 1488

    pairs = {r["pair_id"]: r for r in read_jsonl(a.pairs_path)}
    bctrls = {r.get("b_ctrl_id", r["pair_id"]): r for r in read_jsonl(a.b_ctrl_path)}
    if len(pairs) != CFG.n_pairs_train + CFG.n_pairs_val + CFG.n_pairs_test:
        raise AssertionError(f"Unexpected pair table size: {len(pairs)}")
    input_items = []
    for i, row in enumerate(rows):
        part = row["dataset_part"]
        if part == "bctrl":
            context = bctrls.get(row.get("item_id")) or bctrls.get(row["pair_id"])
            if context is None:
                raise KeyError(f"No B-control prompt for {row.get('item_id')}")
            question = context["prompt"]
        else:
            pair = pairs[row["pair_id"]]
            question = pair["unans_prompt"] if part == "unans" else pair["ans_prompt"]
        input_items.append({"uid": f"{row['condition']}:{row['item_id']}", "question": question,
                            "response": row["generated_text"]})
    records = classify_texts(input_items, offline=a.offline)
    require_low_margin_review(records)
    records_by_condition = defaultdict(list)
    legacy_by_condition = defaultdict(list)
    tids_by_condition = defaultdict(list)
    for row, record in zip(rows, records):
        old = dict(row)
        row.setdefault("abstained_regex_v4", row.get("abstained_regex", False))
        row.setdefault("_v4_abstained_strict", row.get("abstained_strict", False))
        row.update(regex_label=record["regex_label"], judge_label=record["judge_label"],
                   final_label=record["final_label"], classifier_final=record["classifier_final"],
                   classifier_backend=record["classifier_backend"], judge_model_id=record["judge_model_id"],
                   judge_prompt_version=record["judge_prompt_version"], judge_model_revision=record["judge_model_revision"],
                   judge_quantization=record.get("judge_quantization"), judge_logprobs=record.get("judge_logprobs"),
                   judge_margin=record.get("judge_margin"),
                   judge_prompt_sha256=record["judge_prompt_sha256"], regex_judge_agree=record["agree"],
                   abstained_judged=record["final_label"] == "ABSTAIN",
                   degenerate=record["final_label"] == "DEGENERATE")
        records_by_condition[row["condition"]].append(record)
        legacy_by_condition[row["condition"]].append(legacy_v4_abstention_flags(row["generated_text"]))
        tids_by_condition[row["condition"]].append(row["template_id"])

    result = read_json(a.stage6_path)
    old_result = json.loads(json.dumps(result))
    for key in ("generation_rates", "generation_audit", "necessity_curve", "sufficiency_curve",
                "branch_rule4", "logit_gen_discrepancy"):
        result.setdefault(key + "_v4", old_result.get(key))
    rates = {}
    audit_by_name = {r["condition"]: r for r in result["generation_audit"]}
    for condition, condition_rows in by_condition.items():
        condition_records = records_by_condition[condition]
        labels = [r["final_label"] for r in condition_records]
        bad = labels.count("UNPARSEABLE")
        if bad / len(labels) > 0.01:
            raise RuntimeError(f"{condition}: {bad}/{len(labels)} UNPARSEABLE outputs exceeds 1%")
        valid_idx = [i for i, x in enumerate(labels) if x != "UNPARSEABLE"]
        judged = [labels[i] == "ABSTAIN" for i in valid_idx]
        degenerate = [labels[i] == "DEGENERATE" for i in valid_idx]
        legacy = legacy_by_condition[condition]
        strict = [x[0] for x in legacy]
        regex = [x[1] for x in legacy]
        agreement = [condition_records[i]["agree"] is True for i in valid_idx]
        rate = {"strict": sum(strict)/len(strict), "regex": sum(regex)/len(regex),
                "regex_v4": sum(regex)/len(regex), "judged": sum(judged)/len(judged),
                "degenerate": sum(degenerate)/len(degenerate),
                "regex_judge_agreement": sum(agreement)/len(agreement), "unparseable_count": bad}
        rates[condition] = rate
        audit = audit_by_name[condition]
        audit.update(gen_abstention_rate_strict=rate["strict"], gen_abstention_rate_regex=rate["regex_v4"],
                     gen_abstention_rate_regex_v4=rate["regex_v4"], gen_abstention_rate_judged=rate["judged"],
                     gen_degenerate_rate=rate["degenerate"], regex_judge_agreement=rate["regex_judge_agreement"],
                     unparseable_count=bad)
        boot = cluster_bootstrap(judged, [tids_by_condition[condition][i] for i in valid_idx], CFG.n_bootstrap, CFG.seed_bootstrap)
        audit["gen_rate_ci_95_low"], audit["gen_rate_ci_95_high"] = boot["ci95_low"], boot["ci95_high"]
    result["generation_rates"] = rates
    result["generation_audit"] = list(audit_by_name.values())

    _, _, BASE_UNANS = next((name, part, key) for name, part, key in generation_conditions()
                            if name == "test_unans_baseline")
    _, _, BASE_ANS = next((name, part, key) for name, part, key in generation_conditions()
                          if name == "test_ans_baseline")
    conditions_by_key = {key: name for name, _, key in generation_conditions()}
    def update_curve(curve, part):
        for point in curve:
            if point["mode"] == "ablate_all":
                from src.test_evaluation import ckey
                key = ckey("unans", "ablate_all")
            else:
                from src.test_evaluation import ckey
                key = ckey(part, "steer_layer", alpha=float(point["alpha"]))
            name = conditions_by_key[key]
            for dst, src in (("gen_abstention_rate_strict", "strict"),
                             ("gen_abstention_rate_regex", "regex_v4"),
                             ("gen_abstention_rate_regex_v4", "regex_v4"),
                             ("gen_abstention_rate_judged", "judged"),
                             ("gen_degenerate_rate", "degenerate"),
                             ("regex_judge_agreement", "regex_judge_agreement")):
                point[dst] = rates[name][src]
    update_curve(result["necessity_curve"], "unans")
    update_curve(result["sufficiency_curve"], "ans")

    audit = {r["condition"]: r for r in result["generation_audit"]}
    branch = {}
    for label, cond_name, base_name in (("ablate_all_unans", "test_unans_ablate_learned", "test_unans_baseline"),
                                        ("steer_pos1_ans", "test_ans_steer_pos1_learned", "test_ans_baseline")):
        delta_s = audit[cond_name]["mean_log_odds"] - audit[base_name]["mean_log_odds"]
        delta_r = audit[cond_name]["gen_abstention_rate_judged"] - audit[base_name]["gen_abstention_rate_judged"]
        flagged = bool((delta_s * delta_r < 0) or
                       (abs(delta_s) >= CFG.discrepancy_log_odds and abs(delta_r) < CFG.discrepancy_rate))
        branch[label] = {"delta_log_odds": delta_s, "delta_gen_rate_judged": delta_r, "flagged": flagged}
    result["branch_rule4"] = branch
    result["logit_gen_discrepancy"] = any(v["flagged"] for v in branch.values())
    bctrl = rates["b_ctrl_baseline"]
    h3 = result["primary_tests"]["H3"]
    h3["b_ctrl_gen_abstention_rate_regex"] = bctrl["regex_v4"]
    h3["b_ctrl_gen_abstention_rate_regex_v4"] = bctrl["regex_v4"]
    h3["b_ctrl_gen_abstention_rate_judged"] = bctrl["judged"]
    h3["b_ctrl_gen_degenerate_rate"] = bctrl["degenerate"]
    h3["b_ctrl_abstention_ok"] = bool(bctrl["judged"] <= CFG.b_ctrl_abstention_max)
    result["b_ctrl_summary"].update(b_ctrl_gen_abstention_rate_regex=bctrl["regex_v4"],
                                     b_ctrl_gen_abstention_rate_regex_v4=bctrl["regex_v4"],
                                     b_ctrl_gen_abstention_rate_judged=bctrl["judged"],
                                     b_ctrl_gen_degenerate_rate=bctrl["degenerate"])
    prelim = result["preliminary_holm_h1_h3"]
    new_prelim = holm_bonferroni({k: result["primary_tests"][k]["p_wilcoxon"] for k in ("H1", "H2", "H3")},
                                 extra_criteria={"H1": result["primary_tests"]["H1"]["passes_mc_rank_1"],
                                                 "H2": result["primary_tests"]["H2"]["passes_mc_rank_1"],
                                                 "H3": h3["b_ctrl_abstention_ok"]})
    result["preliminary_holm_h1_h3"] = new_prelim
    result["classifier"] = {"backend": CFG.judge_backend, "model_id": CFG.judge_model_id,
                            "prompt_version": "v1", "unparseable_count": sum(r["final_label"] == "UNPARSEABLE" for r in records),
                            "unparseable_by_condition": {k: v["unparseable_count"] for k, v in rates.items()},
                            "classifier_final": all(r["classifier_final"] for r in records)}
    result["passed_acceptance_gate"] = bool(result["classifier"]["classifier_final"] and all(r["final_label"] in {"ANSWER","ABSTAIN","DEGENERATE"} for r in records))

    before = json.dumps(_log_odds_view(old_result), sort_keys=True, separators=(",", ":"))
    after = json.dumps(_log_odds_view(result), sort_keys=True, separators=(",", ":"))
    assert before == after, "Stage 6 rescore changed a log-odds field"
    atomic_write_json(a.stage6_path, result)
    atomic_write_jsonl(a.generation_path, rows)
    from src.results_tables import write_table1
    write_table1(CFG.table1_csv_path, CFG.table1_md_path, stage6=result)
    write_csv_and_md(result["generation_audit"], TABLE4_COLUMNS, a.table4_csv_path, a.table4_md_path,
                     "Table 4. First-token log-odds vs. judged 16-token greedy generation audit")
    print(f"Rescored {len(rows)} generations; log-odds unchanged; Branch Rule 4={result['logit_gen_discrepancy']}")
    verify_stage6_outputs(a.stage6_path, a.generation_path, CFG.table1_csv_path, CFG.table1_md_path, a.table4_csv_path, CFG.stage5_out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
