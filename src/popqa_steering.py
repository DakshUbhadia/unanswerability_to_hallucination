"""Stage 10 (stretch): additive steering on popularity-matched PopQA and the selective-abstention curve."""
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import CFG
from src.eval_common import (ConditionCheckpoint, atomic_write_json, build_chat_batches, evaluate_condition,
                             generate_with_hooks, read_json, slice_vectors_at_star)
from src.abstention_classifier import classify_texts, require_judge_validation, require_low_margin_review
from src.tokenization_contract import CONTRACT_VERSION, canonical_input_ids, chat_template_sha256, input_ids_sha256

STAGE10_KEYS = {"stage", "model_name", "l_star", "i_star", "alpha_grid", "n_pairs", "n_items", "coverage_curve",
                "confidence_threshold_curve", "random_abstention_accuracy", "alpha0_label_identity_passed",
                "classifier_final", "items", "passed_acceptance_gate"}


def _pair_bootstrap(stat_fn, n_pairs: int, n_boot: int, seed: int) -> List[float]:
    rng = np.random.default_rng(int(seed))
    vals = []
    for _ in range(int(n_boot)):
        v = stat_fn(rng.integers(0, n_pairs, n_pairs))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [None, None]


def selective_metrics(grades_wrong: List[str], grades_correct: List[str], n_boot: int = CFG.n_bootstrap,
                      seed: int = CFG.seed_bootstrap) -> Dict[str, Any]:
    """Grades are pair-aligned over the matched pairs (baseline-wrong item i, baseline-correct item i).

    Ambiguous grades count as answered but not correct (conservative).
    """
    gw, gc = np.asarray(grades_wrong), np.asarray(grades_correct)
    n = len(gw)
    assert n == len(gc) and n > 0

    def stats_for(idx):
        w, c = gw[idx], gc[idx]
        allg = np.concatenate([w, c])
        answered = allg != "abstain"
        acc = float((allg[answered] == "correct").mean()) if answered.any() else None
        return {"wrong_abs": float((w == "abstain").mean()), "correct_abs": float((c == "abstain").mean()),
                "coverage": float(answered.mean()), "acc": acc}

    full = stats_for(np.arange(n))
    gain = full["wrong_abs"] - full["correct_abs"]
    return {"wrong_to_abstain_rate": full["wrong_abs"], "correct_to_abstain_rate": full["correct_abs"],
            "selective_abstention_gain": gain,
            "selective_abstention_gain_ci": _pair_bootstrap(lambda i: stats_for(i)["wrong_abs"] - stats_for(i)["correct_abs"], n, n_boot, seed),
            "coverage": full["coverage"], "coverage_ci": _pair_bootstrap(lambda i: stats_for(i)["coverage"], n, n_boot, seed),
            "selective_acc": full["acc"], "selective_acc_ci": _pair_bootstrap(lambda i: stats_for(i)["acc"], n, n_boot, seed),
            "n_correct": int((gw == "correct").sum() + (gc == "correct").sum()),
            "n_abstain": int((gw == "abstain").sum() + (gc == "abstain").sum())}


def confidence_threshold_curve(alpha0_grades, maxp_values=None, item_ids=None, expected_item_ids=None,
                               n_points: int = 11) -> List[Dict[str, Any]]:
    """Threshold Stage 8 maxp against the same Stage 2/8 alpha=0 labels and item order."""
    if maxp_values is None and alpha0_grades and isinstance(alpha0_grades[0], dict):
        rows = alpha0_grades
        alpha0_grades = [r.get("grade", r.get("partition")) for r in rows]
        maxp_values = [r["maxp"] for r in rows]
        item_ids = [r.get("popqa_id", str(i)) for i, r in enumerate(rows)]
    grades = list(alpha0_grades)
    maxp = np.asarray(maxp_values, dtype=float)
    ids = list(item_ids or [])
    if len(grades) != len(maxp) or len(ids) != len(grades):
        raise ValueError("Alpha=0 grades, maxp values, and item IDs must have identical lengths")
    if expected_item_ids is not None and ids != list(expected_item_ids):
        raise ValueError("Stage 8 alpha=0 and maxp item IDs/order differ")
    correct = np.asarray([grade == "correct" for grade in grades])
    rows = []
    for q in np.linspace(0, 90, n_points - 1).tolist() + [None]:
        thr = -np.inf if q == 0 else (np.percentile(maxp, q) if q is not None else None)
        if thr is None:
            continue
        answered = maxp >= thr
        rows.append({"threshold": float(thr) if np.isfinite(thr) else 0.0, "coverage": float(answered.mean()),
                     "selective_acc": float(correct[answered].mean()) if answered.any() else None})
    return rows


def run_stage10_popqa_steering_sweep(raw_model, tokenizer, contract, directions_dict, selection_dict, popqa_groups,
                                     alpha_grid=CFG.grid_alpha_stretch, batch_size=CFG.batch_size_gen,
                                     checkpoint_path=CFG.stage10_checkpoint_path, force=False,
                                     matched_items: Optional[List[Dict[str, Any]]] = None,
                                     source_stage4_sha256="", offline=False, generation_only=False) -> Dict[str, Any]:
    from src.pilot_check import grade_popqa
    started = time.perf_counter()
    l_star, i_star, pos_idx = int(selection_dict["l_star"]), int(selection_dict["i_star"]), int(selection_dict["pos_idx"])
    vectors = slice_vectors_at_star(directions_dict, l_star, pos_idx)
    rows = list(popqa_groups["wrong"]) + list(popqa_groups["correct"])
    n = len(popqa_groups["wrong"])
    prompt_rows = [r["question"] for r in rows]
    suffix_ids = list(contract["suffix_token_ids"])
    template_hash = chat_template_sha256(tokenizer)
    tokenizer_name = contract["model_name"]
    canonical_ids = [canonical_input_ids(tokenizer, p, contract["system_prompt_used"], suffix_ids) for p in prompt_rows]
    actual_hashes = [input_ids_sha256(ids, tokenizer_name, template_hash) for ids in canonical_ids]
    mismatches = [r["popqa_id"] for r, actual in zip(rows, actual_hashes)
                  if r.get("input_ids_sha256") != actual or r.get("tokenization_contract_version") != CONTRACT_VERSION]
    if mismatches:
        raise ValueError(f"Stage 10 canonical input hashes differ from Stage 2 for PopQA IDs: {mismatches}")
    batches_by_size = {size: build_chat_batches(tokenizer, prompt_rows, contract["system_prompt_used"], suffix_ids, size)
                       for size in {1, int(batch_size)}}
    fp = hashlib.sha256(f"{source_stage4_sha256}|{l_star}|{pos_idx}|{[r['popqa_id'] for r in rows]}|{tuple(alpha_grid)}|{CONTRACT_VERSION}|alpha0_batch1|{template_hash}".encode()).hexdigest()
    ckpt = ConditionCheckpoint(checkpoint_path, fp, force=force, every=1)
    if not generation_only and CFG.stage10_generation_path.exists():
        saved=read_jsonl(CFG.stage10_generation_path)
        grouped={}
        for row in saved:
            alpha=float(row["alpha"]);direction=row["direction"]
            key=f"base|{alpha:+.2f}" if alpha==0.0 else f"{direction}|{alpha:+.2f}"
            grouped.setdefault(key,[]).append(row)
        for key,group in grouped.items():
            if key in ckpt:continue
            if len(group)!=len(rows):raise RuntimeError(f"Saved Stage 10 generation {key} has {len(group)} rows; expected {len(rows)}")
            ckpt.put(key,{"texts":[r["generated_text"] for r in group],"log_odds":[r["log_odds"] for r in group],
                          "input_ids_sha256":[r["input_ids_sha256"] for r in group],
                          "generation_batch_size":int(group[0]["generation_batch_size"])})
    curves: Dict[str, List[Dict[str, Any]]] = {"learned": [], "concept": []}
    items = []
    for direction, vec in (("learned", vectors["learned_diff"]), ("concept", vectors["concept_scaled"])):
        for alpha in alpha_grid:
            key = f"base|{alpha:+.2f}" if alpha == 0.0 else f"{direction}|{alpha:+.2f}"
            mode = "none" if alpha == 0.0 else "steer_layer"
            if key not in ckpt:
                if raw_model is None:
                    raise RuntimeError(f"Missing saved Stage 10 generation condition {key}; run generation pass first")
                active_batch_size = 1 if alpha == 0.0 else int(batch_size)
                batches = batches_by_size[active_batch_size]
                lo = evaluate_condition(raw_model, batches, contract, mode, vec, l_star, alpha)
                texts = generate_with_hooks(raw_model, tokenizer, batches, contract, mode, vec, l_star, alpha, CFG.max_new_tokens)
                ckpt.put(key, {"log_odds": lo["log_odds"], "texts": texts,
                               "input_ids_sha256": actual_hashes, "generation_batch_size": active_batch_size})
                print(f"[Stage 10] generated {direction} alpha={alpha:+.2f} ({time.perf_counter() - started:.0f}s)", flush=True)
            res = ckpt.get(key)
            if generation_only:
                for i,(r,t) in enumerate(zip(rows,res["texts"])):
                    items.append({"direction":direction,"alpha":float(alpha),"popqa_id":r["popqa_id"],"matched_pair_id":r.get("matched_pair_id"),
                                  "question":r["question"],"generated_text":t,"final_label":None,"judge_label":None,"judge_margin":None,
                                  "log_odds":float(res["log_odds"][i]),"input_ids_sha256":res["input_ids_sha256"][i],"generation_batch_size":res["generation_batch_size"]})
                continue
            if "grades" not in res:
                records = classify_texts([{"uid":r["popqa_id"],"question":r["question"],"response":t}
                                          for t,r in zip(res["texts"],rows)],offline=offline,batch_size=CFG.judge_batch_size)
                require_low_margin_review(records)
                bad=sum(record["final_label"]=="UNPARSEABLE" for record in records)
                if bad/len(records)>.01:raise RuntimeError(f"{direction} alpha={alpha:+.2f}: {bad}/{len(records)} UNPARSEABLE outputs exceed 1%")
                graded=[grade_popqa(t,r.get("possible_answers",[]),r.get("obj",""),record)
                        for t,r,record in zip(res["texts"],rows,records)]
                grades=[g["grade"] for g in graded];grade_reasons=[g["reason"] for g in graded]
                if alpha==0.0:
                    mismatches=[{"popqa_id":r["popqa_id"],"stage2_grade":r["baseline_partition"],"alpha0_grade":g["grade"],
                                 "stage2_response":r.get("baseline_response"),"alpha0_response":t,
                                 "stage2_input_ids_sha256":r.get("input_ids_sha256"),"alpha0_input_ids_sha256":actual_hashes[i]}
                                for i,(r,t,g) in enumerate(zip(rows,res["texts"],graded)) if r["baseline_partition"]!=g["grade"]]
                    if mismatches:
                        atomic_write_json(CFG.results_dir/"stage10_alpha0_mismatches.json",mismatches)
                        raise AssertionError(f"Stage 10 alpha=0 labels differ for {len(mismatches)} matched items")
                res={**res,"grades":grades,"grade_reasons":grade_reasons,"classifier_records":records};ckpt.put(key,res)
            m = selective_metrics(res["grades"][:n], res["grades"][n:])
            curves[direction].append({"alpha": float(alpha), "mean_log_odds": float(np.mean(res["log_odds"])), **m})
            for i, (r, t, g, score, record) in enumerate(zip(rows, res["texts"], res["grades"], res["log_odds"], res["classifier_records"])):
                items.append({"direction": direction, "alpha": float(alpha), "popqa_id": r["popqa_id"], "matched_pair_id": r["matched_pair_id"],
                              "baseline_partition": r["baseline_partition"], "generated_text": t, "grade": g,
                              "grade_reason": res["grade_reasons"][i], "final_label": record["final_label"],
                              "judge_label": record["judge_label"], "regex_label": record["regex_label"],
                              "judge_model_id": record["judge_model_id"], "judge_model_revision": record["judge_model_revision"],
                              "judge_quantization": record.get("judge_quantization"), "judge_logprobs": record.get("judge_logprobs"),
                              "judge_margin": record.get("judge_margin"), "judge_prompt_version": record["judge_prompt_version"],
                              "regex_judge_agree": record["agree"], "judge_prompt_sha256": record["judge_prompt_sha256"],
                              "input_ids_sha256": res["input_ids_sha256"][i],
                              "generation_batch_size": res["generation_batch_size"], "log_odds": float(score)})
    ckpt.flush()
    if generation_only:
        return {"stage":10,"generation_only":True,"model_name":contract["model_name"],"l_star":l_star,"i_star":i_star,
                "n_items":len(rows),"items":items,"judge_pending":True}
    conf = confidence_threshold_curve([r["grade"] for r in matched_items], [r["maxp"] for r in matched_items],
                                      [r["popqa_id"] for r in matched_items], [r["popqa_id"] for r in rows]) if matched_items else []
    return {"stage": 10, "model_name": contract["model_name"], "l_star": l_star, "i_star": i_star,
            "alpha_grid": [float(a) for a in alpha_grid], "n_pairs": n, "n_items": len(rows), "coverage_curve": curves,
            "confidence_threshold_curve": conf, "random_abstention_accuracy": 0.5,
            "random_abstention_note": "matched groups are 50% baseline-correct by construction", "items": items,
            "ambiguous_grade_policy": "answered, not correct", "alpha0_label_identity_passed": True,
            "classifier_final": all(item.get("final_label") in {"ANSWER", "ABSTAIN", "DEGENERATE"} for item in items),
            "alpha0_generation_batch_size": 1, "other_generation_batch_size": int(batch_size),
            "elapsed_sec": time.perf_counter() - started,
            "passed_acceptance_gate": True}


def plot_figure2b_selective_steering(stage10_dict, out_png_path=CFG.fig2b_png_path, out_pdf_path=None) -> None:
    from src.plotting_and_report import plot_figure2b_selective_steering as _plot
    _plot(stage10_dict, out_png_path, out_pdf_path)


def save_stage10_artifacts(stage10_dict, output_path=CFG.stage10_out_path, fig2b_png_path=CFG.fig2b_png_path,
                           fig2b_pdf_path=None, checkpoint_path=None) -> None:
    atomic_write_json(output_path, stage10_dict)
    plot_figure2b_selective_steering(stage10_dict, fig2b_png_path, fig2b_pdf_path)
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()


def verify_stage10_outputs(output_path=CFG.stage10_out_path, fig2b_png_path=CFG.fig2b_png_path) -> Dict[str, Any]:
    d = read_json(output_path)
    missing = STAGE10_KEYS - d.keys()
    assert not missing, f"Stage 10 JSON missing keys: {sorted(missing)}"
    assert d["stage"] == 10 and d["passed_acceptance_gate"] is True and d["n_items"] == 2 * d["n_pairs"]
    assert d["alpha0_label_identity_passed"] is True and d["classifier_final"] is True
    for direction in ("learned", "concept"):
        curve = d["coverage_curve"][direction]
        assert [r["alpha"] for r in curve] == d["alpha_grid"]
        for r in curve:
            assert 0.0 <= r["coverage"] <= 1.0 and 0.0 <= r["wrong_to_abstain_rate"] <= 1.0
    assert len(d["items"]) == 2 * d["n_items"] * len(d["alpha_grid"])
    p = Path(fig2b_png_path)
    assert p.is_file() and p.stat().st_size > 1000 and p.with_suffix(".pdf").is_file()
    return d
