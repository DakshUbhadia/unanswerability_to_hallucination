"""Stage 10 (stretch): additive steering on popularity-matched PopQA and the selective-abstention curve."""
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import CFG
from src.eval_common import (ConditionCheckpoint, atomic_write_json, build_chat_batches, evaluate_condition,
                             generate_with_hooks, read_json, slice_vectors_at_star)

STAGE10_KEYS = {"stage", "model_name", "l_star", "i_star", "alpha_grid", "n_pairs", "n_items", "coverage_curve",
                "confidence_threshold_curve", "random_abstention_accuracy", "items", "passed_acceptance_gate"}


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


def confidence_threshold_curve(matched_items: List[Dict[str, Any]], n_points: int = 11) -> List[Dict[str, Any]]:
    """Abstain when baseline max first-token probability < threshold; baseline labels give correctness."""
    maxp = np.asarray([r["maxp"] for r in matched_items])
    correct = np.asarray([r["partition"] == "correct" for r in matched_items])
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
                                     source_stage4_sha256="") -> Dict[str, Any]:
    from src.pilot_check import normalize_and_grade_popqa_response
    started = time.perf_counter()
    l_star, i_star, pos_idx = int(selection_dict["l_star"]), int(selection_dict["i_star"]), int(selection_dict["pos_idx"])
    vectors = slice_vectors_at_star(directions_dict, l_star, pos_idx)
    rows = list(popqa_groups["wrong"]) + list(popqa_groups["correct"])
    n = len(popqa_groups["wrong"])
    batches = build_chat_batches(tokenizer, [r["question"] for r in rows], contract["system_prompt_used"],
                                 list(contract["suffix_token_ids"]), batch_size)
    fp = hashlib.sha256(f"{source_stage4_sha256}|{l_star}|{pos_idx}|{[r['popqa_id'] for r in rows]}|{tuple(alpha_grid)}".encode()).hexdigest()
    ckpt = ConditionCheckpoint(checkpoint_path, fp, force=force, every=1)
    curves: Dict[str, List[Dict[str, Any]]] = {"learned": [], "concept": []}
    items = []
    for direction, vec in (("learned", vectors["learned_diff"]), ("concept", vectors["concept_scaled"])):
        for alpha in alpha_grid:
            key = f"base|{alpha:+.2f}" if alpha == 0.0 else f"{direction}|{alpha:+.2f}"
            mode = "none" if alpha == 0.0 else "steer_layer"
            if key not in ckpt:
                lo = evaluate_condition(raw_model, batches, contract, mode, vec, l_star, alpha)
                texts = generate_with_hooks(raw_model, tokenizer, batches, contract, mode, vec, l_star, alpha, CFG.max_new_tokens)
                grades = [normalize_and_grade_popqa_response(t, r.get("possible_answers", []), r.get("obj", ""))
                          for t, r in zip(texts, rows)]
                ckpt.put(key, {"log_odds": lo["log_odds"], "texts": texts, "grades": grades})
                print(f"[Stage 10] {direction} alpha={alpha:+.2f} ({time.perf_counter() - started:.0f}s)", flush=True)
            res = ckpt.get(key)
            m = selective_metrics(res["grades"][:n], res["grades"][n:])
            curves[direction].append({"alpha": float(alpha), "mean_log_odds": float(np.mean(res["log_odds"])), **m})
            if direction == "learned":
                for r, t, g, s in zip(rows, res["texts"], res["grades"], res["log_odds"]):
                    items.append({"alpha": float(alpha), "popqa_id": r["popqa_id"], "matched_pair_id": r["matched_pair_id"],
                                  "baseline_partition": r["baseline_partition"], "generated_text": t, "grade": g,
                                  "log_odds": float(s)})
    ckpt.flush()
    conf = confidence_threshold_curve(matched_items) if matched_items else []
    return {"stage": 10, "model_name": contract["model_name"], "l_star": l_star, "i_star": i_star,
            "alpha_grid": [float(a) for a in alpha_grid], "n_pairs": n, "n_items": len(rows), "coverage_curve": curves,
            "confidence_threshold_curve": conf, "random_abstention_accuracy": 0.5,
            "random_abstention_note": "matched groups are 50% baseline-correct by construction", "items": items,
            "ambiguous_grade_policy": "answered, not correct", "elapsed_sec": time.perf_counter() - started,
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
    for direction in ("learned", "concept"):
        curve = d["coverage_curve"][direction]
        assert [r["alpha"] for r in curve] == d["alpha_grid"]
        for r in curve:
            assert 0.0 <= r["coverage"] <= 1.0 and 0.0 <= r["wrong_to_abstain_rate"] <= 1.0
    assert len(d["items"]) == d["n_items"] * len(d["alpha_grid"])
    p = Path(fig2b_png_path)
    assert p.is_file() and p.stat().st_size > 1000 and p.with_suffix(".pdf").is_file()
    return d
