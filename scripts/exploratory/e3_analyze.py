#!/usr/bin/env python3
"""Run the CPU analyses specified by the E3 instruction-ablation plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import CFG
from src.eval_common import sha256_file, read_jsonl
from src.exploratory import assert_not_fresh, exploratory_header, require_tier1_complete, write_exploratory_json
from src.exploratory.e1_cv import nested_probe_oof, nested_stacking_delta
from src.exploratory.e2_analysis import (crossfit_residual, cosine_spaces, fit_dom, geometry_nulls,
                                         label_part_R)
from src.exploratory.e3_analysis import (direction_from_pairs, e3_verdict, in_context_auroc,
    incremental_value_for_projection, independent_condition_bootstrap, label_concordant_subset,
    label_transition_table, qwen_replication, bootstrap_train_direction_alignment)

OUT = ROOT / "results/exploratory"
MODELS = ("llama32_3b", "qwen25_3b")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _condition_files(slug):
    folder = assert_not_fresh(OUT / "e3_features" / f"{slug}_neut")
    pair_manifest = _json(folder / "pairs_manifest.json")
    feature_manifest = _json(folder / "features_manifest.json")
    for name, manifest, key in (("pairs_suf.npy", pair_manifest, "pairs_suf.npy"),
                                ("popqa_suf.npy", feature_manifest, "popqa_suf.npy"),
                                ("popqa_confidence.npz", feature_manifest, "popqa_confidence.npz"),
                                ("popqa_ans.npy", feature_manifest, "popqa_ans.npy")):
        path = folder / name
        expected = manifest.get("sha256", {}).get(key)
        if not expected:
            raise ValueError(f"{path} has no checksum in its manifest")
        from src.exploratory.tensor_io import download_verified
        download_verified(path, repo_path=f"exploratory/e3_features/{slug}_neut/{name}",
                          expected_sha256=expected)
    return folder, pair_manifest, feature_manifest


def _pair_bootstrap_auc(y, score, groups, n_boot=1000, seed=20260407):
    y, score, groups = np.asarray(y, int), np.asarray(score, float), np.asarray(groups).astype(str)
    point = float(roc_auc_score(y, score))
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(n_boot)):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            values.append(float(roc_auc_score(y[idx], score[idx])))
    if not values:
        raise ValueError("No valid matched-pair bootstrap samples")
    return {"auroc": point, "ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
            "n_boot": len(values), "seed": int(seed), "resampling": "matched_pair"}


def _spot_check_flags():
    judge_path, grader_path = OUT / "e3_judge_validation.json", OUT / "e3_grader_validation.json"
    judge = _json(judge_path) if judge_path.exists() else None
    grader = _json(grader_path) if grader_path.exists() else None
    kappa = None if not judge else judge.get("cohen_kappa", judge.get("kappa"))
    recall = None
    if judge:
        recall = judge.get("recall", {}).get("ABSTAIN", judge.get("abstain_recall"))
    precision = None if not grader else grader.get("precision_correct")
    reliable = kappa is not None and recall is not None and precision is not None and \
        float(kappa) >= .85 and float(recall) >= .90 and float(precision) >= .90
    return {"judge_kappa": kappa, "judge_abstain_recall": recall,
            "grader_precision_correct": precision, "reliability_passed": bool(reliable),
            "provisional": not bool(reliable)}


def _load_labels(slug):
    path = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    rows = read_jsonl(path)
    if not rows or any(r.get("label_source") != "e3_neutral_judged" for r in rows):
        raise RuntimeError(f"E3 neutral labels are incomplete for {slug}; run e3_label_and_match.py first")
    return rows


def _condition(slug, bootstrap, pair_manifest=None, feature_manifest=None):
    folder, pair_manifest, feature_manifest = _condition_files(slug)
    pair_path, suf_path = folder / "pairs_suf.npy", folder / "popqa_suf.npy"
    pairs = np.load(pair_path, mmap_mode="r")
    rows = pair_manifest["rows"]
    diff, unit, train_ids = direction_from_pairs(pairs, rows, split="train")
    inctx = in_context_auroc(pairs, rows, unit, split="test")
    popqa = _load_labels(slug)
    confidence = np.load(folder / "popqa_confidence.npz")
    if len(popqa) != len(confidence["maxp"]):
        raise ValueError(f"{slug} PopQA labels and feature rows are misaligned")
    layer, pos = 11, 1
    preflight = _json(OUT / "e3_preflight.json")
    if slug == "qwen25_3b":
        if not _json(OUT / "e3_qwen_contract.json").get("passed_all_checks"):
            raise RuntimeError("Qwen E3 analysis requested although its preflight failed")
        q = _json(OUT / "e3_qwen_contract.json")
        layer, pos = int(q["qwen_depth_matched_layer"]), int(q["role_matched_suffix_index"])
    target = unit[layer, pos]
    acts = np.load(suf_path, mmap_mode="r")
    raw_score = np.asarray(acts[:, pos, layer, :], dtype=np.float32) @ target.astype(np.float32)
    confidence_df = pd.DataFrame({"maxp": confidence["maxp"], "entropy": confidence["entropy"],
        "log_odds_set": confidence["abstention_set_log_odds"], "proj_own": raw_score,
        "seq_logprob_mean": confidence["seq_logprob_mean"], "popqa_id": [str(r["popqa_id"]) for r in popqa],
        "label": [str(r.get("baseline_partition")) for r in popqa],
        "matched_pair_id": [str(r.get("matched_pair_id") or "") for r in popqa],
        "is_popularity_matched": [bool(r.get("is_popularity_matched")) for r in popqa],
        "log_s_pop": [float(np.log10(max(float(r.get("s_pop", 0)), 0) + 1)) for r in popqa]})
    matched = confidence_df[confidence_df.is_popularity_matched &
                            confidence_df.label.isin(("correct", "wrong"))].copy()
    matched["y"] = (matched.label == "wrong").astype(int)
    if matched.empty or matched.matched_pair_id.nunique() < 2:
        raise RuntimeError(f"{slug} has too few matched correct/wrong PopQA rows for E3 analyses")
    groups = matched.matched_pair_id.to_numpy(str)
    y = matched.y.to_numpy(int)
    aucs = {}
    for name, score in (("proj_own", matched.proj_own), ("maxp", -matched.maxp),
                        ("entropy", matched.entropy), ("abstention_set_log_odds", matched.log_odds_set),
                        ("seq_logprob_mean", -matched.seq_logprob_mean), ("log_s_pop", -matched.log_s_pop)):
        aucs[name] = _pair_bootstrap_auc(y, score, groups, bootstrap, 20260407)
    if slug == "llama32_3b":
        import torch
        directions = torch.load(ROOT / "results/stage4_directions.pt", map_location="cpu", weights_only=False)
        instruction = directions["unit_directions"][11, 1].detach().cpu().numpy()
        proj_instruction = np.asarray(acts[matched.index, pos, layer, :], dtype=np.float32) @ instruction.astype(np.float32)
        matched["proj_instr"] = proj_instruction
        aucs["proj_instr"] = _pair_bootstrap_auc(y, proj_instruction, groups, bootstrap, 20260407)
    matched["log_odds_set"] = matched.log_odds_set
    incremental = incremental_value_for_projection(matched, "proj_own", n_boot=bootstrap, seed=20260414)
    if slug == "llama32_3b":
        incremental_instr = incremental_value_for_projection(matched, "proj_instr", n_boot=bootstrap, seed=20260414)
    else:
        incremental_instr = None
    confounds = matched[["log_odds_set", "maxp", "entropy"]].to_numpy(float)
    residual = crossfit_residual(matched.proj_own, confounds, y, groups, "spline_ridge", 5, 5, 20260414)
    from src.exploratory.e2_analysis import _cluster_boot_auc
    residual_ci = _cluster_boot_auc(y, residual["residual"], groups, bootstrap, 20260407)
    residual_label = label_part_R([residual_ci["low"], residual_ci["high"]])

    # E1's nested probe and nested stacker on neutral suffix -1 and answer-first features.
    ans = np.load(folder / "popqa_ans.npy", mmap_mode="r")
    ids = matched.index.to_numpy(int)
    groups_cv = matched.matched_pair_id.to_numpy(str)
    base = matched[["maxp", "entropy", "log_odds_set"]].to_numpy(float)
    probe = {}
    for name, values in (("C1_suf-1", np.asarray(acts[ids, 4], dtype=np.float32)),
                         ("C2_ans_first", np.asarray(ans[ids, 0], dtype=np.float32))):
        valid = np.isfinite(values).all(axis=(1, 2))
        local_values, local_y, local_groups, local_base = values[valid], y[valid], groups_cv[valid], base[valid]
        if len(local_y) < 10 or len(np.unique(local_y)) != 2:
            probe[name] = {"omitted": "insufficient finite teacher-forced rows"}
            continue
        scores, selected = nested_probe_oof(local_values, local_y, local_groups, 5, 5, 20260414)
        stacked = nested_stacking_delta(local_values, local_base, local_y, local_groups, 5, 5, 20260414)
        probe[name] = {"n": int(len(local_y)), "nested_probe_auroc": float(roc_auc_score(local_y, scores.mean(0))),
                       "selected": selected,
                       "stacking_delta_by_repeat": stacked["delta_by_repeat"].tolist(),
                       "stacking_delta_mean": float(np.mean(stacked["delta_by_repeat"]))}

    out = {"slug": slug, "n_matched_pairs": int(matched.matched_pair_id.nunique()),
        "power_status": "descriptive_only" if matched.matched_pair_id.nunique() < 100 else "powered_by_preregistered_rule",
        "locked_coordinate": {"layer": layer, "suffix_index": pos}, "train_pair_ids": train_ids,
        "in_context_auroc": inctx.tolist(), "in_context_locked_auroc": float(inctx[layer, pos]),
        "popqa_auroc": aucs, "incremental_value": incremental,
        "incremental_value_instruction_direction": incremental_instr,
        "residualization": {"crossfit_r2": residual["crossfit_r2"], "residual_auroc": residual_ci,
                            "label": residual_label}, "supervised_upper_bound": probe,
        "matched_popqa_ids": matched.popqa_id.tolist(), "matched_y": y.tolist(),
        "matched_groups": groups_cv.tolist(), "matched_projection": matched.proj_own.to_numpy(float).tolist(),
        "feature_manifest_sha256": sha256_file(folder / "features_manifest.json"),
        "pairs_manifest_sha256": sha256_file(folder / "pairs_manifest.json")}
    if slug == "llama32_3b":
        # Compare neutral direction with the frozen registered direction in E2's geometry.
        import torch
        directions = torch.load(ROOT / "results/stage4_directions.pt", map_location="cpu", weights_only=False)
        instruction = directions["unit_directions"][11, 1].detach().cpu().numpy()
        X_pairs = np.asarray(pairs[:, 1, 11, :], dtype=np.float32)
        y_pairs = np.asarray([int(r["part"] == "unans") for r in rows], dtype=int)
        X_pop = np.asarray(acts[matched.index, 1, 11, :], dtype=np.float32)
        pop_dom = fit_dom(X_pop, y)
        dims = np.argsort(np.mean(np.abs(X_pairs), axis=0))[-8:]
        geometry_cos = cosine_spaces(instruction, unit[11, 1], X_pairs, dims)
        instruction_geometry_nulls = geometry_nulls(instruction, X_pairs, y_pairs,
                                                     np.cov(X_pairs, rowvar=False))
        alignment_ci = bootstrap_train_direction_alignment(pairs, rows, instruction,
                                                           n_boot=bootstrap, seed=20260407)
        dims_pop = np.argsort(np.mean(np.abs(X_pop), axis=0))[-8:]
        dom_cos = cosine_spaces(unit[11, 1], pop_dom, X_pop, dims_pop)
        nulls = geometry_nulls(unit[11, 1], X_pop, y, np.cov(X_pop, rowvar=False))
        out["geometry"] = {"instruction_vs_neutral": geometry_cos, "neutral_vs_popqa_dom": dom_cos,
                           "neutral_vs_popqa_dom_nulls": nulls,
                           "instruction_geometry_nulls": instruction_geometry_nulls,
                           "instruction_neutral_bootstrap_ci": alignment_ci}
        out["alignment"] = bool(abs(geometry_cos["whitened"]) >
                                  np.percentile(np.abs(instruction_geometry_nulls["covariance_null"]), 99))
    return out


def _instruction_contrast(neutral, bootstrap):
    import torch
    from src.exploratory.e2_data import load_e1_cache
    cache = load_e1_cache()
    meta = cache["meta"].copy()
    directions = torch.load(ROOT / "results/stage4_directions.pt", map_location="cpu", weights_only=False)
    rhat = directions["unit_directions"][11, 1].detach().cpu().numpy()
    ids = {str(v): i for i, v in enumerate(meta.popqa_id.astype(str))}
    neutral_rows_all = _load_labels("llama32_3b")
    neutral_by_id = {str(r["popqa_id"]): r for r in neutral_rows_all}
    neutral_ids = [str(r["popqa_id"]) for r in neutral_rows_all
                   if r.get("is_popularity_matched") and r.get("baseline_partition") in ("correct", "wrong")]
    instru_mask = meta["is_popularity_matched"].fillna(False).astype(bool) & meta["baseline_partition"].astype(str).isin(("correct", "wrong"))
    instru_ids = meta.loc[instru_mask, "popqa_id"].astype(str).tolist()
    n_ix = np.asarray([ids[x] for x in neutral_ids], int)
    i_ix = np.asarray([ids[x] for x in instru_ids], int)
    neutral_y = np.asarray([int(neutral_by_id[x]["baseline_partition"] == "wrong") for x in neutral_ids])
    instruction_y = (meta.iloc[i_ix]["baseline_partition"].astype(str).to_numpy() == "wrong").astype(int)
    neutral_score_by_id = {str(k): float(v) for k, v in zip(neutral["matched_popqa_ids"], neutral["matched_projection"])}
    neutral_ids = [x for x in neutral_ids if x in neutral_score_by_id]
    # E3's neutral projection artifact stores the matched rows in its preregistered order.
    n_ix = np.asarray([ids[x] for x in neutral_ids], int)
    neutral_y = np.asarray([int(neutral_by_id[x]["baseline_partition"] == "wrong") for x in neutral_ids])
    nscore = np.asarray([neutral_score_by_id[x] for x in neutral_ids], float)
    instr_score = np.asarray(cache["suf"][i_ix, 1, 11, :], dtype=np.float32) @ rhat.astype(np.float32)
    independent = independent_condition_bootstrap(instruction_y, instr_score, neutral_y, nscore,
                                                 n_boot=bootstrap, seed=20260407)
    neutral_rows = pd.DataFrame({"popqa_id": [str(r["popqa_id"]) for r in neutral_rows_all],
                                 "label": [str(r.get("baseline_partition")) for r in neutral_rows_all]})
    instru_rows = pd.DataFrame({"popqa_id": meta["popqa_id"].astype(str),
                                "label": meta["baseline_partition"].astype(str)})
    concordant_i, concordant_n = label_concordant_subset(instru_rows, neutral_rows)
    trans = label_transition_table(instru_rows, neutral_rows)
    from src.pilot_check import match_popqa_by_popularity_caliper
    concordant_ids = set(concordant_n.popqa_id.astype(str))
    neutral_candidates = [dict(r) for r in neutral_rows_all
                          if str(r["popqa_id"]) in concordant_ids and r.get("baseline_partition") in ("correct", "wrong")]
    instruction_candidates = [dict(r) for r in meta.to_dict("records")
                              if str(r["popqa_id"]) in concordant_ids and r.get("baseline_partition") in ("correct", "wrong")]
    _ni, neutral_concordant_pairs, neutral_width = match_popqa_by_popularity_caliper(
        neutral_candidates, 0.25, 20260406)
    _ii, instruction_concordant_pairs, instruction_width = match_popqa_by_popularity_caliper(
        instruction_candidates, 0.25, 20260406)
    neutral_folder = OUT / "e3_features" / "llama32_3b_neut"
    neutral_manifest = _json(neutral_folder / "features_manifest.json")
    from src.exploratory.tensor_io import download_verified
    neutral_tensor = download_verified(neutral_folder / "popqa_suf.npy",
        repo_path="exploratory/e3_features/llama32_3b_neut/popqa_suf.npy",
        expected_sha256=neutral_manifest["sha256"]["popqa_suf.npy"])
    neutral_acts = np.load(neutral_tensor, mmap_mode="r")
    neutral_pos, neutral_layer = 1, 11
    neutral_score_all = np.asarray(neutral_acts[:, neutral_pos, neutral_layer, :], dtype=np.float32) @ \
        np.asarray(neutral["unit_direction_11_suf-4"], dtype=np.float32) if "unit_direction_11_suf-4" in neutral else None
    if neutral_score_all is None:
        # The fitted DOM is deterministic from the already verified train pair cache.
        pair_manifest = _json(neutral_folder / "pairs_manifest.json")
        pair_acts = np.load(neutral_folder / "pairs_suf.npy", mmap_mode="r")
        _, pair_unit, _ = direction_from_pairs(pair_acts, pair_manifest["rows"], split="train")
        neutral_score_all = np.asarray(neutral_acts[:, neutral_pos, neutral_layer, :], dtype=np.float32) @ pair_unit[neutral_layer, neutral_pos].astype(np.float32)
    neutral_positions = {str(r["popqa_id"]): i for i, r in enumerate(neutral_rows_all)}
    matched_neutral = [r for r in neutral_candidates if r.get("is_popularity_matched")]
    matched_instruction = [r for r in instruction_candidates if r.get("is_popularity_matched")]
    neutral_ids = [str(r["popqa_id"]) for r in matched_neutral]
    instruction_ids = [str(r["popqa_id"]) for r in matched_instruction]
    if matched_neutral and matched_instruction:
        neutral_y_concordant = np.asarray([int(r["baseline_partition"] == "wrong") for r in matched_neutral])
        instruction_y_concordant = np.asarray([int(r["baseline_partition"] == "wrong") for r in matched_instruction])
        instruction_features = np.asarray(cache["suf"][np.asarray([ids[x] for x in instruction_ids]), 1, 11, :], dtype=np.float32)
        instruction_score_all = instruction_features @ rhat.astype(np.float32)
        neutral_groups = np.asarray([str(r["matched_pair_id"]) for r in matched_neutral])
        instruction_groups = np.asarray([str(r["matched_pair_id"]) for r in matched_instruction])
        concordant_auroc = {"neutral": _pair_bootstrap_auc(neutral_y_concordant,
                np.asarray([neutral_score_all[neutral_positions[x]] for x in neutral_ids]), neutral_groups),
            "instruction": _pair_bootstrap_auc(instruction_y_concordant,
                instruction_score_all, instruction_groups)}
    else:
        concordant_auroc = {"neutral": None, "instruction": None}
    return {"independent_auroc_bootstrap": independent, "transition_table": trans,
            "instruction_n": len(instru_ids), "neutral_n": len(neutral_ids),
            "label_concordant_n": len(concordant_n),
            "concordant_rematched_pairs": {"instruction": int(instruction_concordant_pairs),
                "neutral": int(neutral_concordant_pairs), "caliper_s_pop": {"instruction": float(instruction_width),
                                                                            "neutral": float(neutral_width)}},
            "concordant_matched_auroc": concordant_auroc,
            "concordant_popqa_ids": concordant_n.popqa_id.astype(str).tolist(),
            "instruction_matched_ids": concordant_i.popqa_id.astype(str).tolist()}


def _plot(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    slugs = [s for s in MODELS if s in results["conditions"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    for slug in slugs:
        p = results["pilots"].get(slug, {})
        ax.bar(f"{slug}\nunans", p.get("neutral_unans_abstention_rate", 0))
        ax.bar(f"{slug}\nans", p.get("neutral_ans_abstention_rate", 0))
    ax.set_ylim(0, 1); ax.set_ylabel("Judged abstention rate"); fig.tight_layout()
    fig.savefig(output_dir / "e3_pilot_rates.png", dpi=150); plt.close(fig)
    for slug in slugs:
        matrix = np.asarray(results["conditions"][slug]["in_context_auroc"])
        fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(matrix, aspect="auto", vmin=.4, vmax=1, cmap="viridis")
        ax.set(xlabel="Suffix position", ylabel="Layer", title=f"{slug}: neutral in-context AUROC")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output_dir / f"e3_incontext_auroc_heatmap_{slug}.png", dpi=150); plt.close(fig)
    if "llama32_3b" in slugs:
        vals = results["conditions"]["llama32_3b"]["geometry"]["instruction_vs_neutral"]
        keys = [k for k in ("raw", "outlier_removed", "whitened") if k in vals]
        fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(keys, [vals[k] for k in keys])
        ax.set_ylabel("Cosine"); fig.tight_layout(); fig.savefig(output_dir / "e3_cos_instr_vs_neut.png", dpi=150); plt.close(fig)
    names = ["proj_own", "maxp", "entropy", "abstention_set_log_odds", "seq_logprob_mean", "log_s_pop"]
    fig, ax = plt.subplots(figsize=(9, 4)); width = .8 / max(len(slugs), 1)
    for j, slug in enumerate(slugs):
        a = results["conditions"][slug]["popqa_auroc"]
        ax.bar(np.arange(len(names)) + j * width, [a[n]["auroc"] for n in names], width, label=slug)
    ax.set_xticks(np.arange(len(names)) + width * (len(slugs)-1)/2, names, rotation=25, ha="right")
    ax.set_ylim(.4, 1); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "e3_popqa_auroc_bars.png", dpi=150); plt.close(fig)
    llama = results["conditions"].get("llama32_3b", {}).get("instruction_contrast", {}).get("transition_table")
    if llama:
        rows = sorted(llama)
        cols = sorted({k for v in llama.values() for k in v})
        arr = np.asarray([[llama[r].get(c, 0) for c in cols] for r in rows])
        fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(arr, cmap="Blues")
        ax.set(xticks=range(len(cols)), xticklabels=cols, yticks=range(len(rows)), yticklabels=rows,
               xlabel="Neutral label", ylabel="Registered label")
        for i in range(len(rows)):
            for j in range(len(cols)): ax.text(j, i, str(arr[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output_dir / "e3_label_transitions.png", dpi=150); plt.close(fig)


def _summary(results):
    lines = ["# E3 instruction ablation summary", "", "Exploratory post-registration analyses.", "",
             "| Condition | Pilot unans / ans abstention | In-context AUROC | PopQA matched pairs | Verdict |",
             "|---|---:|---:|---:|---|"]
    for slug, row in results["conditions"].items():
        p = results["pilots"].get(slug, {})
        v = row.get("verdict", "pending")
        lines.append(f"| {slug} | {p.get('neutral_unans_abstention_rate', float('nan')):.3f} / {p.get('neutral_ans_abstention_rate', float('nan')):.3f} | {row['in_context_locked_auroc']:.3f} | {row['n_matched_pairs']} | {v} |")
        lines.append("")
        lines.append(f"Power status: `{row['power_status']}`; residual label: `{row['residualization']['label']}`; incremental ΔAUROC: {row['incremental_value']['delta_auroc_cv']:.3f}.")
    lines += ["", f"Spot-check reliability passed: `{results['spot_checks']['reliability_passed']}`. PopQA-dependent claims are provisional when it is false.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures/exploratory")
    args = parser.parse_args(argv)
    require_tier1_complete(ROOT)
    prereg_path = assert_not_fresh(OUT / "e3_preregistration.json")
    preflight = _json(OUT / "e3_preflight.json")
    if preflight.get("template_date") not in {"static", "pinned"}:
        raise RuntimeError("E3 analysis is blocked until preflight establishes static or approved-pinned chat-template dates")
    reliability = _spot_check_flags()
    pilots = {}
    conditions = {}
    selected_models = args.model
    if not selected_models:
        selected_models = ["llama32_3b"]
        qwen_pilot_path = OUT / "e3_pilot_qwen25_3b.json"
        if qwen_pilot_path.exists() and _json(qwen_pilot_path).get("arm_included"):
            selected_models.append("qwen25_3b")
    for slug in selected_models:
        pilot_path = OUT / f"e3_pilot_{slug}.json"
        pilots[slug] = _json(pilot_path)
        if slug == "qwen25_3b" and not pilots[slug].get("arm_included"):
            raise RuntimeError("Qwen arm was excluded by the pre-registered neutral pilot")
        conditions[slug] = _condition(slug, args.bootstrap)
    llama = conditions.get("llama32_3b")
    for slug, condition in conditions.items():
        residual = condition["residualization"]["label"]
        resid_label = "residual_null_equivalent" if ("equivalent" in residual or "below_margin" in residual) else \
            "residual_signal_present" if "signal_present" in residual else "inconclusive"
        own_auc = condition["popqa_auroc"]["proj_own"]["ci95"]
        separation = "present" if own_auc[0] > .55 else \
            "absent" if own_auc[0] >= .45 and own_auc[1] <= .55 else "inconclusive"
        condition["verdict"] = e3_verdict(in_context_survives=condition["in_context_locked_auroc"] >= .90,
            aligned_with_instructed=condition.get("alignment"), separation=separation, residual=resid_label)
        condition["interpretation_labels"] = {"in_context_survives": condition["in_context_locked_auroc"] >= .90,
            "aligned_with_instructed": condition.get("alignment"), "closed_book_separation": separation,
            "residual": resid_label}
        condition["provisional"] = reliability["provisional"]
        if reliability["provisional"]:
            condition["interpretation_labels"] = {key: (f"{value}_PROVISIONAL"
                if key in {"closed_book_separation", "residual"} else value)
                for key, value in condition["interpretation_labels"].items()}
            condition["verdict"] = f"{condition['verdict']} _PROVISIONAL"
    if llama:
        llama["instruction_contrast"] = _instruction_contrast(llama, args.bootstrap)
    if "qwen25_3b" in conditions and llama:
        conditions["qwen25_3b"]["replication"] = qwen_replication(llama["verdict"], conditions["qwen25_3b"]["verdict"])
    result = {"header": exploratory_header({"analysis": "E3 A-F", "status": "exploratory_post_registration"}, ROOT),
        "preregistration_sha256": hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        "preflight": preflight, "pilots": pilots, "conditions": conditions,
        "spot_checks": reliability, "analysis_ids": ["A", "B", "C", "D", "E", "F"]}
    outdir = assert_not_fresh(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    write_exploratory_json(outdir / "e3_results.json", result, ROOT)
    (outdir / "e3_summary.md").write_text(_summary(result), encoding="utf-8", newline="\n")
    _plot(result, args.figure_dir)
    print(json.dumps({"conditions": {k: {"n_matched_pairs": v["n_matched_pairs"], "verdict": v.get("verdict")}
                       for k, v in conditions.items()}, "spot_checks": reliability}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
