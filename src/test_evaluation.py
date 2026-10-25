"""Stage 6: held-out test evaluation of H1-H3 with 119 control directions and a generation audit."""
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import CFG
from src.abstention_classifier import classify_texts
from src.eval_common import (ConditionCheckpoint, atomic_write_json, atomic_write_jsonl, build_chat_batches,
                             capture_residuals_and_logits, control_specs, evaluate_condition,
                             generate_with_hooks, load_b_ctrl_rows, load_split_pairs, log_odds_and_p_abs,
                             read_json, read_jsonl, slice_vectors_at_star, verify_native_hooks_match_nnsight,
                             write_csv_and_md)
from src.stats_utils import (binary_auroc, cluster_bootstrap, cluster_means, holm_bonferroni,
                             monte_carlo_p_greater, summarize_null, wilcoxon_greater)

TABLE4_COLUMNS = ["condition", "dataset_partition", "n_prompts", "mean_log_odds", "log_odds_ci_95_low",
                  "log_odds_ci_95_high", "mean_p_unknown", "gen_abstention_rate_strict",
                  "gen_abstention_rate_regex_v4", "gen_abstention_rate_judged", "gen_degenerate_rate",
                  "regex_judge_agreement", "unparseable_count", "gen_rate_ci_95_low", "gen_rate_ci_95_high",
                  "discrepancy_flagged"]
STAGE6_KEYS = {"stage", "model_name", "l_star", "i_star", "pos_idx", "n_test_pairs", "n_b_ctrl", "primary_tests",
               "secondary_tests", "necessity_curve", "sufficiency_curve", "controls_summary", "b_ctrl_summary",
               "family_breakdown", "generation_audit", "logit_gen_discrepancy", "branch_rule4", "projections",
               "per_prompt", "generation_rates", "classifier", "native_vs_nnsight_max_abs_diff", "source_stage4_sha256", "source_stage5_sha256",
               "elapsed_sec", "passed_acceptance_gate"}


def ckey(part: str, mode: str, tier: str = "learned", m: int = 0, alpha: float = 0.0) -> str:
    return f"{part}|{mode}|{tier}|{m}|{float(alpha):+.2f}"


BASE = {part: ckey(part, "none", "base") for part in ("unans", "ans", "bctrl")}


def alpha_tag(alpha: float) -> str:
    a = float(alpha)
    return ("neg" if a < 0 else "pos") + f"{abs(a):g}".replace(".", "p")


def generation_conditions() -> List[Tuple[str, str, str]]:
    """(audit condition name, dataset part, condition key) for the multi-token generation audit."""
    conds = [("test_unans_baseline", "unans", BASE["unans"]),
             ("test_unans_ablate_learned", "unans", ckey("unans", "ablate_all"))]
    conds += [(f"test_unans_steer_{alpha_tag(a)}_learned", "unans", ckey("unans", "steer_layer", alpha=a))
              for a in CFG.grid_alpha_nec if a != 0.0]
    conds += [("test_unans_ablate_concept", "unans", ckey("unans", "ablate_all", "concept")),
              ("test_unans_steer_neg1_concept", "unans", ckey("unans", "steer_layer", "concept", alpha=CFG.alpha_primary_neg)),
              ("test_ans_baseline", "ans", BASE["ans"])]
    conds += [(f"test_ans_steer_{alpha_tag(a)}_learned", "ans", ckey("ans", "steer_layer", alpha=a))
              for a in CFG.grid_alpha_suf if a != 0.0]
    conds += [("test_ans_steer_pos1_concept", "ans", ckey("ans", "steer_layer", "concept", alpha=CFG.alpha_primary_pos)),
              ("test_ans_ablate_learned", "ans", ckey("ans", "ablate_all")),
              ("b_ctrl_baseline", "bctrl", BASE["bctrl"])]
    return conds


# ----------------------------- loading ---------------------------------------
def load_and_validate_test_and_b_ctrl_datasets(pairs_path: Path = CFG.data_dir / "pairs.jsonl",
                                               b_ctrl_path: Path = CFG.data_dir / "b_ctrl.jsonl",
                                               splits_path: Path = CFG.data_dir / "splits.json"):
    test_pairs = load_split_pairs("test", pairs_path, splits_path)
    assert len(test_pairs) == CFG.n_pairs_test
    b_ctrl = load_b_ctrl_rows(b_ctrl_path, test_pairs)
    return test_pairs, b_ctrl


# ----------------------------- statistics (CPU) -------------------------------
def compute_template_clustered_bootstrap_ci(prompt_values, template_ids, n_boot=CFG.n_bootstrap,
                                            seed=CFG.seed_bootstrap) -> Dict[str, float]:
    return cluster_bootstrap(prompt_values, template_ids, n_boot, seed)


def compute_empirical_control_p_values(observed_delta: float, random_deltas, shuffled_deltas,
                                       concept_delta: float, direction: str = "less") -> Dict[str, Any]:
    """Rank p-values for the learned effect against each control tier.

    direction="less": the target effect is a decrease (H1); direction="greater": an increase (H2).
    """
    sign = -1.0 if direction == "less" else 1.0
    obs = sign * float(observed_delta)
    rnd = sign * np.asarray(random_deltas, dtype=np.float64)
    shf = sign * np.asarray(shuffled_deltas, dtype=np.float64)
    con = sign * float(concept_delta)
    r_sum, s_sum = summarize_null(obs, rnd), summarize_null(obs, shf)
    ratio = con / obs if obs != 0 else None
    raw_rnd, raw_shf = np.asarray(random_deltas, dtype=np.float64), np.asarray(shuffled_deltas, dtype=np.float64)
    return {"p_random": monte_carlo_p_greater(obs, rnd), "p_shuffled": monte_carlo_p_greater(obs, shf),
            "random_mean": float(raw_rnd.mean()), "random_std": r_sum["std"], "random_min": float(raw_rnd.min()),
            "random_max": float(raw_rnd.max()), "random_z": r_sum["z_score"], "random_rank": r_sum["rank_of_observed"],
            "shuffled_mean": float(raw_shf.mean()), "shuffled_std": s_sum["std"], "shuffled_min": float(raw_shf.min()),
            "shuffled_max": float(raw_shf.max()), "shuffled_z": s_sum["z_score"],
            "shuffled_rank": s_sum["rank_of_observed"], "concept_delta": float(concept_delta),
            "concept_ratio": ratio,
            "passes_random_gate": monte_carlo_p_greater(obs, rnd) <= 1.0 / (len(rnd) + 1) + 1e-12,
            "passes_shuffled_gate": monte_carlo_p_greater(obs, shf) <= 1.0 / (len(shf) + 1) + 1e-12,
            "passes_concept_gate": ratio is not None and ratio <= CFG.concept_effect_ratio_max}


def apply_holm_bonferroni(p_values_dict: Dict[str, float]) -> Dict[str, float]:
    return {r["test_id"]: r["holm_threshold"] for r in holm_bonferroni(p_values_dict)}


def _effect_test(effect: np.ndarray, tids: List[str], null_effects: Dict[str, np.ndarray],
                 n_boot: int, seed: int) -> Dict[str, Any]:
    """Effects are oriented so larger = stronger hypothesised effect."""
    _, cm = cluster_means(effect, tids)
    w = wilcoxon_greater(cm)
    obs = float(np.mean(effect))
    out = {"mean_effect": obs, "cluster_means": cm.tolist(), "wilcoxon_statistic": w["statistic"],
           "p_wilcoxon": w["p_value"], "n_clusters": w["n"], "bootstrap": cluster_bootstrap(effect, tids, n_boot, seed)}
    for tier, null in null_effects.items():
        out[f"{tier}_null"] = summarize_null(obs, null)
    out["p_mc"] = out["random_null"]["p_rank"]
    out["mc_rank_out_of_100"] = out["random_null"]["rank_of_observed"]
    out["p_shuffled"] = out["shuffled_null"]["p_rank"]
    concept = float(null_effects["concept"][0])
    out["concept_effect"] = concept
    out["r_concept_effect_ratio"] = concept / obs if obs != 0 else None
    out["concept_confound_flag"] = bool(obs != 0 and concept / obs > CFG.concept_effect_ratio_max)
    out["passes_mc_rank_1"] = bool(out["p_mc"] <= 1.0 / (len(null_effects["random"]) + 1) + 1e-12)
    return out


def _rate(flags: List[bool]) -> float:
    return float(np.mean(flags)) if len(flags) else float("nan")


def analyze_stage6(cond: Dict[str, Dict[str, np.ndarray]], gens: Dict[str, List[str]],
                   test_pairs: List[Dict[str, Any]], b_ctrl_rows: List[Dict[str, Any]],
                   n_ctrl_random: int = CFG.n_ctrl_random, n_ctrl_shuffled: int = CFG.n_ctrl_shuffled,
                   n_boot: int = CFG.n_bootstrap, seed: int = CFG.seed_bootstrap, offline: bool = False, backend: Optional[str] = None) -> Dict[str, Any]:
    """Pure-numpy H1-H3 analysis from cached per-prompt log-odds, projections and generations."""
    from src.pilot_check import legacy_v4_abstention_flags
    tids = [p["template_id"] for p in test_pairs]
    fams = np.array([p["family"] for p in test_pairs])
    S = {k: np.asarray(v["log_odds"], dtype=np.float64) for k, v in cond.items() if "log_odds" in v}
    P = {k: np.asarray(v["p_abs"], dtype=np.float64) for k, v in cond.items() if "p_abs" in v}
    base_u, base_a = S[BASE["unans"]], S[BASE["ans"]]

    def tier_effects(part: str, mode: str, alpha: float, orient: float) -> Dict[str, np.ndarray]:
        base = S[BASE[part]]
        res = {}
        for tier, n in (("random", n_ctrl_random), ("shuffled", n_ctrl_shuffled), ("concept", 1)):
            res[tier] = np.array([orient * float(np.mean(S[ckey(part, mode, tier, m, alpha)] - base)) for m in range(n)])
        return res

    def tier_means(part: str, mode: str, alpha: float) -> Dict[str, List[float]]:
        return {tier: [float(np.mean(S[ckey(part, mode, tier, m, alpha)])) for m in range(n)]
                for tier, n in (("random", n_ctrl_random), ("shuffled", n_ctrl_shuffled), ("concept", 1))}

    # H1 (all-layer ablation), H1b (single-layer subtraction) and H2 (single-layer addition).
    h1_eff = base_u - S[ckey("unans", "ablate_all")]
    h1b_eff = base_u - S[ckey("unans", "steer_layer", alpha=CFG.alpha_primary_neg)]
    h2_eff = S[ckey("ans", "steer_layer", alpha=CFG.alpha_primary_pos)] - base_a
    h1 = _effect_test(h1_eff, tids, tier_effects("unans", "ablate_all", 0.0, -1.0), n_boot, seed)
    h1b = _effect_test(h1b_eff, tids, tier_effects("unans", "steer_layer", CFG.alpha_primary_neg, -1.0), n_boot, seed)
    h2 = _effect_test(h2_eff, tids, tier_effects("ans", "steer_layer", CFG.alpha_primary_pos, 1.0), n_boot, seed)
    for d, name in ((h1, "H1"), (h1b, "H1b"), (h2, "H2")):
        d["test_id"] = name

    # Plan-style empirical p-values on the raw (signed) deltas, kept for transparency.
    for d, part, mode, alpha, direction in ((h1, "unans", "ablate_all", 0.0, "less"),
                                            (h1b, "unans", "steer_layer", CFG.alpha_primary_neg, "less"),
                                            (h2, "ans", "steer_layer", CFG.alpha_primary_pos, "greater")):
        sgn = -1.0 if direction == "less" else 1.0
        eff = tier_effects(part, mode, alpha, 1.0)
        d["empirical_control_p_values"] = compute_empirical_control_p_values(
            sgn * d["mean_effect"], eff["random"], eff["shuffled"], float(eff["concept"][0]), direction)

    # Family breakdown.
    family = {}
    for fam in ("A", "B"):
        mask = fams == fam
        ft = [t for t, keep in zip(tids, mask) if keep]
        family[fam] = {name: {**cluster_bootstrap(eff[mask], ft, n_boot, seed),
                              "p_wilcoxon": wilcoxon_greater(cluster_means(eff[mask], ft)[1])["p_value"]}
                       for name, eff in (("H1", h1_eff), ("H1b", h1b_eff), ("H2", h2_eff))}

    # Generation rates per audited condition.
    audit_rows, rates, labels_by_key = [], {}, {}
    item_tids = {"unans": tids, "ans": tids, "bctrl": [r["template_id"] for r in b_ctrl_rows]}
    for name, part, key in generation_conditions():
        if key not in gens or key not in S:
            continue
        part_prompts = ([p["unans_prompt"] for p in test_pairs] if part == "unans" else
                        [p["ans_prompt"] for p in test_pairs] if part == "ans" else
                        [r["prompt"] for r in b_ctrl_rows])
        records = classify_texts([{"uid": f"{name}:{i}", "question": q, "response": text}
                                  for i, (q, text) in enumerate(zip(part_prompts, gens[key]))], offline=offline, backend=backend)
        legacy = [legacy_v4_abstention_flags(t) for t in gens[key]]
        strict = [c[0] for c in legacy]
        regex = [c[1] for c in legacy]
        unparseable = sum(r["final_label"] == "UNPARSEABLE" for r in records)
        if len(records) and unparseable / len(records) > 0.01:
            raise RuntimeError(f"{name} has {unparseable}/{len(records)} UNPARSEABLE judge outputs (>1%)")
        valid_idx = [i for i, r in enumerate(records) if r["final_label"] != "UNPARSEABLE"]
        judged = [records[i]["final_label"] == "ABSTAIN" for i in valid_idx]
        degenerate = [records[i]["final_label"] == "DEGENERATE" for i in valid_idx]
        agreement = [records[i]["agree"] for i in valid_idx]
        labels_by_key[key] = records
        lo_b = cluster_bootstrap(S[key], item_tids[part], n_boot, seed)
        rb = cluster_bootstrap(np.asarray(judged, dtype=np.float64), [item_tids[part][i] for i in valid_idx], n_boot, seed)
        rates[name] = {"strict": _rate(strict), "regex": _rate(regex), "regex_v4": _rate(regex),
                       "judged": _rate(judged), "degenerate": _rate(degenerate),
                       "regex_judge_agreement": _rate([x is True for x in agreement]),
                       "unparseable_count": unparseable}
        audit_rows.append({"condition": name, "dataset_partition": part, "n_prompts": len(regex),
                           "mean_log_odds": float(S[key].mean()), "log_odds_ci_95_low": lo_b["ci95_low"],
                           "log_odds_ci_95_high": lo_b["ci95_high"], "mean_p_unknown": float(P[key].mean()),
                           "gen_abstention_rate_strict": _rate(strict), "gen_abstention_rate_regex": _rate(regex),
                           "gen_abstention_rate_regex_v4": _rate(regex), "gen_abstention_rate_judged": _rate(judged),
                           "gen_degenerate_rate": _rate(degenerate), "regex_judge_agreement": rates[name]["regex_judge_agreement"],
                           "unparseable_count": unparseable, "gen_rate_ci_95_low": rb["ci95_low"],
                           "gen_rate_ci_95_high": rb["ci95_high"], "discrepancy_flagged": False, "_key": key})

    # Branch Rule 4: logit vs. generation discrepancy at the primary operating points.
    br4 = {}
    for label, cond_name, base_name in (("ablate_all_unans", "test_unans_ablate_learned", "test_unans_baseline"),
                                        ("steer_pos1_ans", f"test_ans_steer_{alpha_tag(CFG.alpha_primary_pos)}_learned",
                                         "test_ans_baseline")):
        row = next((r for r in audit_rows if r["condition"] == cond_name), None)
        base = next((r for r in audit_rows if r["condition"] == base_name), None)
        if row is None or base is None:
            continue
        d_s = row["mean_log_odds"] - base["mean_log_odds"]
        d_r = row["gen_abstention_rate_judged"] - base["gen_abstention_rate_judged"]
        flag = bool((np.sign(d_s) != np.sign(d_r)) or
                    (abs(d_s) >= CFG.discrepancy_log_odds and abs(d_r) < CFG.discrepancy_rate))
        row["discrepancy_flagged"] = flag
        br4[label] = {"delta_log_odds": float(d_s), "delta_gen_rate_judged": float(d_r),
                      "delta_gen_rate_regex_v4": row["gen_abstention_rate_regex_v4"] - base["gen_abstention_rate_regex_v4"],
                      "flagged": flag}
    discrepancy = any(v["flagged"] for v in br4.values())

    # Dose-response curves.
    def curve(part: str, grid, base_name: str, cond_fmt: str) -> List[Dict[str, Any]]:
        rows = []
        entries = [("steer_layer", a) for a in grid]
        if part == "unans":
            entries.append(("ablate_all", None))
        for mode, a in entries:
            if mode == "steer_layer" and a == 0.0:
                key, gname = BASE[part], base_name
            elif mode == "steer_layer":
                key, gname = ckey(part, mode, alpha=a), cond_fmt.format(alpha_tag(a))
            else:
                key, gname = ckey(part, mode), "test_unans_ablate_learned"
            b = cluster_bootstrap(S[key], tids, n_boot, seed)
            eff = cluster_bootstrap(S[key] - S[BASE[part]], tids, n_boot, seed)
            rate = rates.get(gname, {})
            rows.append({"mode": mode, "alpha": a, "mean_log_odds": b["mean"], "log_odds_ci95": [b["ci95_low"], b["ci95_high"]],
                         "mean_delta_log_odds": eff["mean"], "delta_ci95": [eff["ci95_low"], eff["ci95_high"]],
                         "first_token_abstention_rate": float((S[key] > 0).mean()),
                         "gen_abstention_rate_regex": rate.get("regex_v4"),
                         "gen_abstention_rate_regex_v4": rate.get("regex_v4"),
                         "gen_abstention_rate_judged": rate.get("judged"),
                         "gen_degenerate_rate": rate.get("degenerate"),
                         "regex_judge_agreement": rate.get("regex_judge_agreement"),
                         "unparseable_count": rate.get("unparseable_count", 0),
                         "gen_abstention_rate_strict": rate.get("strict")})
        return rows

    nec = curve("unans", CFG.grid_alpha_nec, "test_unans_baseline", "test_unans_steer_{}_learned")
    suf = curve("ans", CFG.grid_alpha_suf, "test_ans_baseline", "test_ans_steer_{}_learned")

    controls = {
        "ablate_all_unans": {"effects": {k: v.tolist() for k, v in tier_effects("unans", "ablate_all", 0.0, -1.0).items()},
                             "mean_log_odds": tier_means("unans", "ablate_all", 0.0), "learned_effect": h1["mean_effect"]},
        "steer_neg1_unans": {"effects": {k: v.tolist() for k, v in tier_effects("unans", "steer_layer", CFG.alpha_primary_neg, -1.0).items()},
                             "mean_log_odds": tier_means("unans", "steer_layer", CFG.alpha_primary_neg), "learned_effect": h1b["mean_effect"]},
        "steer_pos1_ans": {"effects": {k: v.tolist() for k, v in tier_effects("ans", "steer_layer", CFG.alpha_primary_pos, 1.0).items()},
                           "mean_log_odds": tier_means("ans", "steer_layer", CFG.alpha_primary_pos), "learned_effect": h2["mean_effect"]},
        "effect_orientation": "positive = hypothesised direction (reduction for unans, increase for ans)",
    }

    # H3: Family B unanswerable vs. private-detail answerable control projections.
    proj_u, proj_a, proj_c = (np.asarray(cond[f"proj|{p}"]["proj"], dtype=np.float64) for p in ("unans", "ans", "bctrl"))
    idx_by_pair = {p["pair_id"]: i for i, p in enumerate(test_pairs)}
    paired_u = np.array([proj_u[idx_by_pair[r["pair_id"]]] for r in b_ctrl_rows])
    b_tids = [r["template_id"] for r in b_ctrl_rows]
    b_mask = fams == "B"
    b_test_tids = [t for t, keep in zip(tids, b_mask) if keep]
    ids_u, cm_u = cluster_means(proj_u[b_mask], b_test_tids)
    ids_c, cm_c = cluster_means(proj_c, b_tids)
    assert ids_u == ids_c, "Family B test templates must match B-ctrl templates"
    h3_w = wilcoxon_greater(cm_u - cm_c)
    bctrl_key = BASE["bctrl"]
    bctrl_records = labels_by_key.get(bctrl_key, [])
    bctrl_regex = [legacy_v4_abstention_flags(t)[1] for t in gens.get(bctrl_key, [])]
    bctrl_valid = [r for r in bctrl_records if r["final_label"] != "UNPARSEABLE"]
    bctrl_judged = [r["final_label"] == "ABSTAIN" for r in bctrl_valid]
    bctrl_degenerate = [r["final_label"] == "DEGENERATE" for r in bctrl_valid]
    bctrl_rate = _rate(bctrl_judged) if bctrl_judged else float("nan")
    h3 = {"test_id": "H3", "mean_effect": float((cm_u - cm_c).mean()), "cluster_differences": (cm_u - cm_c).tolist(),
          "wilcoxon_statistic": h3_w["statistic"], "p_wilcoxon": h3_w["p_value"], "n_clusters": h3_w["n"],
          "bootstrap": cluster_bootstrap(paired_u - proj_c, b_tids, n_boot, seed),
          "b_ctrl_gen_abstention_rate_regex": _rate(bctrl_regex),
          "b_ctrl_gen_abstention_rate_judged": bctrl_rate,
          "b_ctrl_gen_degenerate_rate": _rate(bctrl_degenerate),
          "b_ctrl_first_token_abstention_rate": float((S[bctrl_key] > 0).mean()),
          "b_ctrl_abstention_ok": bool(bctrl_rate <= CFG.b_ctrl_abstention_max),
          "auroc_unansB_vs_bctrl": binary_auroc(proj_u[b_mask], proj_c),
          "auroc_unansB_vs_ansB": binary_auroc(proj_u[b_mask], proj_a[b_mask]),
          "auroc_bctrl_vs_ansB": binary_auroc(proj_c, proj_a[b_mask])}
    b_summary = {"mean_proj_unans_B": float(proj_u[b_mask].mean()), "mean_proj_ans_B": float(proj_a[b_mask].mean()),
                 "mean_proj_b_ctrl": float(proj_c.mean()), "mean_log_odds_unans_B": float(base_u[b_mask].mean()),
                 "mean_log_odds_ans_B": float(base_a[b_mask].mean()), "mean_log_odds_b_ctrl": float(S[bctrl_key].mean()),
                 "b_ctrl_gen_abstention_rate_regex": bctrl_rate,
                 "b_ctrl_gen_abstention_rate_strict": _rate([legacy_v4_abstention_flags(t)[0] for t in gens.get(bctrl_key, [])])
                 if bctrl_regex else None}

    prelim = holm_bonferroni({"H1": h1["p_wilcoxon"], "H2": h2["p_wilcoxon"], "H3": h3["p_wilcoxon"]},
                             extra_criteria={"H1": h1["passes_mc_rank_1"], "H2": h2["passes_mc_rank_1"],
                                             "H3": h3["b_ctrl_abstention_ok"]})
    return {"primary_tests": {"H1": h1, "H2": h2, "H3": h3}, "secondary_tests": {"H1b_steer_neg1": h1b},
            "preliminary_holm_h1_h3": prelim, "necessity_curve": nec, "sufficiency_curve": suf,
            "controls_summary": controls, "b_ctrl_summary": b_summary, "family_breakdown": family,
            "generation_audit": [{k: v for k, v in r.items() if k != "_key"} for r in audit_rows],
            "generation_rates": rates, "generation_classifier_records": labels_by_key, "branch_rule4": br4, "logit_gen_discrepancy": discrepancy,
            "projections": {"test_unans": proj_u.tolist(), "test_ans": proj_a.tolist(), "b_ctrl": proj_c.tolist()},
            "per_prompt": {"test_unans_baseline": base_u.tolist(), "test_ans_baseline": base_a.tolist(),
                           "test_unans_ablate_learned": S[ckey("unans", "ablate_all")].tolist(),
                           "test_unans_steer_neg1_learned": S[ckey("unans", "steer_layer", alpha=CFG.alpha_primary_neg)].tolist(),
                           "test_ans_steer_pos1_learned": S[ckey("ans", "steer_layer", alpha=CFG.alpha_primary_pos)].tolist(),
                           "b_ctrl_baseline": S[bctrl_key].tolist(), "template_ids": tids,
                           "families": fams.tolist(), "pair_ids": [p["pair_id"] for p in test_pairs]}}


# ----------------------------- GPU runner -----------------------------------
def generate_with_active_intervention(raw_model, tokenizer, batch_inputs, is_layer_output_tuple, mode="none",
                                      r_vec=None, target_layer=None, alpha=0.0, max_new_tokens=CFG.max_new_tokens):
    contract = {"is_layer_output_tuple": is_layer_output_tuple}
    batches = batch_inputs if isinstance(batch_inputs, list) else [batch_inputs]
    return generate_with_hooks(raw_model, tokenizer, batches, contract, mode, r_vec, target_layer, alpha, max_new_tokens)


def _condition_plan(vectors: Dict[str, Any], l_star: int) -> List[Tuple[str, str, Any, Optional[int], float]]:
    """(key, mode, vector, layer, alpha) for every intervention condition in a fixed order."""
    plan = [(ckey("unans", "ablate_all"), "ablate_all", vectors["learned_unit"], None, 0.0),
            (ckey("ans", "ablate_all"), "ablate_all", vectors["learned_unit"], None, 0.0)]
    plan += [(ckey("unans", "steer_layer", alpha=a), "steer_layer", vectors["learned_diff"], l_star, a)
             for a in CFG.grid_alpha_nec if a != 0.0]
    plan += [(ckey("ans", "steer_layer", alpha=a), "steer_layer", vectors["learned_diff"], l_star, a)
             for a in CFG.grid_alpha_suf if a != 0.0]
    plan.append((ckey("ans", "ablate_all", "concept"), "ablate_all", vectors["concept_unit"], None, 0.0))
    for tier, m, unit, steer in control_specs(vectors):
        plan.append((ckey("unans", "ablate_all", tier, m), "ablate_all", unit, None, 0.0))
        plan.append((ckey("unans", "steer_layer", tier, m, CFG.alpha_primary_neg), "steer_layer", steer, l_star, CFG.alpha_primary_neg))
        plan.append((ckey("ans", "steer_layer", tier, m, CFG.alpha_primary_pos), "steer_layer", steer, l_star, CFG.alpha_primary_pos))
    return plan


def run_stage6_test_evaluation(raw_model, tokenizer, contract, directions_dict, selection_dict, test_pairs, b_ctrl_rows,
                               batch_size=CFG.batch_size_forward, checkpoint_path=CFG.stage6_checkpoint_path,
                               force=False, source_stage4_sha256="", source_stage5_sha256="", verify_nnsight=True, offline=False, generation_only=False):
    started = time.perf_counter()
    l_star, i_star, pos_idx = int(selection_dict["l_star"]), int(selection_dict["i_star"]), int(selection_dict["pos_idx"])
    vectors = slice_vectors_at_star(directions_dict, l_star, pos_idx)
    sp, suffix = contract["system_prompt_used"], list(contract["suffix_token_ids"])
    batches = {"unans": build_chat_batches(tokenizer, [p["unans_prompt"] for p in test_pairs], sp, suffix, batch_size),
               "ans": build_chat_batches(tokenizer, [p["ans_prompt"] for p in test_pairs], sp, suffix, batch_size),
               "bctrl": build_chat_batches(tokenizer, [r["prompt"] for r in b_ctrl_rows], sp, suffix, batch_size)}
    parity = (verify_native_hooks_match_nnsight(raw_model, tokenizer, batches["unans"][0], contract,
                                                vectors["learned_unit"], vectors["learned_diff"], l_star)
              if verify_nnsight else {})
    print(f"[Stage 6] native-vs-nnsight max |diff| = {parity}", flush=True)
    fp = hashlib.sha256(f"{source_stage4_sha256}|{source_stage5_sha256}|{l_star}|{pos_idx}|{len(test_pairs)}|"
                        f"{len(b_ctrl_rows)}|{sp}|{CFG.max_new_tokens}".encode()).hexdigest()
    ckpt = ConditionCheckpoint(checkpoint_path, fp, force=force, every=12)

    unit_star = vectors["learned_unit"].double()
    for part in ("unans", "ans", "bctrl"):
        if BASE[part] in ckpt and f"proj|{part}" in ckpt:
            continue
        acts, logits = capture_residuals_and_logits(raw_model, batches[part], contract, [l_star], i_star)
        s, pabs = log_odds_and_p_abs(logits, contract["unknown_token_ids"])
        ckpt.put(BASE[part], {"log_odds": s.double().numpy(), "p_abs": pabs.double().numpy()})
        ckpt.put(f"proj|{part}", {"proj": (acts[:, 0, :].double() @ unit_star).numpy()})
    plan = _condition_plan(vectors, l_star)
    todo = [c for c in plan if c[0] not in ckpt]
    print(f"[Stage 6] {len(plan)} intervention conditions, {len(todo)} to run", flush=True)
    for n, (key, mode, vec, layer, alpha) in enumerate(todo, 1):
        part = key.split("|")[0]
        res = evaluate_condition(raw_model, batches[part], contract, mode, vec, layer, alpha)
        ckpt.put(key, res)
        if n % 25 == 0 or n == len(todo):
            print(f"[Stage 6] conditions {n}/{len(todo)} ({time.perf_counter() - started:.0f}s)", flush=True)
    ckpt.flush()

    plan_by_key = {c[0]: c for c in plan}
    gens = {}
    for name, part, key in generation_conditions():
        gkey = f"gen|{key}"
        if gkey not in ckpt:
            if key == BASE[part]:
                mode, vec, layer, alpha = "none", None, None, 0.0
            else:
                _, mode, vec, layer, alpha = plan_by_key[key]
            ckpt.put(gkey, {"texts": generate_with_hooks(raw_model, tokenizer, batches[part], contract, mode, vec,
                                                         layer, alpha, CFG.max_new_tokens)})
            print(f"[Stage 6] generated {name}", flush=True)
        gens[key] = list(ckpt.get(gkey)["texts"])
    ckpt.flush()

    cond = {k: v for k, v in ckpt.results.items() if not k.startswith("gen|")}
    analysis = analyze_stage6(cond, gens, test_pairs, b_ctrl_rows, offline=offline, backend=("regex_only" if generation_only else None))
    classifier_records = analysis.pop("generation_classifier_records")
    gen_samples = build_generation_samples(cond, gens, test_pairs, b_ctrl_rows, classifier_records, offline=offline)
    if generation_only:
        for row in gen_samples:
            row.update(judge_label=None, final_label=None, classifier_final=False, regex_judge_agree=None, judge_margin=None)
    flat_records = [r for group in classifier_records.values() for r in group]
    result = {"stage": 6, "model_name": contract["model_name"], "l_star": l_star, "i_star": i_star, "pos_idx": pos_idx,
              "classifier": {"backend": flat_records[0].get("classifier_backend", CFG.judge_backend) if flat_records else CFG.judge_backend,
                             "model_id": flat_records[0].get("judge_model_id", CFG.judge_model_id) if flat_records else CFG.judge_model_id,
                             "prompt_version": "v1",
                             "unparseable_count": sum(r["final_label"] == "UNPARSEABLE" for r in flat_records),
                             "unparseable_by_condition": {name: sum(r["final_label"] == "UNPARSEABLE" for r in group)
                                                          for name, group in classifier_records.items()},
                             "classifier_final": all(r["classifier_final"] for r in flat_records)},
              "r_norm_at_star": vectors["norm"], "cos_concept_pre_ortho_at_star": float(selection_dict["cos_concept_pre_ortho_at_star"]),
              "n_test_pairs": len(test_pairs), "n_b_ctrl": len(b_ctrl_rows), "n_bootstrap": CFG.n_bootstrap,
              "seed_bootstrap": CFG.seed_bootstrap, "grid_alpha_nec": list(CFG.grid_alpha_nec),
              "grid_alpha_suf": list(CFG.grid_alpha_suf), **analysis, "native_vs_nnsight_max_abs_diff": parity,
              "source_stage4_sha256": source_stage4_sha256, "source_stage5_sha256": source_stage5_sha256,
              "elapsed_sec": time.perf_counter() - started, "passed_acceptance_gate": bool(all(r["classifier_final"] for r in flat_records))}
    return result, gen_samples


def build_generation_samples(cond, gens, test_pairs, b_ctrl_rows, labels_by_key=None, offline=False, generation_only=False) -> List[Dict[str, Any]]:
    from src.pilot_check import legacy_v4_abstention_flags
    from src.abstention_classifier import regex_label
    labels_by_key = labels_by_key or {}
    rows = []
    for name, part, key in generation_conditions():
        if key not in gens:
            continue
        items = b_ctrl_rows if part == "bctrl" else test_pairs
        prompts = ([p["unans_prompt"] for p in test_pairs] if part == "unans" else
                   [p["ans_prompt"] for p in test_pairs] if part == "ans" else [r["prompt"] for r in b_ctrl_rows])
        records = labels_by_key.get(key)
        if records is None and not generation_only:
            records = classify_texts([{"uid": f"{name}:{i}", "question": q, "response": t}
                                      for i, (q, t) in enumerate(zip(prompts, gens[key]))], offline=offline)
        for i, (item, text) in enumerate(zip(items, gens[key])):
            strict, regex = legacy_v4_abstention_flags(text)
            if generation_only:
                rows.append({"condition":name,"dataset_part":part,
                             "item_id":item.get("b_ctrl_id",item.get("pair_id")),"pair_id":item["pair_id"],
                             "template_id":item["template_id"],"family":item.get("family","B"),
                             "log_odds":float(cond[key]["log_odds"][i]),"p_unknown":float(cond[key]["p_abs"][i]),
                             "generated_text":text,"abstained_strict":bool(strict),"abstained_regex":bool(regex),
                             "abstained_regex_v4":bool(regex),"regex_label":regex_label(text),
                             "judge_label":None,"final_label":None,"classifier_final":False,
                             "judge_margin":None,"generation_batch_size":None})
                continue
            record = records[i]
            rows.append({"condition": name, "dataset_part": part,
                         "item_id": item.get("b_ctrl_id", item.get("pair_id")), "pair_id": item["pair_id"],
                         "template_id": item["template_id"], "family": item.get("family", "B"),
                         "log_odds": float(cond[key]["log_odds"][i]), "p_unknown": float(cond[key]["p_abs"][i]),
                         "generated_text": text, "abstained_strict": bool(strict), "abstained_regex": bool(regex),
                         "abstained_regex_v4": bool(regex), "regex_label": record["regex_label"],
                         "judge_label": record["judge_label"], "final_label": record["final_label"],
                         "classifier_final": record["classifier_final"], "regex_judge_agree": record["agree"],
                         "judge_prompt_sha256": record.get("judge_prompt_sha256"),
                         "judge_model_id": record.get("judge_model_id"), "judge_model_revision": record.get("judge_model_revision"),
                         "judge_quantization": record.get("judge_quantization"), "judge_logprobs": record.get("judge_logprobs"),
                         "judge_margin": record.get("judge_margin"), "judge_prompt_version": record.get("judge_prompt_version")})
    return rows

def save_stage6_artifacts(results_dict, gen_samples, results_path=CFG.stage6_out_path, gen_path=CFG.stage6_gen_path,
                          table1_csv_path=CFG.table1_csv_path, table1_md_path=CFG.table1_md_path,
                          checkpoint_path=None, table4_csv_path=CFG.table4_csv_path,
                          table4_md_path=CFG.table4_md_path) -> None:
    from src.results_tables import write_table1
    atomic_write_json(results_path, results_dict)
    atomic_write_jsonl(gen_path, gen_samples)
    write_csv_and_md(results_dict["generation_audit"], TABLE4_COLUMNS, table4_csv_path, table4_md_path,
                     "Table 4. First-token log-odds vs. 16-token greedy generation audit (held-out test)")
    write_table1(table1_csv_path, table1_md_path, stage6=results_dict)
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()


def verify_stage6_outputs(results_path=CFG.stage6_out_path, gen_path=CFG.stage6_gen_path,
                          table1_csv_path=CFG.table1_csv_path, table1_md_path=CFG.table1_md_path,
                          table4_csv_path=CFG.table4_csv_path, stage5_path=None) -> Dict[str, Any]:
    from src.results_tables import TABLE1_COLUMNS
    from src.eval_common import read_csv_rows
    paths = [Path(p) for p in (results_path, gen_path, table1_csv_path, table1_md_path, table4_csv_path)]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths), "Stage 6 output missing or empty"
    d = read_json(paths[0])
    missing = STAGE6_KEYS - d.keys()
    assert not missing, f"Stage 6 JSON missing keys: {sorted(missing)}"
    assert d["stage"] == 6 and d["passed_acceptance_gate"] is True
    n, nb = int(d["n_test_pairs"]), int(d["n_b_ctrl"])
    for hid in ("H1", "H2", "H3"):
        t = d["primary_tests"][hid]
        assert 0.0 <= t["p_wilcoxon"] <= 1.0, f"{hid} p-value out of range"
        assert t["bootstrap"]["ci95_low"] <= t["bootstrap"]["ci95_high"], f"{hid} bootstrap CI is inverted"
    for hid in ("H1", "H2"):
        t = d["primary_tests"][hid]
        assert 1.0 / (CFG.n_ctrl_random + 1) - 1e-12 <= t["p_mc"] <= 1.0
        assert 1.0 / (CFG.n_ctrl_shuffled + 1) - 1e-12 <= t["p_shuffled"] <= 1.0
        assert t["random_null"]["n"] == CFG.n_ctrl_random and t["shuffled_null"]["n"] == CFG.n_ctrl_shuffled
    for k in ("ablate_all_unans", "steer_neg1_unans", "steer_pos1_ans"):
        eff = d["controls_summary"][k]["effects"]
        assert (len(eff["random"]), len(eff["shuffled"]), len(eff["concept"])) == (CFG.n_ctrl_random, CFG.n_ctrl_shuffled, 1)
    assert len(d["necessity_curve"]) == len(CFG.grid_alpha_nec) + 1 and len(d["sufficiency_curve"]) == len(CFG.grid_alpha_suf)
    assert len(d["projections"]["test_unans"]) == len(d["projections"]["test_ans"]) == n
    assert len(d["projections"]["b_ctrl"]) == nb
    assert isinstance(d["logit_gen_discrepancy"], bool)
    gens = read_jsonl(paths[1])
    expected = sum(nb if part == "bctrl" else n for _, part, _ in generation_conditions())
    assert len(gens) == expected, f"Expected {expected} generation records, found {len(gens)}"
    assert {g["condition"] for g in gens} == {c[0] for c in generation_conditions()}
    rows = read_csv_rows(paths[2])
    assert rows and list(rows[0].keys()) == TABLE1_COLUMNS
    assert {"H1", "H2", "H3"} <= {r["test_id"] for r in rows}
    assert len(read_csv_rows(paths[4])) == len(generation_conditions())
    if stage5_path and Path(stage5_path).exists():
        sel = read_json(stage5_path)
        assert (d["l_star"], d["i_star"], d["pos_idx"]) == (sel["l_star"], sel["i_star"], sel["pos_idx"])
    return d
