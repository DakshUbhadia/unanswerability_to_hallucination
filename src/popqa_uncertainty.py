"""Stage 8: closed-book PopQA epistemic-uncertainty test (H4, H5), Holm-Bonferroni over H1-H5, Fig 2A/S1 data."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from src.config import CFG
from src.tokenization_contract import (CONTRACT_VERSION, canonical_input_ids, chat_template_sha256,
                                      input_ids_sha256)
from src.eval_common import (atomic_write_json, build_chat_batches, capture_residuals_and_logits, log_odds_and_p_abs,
                             read_json, read_jsonl, slice_vectors_at_star, write_csv_and_md)
from src.stats_utils import auroc_rows, binary_auroc, holm_bonferroni, paired_bootstrap_auroc, summarize_null

TABLE3_COLUMNS = ["score", "description", "auroc", "ci_95_low", "ci_95_high", "delta_vs_proj", "delta_ci_95_low",
                  "delta_ci_95_high", "p_value", "p_value_test"]
VIOLIN_GROUPS = ["test_unans", "test_ans", "b_ctrl", "popqa_abstain", "popqa_matched_wrong", "popqa_matched_correct"]
VERDICT_B="Hypothesis_B_Internal_Uncertainty_Detector"
VERDICT_A="Hypothesis_A_Contextual_Insufficiency_Detector"
VERDICT_MIXED="Mixed_Not_Supported_Discriminating_Tests_Disagree"
VERDICT_INCONCLUSIVE="Inconclusive_Confound_or_Methodological_Failure"
VERDICT_DOMAIN_FAIL="Inconclusive_Domain_Transfer_Failure"
STAGE8_KEYS = {"stage", "model_name", "l_star", "i_star", "pos_idx", "n_matched_per_group", "n_abstain", "h4", "h5",
               "auroc_proj", "auroc_maxp", "auroc_ent", "holm_bonferroni_table", "interpretation", "violin_6groups",
               "layerwise_auroc_exploratory", "matched_items", "source_stage4_sha256", "source_stage6_sha256",
               "elapsed_sec", "passed_acceptance_gate", "label_source", "verified_input_hash_count"}


def compute_binary_auroc(scores_pos, scores_neg) -> float:
    return binary_auroc(scores_pos, scores_neg)


def compute_paired_bootstrap_auroc_comparison(pos_scores_dict, neg_scores_dict, primary_key="proj_learned",
                                              n_boot=CFG.n_bootstrap, seed=CFG.seed_bootstrap) -> Dict[str, Any]:
    out = paired_bootstrap_auroc(pos_scores_dict, neg_scores_dict, primary_key, n_boot, seed)
    out.pop("_boot_primary", None)
    return out


# ----------------------------- data ------------------------------------------
def load_popqa_groups(popqa_path: Path = CFG.data_dir / "popqa_sample.jsonl",
                      stage2_record: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rows = sorted(read_jsonl(popqa_path), key=lambda r: r["popqa_id"])
    missing_hashes = [r.get("popqa_id", "<unknown>") for r in rows if not r.get("input_ids_sha256")]
    if missing_hashes:
        raise ValueError(f"PopQA rows lack input_ids_sha256: {missing_hashes}")
    bad_contract = [r.get("popqa_id", "<unknown>") for r in rows
                    if r.get("tokenization_contract_version") != CONTRACT_VERSION]
    if bad_contract:
        raise ValueError(f"PopQA rows have unexpected tokenization contract: {bad_contract}")
    if stage2_record:
        if stage2_record.get("label_source") != "canonical_v1_batch1_judged":
            raise ValueError("Stage 2 label_source is not canonical_v1_batch1_judged")
        bad_source = [r.get("popqa_id", "<unknown>") for r in rows
                      if r.get("label_source") != stage2_record["label_source"]]
        if bad_source:
            raise ValueError(f"PopQA rows have inconsistent label_source: {bad_source}")
        counts = {k: sum(r["baseline_partition"] == k for r in rows) for k in ("correct", "wrong", "abstain")}
        assert len(rows) == int(stage2_record["popqa_evaluated_count"]), "PopQA row count differs from Stage 2"
        for k in counts:
            assert counts[k] == int(stage2_record[f"popqa_{k}_count"]), f"PopQA {k} count differs from Stage 2"
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if r.get("is_popularity_matched"):
            by_pair.setdefault(r["matched_pair_id"], {})[r["baseline_partition"]] = r
    pair_ids = sorted(by_pair)
    for pid in pair_ids:
        assert set(by_pair[pid]) == {"wrong", "correct"}, f"Matched pair {pid} is not one wrong + one correct"
    if stage2_record:
        assert len(pair_ids) == int(stage2_record["popqa_matched_pair_count"]), "Matched pair count differs from Stage 2"
    return {"rows": rows, "pair_ids": pair_ids, "wrong": [by_pair[p]["wrong"] for p in pair_ids],
            "correct": [by_pair[p]["correct"] for p in pair_ids],
            "abstain": [r for r in rows if r["baseline_partition"] == "abstain"]}


# ----------------------------- feature extraction ----------------------------
@torch.no_grad()
def extract_all_layer_projections_and_confidence(raw_model, tokenizer, prompts, system_prompt, suffix_token_ids,
                                                 unit_directions_at_pos, unknown_token_ids, is_layer_output_tuple,
                                                 batch_size=CFG.batch_size_forward, i_star=-1, l_star=None) -> Dict[str, np.ndarray]:
    """Unsteered residual projections x_{i*}^{(l)} . r_hat_{i*}^{(l)} for every layer plus output-confidence scores."""
    n_layers = unit_directions_at_pos.shape[0]
    contract = {"is_layer_output_tuple": is_layer_output_tuple}
    batches = build_chat_batches(tokenizer, list(prompts), system_prompt, list(suffix_token_ids), batch_size)
    acts, logits = capture_residuals_and_logits(raw_model, batches, contract, list(range(n_layers)), i_star)
    units = unit_directions_at_pos.double()
    proj = torch.einsum("nld,ld->nl", acts.double(), units).numpy()
    probs = F.softmax(logits.float(), dim=-1).double()
    maxp = probs.max(-1).values.numpy()
    entropy = (-(probs * probs.clamp(min=1e-12).log()).sum(-1)).numpy()
    s, _ = log_odds_and_p_abs(logits, unknown_token_ids)
    out = {"proj": proj, "maxp": maxp, "entropy": entropy, "log_odds": s.double().numpy()}
    if l_star is not None:
        out["acts_star"] = acts[:, int(l_star), :].double().numpy()
    return out


# ----------------------------- analysis (CPU) --------------------------------
def classify_interpretation(holm_rows: List[Dict[str, Any]], stage6: Dict[str, Any], h4: Dict[str, Any],
                            h5: Dict[str, Any], layerwise: List[Dict[str, Any]], l_star: int,
                            auroc_abstain_vs_correct: float) -> Dict[str, Any]:
    """Apply the post-hoc v4.1 correction (DEV-013).

    Dimensions 1-2 (H1-H3) are prerequisites; H4 and H5 discriminate between
    hypotheses. The registered v4 verdict remains in overall_as_registered_v4.
    """
    conf = {row["test_id"]: row["confirmed"] for row in holm_rows}
    primary = stage6["primary_tests"]
    concept_confound = bool(
        primary["H1"].get("concept_confound_flag") or primary["H2"].get("concept_confound_flag")
    )
    causal = bool(conf.get("H1") and conf.get("H2") and not concept_confound)
    deconf = bool(conf.get("H3"))

    if conf.get("H4"):
        closed_book = "B"
    elif auroc_abstain_vs_correct >= CFG.h4_auroc_min:
        closed_book = "A"
    elif max(abs(h4["auroc_proj"] - 0.5), abs(auroc_abstain_vs_correct - 0.5)) < 0.05:
        closed_book = "domain_transfer_failure"
    else:
        closed_book = "A"

    earlier = [row for row in layerwise if row["layer"] < l_star and row["auroc"] > h5["auroc_maxp"]]
    if conf.get("H5") or earlier:
        value = "B"
    elif max(h4["auroc_proj"], h5["auroc_maxp"]) < 0.55:
        value = "neither_separates"
    else:
        value = "A"

    prerequisites_met = bool(causal and deconf)
    if prerequisites_met:
        overall_v4 = VERDICT_B if closed_book == "B" else VERDICT_A
    else:
        overall_v4 = VERDICT_INCONCLUSIVE

    if not prerequisites_met:
        overall = VERDICT_INCONCLUSIVE
        explanation = "Prerequisites H1-H3 do not all pass, so H4 and H5 cannot support either hypothesis."
    elif closed_book == "domain_transfer_failure":
        overall = VERDICT_DOMAIN_FAIL
        explanation = (
            "Prerequisites H1-H3 pass, but H4 indicates domain-transfer failure; "
            f"H5 points to {value}, so neither hypothesis is supported."
        )
    elif closed_book == "B" and value == "B":
        overall = VERDICT_B
        explanation = (
            "Prerequisites H1-H3 pass. H4 points to B (closed-book separation confirmed) and "
            "H5 points to B (the projection adds value over output confidence, or an earlier "
            "layer beats max-probability AUROC). Both discriminating tests support Hypothesis B."
        )
    elif closed_book == "A" and value in {"A", "neither_separates"}:
        overall = VERDICT_A
        h5_detail = (
            "the projection does not add value over output confidence"
            if value == "A" else "neither the projection nor output confidence separates the answers"
        )
        explanation = (
            "Prerequisites H1-H3 pass. H4 points to A (closed-book separation is not confirmed) and "
            f"H5 points to {value} ({h5_detail}). Both discriminating tests support Hypothesis A."
        )
    else:
        overall = VERDICT_MIXED
        h4_detail = (
            "closed-book separation confirmed" if closed_book == "B"
            else "closed-book separation not confirmed"
        )
        h5_detail = (
            "the projection does not add value over output confidence, and no earlier layer beats "
            "max-probability AUROC"
            if value == "A" else
            "the projection adds value over output confidence or an earlier layer beats max-probability AUROC"
        )
        explanation = (
            f"Prerequisites H1\u2013H3 pass. H4 points to {closed_book} ({h4_detail}) but H5 points to "
            f"{value} ({h5_detail}). The discriminating tests disagree, so neither hypothesis is supported."
        )

    return {
        "overall": overall,
        "overall_as_registered_v4": overall_v4,
        "rule_version": "v4.1-posthoc",
        "prerequisites_met": prerequisites_met,
        "discriminating_tests": {
            "H4_closed_book": closed_book,
            "H5_value_over_confidence": value,
        },
        "verdict_explanation": explanation,
        "dimension_1_causal_mediation": "mediates" if causal else "confound_or_failure",
        "concept_confound_flag": concept_confound,
        "dimension_2_private_detail": "passes" if deconf else "private_detail_confound",
        "dimension_3_closed_book": closed_book,
        "dimension_4_value_over_confidence": value,
        "earlier_layers_beating_maxp": [row["layer"] for row in earlier],
        "logit_gen_discrepancy": bool(stage6.get("logit_gen_discrepancy")),
        "note": (
            "Branch Rule 4 flagged a logit/generation discrepancy: qualitative claims must defer to "
            "generation rates."
            if stage6.get("logit_gen_discrepancy") else ""
        ),
    }


def analyze_stage8(groups: Dict[str, Any], feats: Dict[str, Dict[str, np.ndarray]], vectors: Dict[str, Any],
                   stage6: Dict[str, Any], l_star: int, n_boot: int = CFG.n_bootstrap,
                   seed: int = CFG.seed_bootstrap) -> Dict[str, Any]:
    """feats keys: popqa_wrong, popqa_correct (pair-aligned), popqa_abstain, test_unans, test_ans, b_ctrl."""
    fw, fc = feats["popqa_wrong"], feats["popqa_correct"]
    pos = {"proj_learned": fw["proj"][:, l_star], "neg_maxp": -fw["maxp"], "entropy": fw["entropy"],
           "abstention_log_odds": fw["log_odds"]}
    neg = {"proj_learned": fc["proj"][:, l_star], "neg_maxp": -fc["maxp"], "entropy": fc["entropy"],
           "abstention_log_odds": fc["log_odds"]}
    boot = paired_bootstrap_auroc(pos, neg, "proj_learned", n_boot, seed)
    boot.pop("_boot_primary")
    au = boot["auroc"]
    mwu = stats.mannwhitneyu(pos["proj_learned"], neg["proj_learned"], alternative="greater")
    diffs = pos["proj_learned"] - neg["proj_learned"]
    wil = stats.wilcoxon(diffs, alternative="greater") if np.any(diffs != 0) else None

    def ctrl_aurocs(units: np.ndarray) -> np.ndarray:
        return auroc_rows(units @ fw["acts_star"].T, units @ fc["acts_star"].T)

    obs = au["proj_learned"]["point"]
    rnd = ctrl_aurocs(vectors["random_unit"].double().numpy())
    shf = ctrl_aurocs(vectors["shuffled_unit"].double().numpy())
    con = ctrl_aurocs(vectors["concept_unit"].double().numpy()[None, :])
    h4 = {"auroc_proj": obs, "auroc_proj_ci": [au["proj_learned"]["ci95_low"], au["proj_learned"]["ci95_high"]],
          "mwu_u": float(mwu.statistic), "mwu_p": float(mwu.pvalue),
          "mean_within_pair_diff": float(diffs.mean()),
          "cohens_dz": float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else None,
          "wilcoxon_paired_p": float(wil.pvalue) if wil is not None else 1.0,
          "meets_auroc_criterion": bool(obs >= CFG.h4_auroc_min),
          "controls": {"random": {**summarize_null(obs, rnd), "aurocs": rnd.tolist()},
                       "shuffled": {**summarize_null(obs, shf), "aurocs": shf.tolist()},
                       "concept": {"auroc": float(con[0])}}}
    h5 = {"auroc_maxp": au["neg_maxp"]["point"], "auroc_maxp_ci": [au["neg_maxp"]["ci95_low"], au["neg_maxp"]["ci95_high"]],
          "auroc_ent": au["entropy"]["point"], "auroc_ent_ci": [au["entropy"]["ci95_low"], au["entropy"]["ci95_high"]],
          "auroc_log_odds": au["abstention_log_odds"]["point"],
          "delta_auroc_proj_minus_maxp": boot["delta"]["neg_maxp"]["point"],
          "delta_auroc_maxp_ci": [boot["delta"]["neg_maxp"]["ci95_low"], boot["delta"]["neg_maxp"]["ci95_high"]],
          "p_boot": boot["delta"]["neg_maxp"]["p_boot_one_sided"],
          "delta_auroc_proj_minus_ent": boot["delta"]["entropy"]["point"],
          "delta_auroc_ent_ci": [boot["delta"]["entropy"]["ci95_low"], boot["delta"]["entropy"]["ci95_high"]],
          "p_boot_ent": boot["delta"]["entropy"]["p_boot_one_sided"]}

    # Layerwise exploratory sweep (Figure S1).
    n = len(diffs)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n, size=(int(n_boot), n))
    layerwise = []
    for l in range(fw["proj"].shape[1]):
        pw, pc = fw["proj"][:, l], fc["proj"][:, l]
        b = auroc_rows(pw[draws], pc[draws])
        lo, hi = np.percentile(b, [2.5, 97.5])
        layerwise.append({"layer": l, "auroc": binary_auroc(pw, pc), "ci_low": float(lo), "ci_high": float(hi),
                          "auroc_test_ctx": binary_auroc(feats["test_unans"]["proj"][:, l], feats["test_ans"]["proj"][:, l])})

    pt = stage6["primary_tests"]
    p_values = {"H1": pt["H1"]["p_wilcoxon"], "H2": pt["H2"]["p_wilcoxon"], "H3": pt["H3"]["p_wilcoxon"],
                "H4": h4["mwu_p"], "H5": h5["p_boot"]}
    extra = {"H1": bool(pt["H1"]["passes_mc_rank_1"]), "H2": bool(pt["H2"]["passes_mc_rank_1"]),
             "H3": bool(pt["H3"]["b_ctrl_abstention_ok"]), "H4": h4["meets_auroc_criterion"], "H5": True}
    holm = holm_bonferroni(p_values, CFG.alpha_fwer, extra)
    violin = {g: feats[k]["proj"][:, l_star].tolist() for g, k in
              (("test_unans", "test_unans"), ("test_ans", "test_ans"), ("b_ctrl", "b_ctrl"), ("popqa_abstain", "popqa_abstain"),
               ("popqa_matched_wrong", "popqa_wrong"), ("popqa_matched_correct", "popqa_correct"))}
    auroc_abs = binary_auroc(feats["popqa_abstain"]["proj"][:, l_star], fc["proj"][:, l_star])
    interp = classify_interpretation(holm, stage6, h4, h5, layerwise, l_star, auroc_abs)
    matched = []
    for part, rows, f in (("wrong", groups["wrong"], fw), ("correct", groups["correct"], fc)):
        for i, r in enumerate(rows):
            matched.append({"popqa_id": r["popqa_id"], "matched_pair_id": r["matched_pair_id"], "partition": part,
                            "s_pop": float(r["s_pop"]), "proj": float(f["proj"][i, l_star]), "maxp": float(f["maxp"][i]),
                            "entropy": float(f["entropy"][i]), "log_odds": float(f["log_odds"][i]),
            "input_ids_sha256": r["input_ids_sha256"], "label_source": r["label_source"],
            "grade": r.get("grade", part), "grade_reason": r.get("grade_reason")})
    table3 = [{"score": "proj_learned", "description": "x_{i*}^{(l*)} . r_hat (H4)", "auroc": obs,
               "ci_95_low": h4["auroc_proj_ci"][0], "ci_95_high": h4["auroc_proj_ci"][1], "p_value": h4["mwu_p"],
               "p_value_test": "one-sided Mann-Whitney U"}]
    for key, desc, test in (("neg_maxp", "-max first-token probability (H5 baseline)", "paired bootstrap, one-sided"),
                            ("entropy", "first-token predictive entropy", "paired bootstrap, one-sided"),
                            ("abstention_log_odds", "first-token abstention log-odds S(p)", "paired bootstrap, one-sided")):
        d = boot["delta"][key]
        table3.append({"score": key, "description": desc, "auroc": au[key]["point"], "ci_95_low": au[key]["ci95_low"],
                       "ci_95_high": au[key]["ci95_high"], "delta_vs_proj": d["point"], "delta_ci_95_low": d["ci95_low"],
                       "delta_ci_95_high": d["ci95_high"], "p_value": d["p_boot_one_sided"], "p_value_test": test})
    table3.append({"score": "random_99_mean", "description": "mean AUROC of 99 random unit directions",
                   "auroc": float(rnd.mean()), "ci_95_low": float(np.percentile(rnd, 2.5)),
                   "ci_95_high": float(np.percentile(rnd, 97.5)), "p_value": h4["controls"]["random"]["p_rank"],
                   "p_value_test": "Monte Carlo rank of learned AUROC"})
    table3.append({"score": "shuffled_19_mean", "description": "mean AUROC of 19 label-shuffled directions",
                   "auroc": float(shf.mean()), "ci_95_low": float(shf.min()), "ci_95_high": float(shf.max()),
                   "p_value": h4["controls"]["shuffled"]["p_rank"], "p_value_test": "permutation rank of learned AUROC"})
    table3.append({"score": "r_concept", "description": "orthogonalized concept direction", "auroc": float(con[0])})
    return {"n_matched_per_group": n, "n_abstain": len(groups["abstain"]), "h4": h4, "h5": h5,
            "auroc_proj": obs, "auroc_proj_ci": h4["auroc_proj_ci"], "auroc_maxp": h5["auroc_maxp"],
            "auroc_maxp_ci": h5["auroc_maxp_ci"], "auroc_ent": h5["auroc_ent"], "auroc_ent_ci": h5["auroc_ent_ci"],
            "delta_auroc_proj_minus_maxp": h5["delta_auroc_proj_minus_maxp"], "h4_mwu_u": h4["mwu_u"],
            "h4_mwu_p": h4["mwu_p"], "h5_boot_p": h5["p_boot"], "auroc_abstain_vs_correct": auroc_abs,
            "holm_bonferroni_table": holm, "interpretation": interp, "violin_6groups": violin,
            "group_means": {g: float(np.mean(v)) for g, v in violin.items()},
            "layerwise_auroc_exploratory": layerwise, "matched_items": matched, "table3_rows": table3}


# ----------------------------- runner ----------------------------------------
def _verify_popqa_input_identity(tokenizer, rows, system_prompt, model_name, suffix_token_ids):
    template_hash = chat_template_sha256(tokenizer)
    bad = []
    for row in rows:
        ids = canonical_input_ids(tokenizer, row["question"], system_prompt, suffix_token_ids)
        actual = input_ids_sha256(ids, model_name, template_hash)
        if actual != row.get("input_ids_sha256"):
            bad.append(row["popqa_id"])
    if bad:
        raise ValueError(f"Canonical feature input hash mismatch for PopQA IDs: {bad}")
    return len(rows)


def run_stage8_popqa_uncertainty(raw_model, tokenizer, contract, directions_dict, selection_dict, popqa_groups,
                                 test_pairs, b_ctrl_rows, stage6_dict, batch_size=CFG.batch_size_forward,
                                 source_stage4_sha256="", source_stage6_sha256="") -> Dict[str, Any]:
    started = time.perf_counter()
    l_star, i_star, pos_idx = int(selection_dict["l_star"]), int(selection_dict["i_star"]), int(selection_dict["pos_idx"])
    vectors = slice_vectors_at_star(directions_dict, l_star, pos_idx)
    units = directions_dict["unit_directions"][:, pos_idx, :].float()
    popqa_feature_rows = popqa_groups["wrong"] + popqa_groups["correct"] + popqa_groups["abstain"]
    verified_input_hash_count = _verify_popqa_input_identity(
        tokenizer, popqa_feature_rows, contract["system_prompt_used"], contract["model_name"],
        list(contract["suffix_token_ids"]))
    sets = {"popqa_wrong": [r["question"] for r in popqa_groups["wrong"]],
            "popqa_correct": [r["question"] for r in popqa_groups["correct"]],
            "popqa_abstain": [r["question"] for r in popqa_groups["abstain"]],
            "test_unans": [p["unans_prompt"] for p in test_pairs], "test_ans": [p["ans_prompt"] for p in test_pairs],
            "b_ctrl": [r["prompt"] for r in b_ctrl_rows]}
    feats = {}
    for name, prompts in sets.items():
        feats[name] = extract_all_layer_projections_and_confidence(
            raw_model, tokenizer, prompts, contract["system_prompt_used"], contract["suffix_token_ids"], units,
            contract["unknown_token_ids"], bool(contract["is_layer_output_tuple"]), batch_size, i_star, l_star)
        print(f"[Stage 8] extracted {name} ({len(prompts)} prompts, {time.perf_counter() - started:.0f}s)", flush=True)
    # Same unsteered coordinate as Stage 6: projections must agree up to fp16 batch noise.
    s6 = stage6_dict["projections"]
    diffs = {k: float(np.max(np.abs(feats[k]["proj"][:, l_star] - np.asarray(s6[k2]))))
             for k, k2 in (("test_unans", "test_unans"), ("test_ans", "test_ans"), ("b_ctrl", "b_ctrl"))}
    scale = max(1.0, float(np.max(np.abs(np.asarray(s6["test_unans"])))))
    assert max(diffs.values()) <= 0.05 * scale, f"Stage 8 projections differ from Stage 6: {diffs}"
    analysis = analyze_stage8(popqa_groups, feats, vectors, stage6_dict, l_star)
    return {"stage": 8, "model_name": contract["model_name"], "l_star": l_star, "i_star": i_star, "pos_idx": pos_idx,
            **analysis, "stage6_projection_max_abs_diff": diffs, "n_bootstrap": CFG.n_bootstrap,
            "source_stage4_sha256": source_stage4_sha256, "source_stage6_sha256": source_stage6_sha256,
            "label_source": "canonical_v1_batch1_judged", "verified_input_hash_count": verified_input_hash_count,
           "elapsed_sec": time.perf_counter() - started, "passed_acceptance_gate": True}


def save_stage8_artifacts(stage8_dict, output_path=CFG.stage8_out_path, table3_csv_path=CFG.table3_csv_path,
                          table3_md_path=CFG.table3_md_path, table1_csv_path=CFG.table1_csv_path,
                          table1_md_path=CFG.table1_md_path) -> None:
    from src.results_tables import write_table1
    atomic_write_json(output_path, stage8_dict)
    write_csv_and_md(stage8_dict["table3_rows"], TABLE3_COLUMNS, table3_csv_path, table3_md_path,
                     f"Table 3. Popularity-matched PopQA wrong (y=1) vs. correct (y=0), N = {stage8_dict['n_matched_per_group']} per group")
    write_table1(table1_csv_path, table1_md_path, stage8=stage8_dict)


def verify_stage8_outputs(output_path=CFG.stage8_out_path, table3_csv_path=CFG.table3_csv_path,
                          table3_md_path=CFG.table3_md_path, stage5_path=None) -> Dict[str, Any]:
    from src.eval_common import read_csv_rows
    for p in (output_path, table3_csv_path, table3_md_path):
        assert Path(p).is_file() and Path(p).stat().st_size > 0, f"Missing Stage 8 output {p}"
    d = read_json(output_path)
    missing = STAGE8_KEYS - d.keys()
    assert not missing, f"Stage 8 JSON missing keys: {sorted(missing)}"
    assert d["stage"] == 8 and d["passed_acceptance_gate"] is True
    assert d["label_source"] == "canonical_v1_batch1_judged"
    assert int(d["verified_input_hash_count"]) >= 2 * int(d["n_matched_per_group"])
    n = int(d["n_matched_per_group"])
    for k in ("auroc_proj", "auroc_maxp", "auroc_ent"):
        assert 0.0 <= d[k] <= 1.0
    assert d["auroc_proj_ci"][0] <= d["auroc_proj_ci"][1]
    assert 0.0 <= d["h4_mwu_p"] <= 1.0 and 1.0 / (CFG.n_bootstrap + 1) - 1e-12 <= d["h5_boot_p"] <= 1.0
    assert sorted(r["test_id"] for r in d["holm_bonferroni_table"]) == ["H1", "H2", "H3", "H4", "H5"]
    assert set(d["violin_6groups"]) == set(VIOLIN_GROUPS)
    assert len(d["violin_6groups"]["popqa_matched_wrong"]) == len(d["violin_6groups"]["popqa_matched_correct"]) == n
    assert len(d["violin_6groups"]["popqa_abstain"]) == d["n_abstain"]
    assert len(d["matched_items"]) == 2 * n
    assert [r["layer"] for r in d["layerwise_auroc_exploratory"]] == list(range(len(d["layerwise_auroc_exploratory"])))
    assert len(d["h4"]["controls"]["random"]["aurocs"]) == CFG.n_ctrl_random
    assert len(read_csv_rows(table3_csv_path)) == len(d["table3_rows"])
    if stage5_path and Path(stage5_path).exists():
        sel = read_json(stage5_path)
        assert (d["l_star"], d["i_star"], d["pos_idx"]) == (sel["l_star"], sel["i_star"], sel["pos_idx"])
    return d


# Tier 2 v5 rule; v4/v4.1 behavior above remains unchanged.
from src.stage8_v5 import classify_interpretation_v5
