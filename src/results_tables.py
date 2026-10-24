"""Table 1 (confirmatory + control summary) rebuilt from whichever Stage 6/7/8 results exist."""
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import CFG
from src.eval_common import read_json, write_csv_and_md

TABLE1_COLUMNS = ["test_id", "hypothesis_or_control", "metric", "point_estimate", "ci_95_low", "ci_95_high",
                  "raw_p_value", "holm_threshold", "passed_holm_bonferroni", "confirmed", "mc_rank_out_of_100",
                  "mc_p_value", "mc_z_score", "shuffled_p_value", "shuffled_z_score", "r_concept_effect_ratio",
                  "wikitext2_kl_nats", "mmlu_accuracy"]


def find_capability(stage7: Optional[Dict[str, Any]], mode: str, alpha: Optional[float]) -> Dict[str, Any]:
    if not stage7:
        return {}
    for row in stage7.get("learned_direction_capability", []):
        if row["mode"] == mode and (alpha is None or abs(float(row["alpha"]) - float(alpha)) < 1e-9):
            return row
    return {}


def _effect_row(test_id: str, name: str, t: Dict[str, Any], cap: Dict[str, Any]) -> Dict[str, Any]:
    return {"test_id": test_id, "hypothesis_or_control": name, "metric": "mean log-odds shift (hypothesised direction > 0)",
            "point_estimate": t["mean_effect"], "ci_95_low": t["bootstrap"]["ci95_low"],
            "ci_95_high": t["bootstrap"]["ci95_high"], "raw_p_value": t["p_wilcoxon"],
            "mc_rank_out_of_100": t.get("mc_rank_out_of_100"), "mc_p_value": t.get("p_mc"),
            "mc_z_score": t.get("random_null", {}).get("z_score"), "shuffled_p_value": t.get("p_shuffled"),
            "shuffled_z_score": t.get("shuffled_null", {}).get("z_score"),
            "r_concept_effect_ratio": t.get("r_concept_effect_ratio"),
            "wikitext2_kl_nats": cap.get("wikitext_kl"), "mmlu_accuracy": cap.get("mmlu_acc")}


def build_table1_rows(stage6: Optional[Dict[str, Any]] = None, stage7: Optional[Dict[str, Any]] = None,
                      stage8: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if stage6:
        pt = stage6["primary_tests"]
        rows.append(_effect_row("H1", "Causal necessity: all-layer ablation on D_test^unans", pt["H1"],
                                find_capability(stage7, "ablate_all", None)))
        rows.append(_effect_row("H2", "Causal sufficiency: +1.0 r at l* on D_test^ans", pt["H2"],
                                find_capability(stage7, "steer_layer", CFG.alpha_primary_pos)))
        h3 = pt["H3"]
        rows.append({"test_id": "H3", "hypothesis_or_control": "Private-detail de-confounding (Family B unans - B-ctrl projection)",
                     "metric": "mean template projection difference", "point_estimate": h3["mean_effect"],
                     "ci_95_low": h3["bootstrap"]["ci95_low"], "ci_95_high": h3["bootstrap"]["ci95_high"],
                     "raw_p_value": h3["p_wilcoxon"]})
        rows.append({"test_id": "H3_bctrl_rate", "hypothesis_or_control": "B-ctrl generation abstention rate (criterion <= 0.15)",
                     "metric": "regex abstention rate", "point_estimate": h3["b_ctrl_gen_abstention_rate_regex"]})
        rows.append(_effect_row("S1_H1b", "[secondary] Single-layer subtraction -1.0 r on D_test^unans",
                                stage6["secondary_tests"]["H1b_steer_neg1"],
                                find_capability(stage7, "steer_layer", CFG.alpha_primary_neg)))
        rows.append({"test_id": "S2_concept", "hypothesis_or_control": "[secondary] r_concept pre-orthogonalization cosine",
                     "metric": "cos(theta_pre)", "point_estimate": stage6.get("cos_concept_pre_ortho_at_star")})
    if stage8:
        h4, h5 = stage8["h4"], stage8["h5"]
        rows.append({"test_id": "H4", "hypothesis_or_control": "Closed-book wrong vs. correct (popularity-matched PopQA)",
                     "metric": "AUROC_proj (criterion >= 0.60)", "point_estimate": h4["auroc_proj"],
                     "ci_95_low": h4["auroc_proj_ci"][0], "ci_95_high": h4["auroc_proj_ci"][1],
                     "raw_p_value": h4["mwu_p"], "mc_rank_out_of_100": h4["controls"]["random"]["rank_of_observed"],
                     "mc_p_value": h4["controls"]["random"]["p_rank"], "mc_z_score": h4["controls"]["random"]["z_score"],
                     "shuffled_p_value": h4["controls"]["shuffled"]["p_rank"],
                     "shuffled_z_score": h4["controls"]["shuffled"]["z_score"]})
        rows.append({"test_id": "H5", "hypothesis_or_control": "Superiority over max first-token probability",
                     "metric": "AUROC_proj - AUROC_maxp", "point_estimate": h5["delta_auroc_proj_minus_maxp"],
                     "ci_95_low": h5["delta_auroc_maxp_ci"][0], "ci_95_high": h5["delta_auroc_maxp_ci"][1],
                     "raw_p_value": h5["p_boot"]})
        rows.append({"test_id": "S3_entropy", "hypothesis_or_control": "[secondary] AUROC_proj - AUROC_entropy",
                     "metric": "delta AUROC", "point_estimate": h5["delta_auroc_proj_minus_ent"],
                     "ci_95_low": h5["delta_auroc_ent_ci"][0], "ci_95_high": h5["delta_auroc_ent_ci"][1],
                     "raw_p_value": h5["p_boot_ent"]})
        holm = {r["test_id"]: r for r in stage8.get("holm_bonferroni_table", [])}
        for row in rows:
            if row["test_id"] in holm:
                h = holm[row["test_id"]]
                row.update(holm_threshold=h["holm_threshold"], passed_holm_bonferroni=h["passed_holm_bonferroni"],
                           confirmed=h["confirmed"])
    return rows


def write_table1(csv_path: Path = CFG.table1_csv_path, md_path: Path = CFG.table1_md_path,
                 stage6: Optional[Dict[str, Any]] = None, stage7: Optional[Dict[str, Any]] = None,
                 stage8: Optional[Dict[str, Any]] = None, load_missing: bool = True) -> List[Dict[str, Any]]:
    """Write Table 1; results not passed in are loaded from their default paths when present."""
    def _load(given: Optional[Dict[str, Any]], path: Path) -> Optional[Dict[str, Any]]:
        if given is None and load_missing and Path(path).exists():
            return read_json(path)
        return given

    stage6 = _load(stage6, CFG.stage6_out_path)
    stage7 = _load(stage7, CFG.stage7_out_path)
    stage8 = _load(stage8, CFG.stage8_out_path)
    rows = build_table1_rows(stage6, stage7, stage8)
    write_csv_and_md(rows, TABLE1_COLUMNS, csv_path, md_path,
                     "Table 1. Pre-registered tests H1-H5 (Holm-Bonferroni, alpha_FWER = 0.05) and secondary controls")
    return rows
