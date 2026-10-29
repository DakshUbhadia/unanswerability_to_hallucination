#!/usr/bin/env python3
"""Refresh the hand-written Tier 1 README summary from completed gated artifacts."""
import argparse
import json
from pathlib import Path

from src.config import CFG


def _read(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required Tier 1 report artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _auroc_ci(record, value_key, ci_key):
    value = record[value_key]
    ci = record.get(ci_key)
    if ci is None:
        raise KeyError(f"Missing registered AUROC interval {ci_key}")
    if isinstance(ci, dict):
        lo, hi = ci.get("ci95_low", ci.get("low")), ci.get("ci95_high", ci.get("high"))
    else:
        lo, hi = ci[0], ci[1]
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def update_readme(readme_path=CFG.readme_path, results_dir=CFG.results_dir):
    results_dir = Path(results_dir)
    current = _read(results_dir / "stage8_popqa_core.json")
    old = _read(results_dir / "archive_v42" / "results" / "stage8_popqa_core.json")
    judge = _read(results_dir / "judge_validation.json")
    grader = _read(results_dir / "grader_validation.json")
    judge_manifest = _read(results_dir / "judge_validation_manifest.json")
    ambiguous = _read(results_dir / "tier1_ambiguous_sensitivity.json")
    stage10_gate_path = results_dir / "judge_validation_stage10.json"
    stage10_gate = _read(stage10_gate_path) if stage10_gate_path.exists() else None
    stage10_path = results_dir / "stage10_popqa_steering.json"
    stage10_status = "not re-analysed (supplementary judge gate not passed)"
    if stage10_gate and stage10_gate.get("gate_passed") is True and stage10_path.exists():
        stage10_status = "re-analysed after supplementary judge validation passed"

    corrected_h4 = _auroc_ci(current, "auroc_proj", "auroc_proj_ci")
    corrected_h5 = _auroc_ci(current, "auroc_maxp", "auroc_maxp_ci")
    v4_h4 = _auroc_ci(old, "auroc_proj", "auroc_proj_ci")
    v4_h5 = _auroc_ci(old, "auroc_maxp", "auroc_maxp_ci")
    quant = judge_manifest.get("judge_quantization", "unknown")
    recall = judge.get("recall", {}).get("ABSTAIN")
    resolution_count = sum(ambiguous.get("human_ambiguous_resolutions", {}).values())
    sensitivity_pairs = ambiguous.get("n_pairs", 0)

    summary_lines = [
        "## Summary of findings",
        "",
        f"- **Corrected post-hoc results (Addendum B; not confirmatory):** on {current.get('n_matched_per_group', current.get('n_matched', 'the rematched'))} matched pairs, H4 projection AUROC is {corrected_h4} and H5 max-probability AUROC is {corrected_h5}. The archived v4 values were H4 {v4_h4} and H5 {v4_h5}.",
        f"- **Open measurement issues (Tier 1 correction):** judge held-out κ={judge['cohen_kappa']:.3f}, ABSTAIN recall={recall:.3f}, low-margin gate={judge.get('low_margin_gate_passed')}; grader correct precision={grader['precision_correct']:.3f}, recall={grader['recall_correct']:.3f}; judge quantization={quant}; ambiguous sensitivity used {resolution_count} available human resolutions across {sensitivity_pairs} prior matched pairs and does not replace the primary analysis.",
        f"- **Stage 10:** {stage10_status}.",
        "",
        "## Status",
    ]

    readme_path = Path(readme_path)
    text = readme_path.read_text(encoding="utf-8")
    start = text.find("## Summary of findings")
    end = text.find("## Status", start)
    if start < 0 or end < 0:
        raise ValueError("README must contain the hand-written Summary of findings and Status headings")
    text = text[:start] + "\n".join(summary_lines) + "\n" + text[end + len("## Status"):]
    # The generated Stage 9 block remains untouched; the existing exploratory link stays below its end marker.
    readme_path.write_text(text, encoding="utf-8", newline="\n")
    return readme_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=CFG.readme_path)
    parser.add_argument("--results-dir", type=Path, default=CFG.results_dir)
    args = parser.parse_args()
    print(f"Updated {update_readme(args.readme, args.results_dir)}")


if __name__ == "__main__":
    main()
