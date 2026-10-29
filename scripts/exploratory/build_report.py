#!/usr/bin/env python3
"""Build a separate post-registration report from completed E1-E3 JSON outputs."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.exploratory import assert_not_fresh


def _load(path, required=True):
    path = assert_not_fresh(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required exploratory result is missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ci(summary):
    interval = summary.get("ci95") or summary.get("ci") or [None, None]
    if isinstance(interval, dict):
        low, high = interval.get("low"), interval.get("high")
    else:
        low, high = interval[0], interval[1]
    return f"{summary.get('auroc', summary.get('estimate', '')):.3f} [{low:.3f}, {high:.3f}]" if low is not None else "pending"


def _write_report(e1, e2, e3, e3_human, e1_neutral_frozen, e1_neutral_direction,
                  e2_neutral_frozen, e2_neutral_direction, path):
    lines = ["# Exploratory analyses (post-registration)", "",
             "These analyses are exploratory and do not replace or alter confirmatory Stage 9 results.", ""]
    lines += ["## E1: Supervised probe", "", "### Primary matched population", "",
              "| Score | AUROC [95% CI] |", "|---|---:|"]
    primary = e1["primary_and_secondary"]["primary_matched_pairs"]
    for name, summary in primary["baselines"].items():
        if name == "frozen_direction":
            for coordinate, coordinate_summary in sorted(summary.items()):
                lines.append(f"| {coordinate} frozen direction | {_ci(coordinate_summary)} |")
            continue
        lines.append(f"| {name} | {_ci(summary)} |")
    probe = primary["probe"]
    for name, summary in sorted(probe.items()):
        lines.append(f"| nested probe {name} | {_ci(summary)} |")
    incremental = e1["incremental_value_layer11_pos_minus4"]
    lines.extend(["", f"Incremental value over the registered confidence baseline: ΔAUROC {incremental['delta_auroc_cv']:.3f} "
                  f"[{incremental['ci95_low']:.3f}, {incremental['ci95_high']:.3f}].", "",
                  "### Secondary all-graded population", "",
                  "| Probe coordinate | AUROC [95% CI] |", "|---|---:|"])
    secondary = e1["primary_and_secondary"].get("secondary_all_graded_relation_grouped", {}).get("probe", {})
    for name, summary in sorted(secondary.items()):
        lines.append(f"| {name} | {_ci(summary)} |")
    lines.extend(["", "### Sensitivity analyses", "", "| Sensitivity | Result |", "|---|---|"])
    for name, result in e1.get("sensitivity", {}).items():
        if "error" in result:
            lines.append(f"| {name} | Not estimable: {result['error']} |")
        else:
            lines.append(f"| {name} | n={result['n']}; probe metrics reported in the E1 JSON with 95% CIs |")

    if e1_neutral_frozen or e1_neutral_direction:
        lines.extend(["", "### Neutral-prompt E1 direction comparison", "",
                      "| Direction | Coordinate | Probe AUROC [95% CI] |", "|---|---|---:|"])
        for direction_name, artifact in (("Frozen rhat", e1_neutral_frozen),
                                         ("Neutral rhat", e1_neutral_direction)):
            if not artifact:
                continue
            neutral_primary = artifact["primary_and_secondary"]["primary_matched_pairs"]["probe"]
            for coordinate, summary in sorted(neutral_primary.items()):
                lines.append(f"| {direction_name} | {coordinate} | {_ci(summary)} |")

    lines.extend(["", "## E2: Decomposition", "",
                  f"**Pre-declared interpretation:** {e2['interpretation']}.", "",
                  "| Signal | AUROC [95% CI] |", "|---|---:|",
                  f"| Residualized projection, layer 11 / position -4 | {_ci(e2['residualized_proj_layer11_pos_minus4'])} |",
                  f"| Max-probability baseline | {_ci(e2['maxp_baseline'])} |",
                  "", "### Group contrasts by layer", "", "| Layer | Contrast | AUROC [95% CI] |", "|---:|---|---:|"])
    for layer, contrasts in sorted(e2.get("group_contrasts_layer11", {}).items(), key=lambda item: int(item[0])):
        for contrast in contrasts:
            lines.append(f"| {layer} | {contrast['positive']} vs {contrast['negative']} | {_ci(contrast)} |")
    lines.extend([
                  "Probe-weight cosine, partial Spearman correlation, and logit-lens token ranks are descriptive geometric results.", ""])

    for title, artifact in (("E2 neutral-prompt, frozen direction", e2_neutral_frozen),
                            ("E2 neutral-prompt, neutral direction", e2_neutral_direction)):
        if artifact:
            lines.extend([f"### {title}", "", f"Interpretation: **{artifact['interpretation']}**; residualized projection AUROC "
                          f"{_ci(artifact['residualized_proj_layer11_pos_minus4'])}, max-probability baseline "
                          f"{_ci(artifact['maxp_baseline'])}.", ""])

    lines.extend(["## E3: Neutral-prompt instruction ablation", ""])
    if e3:
        lines.extend(["| Outcome | Estimate [95% CI] |", "|---|---:|"])
        for label, key in (("Neutral-prompt abstention rate", "neutral_abstention_rate"),
                           ("Original-prompt abstention rate", "original_abstention_rate"),
                           ("Neutral accuracy among answered items", "neutral_accuracy_among_answered")):
            lines.append(f"| {label} | {_ci(e3[key])} |")
    if e3_human:
        kappa = e3_human.get("cohen_kappa")
        kappa_ci = e3_human.get("cohen_kappa_ci95") or [None, None]
        if kappa is None or any(value is None for value in kappa_ci):
            kappa_text = "κ undefined because the labels lack variation"
        else:
            kappa_text = f"κ {kappa:.3f}, [{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]"
        lines.extend(["", f"Neutral-prompt human/judge agreement is descriptive: {e3_human['agreement']:.3f} "
                      f"[{e3_human['agreement_ci95'][0]:.3f}, {e3_human['agreement_ci95'][1]:.3f}] "
                      f"({kappa_text}); it is not a gate."])
    else:
        lines.append("Neutral-prompt human/judge agreement is pending the descriptive human review sheet.")
    lines.extend(["", "Neutral-prompt E1/E2 direction comparisons are recorded in their respective JSON outputs when both direction modes have been run.", ""])
    path = assert_not_fresh(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=CFG.results_dir / "exploratory")
    parser.add_argument("--output", type=Path, default=CFG.results_dir / "exploratory" / "REPORT.md")
    args = parser.parse_args(argv)
    e1 = _load(args.results_dir / "e1_probe.json")
    e2 = _load(args.results_dir / "e2_decomposition.json")
    e3 = _load(args.results_dir / "e3_neutral_prompt.json", required=False)
    e3_human = _load(args.results_dir / "neutral_judge_agreement.json", required=False)
    e1_neutral_frozen = _load(args.results_dir / "e1_neutral_prompt_frozen_direction.json", required=False)
    e1_neutral_direction = _load(args.results_dir / "e1_neutral_prompt_neutral_direction.json", required=False)
    e2_neutral_frozen = _load(args.results_dir / "e2_neutral_prompt_frozen_direction.json", required=False)
    e2_neutral_direction = _load(args.results_dir / "e2_neutral_prompt_neutral_direction.json", required=False)
    output = _write_report(e1, e2, e3, e3_human, e1_neutral_frozen,
                           e1_neutral_direction, e2_neutral_frozen,
                           e2_neutral_direction, args.output)
    print(f"Wrote report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
