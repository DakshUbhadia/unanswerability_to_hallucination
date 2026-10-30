#!/usr/bin/env python3
"""Export blinded E3 judge and grader spot-check sheets after generations are scored."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.eval_common import read_jsonl
from src.exploratory import assert_not_fresh, exploratory_payload

SEED = 20260409
CONDITIONS = {"llama32_3b": "L_neut", "qwen25_3b": "Q_neut"}


def _conditions():
    selected = ["llama32_3b"]
    pilot = CFG.results_dir / "exploratory" / "e3_pilot_qwen25_3b.json"
    if pilot.exists() and json.loads(pilot.read_text()).get("arm_included"):
        selected.append("qwen25_3b")
    return selected


def _rows(slug):
    path = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    rows = read_jsonl(path)
    if not rows or any(not (r.get("classifier") or {}).get("final_label") for r in rows):
        raise RuntimeError(f"Neutral PopQA must be judged before E3 validation export: {path}")
    if any(not r.get("grade_reason") for r in rows):
        raise RuntimeError(f"Neutral PopQA must be graded before E3 validation export: {path}")
    return rows


def _sample_stratified(rows, n, key, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    keys = sorted(groups, key=str)
    while len(selected) < min(n, len(rows)):
        progressed = False
        for k in keys:
            if groups[k] and len(selected) < n:
                selected.append(groups[k].pop())
                progressed = True
        if not progressed:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CFG.results_dir / "exploratory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    judge_rows, grader_rows = [], []
    for slug in _conditions():
        rows = _rows(slug)
        n_each = 50 if len(_conditions()) == 2 else 100
        judge_rows.extend((slug, row) for row in _sample_stratified(
            rows, n_each, lambda r: (r["classifier"].get("final_label"),
                                     float(r["classifier"].get("judge_margin") or 0) < 1), rng))
        grader_rows.extend((slug, row) for row in _sample_stratified(
            rows, 30 if len(_conditions()) == 2 else 60,
            lambda r: (r.get("baseline_partition"), r.get("grade_reason")), rng))
    judge_key, grader_key = {}, {}
    judge_csv, grader_csv = [], []
    for slug, row in judge_rows:
        uid = f"e3_judge:{slug}:{row['popqa_id']}"
        cls = row["classifier"]
        judge_key[uid] = {"final_label": cls["final_label"], "regex_label": cls.get("regex_label"),
                          "judge_margin": cls.get("judge_margin"), "condition": CONDITIONS[slug]}
        judge_csv.append({"uid": uid, "question": row["question"], "response": row.get("baseline_response", row.get("response", "")),
                          "human_label": "", "notes": ""})
    for slug, row in grader_rows:
        uid = f"e3_grader:{slug}:{row['popqa_id']}"
        grader_key[uid] = {"grade": row["baseline_partition"], "grade_reason": row["grade_reason"],
                           "condition": CONDITIONS[slug]}
        answers = row.get("possible_answers", [])
        grader_csv.append({"uid": uid, "question": row["question"],
                           "possible_answers": json.dumps(answers, ensure_ascii=False),
                           "response": row.get("baseline_response", row.get("response", "")),
                           "human_label": "", "notes": ""})
    for name, rows in (("e3_judge_validation_test.csv", judge_csv),
                       ("e3_grader_validation_test.csv", grader_csv)):
        if not rows:
            raise RuntimeError(f"No rows selected for {name}")
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "e3_judge_validation_key.json").write_text(json.dumps(judge_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "e3_grader_validation_key.json").write_text(json.dumps(grader_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    judge_record = judge_rows[0][1]["classifier"]
    for name, key in (("e3_judge", judge_rows), ("e3_grader", grader_rows)):
        details = {"seed": SEED, "conditions": _conditions(),
            "uids": [f"{name}:{slug}:{row['popqa_id']}" for slug, row in key],
            "n": len(key)}
        if name == "e3_judge":
            details.update({"judge_model_id": judge_record.get("judge_model_id"),
                            "judge_model_revision": judge_record.get("judge_model_revision"),
                            "judge_prompt_version": judge_record.get("judge_prompt_version"),
                            "judge_prompt_sha256": judge_record.get("judge_prompt_sha256"),
                            "judge_quantization": judge_record.get("judge_quantization")})
        manifest = exploratory_payload(details)
        (args.output_dir / f"{name}_validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(judge_csv)} judge and {len(grader_csv)} grader spot-check rows.")


if __name__ == "__main__":
    main()
