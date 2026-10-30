#!/usr/bin/env python3
"""Offline judge/grade neutral PopQA responses, then build E3 matched sets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.abstention_classifier import classify_texts
from src.config import CFG
from src.eval_common import read_jsonl
from src.exploratory import assert_not_fresh, exploratory_payload, require_tier1_complete
from src.pilot_check import grade_popqa, match_popqa_by_popularity_caliper


def _dump_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def label_condition(slug, registered_rows=None):
    path = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    rows = read_jsonl(path)
    if len(rows) != 1000:
        raise ValueError(f"Expected exactly the registered 1,000 PopQA rows for {slug}, found {len(rows)}")
    records = classify_texts([{"uid": f"e3:{slug}:{r['popqa_id']}", "question": r["question"],
                               "response": r.get("response", r.get("baseline_response", ""))} for r in rows],
                             offline=True)
    by_id = {str(r.get("uid", "")).split(":")[-1]: r for r in records}
    for row in rows:
        classifier = by_id[str(row["popqa_id"])]
        graded = grade_popqa(row.get("response", row.get("baseline_response", "")),
                             row.get("possible_answers", []), row.get("obj", ""), classifier)
        row.update({"classifier": classifier, "final_label": classifier.get("final_label"),
                    "grade": graded["grade"], "grade_reason": graded["reason"],
                    "matched_alias": graded.get("matched_alias"), "baseline_partition": graded["grade"],
                    "label_source": "e3_neutral_judged"})
    rows, pair_count, caliper_width = match_popqa_by_popularity_caliper(rows, 0.25, 20260406)
    counts = Counter(r["baseline_partition"] for r in rows)
    transitions = None
    if registered_rows is not None:
        reg = {str(r["popqa_id"]): r for r in registered_rows}
        cells = Counter((str(reg[str(r["popqa_id"])].get("baseline_partition")), r["baseline_partition"])
                        for r in rows if str(r["popqa_id"]) in reg)
        labels = ("correct", "wrong", "abstain", "ambiguous")
        transitions = {a: {b: int(cells[(a, b)]) for b in labels} for a in labels}
    _dump_jsonl(path, rows)
    matched = [{"popqa_id": r["popqa_id"], "matched_pair_id": r["matched_pair_id"],
                "baseline_partition": r["baseline_partition"], "s_pop": r["s_pop"]}
               for r in rows if r.get("is_popularity_matched")]
    result = exploratory_payload({"slug": slug, "n_rows": len(rows), "counts": dict(counts),
        "matched_pair_count": int(pair_count), "caliper_width_s_pop": float(caliper_width),
        "power_status": "descriptive_only" if pair_count < 100 else "powered_by_preregistered_rule",
        "matched_rows": matched, "label_transition_table_vs_registered": transitions,
        "label_source": "e3_neutral_judged"})
    target = CFG.results_dir / "exploratory" / f"e3_labels_{slug}.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama32_3b", "qwen25_3b"), action="append")
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args()
    require_tier1_complete()
    models = args.model or ["llama32_3b"]
    registered = read_jsonl(assert_not_fresh(CFG.data_dir / "popqa_sample.jsonl"))
    for slug in models:
        if slug == "qwen25_3b":
            pilot = CFG.results_dir / "exploratory" / "e3_pilot_qwen25_3b.json"
            if not pilot.exists() or not json.loads(pilot.read_text()).get("arm_included"):
                raise RuntimeError("Qwen arm was not included by the neutral pilot")
        label_condition(slug, registered if slug == "llama32_3b" else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
