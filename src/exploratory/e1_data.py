"""Tier 1 PopQA population assembly for the exploratory E1 probe."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.abstention_classifier import is_answer_then_unknown
from src.config import CFG
from src.exploratory import assert_not_fresh, require_tier1_complete


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_e1_frame(popqa_path: str | Path = CFG.data_dir / "popqa_sample.jsonl",
                  stage2_path: str | Path = CFG.stage2_out_path, dry_run: bool = False) -> pd.DataFrame:
    """Load all PopQA rows, marking analysis eligibility without dropping E2 rows."""
    popqa_path, stage2_path = assert_not_fresh(popqa_path), assert_not_fresh(stage2_path)
    if dry_run:
        root = Path(__file__).resolve().parents[2]
        popqa_path = root / "results/archive_v4/data/popqa_sample.jsonl"
    else:
        require_tier1_complete()
    rows = _read_jsonl(popqa_path)
    if not rows:
        raise ValueError(f"No PopQA rows found in {popqa_path}")

    if dry_run:
        for row in rows:
            row["baseline_response"] = row.get("_v4_baseline_response", row.get("baseline_response", ""))
            row["baseline_partition"] = row.get("_v4_baseline_partition", row.get("baseline_partition"))
            row["is_popularity_matched"] = row.get("_v4_is_popularity_matched", row.get("is_popularity_matched", False))
            row["matched_pair_id"] = row.get("_v4_matched_pair_id", row.get("matched_pair_id"))
            row["label_source"] = "v4_archive_DRYRUN"
    else:
        summary = json.loads(stage2_path.read_text(encoding="utf-8"))
        if summary.get("label_source") != "canonical_v1_batch1_judged":
            raise ValueError("Stage 2 label_source is not canonical_v1_batch1_judged")
        counts = Counter(row.get("baseline_partition") for row in rows)
        expected_keys = {"correct": "popqa_correct_count", "wrong": "popqa_wrong_count",
                         "abstain": "popqa_abstain_count", "ambiguous": "popqa_ambiguous_count"}
        if len(rows) != int(summary.get("popqa_evaluated_count", -1)):
            raise ValueError("PopQA row count differs from Stage 2")
        for label, key in expected_keys.items():
            if counts[label] != int(summary.get(key, -1)):
                raise ValueError(f"PopQA {label} count ({counts[label]}) differs from Stage 2 {key}")
        bad_source = [r.get("popqa_id") for r in rows if r.get("label_source") != summary["label_source"]]
        if bad_source:
            raise ValueError(f"PopQA rows have inconsistent label_source: {bad_source}")

    grouped_counts = Counter(r.get("prop") for r in rows if r.get("baseline_partition") in {"correct", "wrong"})
    for row in rows:
        partition = row.get("baseline_partition")
        final = str(row.get("final_label") or "").upper()
        if final == "DEGENERATE":
            partition = "wrong"
        row["baseline_partition"] = partition
        row["use_in_e1"] = partition in {"correct", "wrong"}
        row["y"] = int(partition == "wrong") if row["use_in_e1"] else None
        row["prop_grouped"] = row.get("prop") if grouped_counts[row.get("prop")] >= 10 else "other"
        row["answer_then_unknown"] = bool(is_answer_then_unknown(row.get("baseline_response", "")))
        row.setdefault("matched_pair_id", None)
    pairs: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("is_popularity_matched"):
            pairs.setdefault(str(row.get("matched_pair_id")), []).append(row)
    for pair_id, pair in pairs.items():
        if len(pair) != 2 or Counter(r["baseline_partition"] for r in pair) != Counter({"wrong": 1, "correct": 1}):
            raise ValueError(f"Matched pair {pair_id} is not exactly one wrong + one correct row")
    frame = pd.DataFrame(rows)
    if dry_run:
        frame["label_source"] = "v4_archive_DRYRUN"
    return frame
