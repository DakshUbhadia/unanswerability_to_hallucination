#!/usr/bin/env python3
"""Export a blinded descriptive human-review sample for the neutral prompt."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.eval_common import read_jsonl
from src.exploratory import assert_not_fresh, git_commit

SEED = 20261029


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CFG.data_dir / "exploratory" / "popqa_neutral.jsonl")
    parser.add_argument("--output-dir", type=Path, default=CFG.results_dir)
    parser.add_argument("--prefix", default="judge_validation_neutral")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args(argv)
    assert_not_fresh(args.input)
    assert_not_fresh(args.output_dir)
    rows = read_jsonl(args.input)
    if len(rows) < args.n:
        raise ValueError(f"Need at least {args.n} neutral-prompt responses, found {len(rows)}")
    rng = random.Random(SEED)
    sample = rng.sample(rows, args.n)
    blinded = {row["popqa_id"]: "N" + hashlib.sha256(row["popqa_id"].encode()).hexdigest()[:18] for row in sample}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.prefix}_test.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "question", "response", "human_label", "notes"])
        writer.writeheader()
        for row in sample:
            writer.writerow({"uid": blinded[row["popqa_id"]], "question": row["question"],
                             "response": row["generated_text"], "human_label": "", "notes": ""})
    key = {"items": {blinded[row["popqa_id"]]: {"popqa_id": row["popqa_id"], "judge_label": row["final_label"]}
                      for row in sample},
           "status": "exploratory_post_registration", "git_commit": git_commit(ROOT)}
    (args.output_dir / f"{args.prefix}_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    manifest = {"seed": SEED, "n": len(sample), "purpose": "descriptive_only_not_a_gate",
                "judge_model_id": sample[0].get("judge_model_id"),
                "judge_model_revision": sample[0].get("judge_model_revision"),
                "judge_quantization": sample[0].get("judge_quantization"),
                "test_uids": sorted(blinded.values()),
                "status": "exploratory_post_registration", "git_commit": git_commit(ROOT)}
    (args.output_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sample)} blinded neutral-prompt responses; agreement is descriptive, not a gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
