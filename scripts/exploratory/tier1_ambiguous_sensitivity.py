#!/usr/bin/env python3
"""Recompute matched AUROC as a post-hoc sensitivity using available human ambiguous labels."""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import CFG
from src.exploratory import assert_not_fresh, git_commit
from src.eval_common import read_jsonl
from src.stats_utils import binary_auroc


def _human_labels(path):
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["uid"].removeprefix("popqa:"): row["human_label"]
                for row in csv.DictReader(handle) if row.get("human_label")}


def _auc_ci(positive, negative, pair_ids, n_boot, seed):
    point = binary_auroc(positive, negative)
    rng = np.random.default_rng(seed)
    n = len(pair_ids)
    boot = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        pos, neg = np.asarray(positive)[ix], np.asarray(negative)[ix]
        boot.append(binary_auroc(pos, neg))
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"auroc": point, "ci95": [float(low), float(high)], "n_pairs": n}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--human-labels", type=Path, default=CFG.results_dir / "grader_validation_test.csv")
    parser.add_argument("--n-bootstrap", type=int, default=CFG.n_bootstrap)
    parser.add_argument("--seed", type=int, default=CFG.seed_bootstrap)
    parser.add_argument("--output", type=Path, default=CFG.results_dir / "tier1_ambiguous_sensitivity.json")
    args = parser.parse_args(argv)
    assert_not_fresh(args.human_labels)
    popqa_path = assert_not_fresh(CFG.data_dir / "popqa_sample.jsonl")
    rows = read_jsonl(popqa_path)
    meta = pd.read_parquet(assert_not_fresh(args.features_dir / "popqa_meta.parquet"))
    resid = np.load(assert_not_fresh(args.features_dir / "popqa_resid.npy"), mmap_mode="r")
    directions = torch.load(assert_not_fresh(CFG.stage4_out_path), map_location="cpu", weights_only=False)
    direction = directions["unit_directions"][11, 1].float().numpy()
    projections = np.einsum("nd,d->n", np.asarray(resid[:, 1, 11, :], dtype=np.float32), direction)
    grader = _human_labels(args.human_labels)
    map_human = {"CORRECT": "correct", "INCORRECT": "wrong", "ABSTAIN": "abstain", "UNGRADEABLE": "ambiguous"}
    index_by_id = {str(row.popqa_id): int(i) for i, row in meta.iterrows()}
    grouped = {}
    resolutions = {}
    for row in rows:
        if not row.get("_v4_is_popularity_matched") or not row.get("_v4_matched_pair_id"):
            continue
        grade = row.get("baseline_partition")
        if grade == "ambiguous" and row["popqa_id"] in grader:
            grade = map_human.get(grader[row["popqa_id"]], grade)
            resolutions[grader[row["popqa_id"]]] = resolutions.get(grader[row["popqa_id"]], 0) + 1
        grouped.setdefault(row["_v4_matched_pair_id"], []).append((row["popqa_id"], grade))
    valid_pairs = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            continue
        by_grade = {grade: popqa_id for popqa_id, grade in members if grade in {"correct", "wrong"}}
        if set(by_grade) == {"correct", "wrong"}:
            valid_pairs.append((pair_id, by_grade["wrong"], by_grade["correct"]))
    if not valid_pairs:
        raise RuntimeError("No prior matched pairs remain after applying available human resolutions")
    wrong_ix = np.asarray([index_by_id[wrong] for _, wrong, _ in valid_pairs])
    correct_ix = np.asarray([index_by_id[correct] for _, _, correct in valid_pairs])
    pair_ids = [pair_id for pair_id, _, _ in valid_pairs]
    output = {}
    for name, values, high_is_wrong in (
        ("projection", projections, True),
        ("maxp", meta["maxp"].to_numpy(float), False),
        ("entropy", meta["entropy"].to_numpy(float), True),
        ("log_odds", meta["log_odds"].to_numpy(float), True),
        ("log_s_pop", meta["log_s_pop"].to_numpy(float), False),
    ):
        pos = np.asarray(values)[wrong_ix]
        neg = np.asarray(values)[correct_ix]
        if not high_is_wrong:
            pos, neg = -pos, -neg
        output[name] = _auc_ci(pos, neg, pair_ids, args.n_bootstrap, args.seed)
    result = {"status": "corrected_post_hoc_not_confirmatory", "purpose": "ambiguous-label sensitivity only",
              "git_commit": git_commit(ROOT),
              "n_pairs": len(valid_pairs), "human_ambiguous_resolutions": resolutions,
              "auc_by_score": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from src.eval_common import atomic_write_json
    atomic_write_json(args.output, result)
    print(f"Wrote {args.output} ({len(valid_pairs)} prior matched pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
