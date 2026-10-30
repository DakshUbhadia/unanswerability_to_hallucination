#!/usr/bin/env python3
"""Report descriptive human/judge agreement for the neutral-prompt review sheet."""
import argparse
import csv
import json
import sys
if __name__ == "__main__":
    if "--legacy" not in sys.argv:
        print("superseded by scripts/import_judge_labels.py; pass --legacy to run", file=sys.stderr)
        raise SystemExit(2)
    sys.argv.remove("--legacy")
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.config import CFG
from src.exploratory import assert_not_fresh, write_exploratory_json

LABELS = ["ANSWER", "ABSTAIN", "DEGENERATE"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CFG.results_dir / "judge_validation_neutral_test.csv")
    parser.add_argument("--prefix", default="judge_validation_neutral")
    parser.add_argument("--output", type=Path, default=CFG.results_dir / "exploratory" / "neutral_judge_agreement.json")
    args = parser.parse_args(argv)
    assert_not_fresh(args.csv)
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8", newline="")))
    if not rows or any(row.get("human_label") not in LABELS for row in rows):
        raise ValueError("Every row needs a human_label of ANSWER, ABSTAIN, or DEGENERATE")
    key_path = assert_not_fresh(args.csv.parent / f"{args.prefix}_key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    human = [row["human_label"] for row in rows]
    judge = [key["items"][row["uid"]]["judge_label"] for row in rows]
    rng = np.random.default_rng(20261029)
    agreement_samples, kappa_samples = [], []
    human_array, judge_array = np.asarray(human), np.asarray(judge)
    with warnings.catch_warnings():
        # Single-class bootstrap samples can make Cohen's κ undefined; keep those
        # replicates out of the interval without flooding the descriptive run log.
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(2000):
            ix = rng.integers(0, len(rows), len(rows))
            agreement_samples.append(float(np.mean(human_array[ix] == judge_array[ix])))
            kappa = float(cohen_kappa_score(human_array[ix], judge_array[ix], labels=LABELS))
            if np.isfinite(kappa):
                kappa_samples.append(kappa)
        observed_kappa = float(cohen_kappa_score(human, judge, labels=LABELS))
    result = {"experiment": "E3_neutral_prompt_human_judge_agreement", "n": len(rows),
              "purpose": "descriptive_only_not_a_gate",
              "agreement": float(np.mean(human_array == judge_array)),
              "agreement_ci95": [float(x) for x in np.percentile(agreement_samples, [2.5, 97.5])],
              "cohen_kappa": observed_kappa if np.isfinite(observed_kappa) else None,
              "cohen_kappa_ci95": ([float(x) for x in np.percentile(kappa_samples, [2.5, 97.5])]
                                   if kappa_samples else [None, None]),
              "confusion_matrix_rows_human_cols_judge": confusion_matrix(human, judge, labels=LABELS).tolist(),
              "labels": LABELS}
    out = write_exploratory_json(args.output, result, ROOT)
    print(f"Wrote descriptive agreement summary to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
