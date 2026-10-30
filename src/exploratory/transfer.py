"""Exploratory cross-source probe transfer interface (dataset execution remains owner-run)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.config import CFG
from src.exploratory import assert_not_fresh

if __name__ == "__main__" and "--legacy" not in sys.argv:
    print("superseded by src/exploratory/e4_transfer.py; pass --legacy to run", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "features"


def load_source(name: str):
    """Return prompts, binary labels (1=unanswerable/abstaining), and leakage groups."""
    if name == "synthetic_pairs":
        pair_path = assert_not_fresh(CFG.data_dir / "pairs.jsonl")
        from src.eval_common import load_split_pairs
        rows = []
        for split in ("train", "val", "test"):
            for pair in load_split_pairs(split, pair_path):
                rows.extend([(pair["ans_prompt"], 0, pair["pair_id"]),
                             (pair["unans_prompt"], 1, pair["pair_id"])])
    elif name == "squad_v2":
        # TODO(owner): check the dataset license/terms and run the GPU feature cache before using this source.
        path = assert_not_fresh(CFG.data_dir / "exploratory" / "squad_v2.jsonl")
        if not path.exists():
            raise FileNotFoundError(f"SQuAD v2 is not downloaded; owner must check licensing first: {path}")
        rows = []
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
            context, question = row.get("context", ""), row.get("question", "")
            prompt = f"Context:\n{context}\n\nQuestion: {question}"
            unanswerable = not bool(row.get("answers", {}).get("text", []))
            rows.append((prompt, int(unanswerable), str(row.get("id", row.get("example_id")))))
    elif name == "popqa_v42":
        path = assert_not_fresh(CFG.data_dir / "popqa_sample.jsonl")
        rows = []
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()):
            grade = row.get("baseline_partition")
            if grade not in {"abstain", "correct", "wrong"}:
                continue
            rows.append((row["question"], int(grade == "abstain"), str(row["popqa_id"])))
    else:
        raise ValueError(f"Unknown exploratory transfer source {name!r}")
    return ([row[0] for row in rows], np.asarray([row[1] for row in rows], dtype=int),
            np.asarray([row[2] for row in rows], dtype=str))


def _source_features(name: str):
    prefix = {"synthetic_pairs": "pairs", "squad_v2": "squad_v2", "popqa_v42": "popqa"}[name]
    meta_path = assert_not_fresh(FEATURES_DIR / f"{prefix}_meta.parquet")
    resid_path = assert_not_fresh(FEATURES_DIR / f"{prefix}_resid.npy")
    import pandas as pd
    meta = pd.read_parquet(meta_path)
    resid = np.load(resid_path, mmap_mode="r")
    if len(meta) != resid.shape[0]:
        raise ValueError(f"{name} feature metadata and residual rows are not aligned")
    if name == "synthetic_pairs":
        labels = (meta["grade"].astype(str) == "unanswerable").to_numpy(dtype=int)
    elif name == "squad_v2":
        labels = meta["label"].to_numpy(dtype=int)
    else:
        labels = (meta["grade"].astype(str) == "abstain").to_numpy(dtype=int)
    return meta, resid, labels


def transfer_matrix(train_source: str, test_source: str, layer: int, pos: int,
                    train_arrays=None, test_arrays=None):
    """Train a linear residual probe on one source; score AUROC and logit-baseline delta on another."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    pos_index = {-5: 0, -4: 1, -3: 2, -2: 3, -1: 4}.get(int(pos))
    if pos_index is None:
        raise ValueError("pos must be one of -5, -4, -3, -2, -1")
    if train_arrays is None:
        _train_meta, train_resid, y_train = _source_features(train_source)
        train_x = np.asarray(train_resid[:, pos_index, int(layer), :], dtype=float)
    else:
        train_x, y_train = np.asarray(train_arrays[0], dtype=float), np.asarray(train_arrays[1], dtype=int)
    if test_arrays is None:
        test_meta, test_resid, y_test = _source_features(test_source)
        test_x = np.asarray(test_resid[:, pos_index, int(layer), :], dtype=float)
        baseline = test_meta["log_odds"].to_numpy(dtype=float)
    else:
        test_x, y_test, baseline = (np.asarray(test_arrays[0], dtype=float),
                                    np.asarray(test_arrays[1], dtype=int),
                                    np.asarray(test_arrays[2], dtype=float))
    if len(train_x) != len(y_train) or len(test_x) != len(y_test) or len(baseline) != len(y_test):
        raise ValueError("Transfer feature, label, and baseline arrays must align")
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(scaler.transform(train_x), y_train)
    prediction = model.predict_proba(scaler.transform(test_x))[:, 1]
    auc = float(roc_auc_score(y_test, prediction))
    baseline_auc = float(roc_auc_score(y_test, baseline))
    return {"train_source": train_source, "test_source": test_source, "layer": int(layer), "pos": int(pos),
            "auroc": auc, "logit_baseline_auroc": baseline_auc, "auroc_minus_logit_baseline": auc - baseline_auc,
            "n_train": len(y_train), "n_test": len(y_test)}
