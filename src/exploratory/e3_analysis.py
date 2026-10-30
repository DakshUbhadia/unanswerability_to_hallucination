"""CPU-side E3 contrasts and preregistered interpretation rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.incremental_value import incremental_value_test


def incremental_value_for_projection(frame, projection_column: str, n_boot=1000, seed=20260414):
    """Map E3's confidence/projection columns to the unchanged registered test API."""
    required = {"maxp", "entropy", "log_odds_set", projection_column, "y", "matched_pair_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"E3 incremental-value frame is missing {sorted(missing)}")
    mapped = frame.copy()
    mapped["log_odds"] = mapped["log_odds_set"]
    mapped["proj"] = mapped[projection_column]
    return incremental_value_test(mapped, n_boot=n_boot, seed=seed,
                                 full_features=["maxp", "entropy", "log_odds", "proj"])


def independent_condition_bootstrap(y_a, score_a, y_b, score_b, *, n_boot=1000, seed=20260407):
    """Bootstrap AUROC contrasts independently; never pairs unrelated rows across prompts."""
    ya, sa, yb, sb = (np.asarray(x) for x in (y_a, score_a, y_b, score_b))
    if len(ya) != len(sa) or len(yb) != len(sb):
        raise ValueError("Labels and scores must align within each condition")
    point = float(roc_auc_score(ya, sa) - roc_auc_score(yb, sb))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_boot)):
        ia = rng.integers(0, len(ya), len(ya))
        ib = rng.integers(0, len(yb), len(yb))
        if len(np.unique(ya[ia])) == 2 and len(np.unique(yb[ib])) == 2:
            boot.append(roc_auc_score(ya[ia], sa[ia]) - roc_auc_score(yb[ib], sb[ib]))
    if not boot:
        raise ValueError("No valid independent bootstrap samples")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta_auroc": point, "ci95_low": float(lo), "ci95_high": float(hi),
            "n_boot": len(boot), "seed": int(seed), "resampling": "independent_by_condition"}


def label_concordant_subset(instr, neutral):
    """Return rows for questions with the same correct/wrong outcome in both conditions."""
    a, b = pd.DataFrame(instr).copy(), pd.DataFrame(neutral).copy()
    for name, frame in (("instructed", a), ("neutral", b)):
        if not {"popqa_id", "label"}.issubset(frame.columns):
            raise ValueError(f"{name} frame requires popqa_id and label")
    joined = a[["popqa_id", "label"]].merge(b[["popqa_id", "label"]], on="popqa_id",
                                               suffixes=("_instr", "_neutral"), validate="one_to_one")
    keep = joined["label_instr"].isin(["correct", "wrong"]) & (joined["label_instr"] == joined["label_neutral"])
    ids = set(joined.loc[keep, "popqa_id"])
    return a[a.popqa_id.isin(ids)].copy(), b[b.popqa_id.isin(ids)].copy()


def label_transition_table(instr, neutral):
    a, b = pd.DataFrame(instr), pd.DataFrame(neutral)
    joined = a[["popqa_id", "label"]].merge(b[["popqa_id", "label"]], on="popqa_id",
                                               suffixes=("_instr", "_neutral"), validate="one_to_one")
    return pd.crosstab(joined.label_instr, joined.label_neutral, dropna=False).to_dict(orient="index")


def e3_verdict(*, in_context_survives: bool, aligned_with_instructed: bool | None,
               separation: str, residual: str) -> str:
    """Apply the exhaustive Llama verdict tree declared in E3 section 4.H."""
    if separation not in {"present", "absent", "inconclusive"}:
        raise ValueError("separation must be present, absent, or inconclusive")
    if residual not in {"residual_null_equivalent", "residual_signal_present", "inconclusive"}:
        raise ValueError("unknown residual label")
    if not in_context_survives:
        return "Direction is instruction-dependent"
    if separation == "absent":
        return "Instruction-dependent transfer"
    if separation == "present" and residual == "residual_signal_present":
        return "Knowledge-state component under the neutral prompt"
    if (aligned_with_instructed is True and separation == "present"
            and residual == "residual_null_equivalent"):
        return "Instruction-robust confidence/abstention readout"
    return ("mixed/inconclusive: "
            f"in_context_survives=True, aligned_with_instructed={aligned_with_instructed}, "
            f"separation={separation}, residual={residual}")


def qwen_replication(llama_verdict: str, qwen_verdict: str) -> str:
    return ("replicates_llama_neutral_verdict" if qwen_verdict == llama_verdict
            else f"does_not_replicate: {qwen_verdict}")


def direction_from_pairs(pair_activations, rows, *, split="train"):
    """Fit mean unanswerable-minus-answerable directions by layer and suffix."""
    acts = np.asarray(pair_activations)
    if acts.ndim != 4:
        raise ValueError("pair activations must have [prompts, suffix, layers, hidden] shape")
    index = {(str(r["pair_id"]), str(r["part"])): i for i, r in enumerate(rows)
             if str(r["split"]) == split}
    pair_ids = sorted({pair for pair, _ in index})
    if not pair_ids:
        raise ValueError(f"No {split} pairs in E3 feature manifest")
    ans = np.stack([acts[index[(pair, "ans")]] for pair in pair_ids])
    unans = np.stack([acts[index[(pair, "unans")]] for pair in pair_ids])
    diff = (unans.astype(np.float64).mean(0) - ans.astype(np.float64).mean(0)).transpose(1, 0, 2)
    norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("E3 train pairs produced a zero direction")
    return diff, diff / norm, pair_ids


def in_context_auroc(pair_activations, rows, direction, *, split="test"):
    """Score answerable/unanswerable synthetic prompts on held-out pairs."""
    acts = np.asarray(pair_activations)
    selected = [(i, r) for i, r in enumerate(rows) if str(r["split"]) == split]
    if not selected:
        raise ValueError(f"No {split} pairs available")
    y = np.asarray([str(r["part"]) == "unans" for _, r in selected], dtype=int)
    x = np.stack([acts[i] for i, _ in selected]).astype(np.float64)
    scores = np.einsum("npld,lpd->npl", x, direction)
    from sklearn.metrics import roc_auc_score
    aurocs = np.empty(direction.shape[:2], dtype=float)
    for layer in range(direction.shape[0]):
        for pos in range(direction.shape[1]):
            aurocs[layer, pos] = roc_auc_score(y, scores[:, pos, layer])
    return aurocs


def bootstrap_train_direction_alignment(pair_activations, rows, reference, *, n_boot=1000, seed=20260407):
    """Refit neutral DOM means over train-pair bootstrap samples for geometry CI."""
    from src.exploratory.e2_analysis import _scale_equivariant_lw_covariance
    acts = np.asarray(pair_activations)
    lookup = {(str(r["pair_id"]), str(r["part"])): i for i, r in enumerate(rows)
              if str(r["split"]) == "train"}
    pair_ids = sorted({p for p, _ in lookup})
    if not pair_ids:
        raise ValueError("No E3 training pairs available for geometry bootstrap")
    delta = np.stack([acts[lookup[(pid, "unans")], 1, 11].astype(np.float64) -
                      acts[lookup[(pid, "ans")], 1, 11].astype(np.float64) for pid in pair_ids])
    pool = np.asarray(acts[[i for i, r in enumerate(rows) if r["split"] == "train"] , 1, 11], dtype=np.float64)
    covariance = _scale_equivariant_lw_covariance(pool)
    vals, vecs = np.linalg.eigh((covariance + covariance.T) / 2)
    floor = 1e-6 * np.trace(covariance) / covariance.shape[0]
    invroot = (vecs * (1 / np.sqrt(np.maximum(vals, floor)))) @ vecs.T
    refw = invroot @ np.asarray(reference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    raw, white = [], []
    for _ in range(int(n_boot)):
        d = delta[rng.integers(0, len(delta), len(delta))].mean(0)
        norm = np.linalg.norm(d)
        if norm <= 1e-12:
            continue
        d /= norm
        den = np.linalg.norm(d) * np.linalg.norm(reference)
        raw.append(float(np.dot(reference, d) / den))
        dw = invroot @ d
        white.append(float(np.dot(refw, dw) / (np.linalg.norm(refw) * np.linalg.norm(dw))))
    if not white:
        raise ValueError("No valid direction bootstrap samples")
    return {"n_boot": len(white), "seed": int(seed),
            "raw_ci95": [float(x) for x in np.percentile(raw, [2.5, 97.5])],
            "whitened_ci95": [float(x) for x in np.percentile(white, [2.5, 97.5])],
            "resampling": "train_pairs_with_replacement"}
