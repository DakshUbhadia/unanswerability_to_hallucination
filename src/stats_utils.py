"""Pre-registered statistics shared by Stages 6-10 (Section 6 of the specification)."""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from src.config import CFG


def binary_auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """Exact Mann-Whitney AUROC P(pos > neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(scores_pos, dtype=np.float64), np.asarray(scores_neg, dtype=np.float64)
    assert pos.ndim == neg.ndim == 1 and len(pos) and len(neg), "AUROC needs two non-empty 1D groups"
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def auroc_rows(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Vectorized AUROC over rows: pos [R, n1], neg [R, n0] -> [R]."""
    pos, neg = np.atleast_2d(pos).astype(np.float64), np.atleast_2d(neg).astype(np.float64)
    allv = np.concatenate([pos, neg], axis=1)
    ranks = stats.rankdata(allv, axis=1)
    n1, n0 = pos.shape[1], neg.shape[1]
    return (ranks[:, :n1].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cluster_means(values: Sequence[float], cluster_ids: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    ids = sorted(set(cluster_ids))
    idx = {c: i for i, c in enumerate(ids)}
    sums, counts = np.zeros(len(ids)), np.zeros(len(ids))
    for val, c in zip(v, cluster_ids):
        sums[idx[c]] += val
        counts[idx[c]] += 1
    return ids, sums / counts


def wilcoxon_greater(x: Sequence[float]) -> Dict[str, float]:
    """One-sided Wilcoxon signed-rank test of median(x) > 0."""
    x = np.asarray(x, dtype=np.float64)
    if np.allclose(x, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "n": int(len(x))}
    res = stats.wilcoxon(x, alternative="greater", zero_method="wilcox")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n": int(len(x))}


def cluster_bootstrap(values: Sequence[float], cluster_ids: Sequence[str], n_boot: int = CFG.n_bootstrap,
                      seed: int = CFG.seed_bootstrap, ci_level: float = CFG.ci_level) -> Dict[str, float]:
    """Template-cluster bootstrap of the cluster-weighted mean (clusters resampled with replacement)."""
    ids, means = cluster_means(values, cluster_ids)
    counts = np.array([sum(1 for c in cluster_ids if c == i) for i in ids], dtype=np.float64)
    sums = means * counts
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(ids), size=(int(n_boot), len(ids)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.percentile(boot, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2])
    mean = float(sums.sum() / counts.sum())
    sd_c = float(means.std(ddof=1)) if len(means) > 1 else 0.0
    p_two = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))) if mean != 0 else 1.0
    return {"mean": mean, "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "se_cluster": float(boot.std(ddof=1)), "ci95_low": float(lo), "ci95_high": float(hi),
            "cohens_dz": float(means.mean() / sd_c) if sd_c > 0 else None,
            "p_boot_two_sided": max(p_two, 1.0 / (n_boot + 1)), "n_clusters": len(ids), "n_items": len(values)}


def item_bootstrap_mean(values: Sequence[float], n_boot: int = CFG.n_bootstrap, seed: int = CFG.seed_bootstrap,
                        ci_level: float = CFG.ci_level) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    boot = v[rng.integers(0, len(v), size=(int(n_boot), len(v)))].mean(1)
    lo, hi = np.percentile(boot, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2])
    return {"mean": float(v.mean()), "ci95_low": float(lo), "ci95_high": float(hi), "n": int(len(v))}


def monte_carlo_p_greater(observed: float, null: Sequence[float]) -> float:
    """(1 + #(null >= observed)) / (M + 1): rank p-value where larger effects are better."""
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def summarize_null(observed: float, null: Sequence[float]) -> Dict[str, Any]:
    null = np.asarray(null, dtype=np.float64)
    sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    return {"n": int(len(null)), "mean": float(null.mean()), "std": sd, "min": float(null.min()),
            "max": float(null.max()), "q025": float(np.percentile(null, 2.5)),
            "q975": float(np.percentile(null, 97.5)),
            "z_score": float((observed - null.mean()) / sd) if sd > 0 else None,
            "percentile_of_observed": float((null < observed).mean() * 100.0),
            "rank_of_observed": int(1 + np.sum(null >= observed)),
            "p_rank": monte_carlo_p_greater(observed, null)}


def holm_bonferroni(p_values: Dict[str, float], alpha: float = CFG.alpha_fwer,
                    extra_criteria: Optional[Dict[str, bool]] = None) -> List[Dict[str, Any]]:
    """Step-down Holm: thresholds alpha/(m-k+1); stop at the first failure.

    `extra_criteria` holds the non-p-value parts of each confirmatory criterion (e.g. MC rank 1);
    they decide `confirmed` but never change the step-down sequence of the p-values themselves.
    """
    items = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    rows, still_testing = [], True
    for k, (name, p) in enumerate(items, start=1):
        thr = alpha / (m - k + 1)
        passed = bool(still_testing and p <= thr)
        if not passed:
            still_testing = False
        extra = True if extra_criteria is None else bool(extra_criteria.get(name, True))
        rows.append({"test_id": name, "raw_p_value": float(p), "rank": k, "holm_threshold": float(thr),
                     "passed_holm_bonferroni": passed, "extra_criteria_met": extra,
                     "confirmed": passed and extra})
    return rows


def paired_bootstrap_auroc(pos_scores: Dict[str, np.ndarray], neg_scores: Dict[str, np.ndarray],
                           primary_key: str, n_boot: int = CFG.n_bootstrap, seed: int = CFG.seed_bootstrap,
                           ci_level: float = CFG.ci_level) -> Dict[str, Any]:
    """Resample matched (pos_i, neg_i) pairs together; AUROC for every score and primary-minus-other deltas."""
    keys = list(pos_scores)
    n = len(pos_scores[primary_key])
    assert all(len(pos_scores[k]) == len(neg_scores[k]) == n for k in keys), "Scores must be pair-aligned"
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, n, size=(int(n_boot), n))
    q = [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2]
    boot = {k: auroc_rows(np.asarray(pos_scores[k])[draws], np.asarray(neg_scores[k])[draws]) for k in keys}
    out: Dict[str, Any] = {"n_pairs": n, "n_boot": int(n_boot), "auroc": {}, "delta": {}}
    for k in keys:
        lo, hi = np.percentile(boot[k], q)
        out["auroc"][k] = {"point": binary_auroc(pos_scores[k], neg_scores[k]), "ci95_low": float(lo),
                           "ci95_high": float(hi)}
    for k in keys:
        if k == primary_key:
            continue
        d = boot[primary_key] - boot[k]
        lo, hi = np.percentile(d, q)
        point = out["auroc"][primary_key]["point"] - out["auroc"][k]["point"]
        out["delta"][k] = {"point": float(point), "ci95_low": float(lo), "ci95_high": float(hi),
                           "p_boot_one_sided": float((1 + np.sum(d <= 0)) / (n_boot + 1)),
                           "p_boot_two_sided": float(min(1.0, 2 * min((1 + np.sum(d <= 0)) / (n_boot + 1),
                                                                      (1 + np.sum(d >= 0)) / (n_boot + 1))))}
    out["_boot_primary"] = boot[primary_key]
    return out


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None
