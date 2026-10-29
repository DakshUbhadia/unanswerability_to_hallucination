"""Group-aware exploratory models and uncertainty summaries."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _validate(X, y, groups, folds):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(groups):
        raise ValueError("X, y, and groups must be aligned one- or two-dimensional arrays")
    if len(np.unique(y)) != 2:
        raise ValueError("Both binary classes are required")
    if min((len(np.unique(groups[y == label])) for label in (0, 1))) < folds:
        raise ValueError(f"Each class needs at least {folds} distinct groups")
    return X, y, groups


def nested_group_probe_oof(X, y, groups, folds=5, repeats=20, seed=0,
                           c_values=(0.01, 0.1, 1.0), inner_folds=3):
    """Return repeated OOF probe predictions with C selected inside every outer fold."""
    X, y, groups = _validate(X, y, groups, int(folds))
    unique_counts = [len(np.unique(groups[y == label])) for label in (0, 1)]
    inner_n = min(int(inner_folds), *unique_counts)
    if inner_n < 2:
        raise ValueError("Nested tuning needs at least two training groups per class")
    predictions = np.full((int(repeats), len(y)), np.nan, dtype=float)
    chosen_c = np.zeros((int(repeats), len(y)), dtype=float)
    for repeat in range(int(repeats)):
        outer = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed) + repeat)
        for train, test in outer.split(X, y, groups):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("An outer group was split across folds")
            inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=int(seed) + repeat + 7919)
            c_scores = []
            for c_value in c_values:
                scores = []
                for inner_train, inner_val in inner.split(X[train], y[train], groups[train]):
                    scaler = StandardScaler().fit(X[train][inner_train])
                    model = LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000)
                    model.fit(scaler.transform(X[train][inner_train]), y[train][inner_train])
                    pred = model.predict_proba(scaler.transform(X[train][inner_val]))[:, 1]
                    scores.append(roc_auc_score(y[train][inner_val], pred))
                c_scores.append(float(np.mean(scores)))
            best_c = float(c_values[int(np.argmax(c_scores))])
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(C=best_c, solver="lbfgs", max_iter=2000)
            model.fit(scaler.transform(X[train]), y[train])
            predictions[repeat, test] = model.predict_proba(scaler.transform(X[test]))[:, 1]
            chosen_c[repeat, test] = best_c
    if not np.isfinite(predictions).all():
        raise RuntimeError("Nested OOF predictions are incomplete or non-finite")
    return predictions, chosen_c


def group_oof_scores(X, y, groups, folds=5, repeats=20, seed=0, c=1.0):
    """Repeated group OOF logistic scores for a non-nested baseline."""
    X, y, groups = _validate(X, y, groups, int(folds))
    predictions = np.full((int(repeats), len(y)), np.nan, dtype=float)
    for repeat in range(int(repeats)):
        cv = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed) + repeat)
        for train, test in cv.split(X, y, groups):
            scaler = StandardScaler().fit(X[train])
            model = LogisticRegression(C=float(c), solver="lbfgs", max_iter=2000)
            model.fit(scaler.transform(X[train]), y[train])
            predictions[repeat, test] = model.predict_proba(scaler.transform(X[test]))[:, 1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Group OOF predictions are incomplete or non-finite")
    return predictions


def bootstrap_auc_ci(y, score, groups, n_boot=1000, seed=0):
    """Cluster bootstrap AUROC by pair/relation group; return percentile 95% CI."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    groups = np.asarray(groups).astype(str)
    if len(y) != len(score) or len(y) != len(groups):
        raise ValueError("Bootstrap vectors must align")
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(n_boot)):
        picked = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in picked])
        if np.unique(y[indices]).size < 2:
            continue
        values.append(float(roc_auc_score(y[indices], score[indices])))
    if not values:
        raise ValueError("No valid cluster bootstrap replicates")
    low, high = np.percentile(values, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "n_valid": len(values)}


def auc_summary(y, scores, groups, n_boot=1000, seed=0):
    y = np.asarray(y, dtype=int)
    score = np.asarray(scores, dtype=float)
    return {"auroc": float(roc_auc_score(y, score)),
            "ci95": bootstrap_auc_ci(y, score, groups, n_boot=n_boot, seed=seed)}
