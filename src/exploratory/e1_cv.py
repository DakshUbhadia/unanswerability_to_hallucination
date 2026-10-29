"""Leakage-conscious nested cross-validation helpers for exploratory E1."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ESTIMATORS = ("logistic", "dom", "pca64_logistic")


def outer_splits(y, groups, folds=5, repeats=5, seed=20260414):
    y, groups = np.asarray(y), np.asarray(groups)
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two distinct groups are required for grouped CV")
    for repeat in range(int(repeats)):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed) + repeat)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            overlap = set(groups[train]) & set(groups[test])
            if overlap:
                raise AssertionError(f"Groups cross folds: {list(overlap)[:5]}")
            yield repeat, fold, train, test


def _balanced(estimator, class_weight):
    return estimator in {"logistic", "pca64_logistic"} and class_weight == "balanced"


def fit_probe(est, X_tr, y_tr, C=0.1, class_weight=None):
    X_tr, y_tr = np.asarray(X_tr, dtype=np.float32), np.asarray(y_tr, dtype=int)
    scaler = StandardScaler().fit(X_tr)
    z = scaler.transform(X_tr)
    if est == "dom":
        wrong = z[y_tr == 1].mean(axis=0)
        correct = z[y_tr == 0].mean(axis=0)
        direction = wrong - correct
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            direction = np.zeros_like(direction)
        return ("dom", scaler, direction)
    if est == "pca64_logistic":
        pca = PCA(n_components=min(64, max(1, len(y_tr) - 1)), random_state=0).fit(z)
        z = pca.transform(z)
    elif est != "logistic":
        raise ValueError(f"Unknown estimator {est!r}")
    clf = LogisticRegression(penalty="l2", C=float(C), solver="lbfgs", max_iter=5000,
                             class_weight=class_weight)
    clf.fit(z, y_tr)
    return (est, scaler, pca if est == "pca64_logistic" else None, clf)


def score_probe(model, X_te):
    X = np.asarray(X_te, dtype=np.float32)
    if model[0] == "dom":
        return model[1].transform(X) @ model[2]
    z = model[1].transform(X)
    if model[0] == "pca64_logistic":
        z = model[2].transform(z)
    clf = model[3]
    return clf.predict_proba(z)[:, 1]


def _auc(y, score):
    if len(np.unique(y)) != 2:
        return 0.5
    return float(roc_auc_score(y, score))


def _oof_single(X, y, groups, folds, seed, est="logistic", C=0.1, class_weight=None):
    scores = np.full(len(y), np.nan, dtype=float)
    for _, _, tr, te in outer_splits(y, groups, folds, 1, seed):
        model = fit_probe(est, X[tr], y[tr], C, class_weight)
        scores[te] = score_probe(model, X[te])
    if np.isnan(scores).any():
        raise AssertionError("OOF prediction missing for one or more rows")
    return scores


def select_layer_C(X_by_layer_tr, y_tr, groups_tr, C_grid=(0.001, 0.01, 0.1, 1.0),
                   inner_folds=5, seed=20260414, class_weight=None):
    """Select layer and C using only arrays supplied from the outer training fold."""
    Xs, y, groups = np.asarray(X_by_layer_tr), np.asarray(y_tr), np.asarray(groups_tr)
    candidates = []
    for layer in range(Xs.shape[1]):
        for C in sorted(map(float, C_grid)):
            pred = _oof_single(Xs[:, layer, :], y, groups, inner_folds, seed, "logistic", C, class_weight)
            candidates.append((_auc(y, pred), -C, -layer, layer, C))
    best = max(candidates)
    return int(best[3]), float(best[4])


def nested_probe_oof(X_by_layer, y, groups, folds=5, repeats=5, seed=20260414,
                     estimator="logistic", C_grid=(0.001, 0.01, 0.1, 1.0), inner_folds=5,
                     class_weight=None):
    Xs, y, groups = np.asarray(X_by_layer), np.asarray(y), np.asarray(groups)
    per_repeat = np.full((repeats, len(y)), np.nan)
    selected = []
    for r, k, tr, te in outer_splits(y, groups, folds, repeats, seed):
        if estimator == "logistic":
            layer, C = select_layer_C(Xs[tr], y[tr], groups[tr], C_grid, inner_folds, seed + r * 100 + k,
                                      class_weight)
        else:
            # Secondary fixed-estimator selection uses inner AUROC at C=0.1.
            layer_scores = []
            for li in range(Xs.shape[1]):
                pred = _oof_single(Xs[tr, li], y[tr], groups[tr], inner_folds, seed + r * 100 + k,
                                   estimator, 0.1, class_weight)
                layer_scores.append(_auc(y[tr], pred))
            layer, C = int(np.argmax(layer_scores)), 0.1
        model = fit_probe(estimator, Xs[tr, layer], y[tr], C, class_weight)
        per_repeat[r, te] = score_probe(model, Xs[te, layer])
        selected.append({"repeat": int(r), "fold": int(k), "layer": int(layer), "C": float(C)})
    if np.isnan(per_repeat).any():
        raise AssertionError("Nested OOF predictions are incomplete")
    return per_repeat, selected


def _inner_oof_probe(X, y, groups, folds, seed, layer, C, class_weight=None):
    return _oof_single(X[:, layer, :], y, groups, folds, seed, "logistic", C, class_weight)


def nested_stacking_delta(X_by_layer, base, y, groups, folds=5, repeats=5, seed=20260414,
                          C_grid=(0.001, 0.01, 0.1, 1.0), inner_folds=5, class_weight=None):
    """Nested OOF scores for base-only and base+probe stackers."""
    Xs, base, y, groups = map(np.asarray, (X_by_layer, base, y, groups))
    full = np.full((repeats, len(y)), np.nan)
    base_only = np.full_like(full, np.nan)
    selected = []
    for r, k, tr, te in outer_splits(y, groups, folds, repeats, seed):
        layer, C = select_layer_C(Xs[tr], y[tr], groups[tr], C_grid, inner_folds,
                                  seed + 1000 + r * 100 + k, class_weight)
        probe_oof = _inner_oof_probe(Xs[tr], y[tr], groups[tr], inner_folds,
                                     seed + 2000 + r * 100 + k, layer, C, class_weight)
        probe_test = score_probe(fit_probe("logistic", Xs[tr, layer], y[tr], C, class_weight), Xs[te, layer])
        stack_x = np.column_stack((base[tr], probe_oof))
        scaler = StandardScaler().fit(stack_x)
        stack = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000,
                                   class_weight=class_weight).fit(scaler.transform(stack_x), y[tr])
        full[ r, te] = stack.predict_proba(scaler.transform(np.column_stack((base[te], probe_test))))[:, 1]
        bscaler = StandardScaler().fit(base[tr])
        bmodel = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000,
                                    class_weight=class_weight).fit(bscaler.transform(base[tr]), y[tr])
        base_only[r, te] = bmodel.predict_proba(bscaler.transform(base[te]))[:, 1]
        selected.append({"repeat": int(r), "fold": int(k), "layer": int(layer), "C": float(C)})
    if np.isnan(full).any() or np.isnan(base_only).any():
        raise AssertionError("Stacking OOF predictions are incomplete")
    return {"full_oof": full, "base_oof": base_only, "selected": selected,
            "delta_by_repeat": np.asarray([_auc(y, full[r]) - _auc(y, base_only[r]) for r in range(repeats)])}


def conditional_bootstrap(pred_a, pred_b, y, units, n=1000, seed=20260407):
    """Paired unit bootstrap for AUROC difference; predictions are held fixed."""
    y, units = np.asarray(y, dtype=int), np.asarray(units).astype(str)
    a, b = np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float)
    if a.ndim == 2:
        a, b = a.mean(axis=0), b.mean(axis=0)
    unique = np.unique(units)
    members = {u: np.flatnonzero(units == u) for u in unique}
    rng = np.random.default_rng(seed)
    deltas, auc_a, auc_b = [], [], []
    for _ in range(int(n)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[u] for u in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        aa, bb = _auc(y[idx], a[idx]), _auc(y[idx], b[idx])
        auc_a.append(aa); auc_b.append(bb); deltas.append(aa - bb)
    if not deltas:
        raise ValueError("Bootstrap produced no samples containing both classes")
    return {"auroc_a": _auc(y, a), "auroc_b": _auc(y, b), "delta": _auc(y, a) - _auc(y, b),
            "ci_95": [float(x) for x in np.percentile(deltas, [2.5, 97.5])],
            "p_one_sided": float((1 + sum(d <= 0 for d in deltas)) / (len(deltas) + 1)),
            "bootstrap_n": len(deltas), "ci_conditional_on_oof_predictions": True}


def run_descriptive_grid(X_by_layer, y, groups, folds=5, repeats=5, seed=20260414, n_jobs=1):
    """Fixed-C descriptive logistic AUROC by layer; optionally parallelized over layers."""
    from joblib import Parallel, delayed
    Xs, y, groups = np.asarray(X_by_layer), np.asarray(y), np.asarray(groups)
    def one(layer):
        pred = np.full((repeats, len(y)), np.nan)
        for r in range(repeats):
            pred[r] = _oof_single(Xs[:, layer], y, groups, folds, seed + r, "logistic", 0.1)
        return layer, pred, [_auc(y, p) for p in pred]
    return {layer: {"oof": pred, "auroc_by_repeat": aucs, "auroc_mean": float(np.mean(aucs)),
                    "auroc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0}
            for layer, pred, aucs in Parallel(n_jobs=n_jobs)(delayed(one)(i) for i in range(Xs.shape[1]))}
