"""CPU analysis helpers for the exploratory E4 cross-source transfer matrix.

The functions in this module deliberately keep training and test selection
explicit. Labels always use 1 for unanswerable / wrong, and probe signs are
never adjusted using the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.exploratory import assert_not_fresh, require_tier1_complete
from src.exploratory.e1_cv import outer_splits, nested_stacking_delta

ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "results/exploratory/e4_features"
SOURCES = ("syn", "squad", "selfaware", "popqa", "triviaqa")
POSITIONS = {"suf-5": 0, "suf-4": 1, "suf-3": 2, "suf-2": 3, "suf-1": 4}


@dataclass
class Source:
    name: str
    X: np.ndarray
    meta: pd.DataFrame
    family: str
    texts: list[str] | None = None

    @property
    def y(self) -> np.ndarray:
        return self.meta["y"].to_numpy(dtype=int)

    @property
    def groups(self) -> np.ndarray:
        col = "group"
        if col not in self.meta:
            raise ValueError(f"{self.name} metadata is missing the preregistered group column")
        return self.meta[col].astype(str).to_numpy()

    def part(self, part: str = "all") -> "Source":
        if part == "all":
            return self
        if self.name == "syn":
            if part not in {"train", "val", "test"}:
                raise ValueError(f"Unknown synthetic split {part!r}")
            keep = self.meta["split"].astype(str).to_numpy() == part
        elif self.name == "popqa" and part in {"P1", "P2"}:
            col = "is_popularity_matched" if part == "P1" else "included_in_v42"
            if part == "P2" and col not in self.meta:
                return self
            if col not in self.meta:
                raise ValueError(f"PopQA metadata lacks {col!r} for view {part}")
            keep = self.meta[col].astype(bool).to_numpy()
        else:
            raise ValueError(f"Part {part!r} is not defined for {self.name}")
        ix = np.flatnonzero(keep)
        return Source(self.name, self.X[ix], self.meta.iloc[ix].reset_index(drop=True), self.family,
                      [self.texts[i] for i in ix] if self.texts is not None else None)


@dataclass
class Probe:
    estimator: str
    scaler: StandardScaler | None
    direction: np.ndarray | None
    model: LogisticRegression | None
    layer: int
    position: int


FAMILY = {"syn": "contextual", "squad": "contextual", "selfaware": "other",
          "popqa": "parametric", "triviaqa": "parametric"}


def load_source(name: str, condition: str = "registered", view: str | None = None) -> Source:
    """Load a feature artifact and optionally select syn or PopQA rows."""
    if name not in FAMILY:
        raise ValueError(f"Unknown E4 source {name!r}")
    if name == "popqa":
        require_tier1_complete(ROOT)
    if name == "popqa" and (FEATURE_ROOT / condition / name / "manifest.json").exists():
        folder = FEATURE_ROOT / condition / name
    elif name == "popqa" and condition == "registered":
        folder = ROOT / "results/exploratory/e1_features"
    elif name == "popqa":
        raise FileNotFoundError("Neutral PopQA requires the E3 cache referenced by the E4 manifest")
    else:
        folder = FEATURE_ROOT / condition / name
    folder = assert_not_fresh(folder)
    manifest_path, meta_path, feat_path = folder / "manifest.json", folder / "meta.parquet", folder / "suf.npy"
    if name == "popqa" and condition == "registered" and not manifest_path.exists():
        manifest_path = folder / "features_manifest.json"
    for path in (manifest_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not feat_path.exists() and manifest.get("suf_path"):
        cached = Path(manifest["suf_path"])
        feat_path = assert_not_fresh(cached if cached.is_absolute() else ROOT / cached)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    meta = pd.read_parquet(meta_path)
    X = np.load(feat_path, mmap_mode="r")
    if len(meta) != X.shape[0]:
        raise ValueError(f"{name} features and metadata are misaligned")
    if name == "popqa":
        eligible = meta.get("use_in_e1", pd.Series(False, index=meta.index)).fillna(False).astype(bool).to_numpy()
        if "baseline_partition" in meta:
            eligible |= meta["baseline_partition"].astype(str).isin(("correct", "wrong")).to_numpy()
        indices = np.flatnonzero(eligible)
        meta = meta.iloc[indices].copy().reset_index(drop=True)
        X = X[indices]
        if "y" not in meta or meta["y"].isna().any():
            meta["y"] = (meta["baseline_partition"].astype(str) == "wrong").astype(int)
        matched = meta.get("matched_pair_id", pd.Series(None, index=meta.index)).fillna("").astype(str)
        popqa_ids = meta.get("popqa_id", pd.Series(np.arange(len(meta)), index=meta.index)).astype(str)
        meta["group"] = np.where(matched.ne("") & matched.ne("None"), matched, popqa_ids)
        if "question" in meta and "user_prompt" not in meta:
            meta["user_prompt"] = meta["question"].astype(str)
    if "y" not in meta or "group" not in meta:
        raise ValueError(f"{name} metadata must include y and group")
    texts = meta["user_prompt"].astype(str).tolist() if "user_prompt" in meta else None
    src = Source(name, X, meta, FAMILY[name], texts)
    return src.part(view) if view else src


def _position_index(pos: int | str) -> int:
    if isinstance(pos, str):
        if pos not in POSITIONS:
            raise ValueError(f"Position must be one of {tuple(POSITIONS)}")
        return POSITIONS[pos]
    if int(pos) not in range(5):
        raise ValueError("Position index must be 0..4 or use a suffix name")
    return int(pos)


def fit_direction(est: str, X_train, y_train, C: float | None = None) -> Probe:
    """Fit DOM in raw activation space or standardized L2 logistic regression."""
    X, y = np.asarray(X_train, dtype=np.float32), np.asarray(y_train, dtype=int)
    if X.ndim != 2 or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Training data must be aligned 2D features with both binary classes")
    if est == "dom":
        direction = X[y == 1].mean(0) - X[y == 0].mean(0)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("DOM direction is zero or non-finite")
        return Probe(est, None, direction / norm, None, -1, -1)
    if est != "logistic":
        raise ValueError(f"Unknown estimator {est!r}")
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=float(C if C is not None else 1.0), penalty="l2",
                               solver="lbfgs", max_iter=5000).fit(scaler.transform(X), y)
    return Probe(est, scaler, None, model, -1, -1)


def verify_dom_parity(X, y, expected, cosine_min=0.9999, relative_norm_max=1e-3):
    """Check raw, unnormalised DOM against a registered difference-of-means vector."""
    X, y, expected = np.asarray(X, float), np.asarray(y, int), np.asarray(expected, float)
    observed = X[y == 1].mean(0) - X[y == 0].mean(0)
    denom = np.linalg.norm(observed) * np.linalg.norm(expected)
    cosine = float(observed @ expected / denom) if denom else float("nan")
    norm = float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-15))
    passed = bool(np.isfinite(cosine) and cosine >= cosine_min and norm <= relative_norm_max)
    return {"cosine": cosine, "relative_norm_error": norm, "passed": passed}


def score(probe: Probe, X_test) -> np.ndarray:
    X = np.asarray(X_test, dtype=np.float32)
    if probe.estimator == "dom":
        return X @ probe.direction
    return probe.model.predict_proba(probe.scaler.transform(X))[:, 1]


def _auc(y, values) -> float:
    y, values = np.asarray(y, dtype=int), np.asarray(values, dtype=float)
    return float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else 0.5


def _groups_bootstrap(y, groups, values: dict[str, np.ndarray], n=1000, seed=20260407):
    y, groups = np.asarray(y, int), np.asarray(groups).astype(str)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng, draws = np.random.default_rng(seed), {k: [] for k in values}
    for _ in range(int(n)):
        idx = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        for key, pred in values.items():
            draws[key].append(_auc(y[idx], np.asarray(pred)[idx]))
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))}
            for k, v in draws.items() if v}


def _bootstrap_auc_difference(y, groups, left, right, n=1000, seed=20260407):
    y, groups = np.asarray(y, int), np.asarray(groups).astype(str)
    left, right = np.asarray(left, float), np.asarray(right, float)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng, values = np.random.default_rng(seed), []
    for _ in range(int(n)):
        idx = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) == 2:
            values.append(_auc(y[idx], left[idx]) - _auc(y[idx], right[idx]))
    if not values:
        raise ValueError("Bootstrap produced no samples containing both classes")
    return {"low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}


def _text_prediction(train_src: Source, test_src: Source) -> np.ndarray | None:
    if train_src.texts is None or test_src.texts is None:
        return None
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, dtype=np.float32)
    xtr = vec.fit_transform(train_src.texts)
    xte = vec.transform(test_src.texts)
    model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000)
    model.fit(xtr, train_src.y)
    return model.predict_proba(xte)[:, 1]


def _diagonal_text_oof(src: Source, rows=None, folds=5, seed=20260414):
    if src.texts is None:
        return None
    ix = np.arange(len(src.y)) if rows is None else np.asarray(rows, dtype=int)
    y, groups = src.y[ix], src.groups[ix]
    texts = [src.texts[i] for i in ix]
    pred = np.full(len(ix), np.nan)
    for _, _, tr, te in outer_splits(y, groups, folds, 1, seed):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, dtype=np.float32)
        xtr = vec.fit_transform([texts[i] for i in tr])
        xte = vec.transform([texts[i] for i in te])
        model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000).fit(xtr, y[tr])
        pred[te] = model.predict_proba(xte)[:, 1]
    return pred


def _select_C(est: str, src: Source, pos: int, layer: int,
              grid=(0.001, 0.01, 0.1, 1.0), seed=20260414):
    if est != "logistic":
        return 1.0
    from src.exploratory.e1_cv import fit_probe, score_probe
    x, y, groups = src.X[:, _position_index(pos), layer], src.y, src.groups
    scores = []
    for c in grid:
        predictions = np.full(len(y), np.nan)
        for _, _, tr, te in outer_splits(y, groups, 5, 1, seed):
            model = fit_probe("logistic", x[tr], y[tr], C=c)
            predictions[te] = score_probe(model, x[te])
        scores.append((_auc(y, predictions), -float(c), float(c)))
    return max(scores)[2]


def text_baseline(train_src: Source, test_src: Source) -> np.ndarray:
    """TF-IDF unigram/bigram logistic score; prompts stay source-local."""
    result = _text_prediction(train_src, test_src)
    if result is None:
        raise ValueError("Text baseline requires user_prompt text in both sources")
    return result


def _stack_oof_fixed(y, groups, base, extra, folds=5, repeats=5, seed=20260414):
    """Compare base-only and base+fixed external probe by grouped OOF stacking."""
    y, groups = np.asarray(y, int), np.asarray(groups).astype(str)
    base, extra = np.asarray(base, float), np.asarray(extra, float)
    full, baseline = np.full((repeats, len(y)), np.nan), np.full((repeats, len(y)), np.nan)
    n = min(folds, len(np.unique(groups)))
    if n < 2:
        raise ValueError("At least two groups are needed for cross-fitted stacking")
    for r in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed + r)
        for tr, te in splitter.split(np.zeros(len(y)), y, groups):
            if set(groups[tr]) & set(groups[te]):
                raise AssertionError("Stacker groups overlap")
            scaler = StandardScaler().fit(base[tr])
            m0 = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(base[tr]), y[tr])
            baseline[r, te] = m0.predict_proba(scaler.transform(base[te]))[:, 1]
            xx = np.column_stack((base[tr], extra[tr]))
            xt = np.column_stack((base[te], extra[te]))
            scaler1 = StandardScaler().fit(xx)
            m1 = LogisticRegression(C=1.0, max_iter=3000).fit(scaler1.transform(xx), y[tr])
            full[r, te] = m1.predict_proba(scaler1.transform(xt))[:, 1]
    if np.isnan(full).any() or np.isnan(baseline).any():
        raise RuntimeError("Stacking OOF predictions are incomplete")
    return baseline.mean(0), full.mean(0)


def _nested_diagonal_stacking(src, pos, est, layer_mode, layer, base, folds=5, repeats=5, seed=20260414):
    """Leakage-safe nested stacking for a diagonal transfer cell."""
    y, groups = src.y, src.groups
    pos_i = _position_index(pos)
    X = src.X[:, pos_i]
    full = np.full((repeats, len(y)), np.nan)
    baseline = np.full_like(full, np.nan)
    for repeat, fold, outer_train, outer_test in outer_splits(y, groups, folds, repeats, seed):
        train_src = Source(src.name, src.X[outer_train], src.meta.iloc[outer_train].reset_index(drop=True),
                           src.family, [src.texts[i] for i in outer_train] if src.texts else None)
        chosen = int(layer) if layer_mode == "fixed" else select_layer_on_source(est, train_src, pos_i,
            seed=seed + repeat * 100 + fold)
        c_value = _select_C(est, train_src, pos_i, chosen, seed=seed + repeat * 100 + fold)
        inner_score = np.full(len(outer_train), np.nan)
        for _, _, inner_train, inner_test in outer_splits(y[outer_train], groups[outer_train],
                min(folds, len(np.unique(groups[outer_train]))), 1, seed + 1000 + repeat * 100 + fold):
            inner_src = Source(src.name, src.X[outer_train[inner_train]],
                src.meta.iloc[outer_train[inner_train]].reset_index(drop=True), src.family,
                [src.texts[i] for i in outer_train[inner_train]] if src.texts else None)
            inner_layer = int(layer) if layer_mode == "fixed" else select_layer_on_source(
                est, inner_src, pos_i, seed=seed + 2000 + repeat * 100 + fold)
            inner_c = _select_C(est, inner_src, pos_i, inner_layer,
                                seed=seed + 3000 + repeat * 100 + fold)
            probe = fit_direction(est, X[outer_train[inner_train], inner_layer],
                                  y[outer_train[inner_train]], C=inner_c)
            inner_score[inner_test] = score(probe, X[outer_train[inner_test], inner_layer])
        if np.isnan(inner_score).any():
            raise RuntimeError("Nested diagonal probe OOF scores are incomplete")
        fitted = fit_direction(est, X[outer_train, chosen], y[outer_train], C=c_value)
        external_score = score(fitted, X[outer_test, chosen])
        stack_train = np.column_stack((base[outer_train], inner_score))
        stack_test = np.column_stack((base[outer_test], external_score))
        scaler = StandardScaler().fit(stack_train)
        model = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(stack_train), y[outer_train])
        full[repeat, outer_test] = model.predict_proba(scaler.transform(stack_test))[:, 1]
        base_scaler = StandardScaler().fit(base[outer_train])
        base_model = LogisticRegression(C=1.0, max_iter=3000).fit(
            base_scaler.transform(base[outer_train]), y[outer_train])
        baseline[repeat, outer_test] = base_model.predict_proba(
            base_scaler.transform(base[outer_test]))[:, 1]
    if np.isnan(full).any() or np.isnan(baseline).any():
        raise RuntimeError("Nested diagonal stacking is incomplete")
    return baseline.mean(0), full.mean(0)


def _diagonal_predictions(src, est, pos, layer, layer_mode):
    """Grouped diagonal probe OOF scores. Layer selection is confined to train folds."""
    X, y, groups = src.X, src.y, src.groups
    pos_i = _position_index(pos)
    if src.name == "syn":
        tr = np.flatnonzero(src.meta["split"].astype(str).to_numpy() == "train")
        te = np.flatnonzero(src.meta["split"].astype(str).to_numpy() == "test")
        if not len(tr) or not len(te):
            raise ValueError("Synthetic diagonal needs registered train and test splits")
        train_src = Source(src.name, src.X[tr], src.meta.iloc[tr].reset_index(drop=True), src.family,
                           [src.texts[i] for i in tr] if src.texts is not None else None)
        selected = int(layer)
        if layer_mode == "selected":
            selected = select_layer_on_source(est, train_src, pos_i)
        c_value = _select_C(est, train_src, pos_i, selected)
        probe = fit_direction(est, X[tr, pos_i, selected], y[tr], C=c_value)
        pred = np.full(len(y), np.nan)
        pred[te] = score(probe, X[te, pos_i, selected])
        valid = te
        return y[valid], groups[valid], pred[valid], selected, valid
    pred = np.full(len(y), np.nan)
    selected_layers = []
    for _, _, tr, te in outer_splits(y, groups, 5, 1, 20260414):
        train_src = Source(src.name, src.X[tr], src.meta.iloc[tr].reset_index(drop=True), src.family,
                           [src.texts[i] for i in tr] if src.texts is not None else None)
        selected = int(layer) if layer_mode == "fixed" else select_layer_on_source(est, train_src, pos_i)
        c_value = _select_C(est, train_src, pos_i, selected)
        probe = fit_direction(est, X[tr, pos_i, selected], y[tr], C=c_value)
        pred[te] = score(probe, X[te, pos_i, selected])
        selected_layers.append(selected)
    return y, groups, pred, selected_layers, np.arange(len(y))


def select_layer_on_source(est: str, src: Source, pos: int, layers=range(28), inner_folds=5,
                           seed=20260414, rows: Iterable[int] | None = None) -> int:
    """Pick the layer using only the supplied training rows and grouped inner CV."""
    from src.exploratory.e1_cv import fit_probe, score_probe
    ix = np.arange(len(src.y)) if rows is None else np.asarray(list(rows), dtype=int)
    y, groups = src.y[ix], src.groups[ix]
    X = src.X[ix, _position_index(pos)]
    candidates = []
    for layer in layers:
        oof = np.full(len(ix), np.nan)
        for _, _, tr, te in outer_splits(y, groups, inner_folds, 1, seed):
            model = fit_probe(est, X[tr, int(layer)], y[tr], C=1.0)
            oof[te] = score_probe(model, X[te, int(layer)])
        if np.isfinite(oof).all():
            candidates.append((_auc(y, oof), -int(layer), int(layer)))
    if not candidates:
        raise ValueError(f"Cannot select layer for {src.name}: insufficient grouped CV rows")
    return max(candidates)[2]


def cell(train_src: Source, test_src: Source, est="dom", pos="suf-4", layer_mode="fixed",
         layer=11, bootstrap=1000, seed=20260407) -> dict:
    """Compute one transfer cell and cluster bootstrap intervals."""
    if train_src is test_src:
        y, groups, probe_score, selected_layer, valid = _diagonal_predictions(
            train_src, est, pos, layer, layer_mode)
        tested = test_src.meta.iloc[valid].reset_index(drop=True)
        ys = np.asarray(y, int)
        baseline = (tested["log_odds"].to_numpy(float) if _position_index(pos) != 4
                    else -tested["maxp"].to_numpy(float))
        maxp = tested["maxp"].to_numpy(float)
        entropy = tested["entropy"].to_numpy(float)
        # Diagonal stacker uses E1 nested logic; the selected feature layer must be
        # nested as well, so the stacker is trained on layerwise features.
        pos_i = _position_index(pos)
        X = test_src.X[valid, pos_i]
        base_matrix = np.column_stack((tested["log_odds"].to_numpy(float), maxp, entropy))
        if train_src.name == "syn":
            pred_base, pred_full = _stack_oof_fixed(ys, groups, base_matrix, probe_score,
                                                     seed=20260414)
        else:
            pred_base, pred_full = _nested_diagonal_stacking(test_src, pos, est, layer_mode,
                layer, base_matrix, folds=5, repeats=5, seed=20260414)
        if train_src.name == "syn":
            text = _text_prediction(train_src.part("train"), test_src.part("test"))
        else:
            text = _diagonal_text_oof(test_src, valid)
    else:
        train_part = train_src.part("train") if train_src.name == "syn" else train_src
        test_part = test_src.part("test") if test_src.name == "syn" else test_src
        if test_src.name == "popqa":
            test_part = test_src
        pos_i = _position_index(pos)
        selected_layer = int(layer)
        if layer_mode == "selected":
            selected_layer = select_layer_on_source(est, train_part, pos_i)
        c_value = _select_C(est, train_part, pos_i, selected_layer)
        probe = fit_direction(est, train_part.X[:, pos_i, selected_layer], train_part.y, C=c_value)
        probe_score = score(probe, test_part.X[:, pos_i, selected_layer])
        ys, groups = test_part.y, test_part.groups
        baseline = (test_part.meta["log_odds"].to_numpy(float) if pos_i != 4
                    else -test_part.meta["maxp"].to_numpy(float))
        maxp = test_part.meta["maxp"].to_numpy(float)
        entropy = test_part.meta["entropy"].to_numpy(float)
        pred_base, pred_full = _stack_oof_fixed(ys, groups, np.column_stack((baseline, maxp, entropy)),
                                                  probe_score, seed=20260414)
        text = _text_prediction(train_part, test_part)
        tested = test_part.meta
    if text is None:
        auroc_text = None
    else:
        auroc_text = _auc(ys, text)
    auroc_probe, auroc_logit = _auc(ys, probe_score), _auc(ys, baseline)
    incremental_delta = _auc(ys, pred_full) - _auc(ys, pred_base)
    metrics = {"auroc_probe": probe_score, "auroc_logit": baseline}
    if text is not None:
        metrics["auroc_text"] = text
    ci_raw = _groups_bootstrap(ys, groups, metrics, bootstrap, seed)
    cis = {
        "auroc_probe": ci_raw["auroc_probe"], "auroc_logit": ci_raw["auroc_logit"],
        "delta": _bootstrap_auc_difference(ys, groups, probe_score, baseline, bootstrap, seed),
        "incremental_delta": _bootstrap_auc_difference(ys, groups, pred_full, pred_base, bootstrap, seed),
        "auroc_text": ci_raw.get("auroc_text"),
    }
    result = {"train_source": train_src.name, "test_source": test_src.name, "estimator": est,
              "position": pos, "layer": int(layer), "selected_layer": selected_layer,
              "n_train": len(train_part.y) if train_src.name != test_src.name else len(train_src.y),
              "n_test": len(ys), "auroc_probe": auroc_probe, "auroc_logit": auroc_logit,
              "delta": auroc_probe - auroc_logit, "auroc_text": auroc_text,
              "incremental_delta": incremental_delta, "ci95": cis}
    result["labels"] = label_cell(result)
    return result


def label_cell(result: dict, sesoi=0.02) -> list[str]:
    """Apply the E4 preregistered ordered cell labels."""
    ci = result["ci95"]
    p, inc, text, logit = ci["auroc_probe"], ci["incremental_delta"], ci.get("auroc_text"), ci["auroc_logit"]
    labels = []
    if text is not None and text["low"] >= p["low"] - sesoi:
        labels.append("surface_explainable")
    if inc["low"] > 0 and result["incremental_delta"] >= sesoi:
        labels.append("transfers_beyond_logits")
    elif p["low"] > 0.60 and inc["high"] < sesoi:
        labels.append("transfers_but_explained_by_logits")
    elif p["low"] >= 0.45 and p["high"] <= 0.55:
        labels.append("no_transfer")
    elif p["high"] < 0.45:
        labels.append("reversed_transfer")
    elif result["auroc_logit"] >= 0.95:
        labels.append("ceiling_limited")
    else:
        labels.append("inconclusive")
    return labels


def transfer_matrix(sources, est="dom", pos="suf-4", layer_mode="fixed", condition="registered",
                    layer=11, include_rhat=True, bootstrap=1000):
    """Create all available train × test cells for a condition."""
    names = [s.name for s in sources]
    matrix = {}
    for a in sources:
        matrix[a.name] = {}
        for b in sources:
            matrix[a.name][b.name] = cell(a, b, est=est, pos=pos, layer_mode=layer_mode,
                                          layer=layer, bootstrap=bootstrap)
    return {"condition": condition, "estimator": est, "position": pos, "layer_mode": layer_mode,
            "sources": names, "matrix": matrix, "rhat_included": bool(include_rhat)}

