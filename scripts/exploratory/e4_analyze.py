#!/usr/bin/env python3
"""Build E4 transfer matrices and figures from owner-produced feature caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.config import CFG
from src.eval_common import sha256_file
from src.exploratory import assert_not_fresh, exploratory_header, require_tier1_complete
from src.exploratory.e4_transfer import FEATURE_ROOT, Source, cell, load_source, transfer_matrix
from src.exploratory.e4_transfer import fit_direction, score, select_layer_on_source

OUT = ROOT / "results/exploratory"
FIG = ROOT / "figures/exploratory"
PREREG = OUT / "e4_preregistration.json"


def _verify_inputs(condition, sources):
    manifest = {}
    prereg_hash = sha256_file(PREREG)
    for source in sources:
        folder = FEATURE_ROOT / condition / source
        path = assert_not_fresh(folder / "manifest.json")
        if not path.exists():
            raise FileNotFoundError(path)
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("preregistration_sha256") != prereg_hash:
            raise RuntimeError(f"Feature cache was not captured against the current preregistration: {source}")
        for filename, key in (("suf.npy", "suf_sha256"), ("meta.parquet", "meta_sha256")):
            if filename == "suf.npy" and info.get("suf_path"):
                cached = Path(info["suf_path"])
                file = cached if cached.is_absolute() else ROOT / cached
            else:
                file = folder / filename
            file = assert_not_fresh(file)
            expected = info.get(key) or info.get("sha256", {}).get(filename)
            if not file.exists() or not expected or sha256_file(file) != expected:
                raise RuntimeError(f"Feature manifest hash mismatch: {file}")
        data_manifest_path = ROOT / "data/exploratory/e4/MANIFEST.json"
        data_entry = None
        if source != "popqa" and data_manifest_path.exists():
            dm = json.loads(data_manifest_path.read_text(encoding="utf-8"))
            data_entry = dm.get("sources", {}).get(source)
            data_file = assert_not_fresh(ROOT / "data/exploratory/e4" / f"{source}.jsonl")
            if data_entry and sha256_file(data_file) != data_entry.get("sha256"):
                raise RuntimeError(f"Prepared data hash mismatch: {data_file}")
        manifest[source] = {"manifest_sha256": sha256_file(path), "feature_manifest": info,
                            "data_manifest": data_entry}
    return manifest


def _popqa_views(condition):
    base = load_source("popqa", condition)
    return {"popqa_P1": base.part("P1"), "popqa_P2": base.part("P2")}


def _load_available(condition):
    available, pending = {}, {}
    for name in ("syn", "squad", "selfaware"):
        try:
            available[name] = load_source(name, condition)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            pending[name] = str(exc)
    try:
        available.update(_popqa_views(condition))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        pending["popqa"] = str(exc)
    try:
        available["triviaqa"] = load_source("triviaqa", condition)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        pending["triviaqa"] = str(exc)
    return available, pending


def _panel_spec(name):
    return {"M1": ("dom", "suf-4", "fixed", 11),
            "M2": ("dom", "suf-1", "selected", 11),
            "M3": ("logistic", "suf-4", "fixed", 11),
            "M4": ("logistic", "suf-1", "selected", 11)}[name]


def _test_view(source):
    if source.name == "syn":
        return source.part("test")
    return source


def _matrix_for_sources(sources, panel, condition, n_bootstrap, n_jobs=1):
    from joblib import Parallel, delayed
    est, pos, layer_mode, layer = _panel_spec(panel)
    keys = [k for k in sources if k != "popqa_P2"]
    test_sources = dict(sources)
    columns = list(test_sources)
    tasks = [(train_key, test_key, sources[train_key], test_sources[test_key])
             for train_key in keys for test_key in columns]
    computed = Parallel(n_jobs=n_jobs)(delayed(cell)(train, test, est=est, pos=pos,
        layer_mode=layer_mode, layer=layer, bootstrap=n_bootstrap)
        for _train_key, _test_key, train, test in tasks)
    matrix = {key: {} for key in keys}
    for (train_key, test_key, _train, _test), result in zip(tasks, computed):
        matrix[train_key][test_key] = result
    # A single logit baseline row is target-source-specific; carry it for plotting and audit.
    return {"estimator": est, "position": pos, "layer_mode": layer_mode,
            "rows": keys, "columns": columns, "cells": matrix}


def _rhat_row(sources, condition, pos):
    directions = torch.load(CFG.stage4_out_path, map_location="cpu", weights_only=False)
    pos_i = {"suf-4": 1, "suf-1": 4}[pos]
    direction = directions["unit_directions"][11, pos_i].detach().cpu().numpy().astype(np.float32)
    cells = {}
    for key, source in sources.items():
        test = _test_view(source)
        score = np.asarray(test.X[:, pos_i, 11], np.float32) @ direction
        y, groups = test.y, test.groups
        logit = (test.meta["log_odds"].to_numpy(float) if pos_i != 4
                 else -test.meta["maxp"].to_numpy(float))
        from sklearn.metrics import roc_auc_score
        auc = lambda v: float(roc_auc_score(y, v)) if len(np.unique(y)) == 2 else .5
        from src.exploratory.e4_transfer import _stack_oof_fixed, _bootstrap_auc_difference, _groups_bootstrap
        b0, b1 = _stack_oof_fixed(y, groups, np.column_stack((test.meta["log_odds"],
                 test.meta["maxp"], test.meta["entropy"])), score)
        inc = auc(b1) - auc(b0)
        cis = {"auroc_probe": _groups_bootstrap(y, groups, {"p": score})["p"],
               "auroc_logit": _groups_bootstrap(y, groups, {"l": logit})["l"],
               "delta": _bootstrap_auc_difference(y, groups, score, logit),
               "incremental_delta": _bootstrap_auc_difference(y, groups, b1, b0), "auroc_text": None}
        result = {"train_source": "r_hat", "test_source": key, "auroc_probe": auc(score),
                  "auroc_logit": auc(logit), "delta": auc(score) - auc(logit),
                  "incremental_delta": inc, "auroc_text": None, "ci95": cis}
        from src.exploratory.e4_transfer import label_cell
        result["labels"] = label_cell(result)
        cells[key] = result
    return cells


def _draw_heatmap(matrix, metric, path, title, annotate=False, ceiling=False, markers=False):
    rows, cols = matrix["rows"], matrix["columns"]
    values = np.full((len(rows), len(cols)), np.nan)
    annotations = np.full(values.shape, "", dtype=object)
    ceiling_cells = []
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            result = matrix["cells"].get(row, {}).get(col)
            if not result or result.get("status") == "pending":
                continue
            value = result.get(metric)
            if value is None:
                continue
            values[i, j] = value
            if metric == "delta":
                low, high = result["ci95"]["delta"]["low"], result["ci95"]["delta"]["high"]
                mark = "▲" if low > .02 else "▼" if high < -.02 else "·"
                annotations[i, j] = f"{value:+.2f}\n{result['auroc_probe']:.2f} {mark}"
                ceiling_cells.append((i, j, result["auroc_logit"] >= .95))
            elif metric == "auroc_probe":
                annotations[i, j] = f"{value:.2f}"
            else:
                annotations[i, j] = f"{value:.2f}"
    width = max(7, .85 * len(cols) + 3)
    height = max(5, .72 * len(rows) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(values, mask=~np.isfinite(values), cmap="vlag" if metric in {"delta", "incremental_delta"} else "viridis",
                center=0 if metric in {"delta", "incremental_delta"} else None,
                vmin=-.5 if metric == "delta" else None, vmax=.5 if metric == "delta" else None,
                annot=annotations if annotate else False, fmt="", linewidths=.5,
                xticklabels=cols, yticklabels=rows, cbar_kws={"label": metric}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Test source")
    ax.set_ylabel("Training source")
    if ceiling:
        for i, j, is_ceiling in ceiling_cells:
            if is_ceiling:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch="////", edgecolor="black", linewidth=0))
        fig.text(.5, .01, "Hatched columns/cells: test logit AUROC ≥ 0.95 (ceiling-limited).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04 if ceiling else 0, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(m1):
    def labels(a, b):
        cell_result = m1["cells"].get(a, {}).get(b)
        return set(cell_result.get("labels", [])) if cell_result else set()
    planned = {"syn_to_squad": labels("syn", "squad"), "squad_to_syn": labels("squad", "syn"),
               "syn_to_popqa": labels("syn", "popqa_P1"), "squad_to_popqa": labels("squad", "popqa_P1"),
               "popqa_to_syn": labels("popqa_P1", "syn"), "popqa_to_squad": labels("popqa_P1", "squad"),
               "syn_to_selfaware": labels("syn", "selfaware"), "squad_to_selfaware": labels("squad", "selfaware"),
               "selfaware_to_popqa": labels("selfaware", "popqa_P1"),
               "popqa_diagonal": labels("popqa_P1", "popqa_P1")}
    syn_squad = m1["cells"].get("syn", {}).get("squad", {})
    valid = bool(set(syn_squad.get("labels", [])) & {"transfers_beyond_logits", "transfers_but_explained_by_logits"}
                 and syn_squad.get("auroc_probe", 0) >= .8)
    if not valid:
        return {"synthetic_validity": "failed_or_pending",
                "headline": "Every other conclusion is conditional on synthetic-to-SQuAD validity; report results as template-sensitive until this contrast passes.",
                "planned_contrasts": planned}
    if ("transfers_but_explained_by_logits" in planned["syn_to_popqa"]
            and "transfers_but_explained_by_logits" in planned["squad_to_popqa"]
            and "transfers_beyond_logits" not in planned["popqa_to_syn"] | planned["popqa_to_squad"]):
        headline = "Shared component is output-level abstention or confidence, not knowledge state."
    elif ("transfers_beyond_logits" in planned["syn_to_popqa"]
            and "transfers_beyond_logits" in planned["squad_to_popqa"]
            and planned["popqa_to_syn"] & {"transfers_beyond_logits", "transfers_but_explained_by_logits"}):
        headline = "Shared knowledge-state component across contextual and parametric unanswerability."
    elif ("no_transfer" in planned["syn_to_popqa"] and "no_transfer" in planned["squad_to_popqa"]
            and "transfers_beyond_logits" in planned["popqa_diagonal"]):
        headline = "Separate representations."
    else:
        headline = "mixed; see the planned-contrast labels"
    selfaware = planned["syn_to_selfaware"] | planned["squad_to_selfaware"]
    if (selfaware & {"transfers_beyond_logits", "transfers_but_explained_by_logits"}
            and "surface_explainable" not in selfaware):
        selfaware_text = "Lavi et al.'s SelfAware generalisation replicates on Llama-3.2-3B, subject to the text baseline."
    else:
        selfaware_text = "SelfAware transfer was not established; inspect the corresponding text-baseline cells."
    return {"synthetic_validity": "passed", "headline": headline, "selfaware": selfaware_text,
            "planned_contrasts": planned}


def _geometry(sources, n_null=499, seed=20260410):
    """Raw/whitened direction cosine matrices and covariance-null bands."""
    from sklearn.covariance import LedoitWolf
    from src.exploratory.e2_analysis import cosine_spaces
    from src.null_directions import sample_covariance_null
    selected = {k: v for k, v in sources.items() if k not in {"popqa_P2"}}
    directions = {"fixed_11_suf-4": {}, "selected_suf-4": {}}
    pools = {"fixed_11_suf-4": [], "selected_suf-4": []}
    for key, source in selected.items():
        train = source.part("train") if source.name == "syn" else source
        x_fixed = train.X[:, 1, 11]
        directions["fixed_11_suf-4"][key] = fit_direction("dom", x_fixed, train.y).direction
        pools["fixed_11_suf-4"].append(x_fixed)
        layer = select_layer_on_source("dom", train, 1)
        x_selected = train.X[:, 1, layer]
        directions["selected_suf-4"][key] = fit_direction("dom", x_selected, train.y).direction
        pools["selected_suf-4"].append(x_selected)
    outputs = {}
    for family in directions:
        pooled = np.concatenate(pools[family], axis=0).astype(np.float64)
        if len(pooled) > 5000:
            rng = np.random.default_rng(seed)
            pooled = pooled[rng.choice(len(pooled), 5000, replace=False)]
        cov = LedoitWolf().fit(pooled).covariance_
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
        floor = 1e-6 * np.trace(cov) / cov.shape[0]
        invroot = (vecs * (1 / np.sqrt(np.maximum(vals, floor)))) @ vecs.T
        keys = list(directions[family])
        vectors = directions[family]
        raw = {a: {b: float(vectors[a] @ vectors[b] /
                        max(np.linalg.norm(vectors[a]) * np.linalg.norm(vectors[b]), 1e-15)) for b in keys}
               for a in keys}
        white_vec = {k: invroot @ vectors[k] for k in keys}
        whitened = {a: {b: float(white_vec[a] @ white_vec[b] /
                             max(np.linalg.norm(white_vec[a]) * np.linalg.norm(white_vec[b]), 1e-15)) for b in keys}
                     for a in keys}
        null = sample_covariance_null(cov, n_null, seed)["unit"].astype(np.float64)
        null_w = null @ invroot.T
        norms = np.linalg.norm(null_w, axis=1)
        null_w = null_w / np.maximum(norms[:, None], 1e-15)
        pair = null_w @ null_w.T
        null_values = pair[np.triu_indices(len(null_w), k=1)]
        # Exercise the shared E2 geometry utility on a representative pair; the
        # full matrix above reuses this exact pooled covariance without refitting it.
        first_two = list(vectors)[:2]
        utility_check = cosine_spaces(vectors[first_two[0]], vectors[first_two[1]], pooled, []) if len(first_two) > 1 else None
        outputs[family] = {"raw": raw, "whitened": whitened, "selected_layers": {
            k: int(select_layer_on_source("dom", (v.part("train") if v.name == "syn" else v), 1))
            for k, v in selected.items()},
            "pooled_rows": len(pooled), "covariance_shrinkage": float(LedoitWolf().fit(pooled).shrinkage_),
            "null_n": n_null, "null_95_abs": float(np.percentile(np.abs(null_values), 95)),
            "null_99_abs": float(np.percentile(np.abs(null_values), 99)),
            "null_cosine_95": [float(x) for x in np.percentile(null_values, [2.5, 97.5])],
            "null_cosine_99": [float(x) for x in np.percentile(null_values, [.5, 99.5])],
            "e2_utility_check": utility_check}
    return outputs


def _layer_curves(sources):
    """Descriptive DOM layer AUROCs for the preregistered directional contrasts."""
    contrasts = (("syn", "squad"), ("squad", "syn"), ("syn", "popqa_P1"),
                 ("squad", "popqa_P1"), ("popqa_P1", "syn"), ("popqa_P1", "squad"),
                 ("syn", "selfaware"), ("squad", "selfaware"), ("selfaware", "popqa_P1"))
    curves = {pos: {} for pos in (1, 4)}
    for pos in curves:
        for a, b in contrasts:
            if a not in sources or b not in sources:
                continue
            train = sources[a].part("train") if sources[a].name == "syn" else sources[a]
            test = _test_view(sources[b])
            curves[pos][f"{a}->{b}"] = []
            for layer in range(28):
                probe = fit_direction("dom", train.X[:, pos, layer], train.y)
                prediction = score(probe, test.X[:, pos, layer])
                from sklearn.metrics import roc_auc_score
                auc = float(roc_auc_score(test.y, prediction)) if len(np.unique(test.y)) == 2 else .5
                curves[pos][f"{a}->{b}"].append(auc)
    return curves


def _draw_geometry(geometry, path):
    families = list(geometry)
    fig, axes = plt.subplots(1, len(families), figsize=(max(12, 6 * len(families)), 6), squeeze=False)
    for ax, family in zip(axes[0], families):
        matrix = geometry[family]["whitened"]
        labels = list(matrix)
        values = np.asarray([[matrix[a][b] for b in labels] for a in labels])
        sns.heatmap(values, annot=True, fmt=".2f", cmap="vlag", center=0,
                    xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        g = geometry[family]
        ax.set_title(f"{family.replace('_', ', ')}")
        ax.set_xlabel(f"Null bands ±{g['null_95_abs']:.2f} / ±{g['null_99_abs']:.2f} (n=499)")
        ax.set_ylabel("Training source")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_layer_curves(curves, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pos in zip(axes, (1, 4)):
        for label, values in curves[pos].items():
            ax.plot(range(28), values, marker=".", label=label)
        ax.set_title("Suffix −4" if pos == 1 else "Suffix −1")
        ax.set_xlabel("Layer")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_matrix_panels(matrices, panel_names, metric, path, title):
    count = len(panel_names)
    fig, axes = plt.subplots(1, count, figsize=(7 * count, 6), squeeze=False)
    for ax, name in zip(axes[0], panel_names):
        matrix = matrices.get(name, {})
        rows, cols = matrix.get("rows", []), matrix.get("columns", [])
        values = np.full((len(rows), len(cols)), np.nan)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                result = matrix.get("cells", {}).get(row, {}).get(col, {})
                value = result.get(metric)
                if value is not None:
                    values[i, j] = value
        sns.heatmap(values, mask=~np.isfinite(values), cmap="vlag" if "delta" in metric else "viridis",
                    center=0 if "delta" in metric else None, annot=True, fmt=".2f",
                    xticklabels=cols, yticklabels=rows, ax=ax, cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Test source")
        ax.set_ylabel("Training source")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_auroc_companions(matrix, path):
    metrics = ("auroc_probe", "auroc_logit", "auroc_text", "incremental_delta")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    rows, cols = matrix.get("rows", []), matrix.get("columns", [])
    for ax, metric in zip(axes.flat, metrics):
        values = np.full((len(rows), len(cols)), np.nan)
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                result = matrix.get("cells", {}).get(row, {}).get(col, {})
                value = result.get(metric)
                if value is not None:
                    values[i, j] = value
        sns.heatmap(values, mask=~np.isfinite(values), cmap="vlag" if "delta" in metric else "viridis",
                    center=0 if "delta" in metric else None, annot=True, fmt=".2f",
                    xticklabels=cols, yticklabels=rows, ax=ax, cbar=False)
        ax.set_title(metric)
        ax.set_xlabel("Test source")
        ax.set_ylabel("Training source")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(condition="registered", panels=("M1", "M2", "M3", "M4"), n_jobs=1,
            bootstrap=1000, allow_pending=True):
    if condition not in {"registered", "neutral", "both"}:
        raise ValueError("condition must be registered, neutral, or both")
    conditions = ("registered", "neutral") if condition == "both" else (condition,)
    prereg_hash = sha256_file(PREREG)
    payload = {"header": exploratory_header({"protocol": "E4 transfer matrix"}, ROOT),
               "preregistration_sha256": prereg_hash, "conditions": {}}
    summaries = []
    FIG.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        sources, pending = _load_available(cond)
        manifest = {}
        for name in sources:
            if name == "popqa_P1":
                try:
                    manifest.update(_verify_inputs(cond, ["popqa"]))
                except FileNotFoundError:
                    pending["popqa"] = "PopQA E4 cache manifest missing"
                continue
            if name.endswith("_P2"):
                continue
            try:
                manifest.update(_verify_inputs(cond, [name]))
            except FileNotFoundError:
                pending[name] = f"Feature manifest missing for {name}"
        record = {"pending": pending, "data_manifest": manifest, "panels": {}}
        base_sources = dict(sources)
        if not base_sources:
            record["status"] = "pending_features"
            payload["conditions"][cond] = record
            continue
        for panel in panels:
            est, pos, layer_mode, layer = _panel_spec(panel)
            try:
                matrix = _matrix_for_sources(base_sources, panel, cond, bootstrap, n_jobs)
                matrix["r_hat"] = _rhat_row(sources, cond, pos)
                matrix["rows"].append("r_hat")
                matrix["cells"]["r_hat"] = matrix["r_hat"]
                record["panels"][panel] = matrix
                if panel == "M1":
                    _draw_heatmap(matrix, "delta", FIG / f"e4_transfer_headline{'_neutral' if cond == 'neutral' else ''}.png",
                                  f"E4 {cond}: probe AUROC gain over logits", annotate=True, ceiling=True)
                    for metric, suffix in (("auroc_probe", "probe"), ("auroc_logit", "logit"),
                                           ("auroc_text", "text"), ("incremental_delta", "incremental")):
                        _draw_heatmap(matrix, metric, FIG / f"e4_M1_{suffix}_{cond}.png", f"E4 M1 {metric}")
                else:
                    _draw_heatmap(matrix, "delta", FIG / f"e4_{panel}_{cond}.png", f"E4 {panel}: probe AUROC gain")
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                record["panels"][panel] = {"status": "pending", "reason": str(exc)}
        m1 = record["panels"].get("M1", {})
        if m1.get("cells") and "popqa" in pending:
            synthetic = m1["cells"].get("syn", {}).get("squad", {})
            valid = bool(set(synthetic.get("labels", [])) &
                         {"transfers_beyond_logits", "transfers_but_explained_by_logits"}
                         and synthetic.get("auroc_probe", 0) >= .8)
            summary = {"synthetic_validity": "passed" if valid else "failed_or_pending",
                       "headline": "PopQA-dependent summary rules are pending Tier 1.",
                       "planned_contrasts": {"syn_to_squad": synthetic.get("labels", []),
                                             "ctx_to_popqa": "pending_tier1",
                                             "popqa_to_ctx": "pending_tier1",
                                             "selfaware_to_popqa": "pending_tier1"}}
            record["summary"] = summary
            summaries.append((cond, summary))
        elif m1.get("cells"):
            summary = _summary(m1)
            record["summary"] = summary
            summaries.append((cond, summary))
        if all(name in sources for name in ("syn", "squad", "selfaware")):
            try:
                record["geometry"] = _geometry(sources, n_null=499)
                record["layer_curves"] = _layer_curves(sources)
                if cond == "registered":
                    _draw_geometry(record["geometry"], FIG / "e4_direction_cosines.png")
                    _draw_layer_curves(record["layer_curves"], FIG / "e4_layer_curves.png")
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                record["geometry_pending"] = str(exc)
        if cond == "registered":
            if "M1" in record["panels"] and record["panels"]["M1"].get("cells"):
                _draw_auroc_companions(record["panels"]["M1"], FIG / "e4_transfer_auroc_panels.png")
            _draw_matrix_panels(record["panels"], [p for p in ("M2", "M4") if p in record["panels"]],
                                "delta", FIG / "e4_transfer_M2_M4.png", "E4 secondary transfer panels")
        record["status"] = "partial" if pending else "complete"
        payload["conditions"][cond] = record
    e1_path, e2_path = OUT / "e1_results.json", OUT / "e2_results.json"
    e1_label, e2_verdict = "pending", "pending"
    if e1_path.exists():
        e1_data = json.loads(e1_path.read_text(encoding="utf-8"))
        e1_label = (e1_data.get("populations", {}).get("P1", {}).get("contrasts", {})
                    .get("C1_pregen", {}).get("interpretation", "pending"))
    if e2_path.exists():
        e2_data = json.loads(e2_path.read_text(encoding="utf-8"))
        e2_verdict = e2_data.get("combined_verdict", {}).get("verdict", "pending")
    payload["E1_C1_label"] = e1_label
    payload["E2_combined_verdict"] = e2_verdict
    out_path = assert_not_fresh(OUT / "e4_results.json")
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# E4 transfer matrix summary", "", f"Preregistration SHA-256: `{prereg_hash}`", ""]
    for cond, summary in summaries:
        lines += [f"## {cond}", "", summary["headline"], "", summary.get("selfaware", ""), "",
                  f"Synthetic validity: {summary['synthetic_validity']}", ""]
    if not summaries:
        lines += ["Feature extraction is pending. No E4 transfer metrics are reported.", ""]
    lines += ["Contextual test columns under the registered prompt were pre-declared as ceiling-limited when the logit AUROC is at least 0.95.",
              "SelfAware results must be read beside the text-only baseline. CREPE, RepliQA, NQ, MuSiQue and TriviaQA remain out of scope unless explicitly prepared and judged.",
              "", f"E1 C1: {e1_label}", f"E2 combined verdict: {e2_verdict}", ""]
    (OUT / "e4_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("registered", "neutral", "both"), default="registered")
    parser.add_argument("--panels", default="M1,M2,M3,M4")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    payload = analyze(args.condition, tuple(x.strip() for x in args.panels.split(",") if x.strip()),
                      args.n_jobs, args.bootstrap)
    print(json.dumps({k: v.get("status") for k, v in payload["conditions"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
