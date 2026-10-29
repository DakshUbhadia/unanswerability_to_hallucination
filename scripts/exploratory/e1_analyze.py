#!/usr/bin/env python3
"""CPU analysis for the pre-registered exploratory E1 supervised probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.exploratory import assert_not_fresh, exploratory_header
from src.exploratory.e1_cv import (conditional_bootstrap, fit_probe, nested_probe_oof, outer_splits,
                                   nested_stacking_delta, run_descriptive_grid)
from src.eval_common import sha256_file

POSITIONS = ["suf-5", "suf-4", "suf-3", "suf-2", "suf-1", "ans_first", "ans_last"]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _auroc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else 0.5


def _population(meta, which):
    rows = meta[meta["baseline_partition"].isin(["wrong", "correct"])].copy()
    if which == "P1":
        rows = rows[rows["is_popularity_matched"].astype(bool) & rows["matched_pair_id"].notna()]
        groups = rows["matched_pair_id"].astype(str).to_numpy()
    elif which == "P2":
        groups = rows["popqa_id"].astype(str).to_numpy()
    else:
        rows = rows[rows["prop_grouped"].notna()]
        groups = rows["prop_grouped"].astype(str).to_numpy()
    y = (rows["baseline_partition"].to_numpy() == "wrong").astype(int)
    if len(rows) == 0 or len(np.unique(y)) != 2:
        raise ValueError(f"{which} does not contain both labels")
    units = rows["matched_pair_id"].fillna(rows["popqa_id"]).astype(str).to_numpy() if which == "P1" else rows["popqa_id"].astype(str).to_numpy()
    base_cols = ["maxp", "entropy", "log_odds"]
    if which == "P2":
        base_cols += ["log_s_pop"]
        rel = pd.get_dummies(rows["prop_grouped"].astype(str), prefix="prop", dtype=float)
        base = np.column_stack([rows[base_cols].to_numpy(float), rel.to_numpy(float)])
    else:
        base = rows[base_cols].to_numpy(float)
    return rows, y, groups, units, base


def _base_conf(rows, y, groups, units, extra=None, n_boot=1000, seed=20260407, folds=5, population="P1"):
    out = {}
    for name, score in (("maxp", -rows["maxp"].to_numpy(float)),
                        ("entropy", rows["entropy"].to_numpy(float)),
                        ("log_odds", rows["log_odds"].to_numpy(float))):
        out[name] = {"auroc": _auroc(y, score)}
    if "log_s_pop" in rows:
        out["log_s_pop"] = {"auroc": _auroc(y, -rows["log_s_pop"].to_numpy(float))}
    if population == "P2":
        relation = pd.get_dummies(rows["prop_grouped"].astype(str), dtype=float).to_numpy()
        pred = np.full(len(y), np.nan)
        for _, _, tr, te in outer_splits(y, groups, min(folds, len(np.unique(groups))), 5, seed):
            model = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(relation[tr], y[tr])
            pred[te] = model.predict_proba(relation[te])[:, 1]
        out["relation_one_hot"] = {"auroc": _auroc(y, pred)}
    if extra is not None:
        out["probe"] = conditional_bootstrap(extra, -rows["maxp"].to_numpy(float), y, units, n_boot, seed)
    return out


def _position_tensor(suf, ans, pos):
    if pos.startswith("suf"):
        p = int(pos[3:])
        return np.asarray(suf[:, p, :, :], dtype=np.float32)
    p = 0 if pos == "ans_first" else 1
    return np.asarray(ans[:, p, :, :], dtype=np.float32)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "results/exploratory/e1_features")
    parser.add_argument("--population", choices=["P1", "P2", "P3", "all"], default="all")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    feature_dir = assert_not_fresh(args.features_dir)
    manifest = json.loads((feature_dir / "features_manifest.json").read_text(encoding="utf-8"))
    dry = bool(args.dry_run or manifest.get("dry_run"))
    if dry and "results/exploratory/dryrun" not in str(feature_dir):
        raise ValueError("Dry-run analysis inputs must be under results/exploratory/dryrun/")
    suf_path, ans_path = feature_dir / "suf.npy", feature_dir / "ans.npy"
    for path, key in ((suf_path, "suf.npy"), (ans_path, "ans.npy")):
        if not path.exists():
            repo = __import__("os").environ.get("HF_TENSOR_REPO_ID")
            if not repo:
                raise FileNotFoundError(f"Missing {path}; restore it from HF or set HF_TENSOR_REPO_ID")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=repo, filename=f"exploratory/e1_features/{key}", repo_type="dataset")
            Path(downloaded).replace(path)
        if sha256_file(path) != manifest["sha256"][key]:
            raise ValueError(f"Feature checksum mismatch: {path}")
    meta_path = feature_dir / "meta.parquet"
    meta = pd.read_parquet(meta_path)
    if tuple(meta.shape[:1]) != (manifest["shapes"]["suf"][0],):
        raise ValueError("Feature metadata row count does not match the manifest")
    if meta["popqa_id"].astype(str).tolist() != manifest["row_order"]:
        raise ValueError("meta.parquet row order differs from the feature manifest")
    if not dry:
        from src.popqa_uncertainty import _verify_popqa_input_identity
        from src.eval_common import load_runtime_contract, load_stage_model
        # Tokenizer-only validation avoids loading model weights.
        contract = load_runtime_contract()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(contract["model_name"])
        rows = meta.to_dict("records")
        _verify_popqa_input_identity(tokenizer, rows, contract["system_prompt_used"], contract["model_name"], contract["suffix_token_ids"])
        source = ROOT / "data/popqa_sample.jsonl"
        expected = [json.loads(line)["popqa_id"] for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if meta["popqa_id"].astype(str).tolist() != expected:
            raise ValueError("Feature row order differs from data/popqa_sample.jsonl")
    suf, ans = np.load(suf_path, mmap_mode="r"), np.load(ans_path, mmap_mode="r")
    if list(suf.shape) != manifest["shapes"]["suf"] or list(ans.shape) != manifest["shapes"]["ans"]:
        raise ValueError("Feature tensor shapes differ from the manifest")
    populations = [args.population] if args.population != "all" else ["P1", "P2", "P3"]
    output = {"preregistration_sha256": _sha(ROOT / "results/exploratory/e1_preregistration.json"),
              "dry_run": dry, "population_counts": {}, "populations": {}}
    for pop in populations:
        rows, y, groups, units, base = _population(meta, pop)
        indices = rows.index.to_numpy(dtype=int)
        output["population_counts"][pop] = {"n": len(rows), "n_wrong": int(y.sum()), "n_correct": int(len(y)-y.sum()), "n_groups": len(np.unique(groups))}
        if pop == "P1" and not dry and len(np.unique(groups)) < 100:
            raise RuntimeError(f"P1 has only {len(np.unique(groups))} matched pairs; minimum is 100")
        folds = min(5, len(np.unique(groups)))
        pop_result = {"baselines": _base_conf(rows, y, groups, units, folds=min(5, len(np.unique(groups))), population=pop),
                      "descriptive_grid": {}, "nested": {}, "contrasts": {}}
        positions_by_idx = {}
        for pos_i, pos in enumerate(POSITIONS):
            Xall = _position_tensor(suf, ans, pos)
            valid = np.isfinite(Xall[indices]).all(axis=(1, 2))
            pi = indices[valid]
            if len(pi) < 10 or len(np.unique(meta.loc[pi, "baseline_partition"])) < 2:
                pop_result["nested"][pos] = {"omitted": "insufficient finite answer-position rows"}
                continue
            subrows = meta.loc[pi]
            yy = (subrows["baseline_partition"].to_numpy() == "wrong").astype(int)
            gg = (subrows["matched_pair_id"].fillna(subrows["popqa_id"]).astype(str).to_numpy()
                  if pop == "P1" else groups[valid])
            X = Xall[pi]
            grid = run_descriptive_grid(X, yy, gg, folds, 5, 20260414, args.n_jobs)
            pop_result["descriptive_grid"][pos] = {str(l): {"auroc_mean": v["auroc_mean"], "auroc_sd": v["auroc_sd"]} for l, v in grid.items()}
            class_weight = "balanced" if pop == "P2" else None
            pred, selected = nested_probe_oof(X, yy, gg, folds, 5, 20260414,
                                               class_weight=class_weight)
            mean_pred = pred.mean(axis=0)
            estimators = {"logistic": {"auroc": _auroc(yy, mean_pred), "selected": selected,
                                         "selected_layer_counts": pd.Series([x["layer"] for x in selected]).value_counts().to_dict()}}
            for est in ("dom", "pca64_logistic"):
                epred, eselected = nested_probe_oof(X, yy, gg, folds, 5, 20260414,
                                                     estimator=est, class_weight=class_weight)
                estimators[est] = {"auroc": _auroc(yy, epred.mean(axis=0)), "selected": eselected,
                                   "selected_layer_counts": pd.Series([x["layer"] for x in eselected]).value_counts().to_dict()}
            pop_result["nested"][pos] = estimators
            positions_by_idx[pos] = (X, yy, gg, subrows, pi)
        for name, pos in (("C1_pregen", "suf-1"), ("C2_postans", "ans_first")):
            if pos not in positions_by_idx:
                continue
            X, yy, gg, subrows, pi = positions_by_idx[pos]
            bb = subrows[["maxp", "entropy", "log_odds"]].to_numpy(float)
            if name == "C2_postans":
                bb = np.column_stack((bb, subrows[["seq_logprob_mean", "seq_logprob_min"]].to_numpy(float)))
            if pop == "P2":
                bb = np.column_stack((bb, subrows[["log_s_pop"]].to_numpy(float), pd.get_dummies(subrows["prop_grouped"].astype(str), dtype=float).to_numpy()))
            class_weight = "balanced" if pop == "P2" else None
            stacked = nested_stacking_delta(X, bb, yy, gg, folds, 5, 20260414, class_weight=class_weight)
            boot = conditional_bootstrap(stacked["full_oof"], stacked["base_oof"], yy,
                                         subrows["matched_pair_id"].fillna(subrows["popqa_id"]).to_numpy(), 1000, 20260407)
            boot["delta_repeat_sd"] = float(np.std(stacked["delta_by_repeat"], ddof=1))
            probe_pred, _ = nested_probe_oof(X, yy, gg, folds, 5, 20260414,
                                              class_weight=class_weight)
            probe_vs_maxp = conditional_bootstrap(probe_pred, -subrows["maxp"].to_numpy(float), yy,
                subrows["matched_pair_id"].fillna(subrows["popqa_id"]).to_numpy(), 1000, 20260408)
            pop_result["contrasts"][name] = {**boot, "selected": stacked["selected"],
                                              "probe_alone_beats_maxp": probe_vs_maxp}
        if pop == "P1" and "suf-4" in positions_by_idx:
            X, yy, gg, subrows, _pi = positions_by_idx["suf-4"]
            direction_file = ROOT / "results/stage4_directions.pt"
            if direction_file.exists():
                import torch
                directions = torch.load(direction_file, map_location="cpu", weights_only=False)
                units = directions["unit_directions"].detach().cpu().numpy()
                frozen = X[:, 11, :] @ units[11, 1, :].astype(np.float32)
                dom_scores = np.full(len(yy), np.nan)
                for _, _, tr, te in outer_splits(yy, gg, folds, 5, 20260414):
                    model = fit_probe("dom", X[tr, 11], yy[tr])
                    from src.exploratory.e1_cv import score_probe
                    dom_scores[te] = score_probe(model, X[te, 11])
                nested = nested_probe_oof(X, yy, gg, folds, 5, 20260414)[0].mean(axis=0)
                pair_units = subrows["matched_pair_id"].astype(str).to_numpy()
                pop_result["transfer_gap"] = {
                    "frozen_rhat_auroc": _auroc(yy, frozen), "nested_probe_auroc": _auroc(yy, nested),
                    "dom_layer11_auroc": _auroc(yy, dom_scores),
                    "nested_minus_rhat": conditional_bootstrap(nested, frozen, yy, pair_units, 1000, 20260407),
                    "dom_minus_rhat": conditional_bootstrap(dom_scores, frozen, yy, pair_units, 1000, 20260407)}
            else:
                pop_result["transfer_gap"] = {"pending": "Stage 4 directions artifact is unavailable"}
        contrast_items = list(pop_result["contrasts"].items())
        ordered = sorted(range(len(contrast_items)), key=lambda i: contrast_items[i][1]["p_one_sided"])
        adjusted, running = [0.0] * len(contrast_items), 0.0
        for rank, idx_adj in enumerate(ordered):
            running = max(running, min(1.0, (len(ordered) - rank) * contrast_items[idx_adj][1]["p_one_sided"]))
            adjusted[idx_adj] = running
        for i, (_name, result) in enumerate(contrast_items):
            result["holm_adjusted_p"] = adjusted[i]
            lo, hi = result["ci_95"]
            delta = result["delta"]
            if result["holm_adjusted_p"] < 0.05 and delta >= 0.02:
                label = "knowledge_state_signal_beyond_output_confidence"
            elif hi < 0.02:
                label = "no_incremental_signal_equivalence"
            elif lo > 0 and delta < 0.02:
                label = "statistically_detectable_but_below_sesoi"
            else:
                label = "inconclusive_underpowered"
            result["interpretation"] = label
        if pop == "P1":
            sensitivity = {}
            filters = {
                "exclude_answer_then_unknown": ~rows["answer_then_unknown"].fillna(False).astype(bool).to_numpy(),
                "degenerate_excluded": rows["final_label"].fillna("").astype(str).str.upper().ne("DEGENERATE").to_numpy(),
                "exclude_greedy_mismatch_rows": rows["greedy_consistent"].fillna(False).astype(bool).to_numpy(),
            }
            for filter_name, keep in filters.items():
                sensitivity[filter_name] = {}
                for contrast, pos in (("C1_pregen", "suf-1"), ("C2_postans", "ans_first")):
                    if pos not in positions_by_idx:
                        continue
                    X, yy, gg, subrows, _pi = positions_by_idx[pos]
                    # rows/subrows retain the same source order for these matched-only populations.
                    submask = np.asarray([bool(filters[filter_name][np.flatnonzero(rows.index == idx)[0]]) for idx in subrows.index])
                    if submask.sum() < 20 or len(np.unique(yy[submask])) < 2:
                        sensitivity[filter_name][contrast] = {"omitted": "insufficient rows after filter"}
                        continue
                    sx, sy, sg, sm = X[submask], yy[submask], gg[submask], subrows.iloc[np.flatnonzero(submask)]
                    bcols = ["maxp", "entropy", "log_odds"]
                    if contrast == "C2_postans":
                        bcols += ["seq_logprob_mean", "seq_logprob_min"]
                    sb = sm[bcols].to_numpy(float)
                    ss = nested_stacking_delta(sx, sb, sy, sg, folds, 5, 20260414)
                    sensitivity[filter_name][contrast] = conditional_bootstrap(
                        ss["full_oof"], ss["base_oof"], sy, sm["matched_pair_id"].astype(str).to_numpy(), 1000, 20260407)
            pop_result["sensitivity"] = sensitivity
        output["populations"][pop] = pop_result
        print(f"[E1] finished {pop}: {len(rows)} graded rows", flush=True)
    target = ROOT / ("results/exploratory/dryrun" if dry else "results/exploratory")
    target.mkdir(parents=True, exist_ok=True)
    payload = exploratory_header({**output, "watermark": "DRY RUN – v4 labels – not reportable" if dry else None}, ROOT)
    for pop_result in payload["populations"].values():
        for pos_result in pop_result.get("nested", {}).values():
            for est_result in pos_result.values() if isinstance(pos_result, dict) else ():
                if isinstance(est_result, dict) and "auroc" in est_result and not (0 <= est_result["auroc"] <= 1):
                    raise AssertionError("Nested AUROC outside [0, 1]")
        for contrast in pop_result.get("contrasts", {}).values():
            if not (0 <= contrast["auroc_a"] <= 1 and 0 <= contrast["auroc_b"] <= 1):
                raise AssertionError("Contrast AUROC outside [0, 1]")
            if contrast["ci_95"][0] > contrast["ci_95"][1]:
                raise AssertionError("Contrast confidence interval is unordered")
    results_path = target / "e1_results.json"
    results_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n", encoding="utf-8")
    summary = ["# E1 supervised probe summary", "", "Exploratory post-registration analysis; all estimates are conditional on the preregistered folds.", ""]
    if dry:
        summary.insert(1, "**DRY RUN – v4 labels – not reportable**")
    for pop, result in payload["populations"].items():
        summary += [f"## {pop}", "", "| Contrast | ΔAUROC | 95% CI |", "|---|---:|---:|"]
        for key, value in result["contrasts"].items():
            summary.append(f"| {key} | {value['delta']:.3f} | [{value['ci_95'][0]:.3f}, {value['ci_95'][1]:.3f}] |")
        summary.append("")
    (target / "e1_summary.md").write_text("\n".join(summary), encoding="utf-8")
    figure_dir = (target / "figures" if dry else ROOT / "figures/exploratory")
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p1 = payload["populations"].get("P1")
    if p1:
        layers = len(p1["descriptive_grid"].get("suf-1", {}))
        heat = np.full((layers, 5), np.nan)
        p1_rows, p1_y, *_ = _population(meta, "P1")
        maxp_auc = _auroc(p1_y, -p1_rows["maxp"].to_numpy(float))
        for pi, pos in enumerate(POSITIONS[:5]):
            for li, val in p1["descriptive_grid"].get(pos, {}).items():
                if int(li) < layers:
                    heat[int(li), pi] = val["auroc_mean"] - maxp_auc
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-0.4, vmax=0.4)
        mark = "DRY RUN – v4 labels – not reportable: " if dry else ""
        ax.set(title=mark + "Descriptive, not selection-adjusted", xlabel="Suffix position", ylabel="Layer",
               xticks=range(5), xticklabels=["suf-5", "suf-4", "suf-3", "suf-2", "suf-1"])
        fig.colorbar(im, ax=ax, label="Probe AUROC − maxp AUROC")
        fig.tight_layout(); fig.savefig(figure_dir / "e1_grid_auroc_P1.png", dpi=160); plt.close(fig)
        labels, values = [], []
        for pos in POSITIONS:
            entry = p1["nested"].get(pos, {}).get("logistic", {})
            if "auroc" in entry:
                labels.append(pos); values.append(entry["auroc"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(labels, values, marker="o", label="Nested probe")
        ax.axhline(maxp_auc, linestyle="--", label="maxp")
        ax.axhline(_auroc(p1_y, p1_rows["entropy"].to_numpy(float)), linestyle=":", label="entropy")
        if "transfer_gap" in p1 and "frozen_rhat_auroc" in p1["transfer_gap"]:
            ax.axhline(p1["transfer_gap"]["frozen_rhat_auroc"], linestyle="-.", label="frozen rhat")
        ax.set_ylim(0.35, 1.0); ax.set_ylabel("AUROC"); ax.set_title(mark + "Nested probe AUROC by position"); ax.legend()
        fig.tight_layout(); fig.savefig(figure_dir / "e1_nested_by_position.png", dpi=160); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        names, deltas, lows, highs = [], [], [], []
        for name, value in p1["contrasts"].items():
            names.append(name); deltas.append(value["delta"]); lows.append(value["ci_95"][0]); highs.append(value["ci_95"][1])
        if names:
            ax.errorbar(names, deltas, yerr=[np.array(deltas)-np.array(lows), np.array(highs)-np.array(deltas)], fmt="o", capsize=4)
        ax.axhspan(-0.02, 0.02, alpha=0.15, color="grey", label="SESOI ±0.02")
        ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("ΔAUROC"); ax.set_title(mark + "Nested incremental AUROC"); ax.legend()
        fig.tight_layout(); fig.savefig(figure_dir / "e1_delta_auroc.png", dpi=160); plt.close(fig)
    print(f"Wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
