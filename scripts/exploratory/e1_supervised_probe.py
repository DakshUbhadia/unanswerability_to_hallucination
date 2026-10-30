#!/usr/bin/env python3
"""Nested group-CV supervised upper bound on exploratory PopQA residual features."""
import argparse
import json
import sys
if __name__ == "__main__":
    if "--legacy" not in sys.argv:
        print("superseded by scripts/exploratory/e1_analyze.py; pass --legacy to run", file=sys.stderr)
        raise SystemExit(2)
    sys.argv.remove("--legacy")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config import CFG
from src.exploratory import assert_not_fresh, write_exploratory_json
from src.exploratory.analysis import auc_summary, group_oof_scores, nested_group_probe_oof
from src.incremental_value import BASE_FEATURES, incremental_value_test

GRADES = {"wrong", "correct"}


def _prepare(features_dir, feature_name, directions_path, neutral_direction=False, position_indices=None):
    meta_path = assert_not_fresh(features_dir / f"{feature_name}_meta.parquet")
    resid_path = assert_not_fresh(features_dir / f"{feature_name}_resid.npy")
    meta = pd.read_parquet(meta_path)
    resid = np.load(resid_path, mmap_mode="r")
    directions = torch.load(assert_not_fresh(directions_path), map_location="cpu", weights_only=False)
    units = directions["unit_directions"].detach().cpu().numpy().copy()
    if neutral_direction:
        pair_meta = pd.read_parquet(assert_not_fresh(features_dir / "neutral_pairs_meta.parquet"))
        pair_acts = np.load(assert_not_fresh(features_dir / "neutral_pairs_resid.npy"), mmap_mode="r")
        train = pair_meta[pair_meta["split"] == "train"]
        ans_idx = train.index[train["part"] == "ans"].to_numpy()
        unans_idx = train.index[train["part"] == "unans"].to_numpy()
        mean_ans = np.asarray(pair_acts[ans_idx, 1, 11, :], dtype=np.float32).mean(0)
        mean_unans = np.asarray(pair_acts[unans_idx, 1, 11, :], dtype=np.float32).mean(0)
        neutral = mean_unans - mean_ans
        norm = np.linalg.norm(neutral)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Neutral-prompt train-pair direction is degenerate")
        units[11, 1, :] = neutral / norm
    pos_indices = list(range(resid.shape[1])) if position_indices is None else list(position_indices)
    if resid.shape[2] != units.shape[0] or resid.shape[3] != units.shape[2]:
        raise ValueError("Feature cache and Stage 4 direction shapes differ")
    if len(meta) != resid.shape[0]:
        raise ValueError("Feature metadata and residual cache row counts differ")
    projections = {}
    for position_index in pos_indices:
        for layer in range(resid.shape[2]):
            projections[(layer, position_index)] = np.einsum(
                "nd,d->n", np.asarray(resid[:, position_index, layer, :], dtype=np.float32),
                np.asarray(units[layer, position_index], dtype=np.float32))
    return meta, resid, units, projections


def _population(meta, primary):
    selected = meta[meta["grade"].isin(GRADES)].copy()
    if primary:
        selected = selected[selected["matched_pair_id"].notna()]
        groups = selected["matched_pair_id"].astype(str).to_numpy()
    else:
        selected = selected[selected["prop"].notna()]
        groups = selected["prop"].astype(str).to_numpy()
    if selected.empty:
        raise ValueError("No correct/wrong rows are available for this population")
    y = (selected["grade"].to_numpy() == "wrong").astype(int)
    if np.unique(y).size != 2:
        raise ValueError("Both correct and wrong rows are required")
    return selected, y, groups


def _run_population(meta, projections, folds, repeats, n_boot, seed, sensitivity="primary"):
    selected, y, groups = _population(meta, primary=sensitivity != "secondary")
    idx = selected.index.to_numpy(dtype=int)
    groups_for_boot = (selected["matched_pair_id"].fillna(selected["popqa_id"]).astype(str).to_numpy()
                       if "matched_pair_id" in selected else groups)
    summaries = {}
    for name, score in (("maxp", -selected["maxp"].to_numpy(float)),
                        ("entropy", selected["entropy"].to_numpy(float)),
                        ("log_odds", selected["log_odds"].to_numpy(float)),
                        ("log_s_pop", -selected["log_s_pop"].to_numpy(float))):
        summaries[name] = auc_summary(y, score, groups_for_boot, n_boot, seed)
    relation = pd.get_dummies(selected["prop"].astype(str), dtype=float).to_numpy()
    relation_pred = group_oof_scores(relation, y, groups, folds, repeats, seed)
    summaries["relation_one_hot"] = auc_summary(y, relation_pred.mean(axis=0), groups_for_boot, n_boot, seed)
    layer_position = {}
    probe_predictions = {}
    for (layer, pos_index), all_projection in projections.items():
        projection = all_projection[idx]
        summaries.setdefault("frozen_direction", {})[f"layer_{layer}_pos_{pos_index}"] = auc_summary(
            y, projection, groups_for_boot, n_boot, seed + layer + pos_index)
        X = np.asarray(all_acts := _ACTIVATIONS[idx, pos_index, layer, :], dtype=float)
        pred_repeats, chosen = nested_group_probe_oof(X, y, groups, folds, repeats, seed + layer * 101 + pos_index)
        pred = pred_repeats.mean(axis=0)
        probe_predictions[(layer, pos_index)] = pred
        probe_summary = auc_summary(y, pred, groups_for_boot, n_boot, seed + layer + pos_index)
        layer_position[f"layer_{layer}_pos_{pos_index}"] = {
            **probe_summary,
            "maxp_auroc": summaries["maxp"]["auroc"],
            "delta_vs_maxp": probe_summary["auroc"] - summaries["maxp"]["auroc"],
            "selected_c_counts": {str(c): int(np.sum(chosen == c)) for c in (0.01, 0.1, 1.0)},
        }
    return selected, y, groups, summaries, layer_position, probe_predictions


_ACTIVATIONS = None


def _sensitivity(meta, projections, activations, flags, folds, repeats, n_boot, seed):
    scenarios = {}
    masks = {
        "exclude_answer_then_unknown": ~meta["answer_then_unknown"].fillna(False).astype(bool),
        "exclude_degenerate": meta["final_label"].fillna("").astype(str).str.upper().ne("DEGENERATE"),
    }
    if flags:
        masks["grader_human_labels_where_available"] = pd.Series(True, index=meta.index)
        for popqa_id, label in flags.items():
            row_idx = meta.index[meta["popqa_id"].astype(str) == str(popqa_id)]
            if not len(row_idx):
                continue
            if not bool((meta.loc[row_idx, "grade"].astype(str) == "ambiguous").all()):
                continue
            mapped = {"CORRECT": "correct", "INCORRECT": "wrong", "ABSTAIN": "abstain", "UNGRADEABLE": "ambiguous"}.get(label)
            if mapped is not None:
                meta.loc[row_idx, "grade"] = mapped
    global _ACTIVATIONS
    _ACTIVATIONS = activations
    for name, mask in masks.items():
        subset = meta[mask].copy()
        try:
            _selected, _y, _groups, summaries, layer_scores, _pred = _run_population(
                subset, projections, folds, repeats, n_boot, seed, "primary")
            scenarios[name] = {"n": len(subset), "baselines": summaries, "probe": layer_scores}
        except ValueError as exc:
            scenarios[name] = {"error": str(exc), "n": len(subset)}
    return scenarios


def _load_human_grader_resolutions(csv_path):
    csv_path = assert_not_fresh(csv_path)
    if not csv_path.exists():
        return {}
    import csv
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return {row["uid"].removeprefix("popqa:"): row["human_label"]
                for row in csv.DictReader(handle) if row.get("human_label")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--feature-name", default="popqa")
    parser.add_argument("--neutral-direction", action="store_true",
                        help="Use the train-pair neutral-prompt direction at layer 11, position -4")
    parser.add_argument("--directions", type=Path, default=CFG.stage4_out_path)
    parser.add_argument("--grader-labels", type=Path, default=CFG.results_dir / "grader_validation_test.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=CFG.n_bootstrap)
    parser.add_argument("--seed", type=int, default=CFG.seed_incremental_cv)
    parser.add_argument("--output", type=Path, default=CFG.results_dir / "exploratory" / "e1_probe.json")
    parser.add_argument("--figure-output", type=Path,
                        default=ROOT / "figures" / "exploratory" / "e1_layer_position_auroc.png")
    args = parser.parse_args(argv)
    meta, resid, _units, projections = _prepare(args.features_dir, args.feature_name, args.directions,
                                                neutral_direction=args.neutral_direction)
    global _ACTIVATIONS
    _ACTIVATIONS = resid
    primary, y, groups, baselines, layer_scores, probe_predictions = _run_population(
        meta, projections, args.folds, args.repeats, args.bootstrap, args.seed)
    idx = primary.index.to_numpy(dtype=int)
    scores = {"primary_matched_pairs": {"n": len(primary), "baselines": baselines, "probe": layer_scores}}
    all_valid, all_y, all_groups = _population(meta, primary=False)
    if all_valid["prop"].nunique() >= args.folds:
        all_scores = {}
        for (layer, pos_index), all_projection in projections.items():
            selected_idx = all_valid.index.to_numpy(dtype=int)
            X = np.asarray(resid[selected_idx, pos_index, layer, :], dtype=float)
            pred, chosen = nested_group_probe_oof(X, all_y, all_groups, args.folds, args.repeats,
                                                  args.seed + layer * 101 + pos_index)
            all_scores[f"layer_{layer}_pos_{pos_index}"] = {
                **auc_summary(all_y, pred.mean(axis=0), all_groups, args.bootstrap, args.seed + layer + pos_index),
                "selected_c_counts": {str(c): int(np.sum(chosen == c)) for c in (0.01, 0.1, 1.0)},
            }
        scores["secondary_all_graded_relation_grouped"] = {"n": len(all_valid), "probe": all_scores}
    # Nested scores enter the registered incremental-value utility as an extra, OOF-only feature.
    chosen_probe = probe_predictions[(11, 1)] if (11, 1) in probe_predictions else next(iter(probe_predictions.values()))
    frame = primary.copy().reset_index(drop=True)
    frame["y"] = y
    frame["matched_pair_id"] = groups
    frame["probe_oof"] = chosen_probe
    for feature in BASE_FEATURES:
        frame[feature] = primary[feature].to_numpy(float)
    incremental = incremental_value_test(frame, n_boot=args.bootstrap, seed=args.seed,
                                         full_features=list(BASE_FEATURES) + ["probe_oof"])
    grader = _load_human_grader_resolutions(args.grader_labels)
    sensitivities = _sensitivity(meta.copy(), projections, resid, grader, args.folds, args.repeats,
                                 args.bootstrap, args.seed + 1000)
    payload = {"experiment": "E1_supervised_probe", "feature_name": args.feature_name,
               "folds": args.folds, "repeats": args.repeats,
               "c_grid": [0.01, 0.1, 1.0], "direction_mode": "rhat_neutral_layer11_pos_minus4" if args.neutral_direction else "frozen_rhat",
               "primary_and_secondary": scores,
               "incremental_value_layer11_pos_minus4": incremental, "sensitivity": sensitivities}
    out = write_exploratory_json(args.output, payload, ROOT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers, pos_indices = resid.shape[2], list(range(resid.shape[1]))
    heat = np.array([[layer_scores[f"layer_{layer}_pos_{pos}"]["delta_vs_maxp"]
                      for pos in pos_indices] for layer in range(layers)])
    figure_path = assert_not_fresh(args.figure_output)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 10))
    image = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax.set(xticks=range(len(pos_indices)), xticklabels=[-5, -4, -3, -2, -1],
           xlabel="Suffix position", ylabel="Layer", title="Probe AUROC minus maxp AUROC")
    fig.colorbar(image, ax=ax, label="AUROC difference")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
