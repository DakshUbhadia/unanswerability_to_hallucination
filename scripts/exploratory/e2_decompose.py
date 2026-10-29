#!/usr/bin/env python3
"""Decompose the exploratory residual probe signal and apply the frozen interpretation rule."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import CFG
from src.exploratory import assert_not_fresh, write_exploratory_json
from src.exploratory.analysis import auc_summary, nested_group_probe_oof

INTERPRETATION_RULE = {
    "residual_ci_includes_0.5": "abstention-intent readout",
    "residual_ci_excludes_0.5_and_probe_beats_maxp": "knowledge-state signal beyond output confidence",
    "otherwise": "inconclusive",
}


def _token_lens(direction, model_name, unknown_token_ids, model_revision=None):
    """Approximate an RMSNorm logit lens; only the final norm and unembedding are loaded."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    resolved_revision = model_revision or getattr(config, "_commit_hash", None)
    index_path = None
    try:
        index_path = Path(hf_hub_download(model_name, "model.safetensors.index.json", revision=resolved_revision))
    except Exception:
        pass
    if index_path and index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weights_map = index["weight_map"]
        norm_key = next((k for k in ("model.norm.weight", "transformer.norm.weight") if k in weights_map), None)
        unembed_key = next((k for k in ("lm_head.weight", "model.embed_tokens.weight",
                                        "transformer.wte.weight") if k in weights_map), None)
        if norm_key is None or unembed_key is None:
            raise KeyError("Could not locate final RMSNorm and output embedding tensors in safetensors index")
        shards = {weights_map[norm_key], weights_map[unembed_key]}
        shard_paths = {name: Path(hf_hub_download(model_name, name, revision=resolved_revision)) for name in shards}
        tensors = {}
        for shard_name, shard_path in shard_paths.items():
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in (norm_key, unembed_key):
                    if key in handle.keys():
                        tensors[key] = handle.get_tensor(key)
    else:
        weight_path = Path(hf_hub_download(model_name, "model.safetensors", revision=resolved_revision))
        with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            norm_key = next((k for k in ("model.norm.weight", "transformer.norm.weight") if k in keys), None)
            unembed_key = next((k for k in ("lm_head.weight", "model.embed_tokens.weight",
                                            "transformer.wte.weight") if k in keys), None)
            if norm_key is None or unembed_key is None:
                raise KeyError("Could not locate final RMSNorm and output embedding tensors")
            tensors = {norm_key: handle.get_tensor(norm_key), unembed_key: handle.get_tensor(unembed_key)}
    norm_weight, unembed = tensors[norm_key].float(), tensors[unembed_key]
    vector = torch.as_tensor(direction, dtype=unembed.dtype, device="cpu")
    logits = torch.mv(unembed, norm_weight.to(unembed.dtype) * vector).float()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=resolved_revision)
    order = torch.argsort(logits, descending=True)
    promoted = [{"token_id": int(i), "token": tokenizer.decode([int(i)]), "logit": float(logits[i])}
                for i in order[:20]]
    demoted = [{"token_id": int(i), "token": tokenizer.decode([int(i)]), "logit": float(logits[i])}
               for i in order[-20:].flip(0)]
    unknown_ranks = {}
    for token_id in map(int, unknown_token_ids):
        unknown_ranks[str(token_id)] = int((logits > logits[token_id]).sum().item()) + 1
    return {"model_name": model_name, "resolved_revision": resolved_revision,
            "approximation": "Apply model.norm.weight elementwise to the direction and omit RMS normalization's scalar denominator; ranking is unchanged by that positive scalar.",
            "promoted_top20": promoted, "demoted_top20": demoted, "unknown_token_ranks": unknown_ranks}


def _partial_spearman(projection, y, log_odds, maxp):
    controls = np.column_stack([rankdata(log_odds), rankdata(maxp)])
    p_rank, y_rank = rankdata(projection), rankdata(y)
    model_p = LinearRegression().fit(controls, p_rank)
    model_y = LinearRegression().fit(controls, y_rank)
    return float(spearmanr(p_rank - model_p.predict(controls), y_rank - model_y.predict(controls)).statistic)


def _residualized_oof(projection, log_odds, y, groups, folds, repeats, seed):
    prediction = np.full((repeats, len(y)), np.nan)
    from sklearn.model_selection import StratifiedGroupKFold
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for train, test in splitter.split(log_odds[:, None], y, groups):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("A pair crossed residualization folds")
            model = LinearRegression().fit(log_odds[train, None], projection[train])
            prediction[repeat, test] = projection[test] - model.predict(log_odds[test, None])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Residualized OOF scores are incomplete")
    return prediction.mean(0)


def _group_contrasts(meta, projection, n_boot, seed):
    rows = []
    grade_pairs = (("abstain", "correct"), ("wrong", "correct"), ("abstain", "wrong"))
    for positive, negative in grade_pairs:
        selected = meta[meta["grade"].isin([positive, negative])]
        mask = selected.index.to_numpy(dtype=int)
        y = (selected["grade"].to_numpy() == positive).astype(int)
        groups = selected["matched_pair_id"].fillna(selected.get("popqa_id", pd.Series(index=selected.index, dtype=str))).astype(str).to_numpy()
        if np.unique(y).size < 2:
            continue
        rows.append({"positive": positive, "negative": negative,
                     **auc_summary(y, projection[mask], groups, n_boot, seed)})
    return rows


def _neutral_direction(features_dir):
    meta = pd.read_parquet(assert_not_fresh(features_dir / "neutral_pairs_meta.parquet"))
    acts = np.load(assert_not_fresh(features_dir / "neutral_pairs_resid.npy"), mmap_mode="r")
    train = meta[meta["split"] == "train"]
    ans_ix = train.index[train["part"] == "ans"].to_numpy()
    unans_ix = train.index[train["part"] == "unans"].to_numpy()
    l_star, pos_index = 11, 1  # registered coordinate: layer 11, suffix position -4
    mean_ans = np.asarray(acts[ans_ix, pos_index, l_star, :], dtype=np.float32).mean(0)
    mean_unans = np.asarray(acts[unans_ix, pos_index, l_star, :], dtype=np.float32).mean(0)
    direction = mean_unans - mean_ans
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Neutral-prompt train-pair direction is degenerate")
    return direction / norm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--feature-name", default="popqa")
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--directions", type=Path, default=CFG.stage4_out_path)
    parser.add_argument("--model-revision", default=None, help="Optional immutable HF commit SHA for the token lens")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=CFG.n_bootstrap)
    parser.add_argument("--seed", type=int, default=CFG.seed_incremental_cv)
    parser.add_argument("--position", type=int, default=-4, choices=(-5, -4, -3, -2, -1))
    parser.add_argument("--output", type=Path, default=CFG.results_dir / "exploratory" / "e2_decomposition.json")
    args = parser.parse_args(argv)
    meta = pd.read_parquet(assert_not_fresh(args.features_dir / f"{args.feature_name}_meta.parquet"))
    resid = np.load(assert_not_fresh(args.features_dir / f"{args.feature_name}_resid.npy"), mmap_mode="r")
    directions = torch.load(assert_not_fresh(args.directions), map_location="cpu", weights_only=False)
    units = directions["unit_directions"].detach().cpu().numpy()
    pos_index = {-5: 0, -4: 1, -3: 2, -2: 3, -1: 4}[args.position]
    direction = _neutral_direction(args.features_dir) if args.neutral else units[11, pos_index]
    rows = meta[meta["grade"].isin(["wrong", "correct"]) & meta["matched_pair_id"].notna()].copy()
    if rows.empty:
        raise ValueError("E2 primary matched-population analysis needs matched correct/wrong rows")
    y = (rows["grade"].to_numpy() == "wrong").astype(int)
    groups = rows["matched_pair_id"].fillna(rows.get("popqa_id", pd.Series(index=rows.index, dtype=str))).astype(str).to_numpy()
    popqa_idx = rows.index.to_numpy(dtype=int)
    projections = np.einsum("nd,d->n", np.asarray(resid[popqa_idx, pos_index, 11, :], dtype=np.float32), direction)
    log_odds = rows["log_odds"].to_numpy(float)
    maxp = rows["maxp"].to_numpy(float)
    labels = rows["grade"].to_numpy()
    residual = _residualized_oof(projections, log_odds, y, groups, args.folds, args.repeats, args.seed)
    residual_summary = auc_summary(y, residual, groups, args.bootstrap, args.seed)
    maxp_summary = auc_summary(y, -maxp, groups, args.bootstrap, args.seed)
    probe_x = np.asarray(resid[popqa_idx, pos_index, 11, :], dtype=np.float32)
    probe_repeats, probe_c = nested_group_probe_oof(probe_x, y, groups, args.folds, args.repeats,
                                                    args.seed + 11 * 101 + pos_index)
    probe_summary = auc_summary(y, probe_repeats.mean(axis=0), groups, args.bootstrap, args.seed)
    residual_ci = residual_summary["ci95"]
    includes_half = residual_ci["low"] <= 0.5 <= residual_ci["high"]
    if includes_half:
        interpretation = INTERPRETATION_RULE["residual_ci_includes_0.5"]
    elif probe_summary["auroc"] > maxp_summary["auroc"]:
        interpretation = INTERPRETATION_RULE["residual_ci_excludes_0.5_and_probe_beats_maxp"]
    else:
        interpretation = INTERPRETATION_RULE["otherwise"]
    # Full-data refit is limited to probe-weight geometry; its scores are never evaluated.
    train_rows = meta[meta["grade"].isin(["wrong", "correct"])].index.to_numpy(dtype=int)
    y_train = (meta.loc[train_rows, "grade"].to_numpy() == "wrong").astype(int)
    X_train = np.asarray(resid[train_rows, pos_index, :, :], dtype=np.float32)
    geometry = {}
    for geometry_pos in (1, 4):  # suffix offsets -4 and -1
        layer_rows = {}
        X_geometry = np.asarray(resid[train_rows, geometry_pos, :, :], dtype=np.float32)
        for layer in range(resid.shape[2]):
            x = X_geometry[:, layer, :]
            scaler = StandardScaler().fit(x)
            probe = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(scaler.transform(x), y_train)
            weight = probe.coef_[0] / np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
            unit = units[layer, geometry_pos]
            layer_rows[str(layer)] = float(np.dot(weight, unit) / (np.linalg.norm(weight) * np.linalg.norm(unit)))
        geometry[str((-5, -4, -3, -2, -1)[geometry_pos])] = layer_rows
    contrasts = {}
    for layer in range(resid.shape[2]):
        layer_projection = np.einsum("nd,d->n", np.asarray(resid[:, pos_index, layer, :], dtype=np.float32), direction)
        contrasts[str(layer)] = _group_contrasts(meta, layer_projection, args.bootstrap, args.seed + layer)
    contract = json.loads(CFG.stage0_out_path.read_text(encoding="utf-8"))
    lens = _token_lens(direction, contract["model_name"], [contract["unknown_token_id_nospace"],
                                                              contract["unknown_token_id_space"]],
                       args.model_revision)
    partial = _partial_spearman(projections, y, log_odds, maxp)
    if not np.isfinite(partial):
        partial = None
    payload = {"experiment": "E2_decomposition", "feature_name": args.feature_name,
               "folds": args.folds, "repeats": args.repeats,
               "interpretation_rule": INTERPRETATION_RULE, "interpretation": interpretation,
               "residualized_proj_layer11_pos_minus4": residual_summary,
               "probe_layer11_pos_minus4": {**probe_summary,
                                             "selected_c_counts": {str(c): int(np.sum(probe_c == c)) for c in (0.01, 0.1, 1.0)}},
               "maxp_baseline": maxp_summary, "partial_spearman_proj_controlling_log_odds_maxp": partial,
               "probe_weight_cosine_with_frozen_direction_by_layer": geometry,
               "group_contrasts_layer11": contrasts, "logit_lens": lens}
    out = write_exploratory_json(args.output, payload, ROOT)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
