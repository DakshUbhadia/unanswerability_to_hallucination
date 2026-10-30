#!/usr/bin/env python3
"""Cache exploratory all-layer residuals for judged PopQA and the synthetic pairs."""
import argparse
import hashlib
import json
import sys
if __name__ == "__main__":
    if "--legacy" not in sys.argv:
        print("superseded by scripts/exploratory/e1_extract_features.py; pass --legacy to run", file=sys.stderr)
        raise SystemExit(2)
    sys.argv.remove("--legacy")
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import CFG
from src.eval_common import (build_chat_batches, capture_residuals_and_logits, ensure_stage4_directions,
                             load_b_ctrl_rows, load_runtime_contract, load_stage_model, load_split_pairs,
                             read_json, read_jsonl)
from src.exploratory import assert_not_fresh, write_exploratory_json
from src.popqa_uncertainty import _verify_popqa_input_identity, load_popqa_groups
from src.tokenization_contract import canonical_input_ids, chat_template_sha256, input_ids_sha256

POSITIONS = (-5, -4, -3, -2, -1)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_rows(popqa_path, neutral=False):
    assert_not_fresh(popqa_path)
    rows = read_jsonl(popqa_path)
    if not rows or any(r.get("baseline_partition") not in {"correct", "wrong", "abstain", "ambiguous"} for r in rows):
        raise RuntimeError("Feature caching requires every PopQA row to have a final judged grade")
    if not neutral and any(r.get("label_source") != "canonical_v1_batch1_judged" for r in rows):
        raise RuntimeError("Feature caching requires the judged canonical v4.2 PopQA sample")
    return rows


def _metadata_for_popqa(rows):
    from src.abstention_classifier import is_answer_then_unknown

    output = []
    for row in rows:
        classifier = row.get("classifier") or {}
        output.append({
            "popqa_id": row["popqa_id"], "grade": row["baseline_partition"],
            "matched_pair_id": row.get("matched_pair_id"), "prop": row.get("prop"),
            "log_s_pop": float(row.get("log_s_pop", row.get("log_popularity", 0.0))),
            "maxp": None, "entropy": None, "log_odds": None,
            "input_ids_sha256": row.get("input_ids_sha256"),
            "final_label": classifier.get("final_label"), "judge_margin": classifier.get("judge_margin"),
            "answer_then_unknown": is_answer_then_unknown(row.get("baseline_response", "")),
            "question": row["question"], "response": row.get("baseline_response", ""),
        })
    return output


def _confidence(logits, unknown_ids):
    probs = torch.softmax(logits.float(), dim=-1)
    maxp = probs.max(-1).values.cpu().numpy()
    entropy = (-(probs * probs.clamp(min=1e-12).log()).sum(-1)).cpu().numpy()
    unknown_mass = probs[:, list(dict.fromkeys(map(int, unknown_ids)))].sum(-1).clamp(1e-7, 1 - 1e-7)
    log_odds = (torch.log(unknown_mass) - torch.log1p(-unknown_mass)).cpu().numpy()
    return maxp, entropy, log_odds


def _cache_set(name, rows, prompts, metadata, raw_model, tokenizer, contract, batch_size, out_dir, system_prompt):
    if len(rows) != len(prompts) or len(rows) != len(metadata):
        raise ValueError(f"{name}: rows, prompts, and metadata are not aligned")
    if not prompts:
        raise ValueError(f"{name}: empty feature set")
    n_layers, d_model = int(contract["n_layers"]), int(contract["d_model"])
    resid_path = assert_not_fresh(out_dir / f"{name}_resid.npy")
    mmap = np.lib.format.open_memmap(resid_path, mode="w+", dtype=np.float16,
                                     shape=(len(prompts), len(POSITIONS), n_layers, d_model))
    maxp = entropy = log_odds = None
    for pos_i, pos in enumerate(POSITIONS):
        batches = build_chat_batches(tokenizer, list(prompts), system_prompt,
                                     list(contract["suffix_token_ids"]), batch_size)
        activations, logits = capture_residuals_and_logits(raw_model, batches, contract, list(range(n_layers)), pos)
        activation_np = activations.numpy()
        if not np.isfinite(activation_np).all():
            raise FloatingPointError(f"{name}: non-finite activations at suffix position {pos}")
        mmap[:, pos_i, :, :] = activation_np.astype(np.float16)
        # Position -1 is the last suffix token and its logits predict the first answer token.
        if pos == -1:
            maxp, entropy, log_odds = _confidence(logits, contract["unknown_token_ids"])
    mmap.flush()
    del mmap
    for i, row in enumerate(metadata):
        row["maxp"], row["entropy"], row["log_odds"] = float(maxp[i]), float(entropy[i]), float(log_odds[i])
    import pandas as pd
    meta_path = assert_not_fresh(out_dir / f"{name}_meta.parquet")
    pd.DataFrame(metadata).to_parquet(meta_path, index=False)
    mapped = np.load(resid_path, mmap_mode="r")
    if mapped.shape != (len(prompts), 5, n_layers, d_model) or not np.isfinite(mapped).all():
        raise AssertionError(f"{name}: output shape or finite-value check failed")
    return {"resid": {"path": str(resid_path.relative_to(ROOT)), "sha256": _sha256(resid_path),
                       "size_bytes": resid_path.stat().st_size, "shape": list(mapped.shape), "dtype": str(mapped.dtype)},
            "meta": {"path": str(meta_path.relative_to(ROOT)), "sha256": _sha256(meta_path),
                     "size_bytes": meta_path.stat().st_size, "rows": len(metadata), "format": "parquet"}}


def _make_sets(popqa_rows, pairs_by_split, bctrl_rows):
    popqa_meta = _metadata_for_popqa(popqa_rows)
    sets = {"popqa": (popqa_rows, [r["question"] for r in popqa_rows], popqa_meta)}
    pair_rows, pair_prompts, pair_meta = [], [], []
    for split in ("train", "val", "test"):
        for pair in pairs_by_split[split]:
            for part, key, grade in (("ans", "ans_prompt", "answerable"), ("unans", "unans_prompt", "unanswerable")):
                pair_rows.append(pair); pair_prompts.append(pair[key])
                pair_meta.append({"pair_id": pair["pair_id"], "split": split, "part": part, "grade": grade,
                                  "matched_pair_id": pair["pair_id"], "prop": pair.get("prop"),
                                  "log_s_pop": None, "input_ids_sha256": None,
                                  "template_id": pair.get("template_id"), "family": pair.get("family"),
                                  "maxp": None, "entropy": None, "log_odds": None,
                                  "final_label": None, "judge_margin": None, "answer_then_unknown": False})
    sets["pairs"] = (pair_rows, pair_prompts, pair_meta)
    bctrl_meta = [{"b_ctrl_id": r["b_ctrl_id"], "pair_id": r["pair_id"], "split": "test", "part": "bctrl",
                   "grade": "bctrl", "matched_pair_id": r["pair_id"], "prop": r.get("prop"),
                   "log_s_pop": None, "input_ids_sha256": None,
                   "template_id": r.get("template_id"), "family": r.get("family"), "maxp": None,
                   "entropy": None, "log_odds": None, "final_label": None, "judge_margin": None,
                   "answer_then_unknown": False} for r in bctrl_rows]
    sets["bctrl"] = (bctrl_rows, [r["prompt"] for r in bctrl_rows], bctrl_meta)
    return sets


def _verify_projection_cache(features_dir, directions, stage6_path):
    stage6 = read_json(stage6_path)
    per_prompt = stage6["per_prompt"]
    pair_ids = per_prompt["pair_ids"]
    pair_meta_path = features_dir / "pairs_meta.parquet"
    import pandas as pd
    meta = pd.read_parquet(pair_meta_path)
    resid = np.load(features_dir / "pairs_resid.npy", mmap_mode="r")
    units = directions["unit_directions"].detach().cpu().numpy()
    l_star = int(stage6["l_star"])
    position = int(stage6["i_star"])
    position_index = position + len(POSITIONS)
    pos = POSITIONS.index(position)
    projection = np.einsum("nd,d->n", resid[:, pos, l_star, :].astype(np.float32), units[l_star, position_index].astype(np.float32))
    aligned = {}
    for part in ("unans", "ans"):
        selected = meta[(meta["split"] == "test") & (meta["part"] == part)].set_index("pair_id")
        index_by_pair = {str(row.pair_id): int(i) for i, row in meta.iterrows()
                         if row["split"] == "test" and row["part"] == part}
        values = np.array([projection[index_by_pair[str(pid)]] for pid in pair_ids])
        target = np.asarray(stage6["projections"]["test_unans" if part == "unans" else "test_ans"], dtype=float)
        aligned[part] = {"max_abs_diff": float(np.max(np.abs(values - target))), "n": len(values)}
    bctrl_meta = pd.read_parquet(features_dir / "bctrl_meta.parquet")
    bctrl_projection = np.einsum("nd,d->n", np.load(features_dir / "bctrl_resid.npy", mmap_mode="r")[:, pos, l_star, :].astype(np.float32),
                                 units[l_star, position_index].astype(np.float32))
    target = np.asarray(stage6["projections"]["b_ctrl"], dtype=float)
    aligned["bctrl"] = {"max_abs_diff": float(np.max(np.abs(bctrl_projection - target))), "n": len(target)}
    scale = max(1.0, float(np.max(np.abs(np.asarray(stage6["projections"]["test_unans"], dtype=float)))))
    if max(item["max_abs_diff"] for item in aligned.values()) > 0.05 * scale:
        raise AssertionError(f"Stage 6 projection parity exceeded 0.05 x scale ({scale:g}): {aligned}")
    return {"l_star": l_star, "i_star": position, "scale": scale, "tolerance": 0.05 * scale, "sets": aligned}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size_forward)
    parser.add_argument("--popqa-path", type=Path, default=CFG.data_dir / "popqa_sample.jsonl")
    parser.add_argument("--features-dir", type=Path, default=ROOT / "features")
    parser.add_argument("--neutral", action="store_true", help="Cache the neutral-prompt PopQA and pair features")
    args = parser.parse_args(argv)
    assert_not_fresh(args.popqa_path)
    if args.neutral:
        neutral_path = assert_not_fresh(CFG.data_dir / "exploratory" / "popqa_neutral.jsonl")
        popqa_rows = _feature_rows(neutral_path, neutral=True)
    else:
        popqa_rows = _feature_rows(args.popqa_path)
    # Enforce Stage 2's count, label-source, and exact matched-pair identity contract.
    stage2 = read_json(CFG.stage2_out_path)
    groups = load_popqa_groups(args.popqa_path, stage2)
    if len(popqa_rows) != len(groups["rows"]):
        raise AssertionError("PopQA feature rows and validated Stage 2 rows disagree")
    contract = load_runtime_contract()
    directions, _ = ensure_stage4_directions()
    _selection = read_json(CFG.stage5_out_path)
    raw_model, tokenizer, dtype_name = load_stage_model(contract, args.device, args.dtype)
    tokenizer.padding_side = "left"
    system_prompt = CFG.exploratory_neutral_system_prompt if args.neutral else contract["system_prompt_used"]
    if args.neutral:
        expected_template = chat_template_sha256(tokenizer)
        for row in popqa_rows:
            ids = canonical_input_ids(tokenizer, row["question"], system_prompt, contract["suffix_token_ids"])
            actual = input_ids_sha256(ids, contract["model_name"], expected_template)
            if row.get("input_ids_sha256") != actual:
                raise ValueError(f"Neutral-prompt canonical input hash mismatch for {row['popqa_id']}")
        popqa_verified = len(popqa_rows)
    else:
        popqa_verified = _verify_popqa_input_identity(tokenizer, popqa_rows, system_prompt,
                                                       contract["model_name"], contract["suffix_token_ids"])
    pairs_path = assert_not_fresh(CFG.data_dir / "pairs.jsonl")
    bctrl_path = assert_not_fresh(CFG.data_dir / "b_ctrl.jsonl")
    pairs_by_split = {split: load_split_pairs(split, pairs_path) for split in ("train", "val", "test")}
    bctrl_rows = load_b_ctrl_rows(bctrl_path, pairs_by_split["test"])
    sets = _make_sets(popqa_rows, pairs_by_split, bctrl_rows)
    if args.neutral:
        sets = {f"neutral_{key}": value for key, value in sets.items()}
    template_digest = chat_template_sha256(tokenizer)
    for _name, (set_rows, prompts, metadata) in sets.items():
        for i, (prompt_text, row_meta) in enumerate(zip(prompts, metadata)):
            if row_meta.get("input_ids_sha256"):
                continue
            ids = canonical_input_ids(tokenizer, prompt_text, system_prompt, contract["suffix_token_ids"])
            row_meta["input_ids_sha256"] = input_ids_sha256(ids, contract["model_name"], template_digest)
    args.features_dir.mkdir(parents=True, exist_ok=True)
    files, started = {}, time.perf_counter()
    for name, (rows, prompts, metadata) in sets.items():
        files.update(_cache_set(name, rows, prompts, metadata, raw_model, tokenizer, contract,
                                args.batch_size, args.features_dir, system_prompt))
        print(f"[exploratory cache] {name}: {len(rows)} rows", flush=True)
    del raw_model
    projection_check = None if args.neutral else _verify_projection_cache(args.features_dir, directions, CFG.stage6_out_path)
    manifest = {"model_name": contract["model_name"], "runtime_dtype": dtype_name,
                "n_layers": int(contract["n_layers"]), "d_model": int(contract["d_model"]),
                "positions": list(POSITIONS), "sets": {name: len(rows) for name, (rows, _, _) in sets.items()},
                "popqa_input_hash_count": popqa_verified, "files": files,
                "stage6_projection_check": projection_check, "elapsed_seconds": time.perf_counter() - started}
    manifest_name = "features_manifest_neutral.json" if args.neutral else "features_manifest.json"
    out = write_exploratory_json(CFG.results_dir / "exploratory" / manifest_name, manifest, ROOT)
    print(f"Wrote exploratory feature manifest {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
