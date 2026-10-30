#!/usr/bin/env python3
"""Extract suffix and teacher-forced answer activations for exploratory E1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.config import CFG
from src.eval_common import (ConditionCheckpoint, capture_residuals_and_logits, load_runtime_contract,
                             load_stage_model, read_json, sha256_file)
from src.exploratory import assert_not_fresh, exploratory_header
from src.exploratory.capture import capture_answer_positions, capture_suffix_all_layers
from src.exploratory.e1_data import load_e1_frame
from src.tokenization_contract import canonical_input_ids, input_ids_sha256, chat_template_sha256


def _hash(value):
    h = hashlib.sha256()
    h.update(value)
    return h.hexdigest()


def _input_hashes(tokenizer, frame, contract, dry_run):
    rows = frame.to_dict("records")
    if not dry_run:
        from src.popqa_uncertainty import _verify_popqa_input_identity
        return _verify_popqa_input_identity(tokenizer, rows, contract["system_prompt_used"],
                                            contract["model_name"], contract["suffix_token_ids"])
    template = chat_template_sha256(tokenizer)
    for row in rows:
        ids = canonical_input_ids(tokenizer, row["question"], contract["system_prompt_used"],
                                  contract["suffix_token_ids"])
        row["input_ids_sha256"] = input_ids_sha256(ids, contract["model_name"], template)
    frame["input_ids_sha256"] = [r["input_ids_sha256"] for r in rows]
    return len(rows)


def _parity(raw_model, tokenizer, contract, rows, batch_size):
    from src.eval_common import build_chat_batches
    sample = rows[:32]
    prompts = [r["question"] for r in sample]
    captured = capture_suffix_all_layers(raw_model, tokenizer, contract, prompts, (-4,), batch_size)
    batches = build_chat_batches(tokenizer, prompts, contract["system_prompt_used"],
                                 contract["suffix_token_ids"], batch_size)
    reference, ref_logits = capture_residuals_and_logits(raw_model, batches, contract, [11], -4)
    got = captured["resid"][:, 0, 11].astype(np.float32)
    ref = reference[:, 0].numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(got - ref)))
    tolerance = 0.05 * max(1.0, float(np.max(np.abs(ref))))
    probs = torch.softmax(ref_logits.float(), dim=-1).max(-1).values.numpy()
    maxp_diff = float(np.max(np.abs(captured["maxp"] - probs)))
    if max_abs > tolerance or maxp_diff > 1e-3:
        raise RuntimeError(f"Capture parity failed: residual max diff {max_abs:g} (limit {tolerance:g}), maxp diff {maxp_diff:g}")
    return {"resid_max_abs_diff": max_abs, "resid_tolerance": tolerance, "maxp_max_abs_diff": maxp_diff,
            "n_rows": len(sample)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/exploratory/e1_features")
    parser.add_argument("--push-hf", action="store_true")
    args = parser.parse_args(argv)
    out_dir = assert_not_fresh(args.out_dir)
    if args.dry_run and "results/exploratory/dryrun" not in str(out_dir):
        raise ValueError("Dry-run outputs must be under results/exploratory/dryrun/")
    out_dir.mkdir(parents=True, exist_ok=True)
    contract = load_runtime_contract()
    raw_model, tokenizer, _dtype = load_stage_model(contract, "cuda:0", "auto")
    frame = load_e1_frame(dry_run=args.dry_run)
    rows = frame.to_dict("records")
    _input_hashes(tokenizer, frame, contract, args.dry_run)
    parity = _parity(raw_model, tokenizer, contract, rows, args.batch_size)

    n, nl, d = len(rows), int(contract["n_layers"]), int(contract["d_model"])
    suffix_path, answer_path = out_dir / "suf.npy", out_dir / "ans.npy"
    stage2_sha = sha256_file(CFG.stage2_out_path) if CFG.stage2_out_path.exists() else "dry-run-v4"
    source = ROOT / ("results/archive_v4/data/popqa_sample.jsonl" if args.dry_run else "data/popqa_sample.jsonl")
    fingerprint = _hash(("|".join((str(__import__("subprocess").run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                    capture_output=True, text=True).stdout.strip()), stage2_sha, sha256_file(source), contract["model_name"]))).encode())
    checkpoint = ConditionCheckpoint(out_dir / "e1_extract_checkpoint.pt", fingerprint, every=100)
    resume = bool(checkpoint.results) and suffix_path.exists() and answer_path.exists()
    suffix = np.lib.format.open_memmap(suffix_path, mode="r+" if resume else "w+", dtype=np.float16, shape=(n, 5, nl, d))
    answer = np.lib.format.open_memmap(answer_path, mode="r+" if resume else "w+", dtype=np.float16, shape=(n, 2, nl, d))
    if not resume:
        suffix[:] = np.nan
        answer[:] = np.nan
    scalar_path = out_dir / "e1_scalars.npz"
    if resume and scalar_path.exists():
        old = np.load(scalar_path)
        for key in old.files:
            if key in frame:
                frame[key] = old[key]
    suffix_positions = (-5, -4, -3, -2, -1)
    confidence = {k: np.full(n, np.nan, dtype=np.float32) for k in ("maxp", "entropy", "log_odds")}
    for start in range(0, n, 100):
        key = f"rows_{start}_{min(n, start+100)}"
        if key in checkpoint:
            continue
        stop = min(n, start + 100)
        sf = capture_suffix_all_layers(raw_model, tokenizer, contract, [r["question"] for r in rows[start:stop]],
                                       suffix_positions, args.batch_size)
        suffix[start:stop] = sf["resid"]
        for k in confidence:
            confidence[k][start:stop] = sf[k]
            frame.loc[start:stop-1, k] = sf[k]
        af = capture_answer_positions(raw_model, tokenizer, contract, rows[start:stop])
        answer[start:stop] = af["resid_ans"]
        for j, name in enumerate(("seq_logprob_mean", "seq_logprob_min", "greedy_consistent", "first_token_match", "n_answer_tokens")):
            frame.loc[start:stop-1, name] = af[name]
        np.savez(scalar_path, **{k: frame[k].to_numpy() for k in ("maxp", "entropy", "log_odds", "seq_logprob_mean", "seq_logprob_min", "greedy_consistent", "first_token_match", "n_answer_tokens")})
        checkpoint.put(key, {"done": True})
        suffix.flush(); answer.flush()
        print(f"[E1] captured rows {start}:{stop}/{n}", flush=True)
    checkpoint.flush()
    greedy_rate = float(pd.to_numeric(frame.get("greedy_consistent"), errors="coerce").mean())
    answer_valid = pd.to_numeric(frame.get("n_answer_tokens"), errors="coerce").fillna(0).to_numpy() > 0
    for start in range(0, n, 100):
        stop = min(n, start + 100)
        if not np.isfinite(suffix[start:stop]).all():
            raise RuntimeError(f"Suffix residual features contain non-finite values in rows {start}:{stop}")
        local_valid = answer_valid[start:stop]
        if local_valid.any() and not np.isfinite(answer[start:stop][local_valid]).all():
            raise RuntimeError(f"Non-empty answer rows contain non-finite answer residual features in rows {start}:{stop}")
    if greedy_rate < 0.90:
        bad = frame.loc[~frame["greedy_consistent"].fillna(False).astype(bool), "popqa_id"].astype(str).tolist()
        raise RuntimeError(f"Teacher-forcing greedy consistency is below 90% ({greedy_rate:.3%}); mismatches: {bad[:32]}")
    # Recompute scalar features in row order for resumed runs as well.
    frame["artifact_status"] = "DRY RUN – v4 labels – not reportable" if args.dry_run else "exploratory_post_registration"
    meta_path = out_dir / "meta.parquet"
    frame.to_parquet(meta_path, index=False)
    stage8_parity = None
    if not args.dry_run:
        stage8_path = CFG.stage8_out_path
        stage8 = read_json(stage8_path)
        by_id = {str(item["popqa_id"]): item for item in stage8.get("matched_items", [])}
        diffs = {key: [] for key in ("maxp", "entropy", "log_odds")}
        for _, row in frame[frame["is_popularity_matched"].astype(bool)].iterrows():
            previous = by_id.get(str(row["popqa_id"]))
            if previous is None:
                raise RuntimeError(f"Stage 8 lacks matched PopQA row {row['popqa_id']}")
            for key in diffs:
                diffs[key].append(abs(float(row[key]) - float(previous[key])))
        stage8_parity = {key: float(max(values, default=0.0)) for key, values in diffs.items()}
        if any(value > 1e-3 for value in stage8_parity.values()):
            raise RuntimeError(f"Recomputed Stage 8 confidence differs by >1e-3: {stage8_parity}")
    manifest = exploratory_header({"dry_run": bool(args.dry_run), "label_source": "v4_archive_DRYRUN" if args.dry_run else "canonical_v1_batch1_judged",
        "shapes": {"suf": [n, 5, nl, d], "ans": [n, 2, nl, d]}, "dtypes": {"suf": "float16", "ans": "float16"},
        "sha256": {"suf.npy": sha256_file(suffix_path), "ans.npy": sha256_file(answer_path),
                   "meta.parquet": sha256_file(meta_path)}, "parity": parity,
        "stage8_confidence_parity_max_abs_diff": stage8_parity,
        "greedy_consistency_rate": greedy_rate,
        "row_order": frame["popqa_id"].astype(str).tolist(), "verified_input_hash_count": len(frame)}, ROOT)
    if args.dry_run:
        manifest["watermark"] = "DRY RUN – v4 labels – not reportable"
    (out_dir / "features_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.push_hf:
        repo = os.environ.get("HF_TENSOR_REPO_ID")
        if not repo:
            raise RuntimeError("--push-hf requires HF_TENSOR_REPO_ID")
        from huggingface_hub import upload_folder
        upload_folder(repo_id=repo, folder_path=str(out_dir), path_in_repo="exploratory/e1_features",
                      token=os.environ.get("HF_TOKEN"), ignore_patterns=["e1_extract_checkpoint.pt"])
    print(f"Wrote E1 features to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
