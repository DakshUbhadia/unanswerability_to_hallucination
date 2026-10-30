#!/usr/bin/env python3
"""Capture E4 suffix residuals and first-token baselines (owner-run GPU job)."""
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
import torch

from src.config import CFG
from src.eval_common import load_runtime_contract, load_stage_model, sha256_file
from src.exploratory import assert_not_fresh, exploratory_header
from src.exploratory.capture import capture_suffix_all_layers, make_condition_contract
from src.tokenization_contract import canonical_input_ids, chat_template_sha256, input_ids_sha256

DATA = ROOT / "data/exploratory/e4"
OUT = ROOT / "results/exploratory/e4_features"


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompts(source, rows):
    prompts = []
    for row in rows:
        if source == "syn":
            prompts.append(row["user_prompt"])
        elif source == "squad":
            prompts.append(f"Passage: {row['context']}\n\nQuestion: {row['question']}")
        elif source in {"selfaware", "triviaqa", "popqa"}:
            prompts.append(str(row.get("user_prompt", row.get("question", ""))))
    return prompts


def _metadata(source, rows, prompts, tokenizer, contract, template_digest):
    meta = []
    for row, prompt in zip(rows, prompts):
        ids = canonical_input_ids(tokenizer, prompt, contract["system_prompt_used"], contract["suffix_token_ids"])
        if source == "syn":
            group = row["pair_id"]
            item = {"y": int(row["y"]), "split": row["split"], "group": group, "pair_id": group}
        elif source == "squad":
            item = {"y": int(row["y"]), "group": str(row["title"]), "source_id": str(row["id"]),
                    "title": str(row["title"])}
        elif source == "selfaware":
            item = {"y": int(row["y"]), "group": str(row["question_id"]), "question_id": str(row["question_id"]),
                    "source": str(row["source"])}
        else:
            item = {"y": int(row["y"]), "group": str(row.get("group", row.get("question_id", row.get("popqa_id")))),
                    "source_id": str(row.get("source_id", row.get("question_id", row.get("popqa_id", ""))))}
        item.update({"user_prompt": prompt, "input_ids_sha256": input_ids_sha256(ids, contract["model_name"], template_digest),
                     "n_input_tokens": len(ids)})
        for key in ("log_odds", "maxp", "entropy"):
            if key in row:
                item[key] = float(row[key])
        meta.append(item)
    return pd.DataFrame(meta)


def _condition_contract(condition, base):
    if condition == "registered":
        return base, None
    preflight_path = ROOT / "results/exploratory/e3_preflight.json"
    if not preflight_path.exists():
        raise RuntimeError("Neutral E4 capture requires the E3 preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    date = preflight.get("template_date")
    if date not in {"static", "pinned"}:
        raise RuntimeError("E3 preflight did not establish a static or pinned date; neutral capture stopped")
    date_string = preflight.get("template_date_value") if date == "pinned" else None
    cond = make_condition_contract(base, model_name=base["model_name"],
        system_prompt=CFG.exploratory_neutral_system_prompt, suffix_ids=base["suffix_token_ids"],
        n_layers=base["n_layers"], d_model=base["d_model"],
        is_layer_output_tuple=base["is_layer_output_tuple"], unknown_token_ids=base["unknown_token_ids"],
        date_string=date_string)
    return cond, {"preflight_sha256": sha256_file(preflight_path), "template_date": date,
                  "date_string": date_string}


def _load_popqa(condition):
    if condition == "registered":
        folder = assert_not_fresh(ROOT / "results/exploratory/e1_features")
        meta_path, acts_path = folder / "meta.parquet", folder / "suf.npy"
        if not meta_path.exists() or not acts_path.exists():
            raise FileNotFoundError(f"PopQA registered E1 feature cache unavailable: {folder}")
        return pd.read_parquet(meta_path), np.load(acts_path, mmap_mode="r"), acts_path
    from src.exploratory.e1_data import load_e1_frame
    slug = "llama32_3b"
    folder = assert_not_fresh(ROOT / "results/exploratory/e3_features" / f"{slug}_neut")
    acts_path = folder / "popqa_suf.npy"
    confidence_path = folder / "popqa_confidence.npz"
    rows_path = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    if not acts_path.exists() or not confidence_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"PopQA neutral E3 feature cache unavailable: {folder}")
    labels = load_e1_frame().set_index("popqa_id")
    rows = _read_jsonl(rows_path)
    ids = [str(row["popqa_id"]) for row in rows]
    meta = labels.loc[ids].reset_index()
    conf = np.load(confidence_path)
    for key, name in (("maxp", "maxp"), ("entropy", "entropy"),
                      ("abstention_set_log_odds", "log_odds")):
        meta[name] = conf[key]
    meta["user_prompt"] = [str(row["question"]) for row in rows]
    matched = meta.get("matched_pair_id", pd.Series("", index=meta.index)).fillna("").astype(str)
    meta["group"] = [str(pair if pair else pid) for pair, pid in zip(matched, ids)]
    return meta, np.load(acts_path, mmap_mode="r"), acts_path


def _check_syn_parity(meta, model, tokenizer, contract, prompts, batch_size):
    from src.eval_common import ensure_stage4_directions
    from src.exploratory.capture import capture_suffix_all_layers
    from src.exploratory.capture import make_condition_contract
    dirs, _ = ensure_stage4_directions(CFG.stage4_out_path, CFG.stage4_manifest_path, CFG.stage0_out_path)
    train = meta["split"].astype(str).to_numpy() == "train"
    y = meta.loc[train, "y"].to_numpy(dtype=int)
    train_contract = make_condition_contract(contract, model_name=contract["model_name"],
        system_prompt=contract["system_prompt_used"], suffix_ids=contract["suffix_token_ids"],
        n_layers=contract["n_layers"], d_model=contract["d_model"],
        is_layer_output_tuple=contract["is_layer_output_tuple"],
        unknown_token_ids=contract["unknown_token_ids"], date_string=CFG.chat_template_date_stage3_4)
    train_capture = capture_suffix_all_layers(model, tokenizer, train_contract,
        [p for p, keep in zip(prompts, train) if keep], (-5, -4, -3, -2, -1), batch_size)
    x = np.asarray(train_capture["resid"][:, 1, 11], dtype=np.float64)
    observed = x[y == 1].mean(0) - x[y == 0].mean(0)
    expected = dirs["diff_in_means"][11, 1].detach().cpu().numpy().astype(np.float64)
    cosine = float(observed @ expected / (np.linalg.norm(observed) * np.linalg.norm(expected)))
    rel = float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))
    if cosine < 0.9999 or rel > 1e-3:
        raise RuntimeError(f"E4 syn Stage 4 parity failed: cosine={cosine:.8f}, relative norm error={rel:.6g}")
    val = meta["split"].astype(str).to_numpy() == "val"
    yv = meta.loc[val, "y"].to_numpy(dtype=int)
    val_contract = make_condition_contract(contract, model_name=contract["model_name"],
        system_prompt=contract["system_prompt_used"], suffix_ids=contract["suffix_token_ids"],
        n_layers=contract["n_layers"], d_model=contract["d_model"],
        is_layer_output_tuple=contract["is_layer_output_tuple"],
        unknown_token_ids=contract["unknown_token_ids"], date_string=CFG.chat_template_date_stage5_10)
    val_capture = capture_suffix_all_layers(model, tokenizer, val_contract,
        [p for p, keep in zip(prompts, val) if keep], (-5, -4, -3, -2, -1), batch_size)
    logodds = np.asarray(val_capture["log_odds"], dtype=float)
    means = {"unans": float(logodds[yv == 1].mean()), "ans": float(logodds[yv == 0].mean())}
    selection = json.loads(CFG.stage5_out_path.read_text(encoding="utf-8"))
    expected_means = {"unans": float(selection["val_unans_baseline_log_odds"]),
                      "ans": float(selection["val_ans_baseline_log_odds"])}
    diffs = {k: abs(means[k] - expected_means[k]) for k in means}
    if max(diffs.values()) > 0.05:
        raise RuntimeError(f"E4 syn validation log-odds parity failed: got={means}, expected={expected_means}")
    return {"direction_cosine": cosine, "relative_norm_error": rel, "val_log_odds_mean": means,
            "val_log_odds_expected": expected_means, "val_log_odds_abs_error": diffs, "passed": True}


def extract(sources, condition="registered", batch_size=16, data_dir=DATA, out_root=OUT):
    for source in sources:
        assert_not_fresh(source)
    needs_capture = any(source != "popqa" for source in sources)
    if needs_capture:
        base = load_runtime_contract()
        contract, condition_info = _condition_contract(condition, base)
        model, tokenizer, _ = load_stage_model(contract, "cuda:0", "auto")
        template_digest = chat_template_sha256(tokenizer)
    else:
        contract, condition_info, model, tokenizer, template_digest = None, None, None, None, None
    output = assert_not_fresh(out_root / condition)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for source in sources:
        target = assert_not_fresh(output / source)
        target.mkdir(parents=True, exist_ok=True)
        if source == "popqa":
            meta, acts, cache_path = _load_popqa(condition)
            if condition == "registered":
                eligible = meta.get("use_in_e1", pd.Series(False, index=meta.index)).fillna(False).astype(bool).to_numpy()
                if "baseline_partition" in meta:
                    eligible |= meta["baseline_partition"].astype(str).isin(("correct", "wrong")).to_numpy()
                ix = np.flatnonzero(eligible)
                meta = meta.iloc[ix].copy().reset_index(drop=True)
                acts = acts[ix]
                if "y" not in meta or meta["y"].isna().any():
                    meta["y"] = (meta["baseline_partition"].astype(str) == "wrong").astype(int)
                matched = meta.get("matched_pair_id", pd.Series(None, index=meta.index)).fillna("").astype(str)
                pids = meta.get("popqa_id", pd.Series(np.arange(len(meta)), index=meta.index)).astype(str)
                meta["group"] = np.where(matched.ne("") & matched.ne("None"), matched, pids)
                meta["user_prompt"] = meta["question"].astype(str)
            meta_path = target / "meta.parquet"
            meta.to_parquet(meta_path, index=False)
            manifest = exploratory_header({"source": source, "condition": condition,
                "suf_path": str(cache_path.relative_to(ROOT)), "suf_sha256": sha256_file(cache_path),
                "meta_sha256": sha256_file(meta_path), "n_rows": len(meta),
                "preregistration_sha256": sha256_file(ROOT / "results/exploratory/e4_preregistration.json")}, ROOT)
            (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            manifests[source] = {"manifest_sha256": sha256_file(target / "manifest.json"),
                                 "cache_path": str(cache_path.relative_to(ROOT))}
            continue
        data_path = assert_not_fresh(data_dir / f"{source}.jsonl")
        if not data_path.exists():
            raise FileNotFoundError(f"Prepare E4 source data first: {data_path}")
        rows = _read_jsonl(data_path)
        prompts = _prompts(source, rows)
        meta = _metadata(source, rows, prompts, tokenizer, contract, template_digest)
        acts = np.lib.format.open_memmap(target / "suf.npy", mode="w+", dtype=np.float16,
                                         shape=(len(rows), 5, int(contract["n_layers"]), int(contract["d_model"])))
        scalar = {k: np.full(len(rows), np.nan, dtype=np.float32) for k in ("log_odds", "maxp", "entropy")}
        for start in range(0, len(rows), 64):
            stop = min(len(rows), start + 64)
            captured = capture_suffix_all_layers(model, tokenizer, contract, prompts[start:stop],
                                                 (-5, -4, -3, -2, -1), batch_size)
            acts[start:stop] = captured["resid"]
            for key in scalar:
                scalar[key][start:stop] = captured[key]
            print(f"[{source}] captured {stop}/{len(rows)}", flush=True)
        for key in scalar:
            meta[key] = scalar[key]
        meta_path = target / "meta.parquet"
        meta.to_parquet(meta_path, index=False)
        acts.flush()
        parity = _check_syn_parity(meta, model, tokenizer, contract, prompts, batch_size) if source == "syn" else None
        input_hash = hashlib.sha256("".join(meta["input_ids_sha256"].astype(str)).encode()).hexdigest()
        manifest = exploratory_header({"source": source, "condition": condition, "model_name": contract["model_name"],
            "system_prompt": contract["system_prompt_used"], "condition_info": condition_info,
            "n_rows": len(meta), "shape": list(acts.shape), "dtype": "float16",
            "data_sha256": sha256_file(data_path), "meta_sha256": sha256_file(meta_path),
            "suf_sha256": sha256_file(target / "suf.npy"), "input_hashes_sha256": input_hash,
            "preregistration_sha256": sha256_file(ROOT / "results/exploratory/e4_preregistration.json"),
            "chat_template_sha256": template_digest, "parity": parity}, ROOT)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifests[source] = {"manifest_sha256": sha256_file(target / "manifest.json"), "parity": parity}
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="syn,squad,selfaware")
    parser.add_argument("--condition", choices=("registered", "neutral"), default="registered")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--push-hf", action="store_true", help="Upload extracted tensors to HF dataset storage")
    args = parser.parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    result = extract(sources, args.condition, args.batch_size)
    if args.push_hf:
        from src.exploratory.tensor_io import upload_folder
        for source in sources:
            folder = OUT / args.condition / source
            if folder.exists():
                upload_folder(folder, path_in_repo=f"exploratory/e4_features/{args.condition}/{source}",
                              ignore_patterns=["*checkpoint*", "*.pt"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
