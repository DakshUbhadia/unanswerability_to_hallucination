#!/usr/bin/env python3
"""Run the Stage 3 training activation cache, with verified-output resume support."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_cache import (load_and_validate_train_pairs, run_training_activation_caching,
                                  save_stage3_artifacts, verify_stage3_outputs)
from src.config import CFG
from src.model_utils import resolve_runtime_dtype
from src.stage_contract import load_stage2_model_contract


def _contract(stage0_path: Path, stage2_path: Path) -> dict:
    with Path(stage0_path).open(encoding="utf-8") as f:
        stage0 = json.load(f)
    stage2 = load_stage2_model_contract(stage2_path)
    if stage0.get("passed_all_checks") is not True:
        raise RuntimeError(f"Stage 0 gate did not pass in {stage0_path}")
    if stage0.get("model_name") != stage2.get("model_name"):
        raise RuntimeError("Stage 0 and Stage 2 model names differ")
    keys = ("n_layers", "d_model", "k_suffix", "suffix_token_ids", "is_layer_output_tuple")
    result = {key: stage0[key] for key in keys}
    result.update(model_name=stage2["model_name"], system_prompt_used=stage2["system_prompt_used"],
                  dtype=stage0.get("dtype", "torch.float16"))
    if len(result["suffix_token_ids"]) != int(result["k_suffix"]):
        raise RuntimeError("Stage 0 suffix_token_ids length does not match k_suffix")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size_forward)
    parser.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    parser.add_argument("--splits-path", type=Path, default=CFG.data_dir / "splits.json")
    parser.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    parser.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    parser.add_argument("--output-path", type=Path, default=CFG.stage3_out_path)
    parser.add_argument("--manifest-path", type=Path, default=CFG.stage3_manifest_path)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.force and args.output_path.exists() and args.manifest_path.exists():
        try:
            with args.manifest_path.open(encoding="utf-8") as f:
                existing_manifest = json.load(f)
            verify_stage3_outputs(args.output_path, args.manifest_path,
                                  expected_n_pairs=int(existing_manifest.get("n_train_pairs", CFG.n_pairs_train)))
        except (AssertionError, KeyError, ValueError, RuntimeError):
            pass
        else:
            print(f"[SKIP] Stage 3 already verified in {args.output_path}")
            return 0

    contract = _contract(args.stage0_path, args.stage2_path)
    pairs = load_and_validate_train_pairs(args.pairs_path, args.splits_path)
    runtime_dtype, dtype_name, _ = resolve_runtime_dtype(args.dtype, args.device)
    print(f"[Stage 3] Loading {contract['model_name']} with {dtype_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(contract["model_name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_model = AutoModelForCausalLM.from_pretrained(
        contract["model_name"], torch_dtype=runtime_dtype, device_map=args.device, low_cpu_mem_usage=True)
    raw_model.eval()
    llm = LanguageModel(raw_model, tokenizer=tokenizer, dispatch=True)
    cache, manifest = run_training_activation_caching(llm, contract, pairs, args.batch_size)
    manifest["runtime_dtype"] = dtype_name.replace("torch.", "")
    save_stage3_artifacts(cache, manifest, args.output_path, args.manifest_path)
    verify_stage3_outputs(args.output_path, args.manifest_path,
                          expected_n_pairs=len(pairs), expected_n_layers=int(contract["n_layers"]),
                          expected_k_suffix=int(contract["k_suffix"]), expected_d_model=int(contract["d_model"]))
    print(f"[Stage 3] Verified cache at {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
