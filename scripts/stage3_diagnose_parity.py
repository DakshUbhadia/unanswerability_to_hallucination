#!/usr/bin/env python3
"""Diagnose cached Stage 3 batch-vs-single activation differences without rebuilding the cache."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.stage3_cache_activations import _contract
from src.activation_cache import (
    _single_prompt_inputs,
    extract_batch_post_instruction_activations,
    load_and_validate_train_pairs,
    verify_stage3_outputs,
)
from src.config import CFG
from src.model_utils import resolve_runtime_dtype


def _half_ordered_int(value: float) -> int:
    """Map finite float16 bit patterns to monotone integers for ULP distance."""
    bits = int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])
    return ((~bits) & 0xFFFF) if bits & 0x8000 else (bits | 0x8000)


def diagnose(batched: torch.Tensor, single: torch.Tensor, k_suffix: int,
             top_n: int = 20) -> dict[str, Any]:
    if batched.shape != single.shape or batched.ndim != 3:
        raise ValueError(f"Expected matching [layers, suffix, d_model] tensors; got {batched.shape}, {single.shape}")
    batched = batched.detach().cpu().to(torch.float32)
    single = single.detach().cpu().to(torch.float32)
    if not torch.isfinite(batched).all() or not torch.isfinite(single).all():
        raise ValueError("Parity diagnostic received non-finite activations")

    delta = batched - single
    abs_delta = delta.abs()
    allowed = CFG.stage3_allclose_atol + CFG.stage3_allclose_rtol * single.abs()
    close = abs_delta <= allowed
    ratio = abs_delta / allowed.clamp_min(CFG.eps_norm)
    vector_ref_norm = torch.linalg.vector_norm(single, dim=-1).clamp_min(CFG.eps_norm)
    vector_rel_l2 = torch.linalg.vector_norm(delta, dim=-1) / vector_ref_norm
    min_cosine = float(F.cosine_similarity(batched, single, dim=-1).min().item())

    mismatch_indices = torch.nonzero(~close, as_tuple=False)
    entries = []
    if mismatch_indices.numel():
        mismatch_ratios = ratio[~close]
        order = torch.argsort(mismatch_ratios, descending=True)[:top_n]
        selected = mismatch_indices[order]
        for idx in selected.tolist():
            layer, suffix_pos, dim = idx
            ref_value = float(single[layer, suffix_pos, dim].item())
            batch_value = float(batched[layer, suffix_pos, dim].item())
            entries.append({
                "layer": layer,
                "suffix_position": suffix_pos - k_suffix,
                "dimension": dim,
                "single_value": ref_value,
                "batched_value": batch_value,
                "abs_diff": float(abs_delta[layer, suffix_pos, dim].item()),
                "allclose_allowed_abs_diff": float(allowed[layer, suffix_pos, dim].item()),
                "allclose_violation_ratio": float(ratio[layer, suffix_pos, dim].item()),
                "float16_ulp_distance": abs(_half_ordered_int(ref_value) - _half_ordered_int(batch_value)),
            })

    return {
        "allclose_rtol": CFG.stage3_allclose_rtol,
        "allclose_atol": CFG.stage3_allclose_atol,
        "allclose_passed": bool(close.all().item()),
        "allclose_mismatch_count": int((~close).sum().item()),
        "coordinate_count": int(close.numel()),
        "allclose_mismatch_fraction": float((~close).float().mean().item()),
        "min_cosine": min_cosine,
        "max_relative_l2": float(vector_rel_l2.max().item()),
        "max_absolute_difference": float(abs_delta.max().item()),
        "max_allclose_violation_ratio": float(ratio.max().item()),
        "worst_mismatches": entries,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", type=Path, default=CFG.stage3_out_path)
    parser.add_argument("--manifest-path", type=Path, default=CFG.stage3_manifest_path)
    parser.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    parser.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    parser.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    parser.add_argument("--splits-path", type=Path, default=CFG.data_dir / "splits.json")
    parser.add_argument("--output-path", type=Path, default=CFG.results_dir / "stage3_parity_diagnostic.json")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be positive")

    verify_stage3_outputs(args.cache_path, args.manifest_path)
    cache = torch.load(args.cache_path, map_location="cpu", weights_only=False)
    pairs = load_and_validate_train_pairs(args.pairs_path, args.splits_path)
    if cache["pair_ids"][0] != pairs[0]["pair_id"]:
        raise RuntimeError("Cached first pair does not match sorted training pair data")

    contract = _contract(args.stage0_path, args.stage2_path)
    runtime_dtype, dtype_name, _ = resolve_runtime_dtype(args.dtype, args.device)
    if dtype_name.replace("torch.", "") != "float16":
        raise RuntimeError(f"This diagnostic expects the recorded float16 run; resolved {dtype_name}")
    print(f"[Parity diagnostic] Loading {contract['model_name']} with {dtype_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(contract["model_name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_model = AutoModelForCausalLM.from_pretrained(
        contract["model_name"], torch_dtype=runtime_dtype, device_map=args.device, low_cpu_mem_usage=True)
    raw_model.eval()
    llm = LanguageModel(raw_model, tokenizer=tokenizer, dispatch=True)

    device = str(next(llm._model.parameters()).device)
    inputs = _single_prompt_inputs(tokenizer, pairs[0]["ans_prompt"],
                                   contract["system_prompt_used"], device)
    suffix_ids = [int(x) for x in contract["suffix_token_ids"]]
    actual_suffix = inputs["input_ids"][0, -len(suffix_ids):].detach().cpu().tolist()
    if actual_suffix != suffix_ids:
        raise RuntimeError(f"Single-prompt suffix mismatch: {actual_suffix} != {suffix_ids}")
    single = extract_batch_post_instruction_activations(
        llm, inputs, int(contract["n_layers"]), int(contract["k_suffix"]),
        bool(contract["is_layer_output_tuple"]))[0]

    report = {
        "stage": 3,
        "diagnostic_only": True,
        "decision": "review_required",
        "model_name": contract["model_name"],
        "runtime_dtype": dtype_name.replace("torch.", ""),
        "pair_id": pairs[0]["pair_id"],
        "cache_sha256": json.loads(args.manifest_path.read_text(encoding="utf-8"))["cache_sha256"],
        **diagnose(cache["train_pair_acts_ans"][0], single, int(contract["k_suffix"]), args.top_n),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"[Parity diagnostic] Wrote {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
