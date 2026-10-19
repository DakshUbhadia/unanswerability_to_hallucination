"""Stage 3: cache post-instruction residual activations for training pairs."""
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.config import CFG
from src.model_utils import read_layer_residual_proxy


CACHE_KEYS = {
    "train_ans", "train_unans", "train_pair_acts_ans", "train_pair_acts_unans",
    "pair_ids", "template_ids", "families", "model_name", "system_prompt_used",
    "n_layers", "k_suffix", "d_model", "suffix_indices", "suffix_token_ids",
}
MANIFEST_KEYS = {
    "stage", "model_name", "runtime_dtype", "cache_dtype", "n_train_pairs",
    "n_family_a_pairs", "n_family_b_pairs", "n_layers", "k_suffix", "d_model",
    "suffix_indices", "suffix_token_ids", "train_ans_shape", "train_pair_acts_shape",
    "batch_vs_single_min_cosine", "batch_vs_single_max_rel_l2_diff",
    "batch_vs_single_max_abs_diff", "min_prompt_std_ans",
    "min_prompt_std_unans", "min_diff_in_means_norm", "max_diff_in_means_norm",
    "cache_file_size_bytes", "cache_sha256", "elapsed_sec", "passed_acceptance_gate",
}


def load_and_validate_train_pairs(pairs_path: Path = CFG.data_dir / "pairs.jsonl",
                                  splits_path: Path = CFG.data_dir / "splits.json") -> List[Dict[str, Any]]:
    with Path(splits_path).open(encoding="utf-8") as f:
        splits = json.load(f)
    train_templates = set(splits["train_template_ids"])
    with Path(pairs_path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pairs = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["pair_id"])
    expected = 4 * len(train_templates)
    assert len(pairs) == expected == CFG.n_pairs_train, f"Expected {expected} training pairs, found {len(pairs)}"
    ids = [p["pair_id"] for p in pairs]
    assert len(set(ids)) == len(ids), "Training pair_id values must be unique"
    assert all(p["template_id"] in train_templates for p in pairs), "Training pair has a non-training template"
    families = [p["family"] for p in pairs]
    assert families.count("A") == families.count("B") == len(pairs) // 2, "Training families must be balanced"
    assert all(k in p for p in pairs for k in ("ans_prompt", "unans_prompt")), "Pair is missing a prompt"
    return pairs


def format_and_verify_batch_inputs(tokenizer: Any, user_prompts: List[str], system_prompt: str,
                                   expected_suffix_token_ids: List[int], device: str) -> Dict[str, torch.Tensor]:
    assert tokenizer.padding_side == "left", "Stage 3 requires left padding"
    conversations = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}]
                     for p in user_prompts]
    formatted = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in conversations]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, add_special_tokens=False)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    k = len(expected_suffix_token_ids)
    assert k > 0 and inputs["input_ids"].shape[1] > k, "Input sequence is shorter than suffix"
    expected = torch.tensor(expected_suffix_token_ids, dtype=inputs["input_ids"].dtype)
    tails = inputs["input_ids"][:, -k:].detach().cpu()
    assert torch.equal(tails, expected.expand(tails.shape[0], -1)), "Post-instruction suffix token IDs mismatch"
    assert bool((inputs["attention_mask"][:, -k:] == 1).all()), "Suffix columns include padding"
    return inputs


@torch.no_grad()
def extract_batch_post_instruction_activations(llm: Any, inputs: Dict[str, torch.Tensor], n_layers: int,
                                               k_suffix: int, is_layer_output_tuple: bool) -> torch.Tensor:
    saved_layer_proxies = []
    with llm.trace(inputs):
        for l_idx in range(n_layers):
            resid = read_layer_residual_proxy(llm.model.layers[l_idx], is_layer_output_tuple)
            saved_layer_proxies.append(resid[:, -k_suffix:, :].save())
    layers = [p.value.detach().to(device="cpu", dtype=torch.float32) for p in saved_layer_proxies]
    result = torch.stack(layers, dim=1).contiguous()
    assert torch.isfinite(result).all(), "Non-finite activation in batch"
    del saved_layer_proxies, layers
    return result


def _single_prompt_inputs(tokenizer: Any, prompt: str, system_prompt: str, device: str) -> Dict[str, torch.Tensor]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    encoded = tokenizer([text], return_tensors="pt", padding=False, add_special_tokens=False)
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {k: v.to(device) for k, v in encoded.items()}


def _check_batch_single_parity(batched_first: torch.Tensor, single_first: torch.Tensor,
                               runtime_dtype_str: str) -> Tuple[float, float, float]:
    min_cos = float(F.cosine_similarity(batched_first, single_first, dim=-1).min().item())
    diff_norm = torch.norm(batched_first - single_first, p=2, dim=-1)
    ref_norm = torch.norm(single_first, p=2, dim=-1).clamp(min=CFG.eps_norm)
    max_rel_l2 = float((diff_norm / ref_norm).max().item())
    max_diff = float(torch.max(torch.abs(batched_first - single_first)).item())
    max_abs_limit = (CFG.stage3_max_batch_single_abs_fp16 if runtime_dtype_str == "float16"
                     else CFG.stage3_max_batch_single_abs_fp32)
    assert min_cos >= CFG.stage3_min_batch_single_cosine, (
        f"Batch/single min cosine {min_cos:.6f} < {CFG.stage3_min_batch_single_cosine}")
    assert max_rel_l2 <= CFG.stage3_max_batch_single_rel_l2, (
        f"Batch/single max rel L2 diff {max_rel_l2:.6g} > {CFG.stage3_max_batch_single_rel_l2}")
    # Relative L2, cosine, and dtype-specific absolute checks gate parity.
    assert max_diff <= max_abs_limit, f"Batch/single max abs diff {max_diff:.6g} > {max_abs_limit}"
    return min_cos, max_rel_l2, max_diff


def run_training_activation_caching(llm: Any, contract: Dict[str, Any], train_pairs: List[Dict[str, Any]],
                                    batch_size: int = CFG.batch_size_forward) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.perf_counter()
    n_layers, d_model = int(contract["n_layers"]), int(contract["d_model"])
    k = int(contract["k_suffix"])
    suffix_ids = [int(x) for x in contract["suffix_token_ids"]]
    assert len(train_pairs) == CFG.n_pairs_train and len(suffix_ids) == k
    tokenizer = llm.tokenizer
    tokenizer.padding_side = "left"
    ans = [p["ans_prompt"] for p in train_pairs]
    unans = [p["unans_prompt"] for p in train_pairs]
    single_inputs = _single_prompt_inputs(tokenizer, ans[0], contract["system_prompt_used"],
                                         str(next(llm._model.parameters()).device))
    assert single_inputs["input_ids"].shape[1] > k
    assert single_inputs["input_ids"][0, -k:].detach().cpu().tolist() == suffix_ids
    single_ref = extract_batch_post_instruction_activations(
        llm, single_inputs, n_layers, k, bool(contract["is_layer_output_tuple"]))
    expected_shape = (len(train_pairs), n_layers, k, d_model)
    ans_acts = torch.empty(expected_shape, dtype=torch.float32, device="cpu")
    unans_acts = torch.empty_like(ans_acts)
    device = str(single_inputs["input_ids"].device)
    for prompts, dest in ((ans, ans_acts), (unans, unans_acts)):
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            inputs = format_and_verify_batch_inputs(tokenizer, prompts[start:end], contract["system_prompt_used"], suffix_ids, device)
            batch = extract_batch_post_instruction_activations(llm, inputs, n_layers, k,
                                                               bool(contract["is_layer_output_tuple"]))
            assert batch.shape == (end - start, n_layers, k, d_model)
            dest[start:end].copy_(batch)
            del batch, inputs
            if torch.cuda.is_available() and (start // batch_size + 1) % 4 == 0:
                torch.cuda.empty_cache()

    runtime_dtype_str = str(contract.get("dtype", "torch.float16")).replace("torch.", "")
    min_cos, max_rel_l2, max_diff = _check_batch_single_parity(
        ans_acts[0], single_ref[0], runtime_dtype_str)
    train_ans = ans_acts.to(torch.float64).mean(dim=0).to(torch.float32)
    train_unans = unans_acts.to(torch.float64).mean(dim=0).to(torch.float32)
    diff_norms = torch.linalg.vector_norm(train_unans - train_ans, dim=-1)
    std_ans = float(ans_acts.std(dim=0).min().item())
    std_unans = float(unans_acts.std(dim=0).min().item())
    cache = {
        "train_ans": train_ans, "train_unans": train_unans,
        "train_pair_acts_ans": ans_acts, "train_pair_acts_unans": unans_acts,
        "pair_ids": [p["pair_id"] for p in train_pairs],
        "template_ids": [p["template_id"] for p in train_pairs],
        "families": [p["family"] for p in train_pairs],
        "model_name": contract["model_name"], "system_prompt_used": contract["system_prompt_used"],
        "n_layers": n_layers, "k_suffix": k, "d_model": d_model,
        "suffix_indices": list(range(-k, 0)), "suffix_token_ids": suffix_ids,
    }
    manifest = {
        "stage": 3, "model_name": contract["model_name"],
        "runtime_dtype": str(contract.get("dtype", "torch.float16")).replace("torch.", ""),
        "cache_dtype": "float32", "n_train_pairs": len(train_pairs),
        "n_family_a_pairs": cache["families"].count("A"), "n_family_b_pairs": cache["families"].count("B"),
        "n_layers": n_layers, "k_suffix": k, "d_model": d_model,
        "suffix_indices": list(range(-k, 0)), "suffix_token_ids": suffix_ids,
        "train_ans_shape": list(train_ans.shape), "train_pair_acts_shape": list(ans_acts.shape),
        "batch_vs_single_min_cosine": min_cos,
        "batch_vs_single_max_rel_l2_diff": max_rel_l2,
        "batch_vs_single_max_abs_diff": max_diff,
        "min_prompt_std_ans": std_ans, "min_prompt_std_unans": std_unans,
        "min_diff_in_means_norm": float(diff_norms.min().item()),
        "max_diff_in_means_norm": float(diff_norms.max().item()),
        "elapsed_sec": time.perf_counter() - started, "passed_acceptance_gate": True,
    }
    return cache, manifest


def save_stage3_artifacts(cache_dict: Dict[str, Any], manifest_dict: Dict[str, Any],
                          cache_path: Path = CFG.stage3_out_path,
                          manifest_path: Path = CFG.stage3_manifest_path) -> None:
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(cache_dict, tmp_path)
    tmp_path.replace(cache_path)
    digest = hashlib.sha256()
    with cache_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest_dict["cache_file_size_bytes"] = cache_path.stat().st_size
    manifest_dict["cache_sha256"] = digest.hexdigest()
    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest_dict, f, indent=2, ensure_ascii=False)
        f.write("\n")


def verify_stage3_outputs(cache_path: Path = CFG.stage3_out_path,
                          manifest_path: Path = CFG.stage3_manifest_path,
                          expected_n_pairs: int = CFG.n_pairs_train,
                          expected_n_layers: Optional[int] = None,
                          expected_k_suffix: Optional[int] = None,
                          expected_d_model: Optional[int] = None) -> Dict[str, Any]:
    cache_path, manifest_path = Path(cache_path), Path(manifest_path)
    assert cache_path.is_file() and manifest_path.is_file(), "Stage 3 cache or manifest is missing"
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    missing = CACHE_KEYS - cache.keys()
    assert not missing, f"Cache is missing keys: {sorted(missing)}"
    missing = MANIFEST_KEYS - manifest.keys()
    assert not missing, f"Manifest is missing keys: {sorted(missing)}"
    n, l, k, d = expected_n_pairs, int(cache["n_layers"]), int(cache["k_suffix"]), int(cache["d_model"])
    if expected_n_layers is not None: assert l == expected_n_layers
    if expected_k_suffix is not None: assert k == expected_k_suffix
    if expected_d_model is not None: assert d == expected_d_model
    assert len(cache["pair_ids"]) == len(cache["template_ids"]) == len(cache["families"]) == n
    assert cache["pair_ids"] == sorted(cache["pair_ids"])
    assert cache["families"].count("A") == cache["families"].count("B") == n // 2
    tensors = {name: cache[name] for name in ("train_ans", "train_unans", "train_pair_acts_ans", "train_pair_acts_unans")}
    shapes = {"train_ans": (l, k, d), "train_unans": (l, k, d),
              "train_pair_acts_ans": (n, l, k, d), "train_pair_acts_unans": (n, l, k, d)}
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor) and tensor.shape == shapes[name], f"Invalid shape for {name}: {getattr(tensor, 'shape', None)}"
        assert tensor.dtype == torch.float32, f"{name} must be float32"
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or Inf"
    assert torch.allclose(tensors["train_pair_acts_ans"].mean(0), tensors["train_ans"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(tensors["train_pair_acts_unans"].mean(0), tensors["train_unans"], atol=1e-5, rtol=1e-5)
    min_std_a = float(tensors["train_pair_acts_ans"].std(0).min().item())
    min_std_u = float(tensors["train_pair_acts_unans"].std(0).min().item())
    assert min_std_a > 1e-4 and min_std_u > 1e-4, "Prompt activation variance is too small"
    norms = torch.linalg.vector_norm(tensors["train_unans"] - tensors["train_ans"], dim=-1)
    assert float(norms.min()) > 1e-3, "At least one mean contrast is near zero"
    assert manifest["passed_acceptance_gate"] is True
    assert manifest["n_train_pairs"] == n and manifest["train_ans_shape"] == list(shapes["train_ans"])
    assert manifest["train_pair_acts_shape"] == list(shapes["train_pair_acts_ans"])
    assert manifest["min_prompt_std_ans"] > 1e-4 and manifest["min_prompt_std_unans"] > 1e-4
    assert manifest["min_diff_in_means_norm"] > 1e-3
    assert manifest["batch_vs_single_min_cosine"] >= CFG.stage3_min_batch_single_cosine
    assert manifest.get("batch_vs_single_max_rel_l2_diff", 0.0) <= CFG.stage3_max_batch_single_rel_l2
    max_abs_limit = (CFG.stage3_max_batch_single_abs_fp16 if manifest.get("runtime_dtype") == "float16"
                     else CFG.stage3_max_batch_single_abs_fp32)
    assert manifest["batch_vs_single_max_abs_diff"] <= max_abs_limit
    assert manifest["cache_file_size_bytes"] == cache_path.stat().st_size
    digest = hashlib.sha256()
    with cache_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    assert manifest["cache_sha256"] == digest.hexdigest(), "Cache SHA-256 mismatch"
    return manifest
