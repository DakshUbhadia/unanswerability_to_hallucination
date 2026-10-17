# src/model_utils.py
"""Stage 0: runtime contract checks, fp16-safe intervention operators, nnsight==0.3.7 wrappers."""
import gc
import json
import platform
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

from src.config import CONFIG


# ----------------------------- precision ------------------------------------
def resolve_runtime_dtype(
    requested_dtype: str = "auto", device: str = "cuda:0"
) -> Tuple[torch.dtype, str, bool]:
    """Returns (dtype, dtype_str, precision_deviation_dev001)."""
    if requested_dtype == "bfloat16":
        return torch.bfloat16, "torch.bfloat16", False
    if requested_dtype == "float16":
        return torch.float16, "torch.float16", True
    if requested_dtype == "float32":
        return torch.float32, "torch.float32", True
    if device.startswith("cuda") and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:  # Ampere+ has native bf16 (is_bf16_supported() lies on T4: emulation)
            return torch.bfloat16, "torch.bfloat16", False
        return torch.float16, "torch.float16", True  # Kaggle T4 (7.5) / P100 (6.0)
    return torch.float32, "torch.float32", True


# ----------------------------- native probes --------------------------------
def inspect_native_layer_output(
    raw_model: Any, tokenizer: Any, device: str
) -> Tuple[bool, Tuple[int, int, int]]:
    """Native forward hook on layers[0] OUTSIDE nnsight: tuple-vs-tensor + 3D shape."""
    rec: Dict[str, Any] = {}

    def _probe(module: Any, inputs: Any, output: Any) -> None:
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        rec["is_tuple"] = is_tuple
        rec["ndim"] = hidden.ndim
        rec["shape"] = tuple(hidden.shape)

    handle = raw_model.model.layers[0].register_forward_hook(_probe)
    try:
        enc = tokenizer([" Probe prompt one", " Probe prompt two with extra words"],
                        return_tensors="pt", padding=True).to(device)
        b, s = enc["input_ids"].shape
        with torch.no_grad():
            raw_model(**enc)
    finally:
        handle.remove()

    assert "is_tuple" in rec, "Native forward hook did not fire!"
    d_model = raw_model.config.hidden_size
    assert rec["ndim"] == 3 and rec["shape"] == (b, s, d_model), (
        f"Expected 3D hidden shape {(b, s, d_model)}, got {rec['shape']}"
    )
    return bool(rec["is_tuple"]), rec["shape"]


def verify_unknown_tokens(tokenizer: Any) -> Tuple[int, int]:
    ids_ns = tokenizer.encode("Unknown", add_special_tokens=False)
    assert len(ids_ns) == 1, f"'Unknown' must be exactly 1 token, got {len(ids_ns)}: {ids_ns}"
    ids_sp = tokenizer.encode(" Unknown", add_special_tokens=False)
    assert len(ids_sp) >= 1, "Failed to encode ' Unknown'"
    return int(ids_ns[0]), int(ids_sp[-1])


def _chat_ids(tokenizer: Any, user: str) -> List[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": CONFIG.system_prompt}, {"role": "user", "content": user}],
        tokenize=True, add_generation_prompt=True,
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):  # BatchEncoding in some versions
        ids = ids["input_ids"]
    return list(ids)


def discover_post_instruction_suffix(tokenizer: Any) -> Tuple[int, List[int]]:
    ids_a = _chat_ids(tokenizer, " AlphaUniqueMarker")
    ids_b = _chat_ids(tokenizer, " BetaUniqueMarkerWithDifferentLength")
    k = 0
    while k < min(len(ids_a), len(ids_b)) and ids_a[-(k + 1)] == ids_b[-(k + 1)]:
        k += 1
    assert k >= 1, "No common post-instruction suffix found across chat prompts!"
    return k, [int(x) for x in ids_a[-k:]]


# ----------------------------- operators ------------------------------------
# These run on BOTH concrete tensors and nnsight 0.3.7 proxies. For proxies we only use
# tensor *methods* and python operators (no torch.* calls, no proxy.dtype / proxy.device).
def apply_orthogonal_projection(
    x: Any,
    r_hat: torch.Tensor,
    eps: float = CONFIG.eps_norm,
    target_dtype: Optional[torch.dtype] = None,
) -> Any:
    """x' = x - (x . r_hat) r_hat, computed in float32, cast back to x's dtype."""
    if isinstance(x, torch.Tensor):
        orig_dtype = x.dtype
        r = r_hat.detach().to(device=x.device, dtype=torch.float32)
    else:  # nnsight proxy: r_hat must already live on the right device
        assert target_dtype is not None, "target_dtype required when x is an nnsight proxy"
        orig_dtype = target_dtype
        r = r_hat.detach().to(dtype=torch.float32)
    r_unit = r / (torch.norm(r, p=2) + eps)
    x32 = x.to(torch.float32)
    coeff = (x32 * r_unit).sum(dim=-1, keepdim=True)
    return (x32 - coeff * r_unit).to(orig_dtype)


def apply_norm_scaled_steering(
    x: Any,
    r_vec: torch.Tensor,
    alpha: float,
    target_dtype: Optional[torch.dtype] = None,
) -> Any:
    """x' = x + alpha * r_vec (alpha in units of ||r_vec||_2). Never short-circuits on r_vec == 0."""
    if isinstance(x, torch.Tensor):
        orig_dtype = x.dtype
        r = r_vec.detach().to(device=x.device, dtype=torch.float32)
    else:
        assert target_dtype is not None, "target_dtype required when x is an nnsight proxy"
        orig_dtype = target_dtype
        r = r_vec.detach().to(dtype=torch.float32)
    x32 = x.to(torch.float32)
    return (x32 + float(alpha) * r).to(orig_dtype)


# ----------------------------- nnsight helpers ------------------------------
def _deref_saved(p: Any) -> torch.Tensor:
    return p.value if hasattr(p, "value") else p


def _n_layers(llm: Any) -> int:
    return int(llm._model.config.num_hidden_layers)


def _model_dtype(llm: Any) -> torch.dtype:
    return next(llm._model.parameters()).dtype


def _read_layer_resid_envoy(layer_envoy: Any, is_tuple_output: bool) -> Any:
    return layer_envoy.output[0] if is_tuple_output else layer_envoy.output


def _write_layer_resid_envoy(layer_envoy: Any, new_resid: Any, is_tuple_output: bool) -> None:
    if is_tuple_output:
        layer_envoy.output[0][:] = new_resid
    else:
        layer_envoy.output[:] = new_resid


@torch.no_grad()
def forward_with_intervention(
    llm: LanguageModel,
    is_layer_output_tuple: bool,
    inputs: Any,
    mode: str = "none",
    r_vec: Optional[torch.Tensor] = None,
    target_layer: Optional[int] = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """Final-token logits [batch, vocab] under nnsight==0.3.7 with an optional intervention."""
    n_layers = _n_layers(llm)
    mdtype = _model_dtype(llm)
    dev = inputs["input_ids"].device
    r = None if r_vec is None else r_vec.detach().to(device=dev, dtype=torch.float32)

    with llm.trace(inputs):
        if mode == "ablate_all":
            assert r is not None, "r_vec required for mode='ablate_all'"
            for l_idx in range(n_layers):
                env = llm.model.layers[l_idx]
                resid = _read_layer_resid_envoy(env, is_layer_output_tuple)
                _write_layer_resid_envoy(
                    env, apply_orthogonal_projection(resid, r, target_dtype=mdtype),
                    is_layer_output_tuple)
        elif mode == "steer_layer":
            assert r is not None and target_layer is not None, "r_vec and target_layer required"
            env = llm.model.layers[target_layer]
            resid = _read_layer_resid_envoy(env, is_layer_output_tuple)
            _write_layer_resid_envoy(
                env, apply_norm_scaled_steering(resid, r, alpha=alpha, target_dtype=mdtype),
                is_layer_output_tuple)
        elif mode != "none":
            raise ValueError(f"Unsupported intervention mode: {mode}")
        saved_logits = llm.lm_head.output[:, -1, :].save()

    return _deref_saved(saved_logits).detach()


@torch.no_grad()
def assert_all_layer_activations_finite(
    llm: LanguageModel, is_layer_output_tuple: bool, inputs: Any, k_suffix: int
) -> List[float]:
    n_layers = _n_layers(llm)
    saved = []
    with llm.trace(inputs):
        for l_idx in range(n_layers):
            resid = _read_layer_resid_envoy(llm.model.layers[l_idx], is_layer_output_tuple)
            saved.append(resid.save())
        saved_logits = llm.lm_head.output[:, -1, :].save()
    maxima = []
    mask = inputs["attention_mask"].bool()
    for l_idx, p in enumerate(saved):
        v = _deref_saved(p)
        active = v[mask]
        assert torch.isfinite(active).all(), f"NaN/Inf at layer {l_idx} residual stream ({v.dtype})!"
        maximum = float(active.float().abs().max().item())
        assert maximum < 60000, f"Activation headroom exceeded at layer {l_idx}: {maximum}"
        maxima.append(maximum)
    assert torch.isfinite(_deref_saved(saved_logits)).all(), "NaN/Inf in output logits!"
    return True


@torch.no_grad()
def verify_low_precision_against_fp32(
    model_name: str, probe_inputs: Any, low_prec_logits: torch.Tensor
) -> Dict[str, Any]:
    """Exact fp32 reference from a FRESH CPU load (the GPU model is untouched)."""
    print("[Stage 0] Loading fp32 reference model on CPU (~12 GB RAM)...", flush=True)
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    ref.eval()
    cpu_in = {k: probe_inputs[k].detach().cpu() for k in ("input_ids", "attention_mask")}
    fp32_logits = ref(**cpu_in).logits[:, -1, :].detach().float()
    del ref
    gc.collect()

    lp = low_prec_logits.detach().cpu().float()
    cos_sim = float(F.cosine_similarity(lp, fp32_logits, dim=-1).mean().item())
    log_lp = F.log_softmax(lp.double(), dim=-1)
    log_ref = F.log_softmax(fp32_logits.double(), dim=-1)
    kl_tensor = F.kl_div(log_lp, log_ref, log_target=True, reduction="batchmean")
    kl = float(kl_tensor.item())
    assert kl >= -1e-9, f"KL divergence must be nonnegative, got {kl}"
    top1 = float((lp.argmax(-1) == fp32_logits.argmax(-1)).float().mean().item())
    print(f"[Stage 0] low-prec vs fp32: cos={cos_sim:.6f} KL={kl:.6f} nats top1={top1:.2f}", flush=True)

    assert cos_sim >= CONFIG.min_fp16_fp32_logit_cos, (
        f"logit cosine too low: {cos_sim:.6f} < {CONFIG.min_fp16_fp32_logit_cos}")
    assert kl <= CONFIG.max_fp16_fp32_kl, f"KL too high: {kl:.6f} > {CONFIG.max_fp16_fp32_kl}"
    return {"fp32_logit_cosine_sim": cos_sim, "fp32_logit_mean_kl_nats": kl,
            "fp32_top1_agreement": top1}


# ----------------------------- main entry -----------------------------------
PROBE_QUESTIONS = [
    "Context: Project Helios was directed by Dr. Aris Thorn in 2019.\nQuestion: Who directed Project Helios?",
    "Context: Project Helios was funded by the Department of Energy in 2019.\nQuestion: Who directed Project Helios?",
    "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: In what year was Mara Vance born?",
    "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: What is Mara Vance's childhood pet's name?",
]


def load_fidelity_prompts(tokenizer: Any) -> Tuple[List[str], List[str]]:
    """Return 12 chat prompts (4 fixed + 8 train pairs) and four raw WikiText paragraphs."""
    prompts = list(PROBE_QUESTIONS)
    pair_path = CONFIG.data_dir / "pairs.jsonl"
    if pair_path.exists():
        with pair_path.open(encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                if pair.get("split") == "train":
                    prompts.extend((pair["ans_prompt"], pair["unans_prompt"]))
                    if len(prompts) >= 12:
                        break
    if len(prompts) < 12:
        raise RuntimeError("Stage 0 fidelity probe requires 8 train pairs in data/pairs.jsonl")
    chat = [tokenizer.apply_chat_template(
        [{"role": "system", "content": CONFIG.system_prompt}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True) for q in prompts]
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    raw = []
    for row in ds:
        if len(row["text"].split()) < 64:
            continue
        ids = tokenizer.encode(row["text"], add_special_tokens=False)[:CONFIG.len_wikitext_tokens]
        if len(ids) == CONFIG.len_wikitext_tokens:
            raw.append(tokenizer.decode(ids, skip_special_tokens=True))
            if len(raw) == 4:
                break
    if len(raw) != 4:
        raise RuntimeError("Could not load four WikiText-2 test paragraphs with 256 tokens")
    return chat, raw


def load_model_and_verify_runtime(
    model_name: str = CONFIG.primary_model_name,
    device: str = "cuda:0",
    requested_dtype: str = "auto",
) -> Tuple[LanguageModel, Dict[str, Any]]:
    runtime_dtype, dtype_str, precision_deviation = resolve_runtime_dtype(requested_dtype, device)
    print(f"[Stage 0] runtime dtype = {dtype_str}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == "left"

    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=runtime_dtype, device_map=device, low_cpu_mem_usage=True)
    raw_model.eval()

    # 1-3: native hook / tokens / suffix (all outside nnsight)
    is_tuple, _ = inspect_native_layer_output(raw_model, tokenizer, device)
    unk_ns, unk_sp = verify_unknown_tokens(tokenizer)
    k_suffix, suffix_ids = discover_post_instruction_suffix(tokenizer)
    print(f"[Stage 0] is_layer_output_tuple={is_tuple} k_suffix={k_suffix} unknown_id={unk_ns}", flush=True)

    # Probe batch (chat-templated)
    chat_prompts, raw_prompts = load_fidelity_prompts(tokenizer)
    prompts = chat_prompts + raw_prompts
    probe_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False,
                             max_length=CONFIG.len_wikitext_tokens + 128).to(device)
    for row in range(len(chat_prompts)):
        tail = probe_inputs["input_ids"][row, -k_suffix:].tolist()
        assert tail == suffix_ids, f"Prompt {row} tail {tail} != expected suffix {suffix_ids}"

    # 4: nnsight wrapper + finite check
    llm = LanguageModel(raw_model, tokenizer=tokenizer)
    max_abs_activation_per_layer = assert_all_layer_activations_finite(llm, is_tuple, probe_inputs, k_suffix)
    all_finite = True

    # 5: native baseline + timing, then ACTIVE zero-intervention hook (alpha=1.0 on r=0)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        native_logits = raw_model(**probe_inputs).logits[:, -1, :].detach()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    sec_per_4 = time.perf_counter() - t0

    zero_vec = torch.zeros(raw_model.config.hidden_size, dtype=torch.float32, device=device)
    hook_logits = forward_with_intervention(
        llm, is_tuple, probe_inputs, mode="steer_layer", r_vec=zero_vec, target_layer=0, alpha=1.0)
    max_abs_diff = float(torch.max(torch.abs(native_logits.float() - hook_logits.float())).item())
    assert torch.allclose(native_logits.float(), hook_logits.float(),
                          atol=CONFIG.atol_zero, rtol=CONFIG.rtol_zero), (
        f"Zero-intervention hook failed allclose! max_abs_diff={max_abs_diff:.6e}")

    # 6: low precision vs exact fp32
    if runtime_dtype == torch.float32:
        fid = {"fp32_logit_cosine_sim": 1.0, "fp32_logit_mean_kl_nats": 0.0, "fp32_top1_agreement": 1.0}
    else:
        fid = verify_low_precision_against_fp32(model_name, probe_inputs, native_logits)

    gpu_name = torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else "CPU"
    env_record = {
        "model_name": model_name,
        "n_layers": int(raw_model.config.num_hidden_layers),
        "d_model": int(raw_model.config.hidden_size),
        "dtype": dtype_str,
        "precision_deviation_dev001": bool(precision_deviation),
        "gpu_device_name": gpu_name,
        "is_layer_output_tuple": bool(is_tuple),
        "unknown_token_id_nospace": int(unk_ns),
        "unknown_token_id_space": int(unk_sp),
        "k_suffix": int(k_suffix),
        "suffix_token_ids": [int(x) for x in suffix_ids],
        "all_layer_activations_finite": bool(all_finite),
        "max_abs_activation_per_layer": max_abs_activation_per_layer,
        "zero_intervention_max_abs_diff": float(max_abs_diff),
        "fp32_logit_cosine_sim": float(fid["fp32_logit_cosine_sim"]),
        "fp32_logit_mean_kl_nats": float(fid["fp32_logit_mean_kl_nats"]),
        "fp32_top1_agreement": float(fid["fp32_top1_agreement"]),
        "sec_per_16_prompt_forward": round(sec_per_4, 4),
        "rough_lower_bound_gpu_hours": round((sec_per_4 / len(prompts)) * 180000.0 / 3600.0, 2),
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda or "unavailable",
        "cuda_driver_version": _cuda_driver_version(),
        "passed_all_checks": True,
    }
    return llm, env_record


def _cuda_driver_version() -> str:
    if not torch.cuda.is_available():
        return "unavailable"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=10)
        return result.stdout.splitlines()[0].strip()
    except Exception:
        return "unavailable"
