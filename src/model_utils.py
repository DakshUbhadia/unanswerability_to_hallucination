# src/model_utils.py
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from src.config import CONFIG


def resolve_runtime_dtype(requested_dtype: str = "auto", device: str = "cuda:0") -> Tuple[torch.dtype, str, bool]:
    """
    Resolves the execution torch.dtype:
      - Uses bfloat16 if supported natively by the GPU (sm_80+).
      - Automatically falls back to float16 on Kaggle T4 (sm_75) / P100 (sm_60) and flags
        DEV-001 (precision deviation).
    """
    if requested_dtype == "bfloat16":
        return torch.bfloat16, "torch.bfloat16", False
    if requested_dtype == "float16":
        return torch.float16, "torch.float16", True
    if requested_dtype == "float32":
        return torch.float32, "torch.float32", True

    # "auto" mode: check native hardware bfloat16 support
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "torch.bfloat16", False
        return torch.float16, "torch.float16", True
    return torch.float32, "torch.float32", True


def inspect_native_layer_output(
    raw_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Executes a native PyTorch forward hook on raw_model.model.layers[0] OUTSIDE nnsight
    to determine deterministically whether layer.output is a tuple or a raw Tensor.
    """
    hook_records: Dict[str, Any] = {}

    def _probe_hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hook_records["is_tuple"] = is_tuple
        hook_records["ndim"] = hidden.ndim
        hook_records["shape"] = tuple(hidden.shape)
        hook_records["dtype"] = str(hidden.dtype)

    handle = raw_model.model.layers[0].register_forward_hook(_probe_hook)
    try:
        dummy_texts = [" Probe prompt one", " Probe prompt two with extra words"]
        enc = tokenizer(dummy_texts, return_tensors="pt", padding=True).to(device)
        b_size, seq_len = enc["input_ids"].shape
        with torch.no_grad():
            _ = raw_model(**enc)
    finally:
        handle.remove()

    assert "is_tuple" in hook_records, "Native forward hook did not fire!"
    shape = hook_records["shape"]
    d_model = raw_model.config.hidden_size
    assert hook_records["ndim"] == 3 and shape == (b_size, seq_len, d_model), (
        f"Expected 3D hidden state shape {(b_size, seq_len, d_model)}, got {shape}"
    )
    return bool(hook_records["is_tuple"]), shape


def verify_unknown_tokens(tokenizer: AutoTokenizer) -> Tuple[int, int]:
    """
    Verifies that 'Unknown' encodes to a single token and returns
    (unknown_token_id_nospace, unknown_token_id_space).
    """
    ids_nospace = tokenizer.encode("Unknown", add_special_tokens=False)
    assert len(ids_nospace) == 1, (
        f"'Unknown' must encode to exactly 1 token, got {len(ids_nospace)} tokens: {ids_nospace}"
    )
    ids_space = tokenizer.encode(" Unknown", add_special_tokens=False)
    assert len(ids_space) >= 1, "Failed to encode ' Unknown'"
    return int(ids_nospace[0]), int(ids_space[-1])


def discover_post_instruction_suffix(
    tokenizer: AutoTokenizer,
    system_prompt: str = CONFIG.system_prompt,
) -> Tuple[int, List[int]]:
    """
    Applies the model's chat template to two distinct user prompts and extracts
    the exact trailing post-instruction token sequence following the user content.
    """
    user_a = " AlphaUniqueMarker"
    user_b = " BetaUniqueMarkerWithDifferentLength"

    msgs_a = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_a}]
    msgs_b = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_b}]

    ids_a = tokenizer.apply_chat_template(msgs_a, tokenize=True, add_generation_prompt=True)
    ids_b = tokenizer.apply_chat_template(msgs_b, tokenize=True, add_generation_prompt=True)

    k_suffix = 0
    max_k = min(len(ids_a), len(ids_b))
    while k_suffix < max_k and ids_a[-(k_suffix + 1)] == ids_b[-(k_suffix + 1)]:
        k_suffix += 1

    assert k_suffix >= 1, "No common post-instruction suffix found across chat prompts!"
    suffix_ids = list(ids_a[-k_suffix:])
    return k_suffix, suffix_ids


def apply_orthogonal_projection(
    x: torch.Tensor,
    r_hat: torch.Tensor,
    eps: float = CONFIG.eps_norm,
) -> torch.Tensor:
    """
    Projects out unit direction r_hat from residual tensor x:
        Proj(x) = x - (x . r_hat) r_hat
    All inner products and subtractions execute in float32 before casting back to x.dtype.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    r_f32 = r_hat.to(device=x.device, dtype=torch.float32)
    r_unit = r_f32 / (torch.norm(r_f32, p=2) + eps)
    proj_coeff = torch.matmul(x_f32, r_unit).unsqueeze(-1)
    out_f32 = x_f32 - (proj_coeff * r_unit)
    return out_f32.to(dtype=orig_dtype)


def apply_norm_scaled_steering(
    x: torch.Tensor,
    r_vec: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Applies directional steering in units of ||r_vec||_2:
        Steer(x; r_vec, alpha) = x + alpha * r_vec
    Executes in float32 and does NOT short-circuit when r_vec == 0, allowing
    the active hook write-back path to be verified in Stage 0.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    r_f32 = r_vec.to(device=x.device, dtype=torch.float32)
    out_f32 = x_f32 + (float(alpha) * r_f32)
    return out_f32.to(dtype=orig_dtype)


def _deref_saved(proxy_or_tensor: Any) -> torch.Tensor:
    """Dereferences an nnsight 0.3.7 saved proxy (.value) or returns the tensor directly."""
    return proxy_or_tensor.value if hasattr(proxy_or_tensor, "value") else proxy_or_tensor


def _read_layer_resid_envoy(layer_envoy: Any, is_tuple_output: bool) -> Any:
    """Reads the hidden_states proxy from a decoder layer Envoy using the verified boolean."""
    return layer_envoy.output[0] if is_tuple_output else layer_envoy.output


def _write_layer_resid_envoy(layer_envoy: Any, new_resid: Any, is_tuple_output: bool) -> None:
    """Writes modified hidden_states back to a decoder layer Envoy using the verified boolean."""
    if is_tuple_output:
        layer_envoy.output[0][:] = new_resid
    else:
        layer_envoy.output[:] = new_resid


@torch.no_grad()
def forward_with_intervention(
    llm: LanguageModel,
    is_layer_output_tuple: bool,
    inputs: Dict[str, torch.Tensor],
    mode: str = "none",
    r_vec: Optional[torch.Tensor] = None,
    target_layer: Optional[int] = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """
    Executes a forward pass through `llm` under `nnsight==0.3.7` and returns
    final-token logits of shape [batch_size, vocab_size].
    """
    n_layers = llm.config.num_hidden_layers

    with llm.trace(inputs):
        if mode == "ablate_all":
            assert r_vec is not None, "r_vec required for mode='ablate_all'"
            for l_idx in range(n_layers):
                layer_envoy = llm.model.layers[l_idx]
                resid = _read_layer_resid_envoy(layer_envoy, is_layer_output_tuple)
                modified = apply_orthogonal_projection(resid, r_vec)
                _write_layer_resid_envoy(layer_envoy, modified, is_layer_output_tuple)

        elif mode == "steer_layer":
            assert r_vec is not None and target_layer is not None, (
                "r_vec and target_layer required for mode='steer_layer'"
            )
            layer_envoy = llm.model.layers[target_layer]
            resid = _read_layer_resid_envoy(layer_envoy, is_layer_output_tuple)
            modified = apply_norm_scaled_steering(resid, r_vec, alpha=alpha)
            _write_layer_resid_envoy(layer_envoy, modified, is_layer_output_tuple)

        elif mode != "none":
            raise ValueError(f"Unsupported intervention mode: {mode}")

        saved_logits = llm.lm_head.output[:, -1, :].save()

    return _deref_saved(saved_logits).detach()


@torch.no_grad()
def assert_all_layer_activations_finite(
    llm: LanguageModel,
    is_layer_output_tuple: bool,
    inputs: Dict[str, torch.Tensor],
    k_suffix: int,
) -> bool:
    """
    Stage 0 Check (DEV-001): Hooks all L decoder layers simultaneously and asserts
    that zero residual stream activations (or logits) are NaN or Inf in float16.
    """
    n_layers = llm.config.num_hidden_layers
    saved_resids = []
    with llm.trace(inputs):
        for l_idx in range(n_layers):
            resid = _read_layer_resid_envoy(llm.model.layers[l_idx], is_layer_output_tuple)
            saved_resids.append(resid[:, -k_suffix:, :].save())
        saved_logits = llm.lm_head.output[:, -1, :].save()

    for l_idx, proxy in enumerate(saved_resids):
        tensor_val = _deref_saved(proxy)
        assert torch.isfinite(tensor_val).all(), (
            f"NaN or Inf detected in layer {l_idx} residual stream under {tensor_val.dtype}!"
        )
    logits_val = _deref_saved(saved_logits)
    assert torch.isfinite(logits_val).all(), "NaN or Inf detected in output logits!"
    return True


@torch.no_grad()
def verify_low_precision_against_fp32(
    raw_model: AutoModelForCausalLM,
    probe_inputs: Dict[str, torch.Tensor],
    low_prec_logits: torch.Tensor,
) -> Dict[str, Any]:
    """
    Stage 0 Check (DEV-001): Compares baseline final-token logits in low precision
    (float16/bfloat16) against an exact float32 forward pass on the probe batch.
    Temporarily casting the model weights to float32 on CPU to avoid 16GB VRAM overflow.
    """
    orig_device = next(raw_model.parameters()).device
    orig_dtype = next(raw_model.parameters()).dtype

    # Move model to CPU in float32 briefly to compute exact fp32 reference logits
    # without exceeding 16 GB VRAM on a Kaggle T4
    raw_model.to(device="cpu", dtype=torch.float32)
    try:
        cpu_inputs = {k: v.to("cpu") for k, v in probe_inputs.items()}
        fp32_logits = raw_model(**cpu_inputs).logits[:, -1, :].detach()
    finally:
        raw_model.to(device=orig_device, dtype=orig_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lp_f32 = low_prec_logits.detach().cpu().float()
    cos_sim = float(F.cosine_similarity(lp_f32, fp32_logits, dim=-1).mean().item())

    p_fp32 = F.softmax(fp32_logits, dim=-1).clamp(min=1e-8)
    log_p_fp32 = F.log_softmax(fp32_logits, dim=-1)
    log_p_lp = F.log_softmax(lp_f32, dim=-1)
    mean_kl = float(torch.sum(p_fp32 * (log_p_fp32 - log_p_lp), dim=-1).mean().item())

    top1_agree = float((lp_f32.argmax(dim=-1) == fp32_logits.argmax(dim=-1)).float().mean().item())

    assert cos_sim >= CONFIG.min_fp16_fp32_logit_cos, (
        f"Low-precision vs fp32 logit cosine similarity too low: {cos_sim:.6f} < {CONFIG.min_fp16_fp32_logit_cos}"
    )
    assert mean_kl <= CONFIG.max_fp16_fp32_kl, (
        f"Low-precision vs fp32 KL divergence too high: {mean_kl:.6f} > {CONFIG.max_fp16_fp32_kl}"
    )
    return {
        "fp32_logit_cosine_sim": cos_sim,
        "fp32_logit_mean_kl_nats": mean_kl,
        "fp32_top1_agreement": top1_agree,
    }


def load_model_and_verify_runtime(
    model_name: str = CONFIG.primary_model_name,
    device: str = "cuda:0",
    requested_dtype: str = "auto",
) -> Tuple[LanguageModel, Dict[str, Any]]:
    """
    Loads the model and tokenizer on Kaggle T4 (or local hardware), executes all
    Stage 0 startup checks, verifies zero NaN/Inf activations, benchmarks float16
    against float32 reference logits, and estimates total pipeline GPU hours.
    """
    runtime_dtype, dtype_str, precision_deviation = resolve_runtime_dtype(requested_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == "left", "Tokenizer padding_side must be 'left'"

    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=runtime_dtype,
        device_map=device,
    )
    raw_model.eval()

    # 1. Native PyTorch hook check outside nnsight
    is_tuple, _ = inspect_native_layer_output(raw_model, tokenizer, device=device)

    # 2. Single-token "Unknown" check
    unk_nospace, unk_space = verify_unknown_tokens(tokenizer)

    # 3. Post-instruction suffix discovery
    k_suffix, suffix_ids = discover_post_instruction_suffix(tokenizer)

    # Wrap in nnsight LanguageModel
    llm = LanguageModel(raw_model, tokenizer=tokenizer)

    # 4. Prepare 4 probe prompts (2 answerable, 2 unanswerable)
    raw_probe_questions = [
        "Context: Project Helios was directed by Dr. Aris Thorne in 2019.\nQuestion: Who directed Project Helios?",
        "Context: Project Helios was funded by the Department of Energy in 2019.\nQuestion: Who directed Project Helios?",
        "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: In what year was Mara Vance born?",
        "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: What is Mara Vance's childhood pet's name?",
    ]
    probe_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CONFIG.system_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in raw_probe_questions
    ]
    probe_inputs = tokenizer(probe_prompts, return_tensors="pt", padding=True).to(device)

    for row_idx in range(probe_inputs["input_ids"].shape[0]):
        tail_ids = probe_inputs["input_ids"][row_idx, -k_suffix:].tolist()
        assert tail_ids == suffix_ids, (
            f"Prompt {row_idx} tail {tail_ids} does not match expected suffix {suffix_ids}"
        )

    # 5. All-layer finite activation check (zero NaN/Inf in float16/bfloat16)
    all_finite = assert_all_layer_activations_finite(llm, is_tuple, probe_inputs, k_suffix)

    # 6. Active-hook zero-intervention logit identity check (alpha=1.0, r=0) and timing
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        native_logits = raw_model(**probe_inputs).logits[:, -1, :].detach()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    sec_per_4_prompts = time.perf_counter() - t0

    zero_vec = torch.zeros(raw_model.config.hidden_size, dtype=torch.float32, device=device)
    hook_logits = forward_with_intervention(
        llm=llm,
        is_layer_output_tuple=is_tuple,
        inputs=probe_inputs,
        mode="steer_layer",
        r_vec=zero_vec,
        target_layer=0,
        alpha=1.0,
    )

    max_abs_diff = float(torch.max(torch.abs(native_logits.float() - hook_logits.float())).item())
    is_close = torch.allclose(
        native_logits,
        hook_logits,
        atol=CONFIG.atol_zero,
        rtol=CONFIG.rtol_zero,
    )
    assert is_close, (
        f"Zero-intervention active hook failed allclose! max_abs_diff={max_abs_diff:.6e}"
    )

    # 7. Compare low-precision baseline logits against exact float32 reference
    fp32_fidelity = verify_low_precision_against_fp32(raw_model, probe_inputs, native_logits)

    # Estimate total forward passes across Stages 2-8 (~180,000 prompt evaluations)
    est_total_gpu_hours = float((sec_per_4_prompts / 4.0) * 180000.0 / 3600.0)
    gpu_name = torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else "CPU"

    env_record = {
        "model_name": model_name,
        "n_layers": int(raw_model.config.num_hidden_layers),
        "d_model": int(raw_model.config.hidden_size),
        "dtype": dtype_str,
        "precision_deviation_dev001": precision_deviation,
        "gpu_device_name": gpu_name,
        "is_layer_output_tuple": bool(is_tuple),
        "unknown_token_id_nospace": int(unk_nospace),
        "unknown_token_id_space": int(unk_space),
        "k_suffix": int(k_suffix),
        "suffix_token_ids": [int(x) for x in suffix_ids],
        "all_layer_activations_finite": bool(all_finite),
        "zero_intervention_max_abs_diff": max_abs_diff,
        "fp32_logit_cosine_sim": fp32_fidelity["fp32_logit_cosine_sim"],
        "fp32_logit_mean_kl_nats": fp32_fidelity["fp32_logit_mean_kl_nats"],
        "fp32_top1_agreement": fp32_fidelity["fp32_top1_agreement"],
        "estimated_project_gpu_hours": round(est_total_gpu_hours, 2),
        "passed_all_checks": True,
    }
    return llm, env_record
