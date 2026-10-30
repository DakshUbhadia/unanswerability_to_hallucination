#!/usr/bin/env python3
"""Run E3 chat-template and Qwen runtime preflight before any generation job."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import CFG
from src.exploratory.chat import discover_suffix, render_ids, template_date_probe
from src.model_utils import (assert_all_layer_activations_finite, inspect_native_layer_output,
                             verify_low_precision_against_fp32)
from src.tokenization_contract import canonical_input_ids, chat_template_sha256, input_ids_sha256


LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
QWEN = "Qwen/Qwen2.5-3B-Instruct"
OUT = CFG.results_dir / "exploratory"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _pinned_popqa_hashes_match(tokenizer, rows, suffix_ids) -> bool:
    if not rows:
        return False
    template_digest = chat_template_sha256(tokenizer)
    return all(row.get("input_ids_sha256") == input_ids_sha256(
        canonical_input_ids(tokenizer, row["question"], CFG.system_prompt,
                            suffix_ids, CFG.chat_template_date), LLAMA, template_digest)
        for row in rows)


def _apply_pinned_date_preflight(preflight, tokenizer, rows, suffix_ids, output_path) -> int:
    if not _pinned_popqa_hashes_match(tokenizer, rows, suffix_ids):
        preflight["stop_reason"] = "Pinned-date rendering failed the first-20 PopQA input hash check."
        _write(output_path, preflight)
        return 3
    preflight.update({"template_date": "pinned", "template_date_value": CFG.chat_template_date})
    return 0


def _qwen_contract(model_name, tokenizer, suffix_ids, n_layers, d_model, is_tuple, dtype, fidelity, maxima):
    try:
        unknown_ns = tokenizer.encode("Unknown", add_special_tokens=False)
        unknown_sp = tokenizer.encode(" Unknown", add_special_tokens=False)
        unknown_status = "passed" if len(unknown_ns) == 1 else "informational_failure"
    except Exception as exc:
        unknown_ns, unknown_sp, unknown_status = [], [], f"error: {exc}"
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    try:
        role_pos = len(suffix_ids) - 1 - list(suffix_ids).index(int(im_start))
        role_negative_index = role_pos - len(suffix_ids)
    except ValueError:
        role_pos = role_negative_index = None
    return {
        "model_name": model_name, "n_layers": int(n_layers), "d_model": int(d_model),
        "dtype": str(dtype), "is_layer_output_tuple": bool(is_tuple),
        "unknown_token_id_nospace": int(unknown_ns[0]) if len(unknown_ns) == 1 else None,
        "unknown_token_id_space": int(unknown_sp[-1]) if unknown_sp else None,
        "unknown_token_ids": sorted(set(map(int, (unknown_ns + unknown_sp)))),
        "unknown_token_status": unknown_status, "k_suffix": len(suffix_ids),
        "suffix_token_ids": list(map(int, suffix_ids)), "role_matched_token_id": int(im_start),
        "role_matched_suffix_index": role_pos, "role_matched_negative_index": role_negative_index,
        "all_layer_activations_finite": bool(maxima and all(torch.isfinite(torch.tensor(maxima)))),
        "max_abs_activation_per_layer": list(map(float, maxima)),
        "zero_intervention_max_abs_diff": None,
        "fp32_logit_cosine_sim": float(fidelity.get("fp32_logit_cosine_sim", 1.0)),
        "fp32_logit_mean_kl_nats": float(fidelity.get("fp32_logit_mean_kl_nats", 0.0)),
        "passed_all_checks": bool(maxima and all(torch.isfinite(torch.tensor(maxima))) and fidelity.get("passed", True)),
    }


def _runtime_contract(model_name, tokenizer, suffix_ids, requested_dtype):
    dtype = torch.float16 if requested_dtype == "float16" else torch.float32
    raw = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                device_map="cuda:0", low_cpu_mem_usage=True).eval()
    prompt = "Context: Arin Vale was born in 1982.\nQuestion: When was Arin Vale born?"
    ids = render_ids(tokenizer, prompt, CFG.exploratory_neutral_system_prompt, suffix_ids)
    inputs = tokenizer.pad({"input_ids": [ids]}, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    inputs["position_ids"] = inputs["attention_mask"].long().cumsum(-1) - 1
    inputs["position_ids"].masked_fill_(inputs["attention_mask"] == 0, 0)
    is_tuple, _ = inspect_native_layer_output(raw, tokenizer, "cuda:0")
    from nnsight import LanguageModel
    llm = LanguageModel(raw, tokenizer=tokenizer)
    maxima = assert_all_layer_activations_finite(llm, is_tuple, inputs, len(suffix_ids))
    with torch.no_grad():
        logits = raw(**inputs, use_cache=False).logits[:, -1, :].detach()
    fidelity = {"passed": True, "fp32_logit_cosine_sim": 1.0, "fp32_logit_mean_kl_nats": 0.0}
    if dtype != torch.float32:
        try:
            fidelity.update(verify_low_precision_against_fp32(model_name, inputs, logits))
        except AssertionError as exc:
            fidelity = {"passed": False, "failure": str(exc)}
    result = (int(raw.config.num_hidden_layers), int(raw.config.hidden_size), is_tuple,
              str(dtype).replace("torch.", ""), fidelity, maxima)
    del llm, raw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prereg = json.loads((OUT / "e3_preregistration.json").read_text(encoding="utf-8"))
    if prereg.get("neutral_system_prompt") != CFG.exploratory_neutral_system_prompt:
        raise RuntimeError("Neutral prompt differs from the committed E3 preregistration")
    stage0_path = CFG.results_dir / "stage0_env.json"
    stage0 = json.loads(stage0_path.read_text(encoding="utf-8"))
    llama_tok = _tokenizer(LLAMA)
    date_probe = template_date_probe(llama_tok, CFG.system_prompt)
    normal = date_probe["normal_render"]
    today = __import__("datetime").datetime.now().strftime("%d %b %Y")
    template_date = "pinned" if CFG.chat_template_date else ("dynamic" if date_probe["dynamic"] else "static")
    llama_k, llama_suffix = discover_suffix(llama_tok, CFG.exploratory_neutral_system_prompt)
    if llama_suffix != stage0["suffix_token_ids"]:
        raise RuntimeError("Neutral Llama suffix differs from the Stage 0 locked suffix; stop E3")
    preflight = {"status": "exploratory_post_registration", "template_date": template_date,
                 "template_date_probe": {"dynamic": date_probe["dynamic"],
                    "normal_includes_today": today in normal, "today": today},
                 "llama_neutral_suffix_token_ids": llama_suffix, "llama_k_suffix": llama_k,
                 "llama_chat_template_sha256": chat_template_sha256(llama_tok)}
    if template_date == "pinned":
        rows = [json.loads(line) for line in (CFG.data_dir / "popqa_sample.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:20]
        gate_code = _apply_pinned_date_preflight(preflight, llama_tok, rows,
                                                 stage0["suffix_token_ids"], OUT / "e3_preflight.json")
        if gate_code:
            print("Pinned chat-template date did not match all first 20 PopQA hashes. E3 stopped.")
            return gate_code
    elif date_probe["dynamic"]:
        preflight["stop_reason"] = "Dynamic Llama template date requires owner-approved recovery before E3 jobs."
        _write(OUT / "e3_preflight.json", preflight)
        print("Dynamic chat-template date detected. Recover the Stage 2 regeneration date against five archived input hashes, then obtain owner approval before pinning it. E3 stopped.")
        return 3
    qtok = _tokenizer(QWEN)
    qk, qsuffix = discover_suffix(qtok, CFG.exploratory_neutral_system_prompt)
    precision_attempts = []
    contract_result = None
    for precision in ("float16", "float32"):
        try:
            contract_result = _runtime_contract(QWEN, qtok, qsuffix, precision)
            precision_attempts.append({"precision": precision, "passed": contract_result[4].get("passed", True) and bool(contract_result[5]),
                                       "fidelity": contract_result[4]})
            if precision_attempts[-1]["passed"]:
                break
        except Exception as exc:
            precision_attempts.append({"precision": precision, "passed": False, "failure": str(exc)})
    n_layers, d_model, is_tuple, precision, fidelity, maxima = contract_result or (0, 0, False, "unavailable", {"passed": False}, [])
    q_contract = _qwen_contract(QWEN, qtok, qsuffix, n_layers, d_model, is_tuple, precision, fidelity, maxima)
    q_contract["precision_attempts"] = precision_attempts
    q_contract["chat_template_sha256"] = chat_template_sha256(qtok)
    q_contract["qwen_depth_matched_layer"] = round(11 / 28 * n_layers) if n_layers else None
    q_contract["passed_all_checks"] = bool(q_contract["passed_all_checks"] and contract_result is not None)
    preflight.update({"qwen_suffix_token_ids": qsuffix, "qwen_k_suffix": qk,
                      "qwen_role_matched_token_id": q_contract["role_matched_token_id"],
                      "qwen_role_matched_negative_index": q_contract["role_matched_negative_index"],
                      "qwen_depth_matched_layer": q_contract["qwen_depth_matched_layer"],
                      "qwen_precision": precision, "qwen_passed": q_contract["passed_all_checks"]})
    _write(OUT / "e3_qwen_contract.json", q_contract)
    _write(OUT / "e3_preflight.json", preflight)
    return 0 if q_contract["passed_all_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
