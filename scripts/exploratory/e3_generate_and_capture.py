#!/usr/bin/env python3
"""Generate E3 pilot/PopQA responses and capture pair/PopQA activations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.abstention_classifier import classify_texts, release_local_judge
from src.config import CFG
from src.eval_common import ConditionCheckpoint, read_jsonl
from src.exploratory import assert_not_fresh, exploratory_payload
from src.exploratory.capture import capture_answer_positions, capture_suffix_all_layers, make_condition_contract
from src.exploratory.chat import render_ids
from src.pilot_check import determine_pilot_gate_status
from src.tokenization_contract import CONTRACT_VERSION, chat_template_sha256, input_ids_sha256

MODELS = {"llama": ("meta-llama/Llama-3.2-3B-Instruct", "llama32_3b"),
          "qwen": ("Qwen/Qwen2.5-3B-Instruct", "qwen25_3b")}
OUT = CFG.results_dir / "exploratory"
MAX_NEW_TOKENS = 32


def _read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def _condition(model_key):
    if not (OUT / "e3_preflight.json").exists():
        raise RuntimeError("Run scripts/exploratory/e3_preflight.py before E3 generation")
    preflight = json.loads((OUT / "e3_preflight.json").read_text(encoding="utf-8"))
    if preflight.get("template_date") not in {"static", "pinned"}:
        raise RuntimeError("E3 generation is stopped until a static or approved pinned date is recorded")
    model_name, slug = MODELS[model_key]
    if model_key == "qwen":
        contract = json.loads((OUT / "e3_qwen_contract.json").read_text(encoding="utf-8"))
        if not contract.get("passed_all_checks"):
            raise RuntimeError("Qwen failed its precision preflight; its arm is excluded")
    else:
        contract = json.loads(CFG.stage0_out_path.read_text(encoding="utf-8"))
    return model_name, slug, contract, preflight


def _load(model_name, model_contract, tokenizer=None):
    tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if model_contract.get("dtype") in {"float16", "torch.float16"} else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype,
                                                device_map="cuda:0", low_cpu_mem_usage=True).eval()
    cond = make_condition_contract(model_contract, model_name=model_name,
        system_prompt=CFG.exploratory_neutral_system_prompt,
        suffix_ids=model_contract["suffix_token_ids"], n_layers=model_contract["n_layers"],
        d_model=model_contract["d_model"], is_layer_output_tuple=model_contract["is_layer_output_tuple"],
        unknown_token_ids=model_contract.get("unknown_token_ids", [model_contract.get("unknown_token_id_nospace"),
                                                                     model_contract.get("unknown_token_id_space")]))
    abstention_tokens = ["Unknown", " Unknown", "I", " I", "Sorry", " Sorry", "There", " There",
                         "Unfortunately", " Unfortunately"]
    first_ids = []
    for text in abstention_tokens:
        encoded = tok.encode(text, add_special_tokens=False)
        if encoded:
            first_ids.append(int(encoded[0]))
    cond["abstention_token_strings"] = abstention_tokens
    cond["abstention_token_first_ids"] = list(dict.fromkeys(first_ids))
    cond["unknown_token_ids"] = cond["abstention_token_first_ids"]
    return model, tok, cond


@torch.no_grad()
def _generate(model, tokenizer, contract, prompt, max_new_tokens=MAX_NEW_TOKENS):
    ids = render_ids(tokenizer, prompt, contract["system_prompt_used"], contract["suffix_token_ids"],
                     contract.get("date_string"))
    device = next(model.parameters()).device
    encoded = {"input_ids": torch.tensor([ids], dtype=torch.long, device=device),
               "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
               "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0)}
    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True), ids


def _fingerprint(model_name, tokenizer):
    val = {"model": model_name, "prompt": CFG.exploratory_neutral_system_prompt,
           "template": chat_template_sha256(tokenizer), "max_new_tokens": MAX_NEW_TOKENS}
    return hashlib.sha256(json.dumps(val, sort_keys=True).encode()).hexdigest()


def _pilot(model, tokenizer, contract, model_name, slug, checkpoint):
    pairs = [p for p in read_jsonl(CFG.data_dir / "pairs.jsonl") if p.get("split") == "val"]
    prompts, labels, ids = [], [], []
    for pair in pairs:
        for part, label in (("ans", 0), ("unans", 1)):
            prompts.append(pair[f"{part}_prompt"]); labels.append(label); ids.append(pair["pair_id"])
    responses = []
    for i, prompt in enumerate(prompts):
        key = f"pilot:{i}"
        if key in checkpoint:
            response = checkpoint.get(key)["response"]
        else:
            response, _ = _generate(model, tokenizer, contract, prompt)
            checkpoint.put(key, {"response": response})
        responses.append(response)
    checkpoint.flush()
    # The T4 cannot safely hold the subject model and 7B judge simultaneously.
    model.to("cpu")
    torch.cuda.empty_cache()
    try:
        judged = classify_texts([{"uid": f"{slug}:pilot:{i}", "question": q, "response": r}
                                 for i, (q, r) in enumerate(zip(prompts, responses))])
    finally:
        release_local_judge()
        model.to("cuda:0")
        torch.cuda.empty_cache()
    abstain = [r.get("final_label") == "ABSTAIN" for r in judged]
    unans_rate = float(np.mean([abstain[i] for i, y in enumerate(labels) if y == 1]))
    ans_rate = float(np.mean([abstain[i] for i, y in enumerate(labels) if y == 0]))
    passed, gap, mode = determine_pilot_gate_status(unans_rate, ans_rate)
    included = bool(passed) if slug == "qwen25_3b" else True
    result = exploratory_payload({"model_name": model_name, "system_prompt_used": contract["system_prompt_used"],
        "neutral_unans_abstention_rate": unans_rate, "neutral_ans_abstention_rate": ans_rate,
        "rate_gap": gap, "pilot_mode": mode, "pilot_passed": bool(passed),
        "behaviourally_anchored": unans_rate >= 0.25, "arm_included": included,
        "n_pairs": len(pairs), "responses": responses,
        "judge_model_revision": judged[0].get("judge_model_revision") if judged else None,
        "judge_quantization": judged[0].get("judge_quantization") if judged else None})
    target = OUT / f"e3_pilot_{slug}.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if slug == "qwen25_3b" and not passed:
        return 0
    return 0


def _popqa(model, tokenizer, contract, model_name, slug):
    in_path = assert_not_fresh(CFG.data_dir / "popqa_sample.jsonl")
    rows = _read(in_path)
    template_hash = chat_template_sha256(tokenizer)
    target = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    old = {r.get("popqa_id"): r for r in _read(target)} if target.exists() else {}
    if not target.exists():
        target.touch()
    for row in rows:
        rec = dict(row)
        if rec.get("popqa_id") in old:
            continue
        text, ids = _generate(model, tokenizer, contract, rec["question"])
        rec.update({"baseline_response": text, "response": text,
            "system_prompt_used": contract["system_prompt_used"],
            "input_ids_sha256": input_ids_sha256(ids, model_name, template_hash),
            "chat_template_sha256": template_hash, "tokenization_contract_version": CONTRACT_VERSION,
            "max_new_tokens": MAX_NEW_TOKENS, "precision": contract["dtype"],
            "label_source": "e3_neutral_unjudged", "classifier": None, "final_label": None,
            "grade": None, "grade_reason": None, "baseline_partition": None})
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _pairs(model, tokenizer, contract, model_name, slug):
    pairs = read_jsonl(CFG.data_dir / "pairs.jsonl")
    prompts, meta = [], []
    for pair in sorted(pairs, key=lambda r: (r["split"], r["pair_id"])):
        for part in ("ans", "unans"):
            prompts.append(pair[f"{part}_prompt"])
            meta.append({"pair_id": pair["pair_id"], "split": pair["split"], "part": part})
    path = OUT / "e3_features" / f"{slug}_neut"
    path.mkdir(parents=True, exist_ok=True)
    shape = (len(prompts), 5, int(contract["n_layers"]), int(contract["d_model"]))
    tensor_path = path / "pairs_suf.npy"
    fingerprint = _fingerprint(model_name, tokenizer)
    checkpoint_path = OUT / f"e3_pair_checkpoint_{slug}.pt"
    if not tensor_path.exists() and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = ConditionCheckpoint(checkpoint_path, fingerprint, every=1)
    array = np.lib.format.open_memmap(tensor_path, mode="r+" if tensor_path.exists() else "w+",
                                     dtype=np.float16, shape=shape)
    for start in range(0, len(prompts), 64):
        stop = min(len(prompts), start + 64)
        key = f"pairs_{start}_{stop}"
        if key in checkpoint:
            continue
        captured = capture_suffix_all_layers(model, tokenizer, contract, prompts[start:stop], batch_size=8)
        array[start:stop] = captured["resid"]
        array.flush()
        checkpoint.put(key, {"rows": stop - start})
        print(f"[E3:{slug}] captured pair rows {start}:{stop}/{len(prompts)}", flush=True)
    checkpoint.flush()
    if not np.isfinite(array).all():
        raise RuntimeError("E3 synthetic pair features contain non-finite or incomplete values")
    del array
    array = np.load(tensor_path, mmap_mode="r")
    (path / "pairs_manifest.json").write_text(json.dumps(exploratory_payload({
        "model_name": model_name, "system_prompt_used": contract["system_prompt_used"],
        "shape": list(array.shape), "dtype": str(array.dtype), "rows": meta,
        "sha256": {"pairs_suf.npy": hashlib.sha256(tensor_path.read_bytes()).hexdigest()}}), indent=2) + "\n", encoding="utf-8")


def _features(model, tokenizer, contract, model_name, slug):
    source = assert_not_fresh(CFG.data_dir / "exploratory" / f"popqa_neutral_{slug}.jsonl")
    rows = _read(source)
    path = OUT / "e3_features" / f"{slug}_neut"
    path.mkdir(parents=True, exist_ok=True)
    n, nl, d = len(rows), int(contract["n_layers"]), int(contract["d_model"])
    suffix_path, answer_path = path / "popqa_suf.npy", path / "popqa_ans.npy"
    suffix = np.lib.format.open_memmap(suffix_path, mode="r+" if suffix_path.exists() else "w+",
                                       dtype=np.float16, shape=(n, 5, nl, d))
    answer = np.lib.format.open_memmap(answer_path, mode="r+" if answer_path.exists() else "w+",
                                       dtype=np.float16, shape=(n, 2, nl, d))
    scalar_path = path / "popqa_confidence.npz"
    if scalar_path.exists():
        old = np.load(scalar_path)
        scalar = {k: old[k].copy() for k in old.files}
    else:
        scalar = {k: np.full(n, np.nan, dtype=(bool if k == "greedy_consistent" else np.float32))
                  for k in ("maxp", "entropy", "abstention_set_log_odds", "seq_logprob_mean", "greedy_consistent")}
        scalar["n_answer_tokens"] = np.zeros(n, dtype=np.int32)
    fingerprint = _fingerprint(model_name, tokenizer)
    checkpoint_path = OUT / f"e3_feature_checkpoint_{slug}.pt"
    if (not suffix_path.exists() or not answer_path.exists() or not scalar_path.exists()) and checkpoint_path.exists():
        checkpoint_path.unlink()
    checkpoint = ConditionCheckpoint(checkpoint_path, fingerprint, every=1)
    for start in range(0, n, 100):
        stop = min(n, start + 100)
        key = f"features_{start}_{stop}"
        if key in checkpoint:
            continue
        batch = rows[start:stop]
        features = capture_suffix_all_layers(model, tokenizer, contract,
                                             [r["question"] for r in batch], batch_size=8)
        captured_answer = capture_answer_positions(model, tokenizer, contract, batch)
        suffix[start:stop] = features["resid"]
        answer[start:stop] = captured_answer["resid_ans"]
        scalar["maxp"][start:stop] = features["maxp"]
        scalar["entropy"][start:stop] = features["entropy"]
        scalar["abstention_set_log_odds"][start:stop] = features["log_odds"]
        scalar["seq_logprob_mean"][start:stop] = captured_answer["seq_logprob_mean"]
        scalar["greedy_consistent"][start:stop] = captured_answer["greedy_consistent"]
        scalar["n_answer_tokens"][start:stop] = captured_answer["n_answer_tokens"]
        suffix.flush(); answer.flush()
        np.savez(scalar_path, **scalar)
        checkpoint.put(key, {"rows": stop - start})
        print(f"[E3:{slug}] captured feature rows {start}:{stop}/{n}", flush=True)
    checkpoint.flush()
    if not np.isfinite(suffix).all():
        raise RuntimeError("E3 PopQA feature tensors contain non-finite or incomplete values")
    answer_valid = scalar["n_answer_tokens"] > 0
    if answer_valid.any() and not np.isfinite(answer[answer_valid]).all():
        raise RuntimeError("E3 teacher-forced answer features contain non-finite values")
    if np.isnan(scalar["maxp"]).any() or np.isnan(scalar["seq_logprob_mean"][answer_valid]).any():
        raise RuntimeError("E3 PopQA scalar features are incomplete")
    pd = __import__("pandas")
    pd.DataFrame(rows).to_parquet(path / "meta.parquet", index=False)
    checksums = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                 for name in ("popqa_suf.npy", "popqa_confidence.npz", "popqa_ans.npy", "meta.parquet")}
    (path / "features_manifest.json").write_text(json.dumps(exploratory_payload({
        "model_name": model_name, "system_prompt_used": contract["system_prompt_used"],
        "n_rows": len(rows), "row_order": [str(r["popqa_id"]) for r in rows],
        "suffix_shape": [n, 5, nl, d], "answer_shape": [n, 2, nl, d],
        "greedy_consistency_rate": float(np.mean(scalar["greedy_consistent"])),
        "sha256": checksums}), indent=2) + "\n", encoding="utf-8")


def _push_features(slug):
    from src.exploratory.tensor_io import upload_folder
    upload_folder(OUT / "e3_features" / f"{slug}_neut",
                  path_in_repo=f"exploratory/e3_features/{slug}_neut",
                  ignore_patterns=["*checkpoint*", "*.pt"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--steps", default="pilot", help="comma-separated: pilot,popqa,pairs,features")
    parser.add_argument("--push-hf", action="store_true", help="Upload feature tensors and manifests to the HF dataset repo")
    args = parser.parse_args()
    steps = [x.strip() for x in args.steps.split(",") if x.strip()]
    unknown = set(steps) - {"pilot", "popqa", "pairs", "features"}
    if unknown:
        parser.error(f"unknown steps: {sorted(unknown)}")
    model_name, slug, model_contract, _ = _condition(args.model)
    model, tokenizer, contract = _load(model_name, model_contract)
    checkpoint = ConditionCheckpoint(OUT / f"e3_checkpoint_{slug}.pt", _fingerprint(model_name, tokenizer))
    for step in steps:
        if step == "pilot":
            _pilot(model, tokenizer, contract, model_name, slug, checkpoint)
        else:
            if args.model == "qwen":
                pilot_path = OUT / f"e3_pilot_{slug}.json"
                if not pilot_path.exists() or not json.loads(pilot_path.read_text()).get("arm_included"):
                    raise RuntimeError("Qwen pilot has not passed; later Qwen steps are excluded")
        if step == "popqa":
            _popqa(model, tokenizer, contract, model_name, slug)
        elif step == "pairs":
            _pairs(model, tokenizer, contract, model_name, slug)
            if args.push_hf:
                _push_features(slug)
        elif step == "features":
            _features(model, tokenizer, contract, model_name, slug)
            if args.push_hf:
                _push_features(slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
