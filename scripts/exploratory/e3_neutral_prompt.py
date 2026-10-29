#!/usr/bin/env python3
"""Generate, judge, and grade the post-registration neutral-prompt PopQA sample."""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.abstention_classifier import classify_texts
from src.config import CFG
from src.eval_common import atomic_write_jsonl, load_runtime_contract, load_stage_model, read_jsonl
from src.exploratory import assert_not_fresh, git_commit, write_exploratory_json
from src.pilot_check import grade_popqa
from src.tokenization_contract import CONTRACT_VERSION, canonical_input_ids, chat_template_sha256, input_ids_sha256


def _load_neutral_model(contract, model_name, device, dtype):
    if model_name == contract["model_name"]:
        return load_stage_model(contract, device, dtype)
    from src.model_utils import resolve_runtime_dtype
    runtime_dtype, dtype_name, _ = resolve_runtime_dtype(dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=runtime_dtype,
                                                 device_map=device, low_cpu_mem_usage=True).eval()
    return model, tokenizer, dtype_name


def _generate_one(model, tokenizer, ids, max_new_tokens):
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)


def _bootstrap_rate(values, seed=20261029, n_boot=1000):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"estimate": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n_boot)])
    return {"estimate": float(values.mean()), "ci95": [float(x) for x in np.percentile(samples, [2.5, 97.5])]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--model-name", default=None,
                        help="Override with Qwen/Qwen2.5-3B-Instruct for the exploratory second-model retry")
    parser.add_argument("--offline-judge", action="store_true", help="Require the pinned judge cache to cover all neutral responses")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--input", type=Path, default=CFG.data_dir / "popqa_sample.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    assert_not_fresh(args.input)
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError("The canonical PopQA sample is empty")
    contract = load_runtime_contract()
    model_name = args.model_name or contract["model_name"]
    is_second_model = model_name != contract["model_name"]
    if args.output is None:
        name = "popqa_neutral_qwen25_3b.jsonl" if is_second_model else "popqa_neutral.jsonl"
        args.output = CFG.data_dir / "exploratory" / name
    assert_not_fresh(args.output)
    raw_model, tokenizer, dtype_name = _load_neutral_model(contract, model_name, args.device, args.dtype)
    prompt = CFG.exploratory_neutral_system_prompt
    template_hash = chat_template_sha256(tokenizer)
    suffix_ids = None if is_second_model else contract["suffix_token_ids"]
    existing = {}
    checkpoint_path = assert_not_fresh(args.output.with_name(args.output.stem + "_generation_checkpoint.jsonl"))
    if checkpoint_path.exists() and not args.force:
        existing = {r["popqa_id"]: r for r in read_jsonl(checkpoint_path)}
    generated = []
    source_commit = git_commit(ROOT)
    started = time.perf_counter()
    for index, row in enumerate(rows):
        if row["popqa_id"] in existing:
            cached = existing[row["popqa_id"]]
            cached.setdefault("status", "exploratory_post_registration")
            cached.setdefault("git_commit", source_commit)
            cached.setdefault("model_name", model_name)
            cached.setdefault("runtime_dtype", dtype_name)
            generated.append(cached)
            continue
        if is_second_model:
            rendered = tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}, {"role": "user", "content": row["question"]}],
                tokenize=False, add_generation_prompt=True)
            ids = [int(token_id) for token_id in tokenizer.encode(rendered, add_special_tokens=False)]
        else:
            ids = canonical_input_ids(tokenizer, row["question"], prompt, suffix_ids)
        response = _generate_one(raw_model, tokenizer, ids, args.max_new_tokens)
        generated.append({"popqa_id": row["popqa_id"], "question": row["question"],
                          "generated_text": response, "possible_answers": row.get("possible_answers", []),
                          "obj": row.get("obj", ""), "original_baseline_partition": row.get("baseline_partition"),
                          "baseline_partition": row.get("baseline_partition"),
                          "matched_pair_id": row.get("matched_pair_id"), "prop": row.get("prop"),
                          "log_s_pop": row.get("log_s_pop", row.get("log_popularity", 0.0)),
                          "input_ids_sha256": input_ids_sha256(ids, model_name, template_hash),
                          "tokenization_contract_version": "exploratory_model_native_chat_template" if is_second_model else CONTRACT_VERSION,
                          "system_prompt": prompt, "system_prompt_type": "neutral",
                          "status": "exploratory_post_registration", "git_commit": source_commit,
                          "model_name": model_name, "runtime_dtype": dtype_name,
                          "generation_batch_size": 1, "max_new_tokens": int(args.max_new_tokens)})
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            atomic_write_jsonl(checkpoint_path, generated)
            print(f"[E3] generated {index + 1}/{len(rows)} ({time.perf_counter() - started:.0f}s)", flush=True)
    del raw_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # A separate local judge load after the studied model is released avoids a two-model GPU peak.
    judge_records = classify_texts([{"uid": r["popqa_id"], "question": r["question"],
                                     "response": r["generated_text"]} for r in generated],
                                   offline=args.offline_judge, batch_size=CFG.judge_batch_size)
    output = []
    for row, record in zip(generated, judge_records):
        grade = grade_popqa(row["generated_text"], row["possible_answers"], row["obj"], record)
        output.append({**row, "baseline_partition": grade["grade"], "grade": grade["grade"],
                       "grade_reason": grade["reason"], "final_label": record["final_label"],
                       "judge_label": record["judge_label"], "regex_label": record["regex_label"],
                       "judge_margin": record.get("judge_margin"),
                       "judge_quantization": record.get("judge_quantization"),
                       "judge_model_id": record.get("judge_model_id"),
                       "judge_model_revision": record.get("judge_model_revision"),
                       "classifier": {key: record.get(key) for key in (
                           "final_label", "judge_label", "regex_label", "judge_margin", "judge_quantization",
                           "judge_model_id", "judge_model_revision", "judge_prompt_version", "judge_prompt_sha256")}})
    atomic_write_jsonl(args.output, output)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    import numpy as np
    neutral_grades = [row["grade"] for row in output]
    original_grades = [row.get("original_baseline_partition") for row in output]
    summary = {"experiment": "E3_neutral_prompt", "model_name": model_name, "runtime_dtype": dtype_name,
               "n": len(output), "max_new_tokens": int(args.max_new_tokens),
               "grade_counts": dict(Counter(neutral_grades)),
               "original_grade_counts": dict(Counter(original_grades)),
               "neutral_abstention_rate": _bootstrap_rate([grade == "abstain" for grade in neutral_grades]),
               "original_abstention_rate": _bootstrap_rate([grade == "abstain" for grade in original_grades]),
               "neutral_accuracy_among_answered": _bootstrap_rate([grade == "correct" for grade in neutral_grades if grade != "abstain"]),
               "judge_label_counts": dict(Counter(row["final_label"] for row in output)),
               "offline_judge": bool(args.offline_judge),
               "second_model_retry": bool(is_second_model)}
    summary_path = CFG.results_dir / "exploratory" / ("e3_neutral_qwen.json" if is_second_model else "e3_neutral_prompt.json")
    write_exploratory_json(summary_path, summary, ROOT)
    print(f"Saved {len(output)} judged neutral-prompt rows to {args.output}; summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
