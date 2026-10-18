#!/usr/bin/env python3
"""Measure Stage 1 prompt token parity using the model locked by Stage 2."""
import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG
from src.stage_contract import load_stage2_model_contract


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def _record_deviation_measurement(path: Path, report: dict) -> None:
    path = Path(path)
    if not path.exists():
        return
    pair_summary = (
        f"{report['pairs_with_abs_delta_gt_2']}/384 pairs had |delta_L| > 2; "
        f"max |delta_L|={report['max_abs_token_delta']}; "
        f"absolute-delta distribution={report['absolute_delta_distribution']}"
    )
    bctrl_summary = (
        f"48 B-control prompts tokenized; lengths min/median/max="
        f"{report['b_ctrl_token_length_min']}/{report['b_ctrl_token_length_median']}/"
        f"{report['b_ctrl_token_length_max']}; "
        f"distribution={report['b_ctrl_token_length_distribution']}"
    )
    replacement = (
        f"- Llama token parity measurement: {pair_summary}. {bctrl_summary}. "
        f"Full per-pair details are in `{report['report_path']}`."
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    for i, line in enumerate(lines):
        if line.startswith("- Llama token parity measurement:"):
            lines[i] = replacement
            updated = True
            break
    if updated:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    parser.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    parser.add_argument("--b-ctrl-path", type=Path, default=CFG.data_dir / "b_ctrl.jsonl")
    parser.add_argument("--output-path", type=Path, default=CFG.results_dir / "stage2_llama_parity.json")
    args = parser.parse_args()

    stage2 = load_stage2_model_contract(args.stage2_path)
    model_name = stage2.get("model_name")
    if not model_name:
        raise RuntimeError("results/stage2_pilot.json has no locked model_name")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN") or None)
    with args.pairs_path.open(encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    with args.b_ctrl_path.open(encoding="utf-8") as f:
        b_ctrl = [json.loads(line) for line in f if line.strip()]
    if len(pairs) != 384:
        raise RuntimeError(f"Expected 384 pairs, found {len(pairs)}")
    if len(b_ctrl) != 48:
        raise RuntimeError(f"Expected 48 B-control prompts, found {len(b_ctrl)}")

    pair_deltas = {}
    pair_lengths = []
    for pair in pairs:
        ans_len = len(tokenizer.encode(pair["ans_prompt"], add_special_tokens=False))
        unans_len = len(tokenizer.encode(pair["unans_prompt"], add_special_tokens=False))
        delta = ans_len - unans_len
        pair_deltas[delta] = pair_deltas.get(delta, 0) + 1
        pair_lengths.append({"pair_id": pair["pair_id"], "ans_tokens": ans_len,
                             "unans_tokens": unans_len, "delta_tokens": delta})
    absolute = [abs(row["delta_tokens"]) for row in pair_lengths]
    bctrl_lengths = [len(tokenizer.encode(row["prompt"], add_special_tokens=False)) for row in b_ctrl]
    abs_dist = Counter(absolute)
    bctrl_dist = Counter(bctrl_lengths)
    report = {
        "model_name": model_name,
        "pair_count": len(pairs),
        "max_token_diff_pair_threshold": CFG.max_token_diff_pair,
        "pairs_with_abs_delta_gt_2": sum(delta > 2 for delta in absolute),
        "max_abs_token_delta": max(absolute, default=0),
        "absolute_delta_distribution": {str(k): abs_dist[k] for k in sorted(abs_dist)},
        "signed_delta_distribution": {str(k): pair_deltas[k] for k in sorted(pair_deltas)},
        "pair_token_lengths": pair_lengths,
        "b_ctrl_prompt_count": len(bctrl_lengths),
        "b_ctrl_token_length_min": min(bctrl_lengths),
        "b_ctrl_token_length_median": statistics.median(bctrl_lengths),
        "b_ctrl_token_length_max": max(bctrl_lengths),
        "b_ctrl_token_length_distribution": {str(k): bctrl_dist[k] for k in sorted(bctrl_dist)},
        "report_path": str(args.output_path),
    }
    _atomic_json(args.output_path, report)
    _record_deviation_measurement(ROOT / "DEVIATIONS.md", report)
    print(f"Model: {model_name}")
    print(f"Pairs with |delta_L| > 2: {report['pairs_with_abs_delta_gt_2']}/384")
    print(f"Maximum |delta_L|: {report['max_abs_token_delta']}")
    print(f"Absolute delta distribution: {report['absolute_delta_distribution']}")
    print(f"B-control prompts tokenized: {report['b_ctrl_prompt_count']}/48")
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
