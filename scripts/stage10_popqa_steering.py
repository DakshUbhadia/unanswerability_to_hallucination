#!/usr/bin/env python3
"""Run Stage 10 (optional stretch): selective-abstention steering sweep on matched PopQA and Figure 2B."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import CFG
from src.eval_common import (ensure_stage4_directions, load_locked_selection, load_runtime_contract,
                             load_stage_model, read_json)
from src.popqa_steering import run_stage10_popqa_steering_sweep, save_stage10_artifacts, verify_stage10_outputs
from src.popqa_uncertainty import load_popqa_groups


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=CFG.batch_size_gen)
    p.add_argument("--popqa-path", type=Path, default=CFG.data_dir / "popqa_sample.jsonl")
    p.add_argument("--output-path", type=Path, default=CFG.stage10_out_path)
    a = p.parse_args(argv)
    if not a.force and a.output_path.exists():
        try:
            verify_stage10_outputs(a.output_path)
        except (AssertionError, KeyError, ValueError, RuntimeError, TypeError):
            pass
        else:
            print(f"[SKIP] Stage 10 already verified in {a.output_path}")
            return 0
    contract = load_runtime_contract()
    directions, stage4_sha = ensure_stage4_directions()
    selection = load_locked_selection(CFG.stage5_out_path, stage4_sha)
    groups = load_popqa_groups(a.popqa_path, read_json(CFG.stage2_out_path))
    matched = read_json(CFG.stage8_out_path)["matched_items"] if CFG.stage8_out_path.exists() else None
    raw_model, tokenizer, _ = load_stage_model(contract, a.device, a.dtype)
    result = run_stage10_popqa_steering_sweep(raw_model, tokenizer, contract, directions, selection, groups,
                                              CFG.grid_alpha_stretch, a.batch_size, CFG.stage10_checkpoint_path,
                                              a.force, matched, stage4_sha)
    save_stage10_artifacts(result, a.output_path)
    d = verify_stage10_outputs(a.output_path)
    if CFG.stage10_checkpoint_path.exists():
        CFG.stage10_checkpoint_path.unlink()
    for r in d["coverage_curve"]["learned"]:
        print(f"[Stage 10] alpha={r['alpha']:+.2f} coverage={r['coverage']:.3f} sel_acc={r['selective_acc']} "
              f"wrong->abs={r['wrong_to_abstain_rate']:.3f} correct->abs={r['correct_to_abstain_rate']:.3f}")
    if CFG.stage9_out_path.exists():
        from src.plotting_and_report import run_stage9
        run_stage9()
        print("[Stage 10] refreshed Stage 9 README results with Figure 2B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
