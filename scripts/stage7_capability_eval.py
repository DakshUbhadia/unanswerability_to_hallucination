#!/usr/bin/env python3
"""Run Stage 7: WikiText-2 KL for the learned direction and all 119 controls, plus chat MMLU accuracy."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.capability_eval import run_stage7_capability_evaluation, save_stage7_artifacts, verify_stage7_outputs
from src.config import CFG
from src.eval_common import (ensure_stage4_directions, load_locked_selection, load_runtime_contract,
                             load_stage_model, sha256_file)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=CFG.batch_size_forward)
    p.add_argument("--wikitext-batch-size", type=int, default=CFG.batch_size_wikitext)
    p.add_argument("--skip-mmlu", action="store_true", help="Day 14 Cut 2: WikiText-2 KL only")
    p.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    p.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    p.add_argument("--stage4-directions-path", type=Path, default=CFG.stage4_out_path)
    p.add_argument("--stage4-manifest-path", type=Path, default=CFG.stage4_manifest_path)
    p.add_argument("--stage5-path", type=Path, default=CFG.stage5_out_path)
    p.add_argument("--output-path", type=Path, default=CFG.stage7_out_path)
    p.add_argument("--table1-csv-path", type=Path, default=CFG.table1_csv_path)
    p.add_argument("--table1-md-path", type=Path, default=CFG.table1_md_path)
    a = p.parse_args(argv)
    if not a.force and a.output_path.exists():
        try:
            verify_stage7_outputs(a.output_path, a.stage5_path)
        except (AssertionError, KeyError, ValueError, RuntimeError, TypeError):
            pass
        else:
            print(f"[SKIP] Stage 7 already verified in {a.output_path}")
            return 0

    contract = load_runtime_contract(a.stage0_path, a.stage2_path)
    directions, stage4_sha = ensure_stage4_directions(a.stage4_directions_path, a.stage4_manifest_path, a.stage0_path)
    selection = load_locked_selection(a.stage5_path, stage4_sha)
    raw_model, tokenizer, _ = load_stage_model(contract, a.device, a.dtype)
    result = run_stage7_capability_evaluation(
        raw_model, tokenizer, contract, directions, selection, a.batch_size, CFG.stage7_checkpoint_path, a.force,
        run_mmlu=not a.skip_mmlu, wikitext_batch_size=a.wikitext_batch_size, source_stage4_sha256=stage4_sha,
        source_stage5_sha256=sha256_file(a.stage5_path))
    save_stage7_artifacts(result, a.output_path, a.table1_csv_path, a.table1_md_path, None)
    d = verify_stage7_outputs(a.output_path, a.stage5_path)
    if CFG.stage7_checkpoint_path.exists():
        CFG.stage7_checkpoint_path.unlink()
    print(f"[Stage 7] WikiText-2 baseline NLL = {d['wikitext_baseline_nll']:.4f} over {d['n_kl_positions']} positions")
    for row in d["learned_direction_capability"]:
        print(f"[Stage 7] {row['mode']:<11} alpha={row['alpha']:+.2f} KL={row['wikitext_kl']:.5f} MMLU={row['mmlu_acc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
