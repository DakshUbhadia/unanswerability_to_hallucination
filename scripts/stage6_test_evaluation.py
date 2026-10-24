#!/usr/bin/env python3
"""Run Stage 6: held-out test evaluation (H1-H3), 119 control directions, and the generation audit."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import CFG
from src.eval_common import (ensure_stage4_directions, load_locked_selection, load_runtime_contract,
                             load_stage_model, sha256_file)
from src.test_evaluation import (load_and_validate_test_and_b_ctrl_datasets, run_stage6_test_evaluation,
                                 save_stage6_artifacts, verify_stage6_outputs)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=CFG.batch_size_forward)
    p.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    p.add_argument("--b-ctrl-path", type=Path, default=CFG.data_dir / "b_ctrl.jsonl")
    p.add_argument("--splits-path", type=Path, default=CFG.data_dir / "splits.json")
    p.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    p.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    p.add_argument("--stage4-directions-path", type=Path, default=CFG.stage4_out_path)
    p.add_argument("--stage4-manifest-path", type=Path, default=CFG.stage4_manifest_path)
    p.add_argument("--stage5-path", type=Path, default=CFG.stage5_out_path)
    p.add_argument("--output-path", type=Path, default=CFG.stage6_out_path)
    p.add_argument("--gen-path", type=Path, default=CFG.stage6_gen_path)
    p.add_argument("--table1-csv-path", type=Path, default=CFG.table1_csv_path)
    p.add_argument("--table1-md-path", type=Path, default=CFG.table1_md_path)
    p.add_argument("--table4-csv-path", type=Path, default=CFG.table4_csv_path)
    p.add_argument("--table4-md-path", type=Path, default=CFG.table4_md_path)
    a = p.parse_args(argv)
    outputs = (a.output_path, a.gen_path, a.table1_csv_path, a.table1_md_path, a.table4_csv_path)
    if not a.force and all(x.exists() for x in outputs):
        try:
            verify_stage6_outputs(*outputs, stage5_path=a.stage5_path)
        except (AssertionError, KeyError, ValueError, RuntimeError, TypeError):
            pass
        else:
            print(f"[SKIP] Stage 6 already verified in {a.output_path}")
            return 0

    contract = load_runtime_contract(a.stage0_path, a.stage2_path)
    directions, stage4_sha = ensure_stage4_directions(a.stage4_directions_path, a.stage4_manifest_path, a.stage0_path)
    selection = load_locked_selection(a.stage5_path, stage4_sha)
    test_pairs, b_ctrl = load_and_validate_test_and_b_ctrl_datasets(a.pairs_path, a.b_ctrl_path, a.splits_path)
    print(f"[Stage 6] locked coordinate l*={selection['l_star']} i*={selection['i_star']}; "
          f"{len(test_pairs)} test pairs, {len(b_ctrl)} B-ctrl prompts", flush=True)
    raw_model, tokenizer, _ = load_stage_model(contract, a.device, a.dtype)
    result, gens = run_stage6_test_evaluation(raw_model, tokenizer, contract, directions, selection, test_pairs, b_ctrl,
                                              a.batch_size, CFG.stage6_checkpoint_path, a.force, stage4_sha,
                                              sha256_file(a.stage5_path))
    save_stage6_artifacts(result, gens, a.output_path, a.gen_path, a.table1_csv_path, a.table1_md_path, None,
                          a.table4_csv_path, a.table4_md_path)
    d = verify_stage6_outputs(*outputs, stage5_path=a.stage5_path)
    if CFG.stage6_checkpoint_path.exists():
        CFG.stage6_checkpoint_path.unlink()
    for hid in ("H1", "H2", "H3"):
        t = d["primary_tests"][hid]
        extra = f" p_MC={t['p_mc']:.3f}" if "p_mc" in t else ""
        print(f"[Stage 6] {hid}: effect={t['mean_effect']:.4f} p_wilcoxon={t['p_wilcoxon']:.3g}{extra}")
    print(f"[Stage 6] logit_gen_discrepancy={d['logit_gen_discrepancy']}")
    print(a.table4_md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
