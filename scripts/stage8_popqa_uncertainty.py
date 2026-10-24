#!/usr/bin/env python3
"""Run Stage 8: PopQA closed-book uncertainty test (H4, H5), Holm-Bonferroni across H1-H5, Fig 2A/S1 data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import CFG
from src.eval_common import (ensure_stage4_directions, load_locked_selection, load_runtime_contract,
                             load_stage_model, read_json, sha256_file)
from src.popqa_uncertainty import (load_popqa_groups, run_stage8_popqa_uncertainty, save_stage8_artifacts,
                                   verify_stage8_outputs)
from src.test_evaluation import load_and_validate_test_and_b_ctrl_datasets, verify_stage6_outputs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=CFG.batch_size_forward)
    p.add_argument("--popqa-path", type=Path, default=CFG.data_dir / "popqa_sample.jsonl")
    p.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    p.add_argument("--stage2-path", type=Path, default=CFG.stage2_out_path)
    p.add_argument("--stage4-directions-path", type=Path, default=CFG.stage4_out_path)
    p.add_argument("--stage4-manifest-path", type=Path, default=CFG.stage4_manifest_path)
    p.add_argument("--stage5-path", type=Path, default=CFG.stage5_out_path)
    p.add_argument("--stage6-path", type=Path, default=CFG.stage6_out_path)
    p.add_argument("--output-path", type=Path, default=CFG.stage8_out_path)
    p.add_argument("--table3-csv-path", type=Path, default=CFG.table3_csv_path)
    p.add_argument("--table3-md-path", type=Path, default=CFG.table3_md_path)
    a = p.parse_args(argv)
    if not a.force and a.output_path.exists():
        try:
            verify_stage8_outputs(a.output_path, a.table3_csv_path, a.table3_md_path, a.stage5_path)
        except (AssertionError, KeyError, ValueError, RuntimeError, TypeError):
            pass
        else:
            print(f"[SKIP] Stage 8 already verified in {a.output_path}")
            return 0

    contract = load_runtime_contract(a.stage0_path, a.stage2_path)
    directions, stage4_sha = ensure_stage4_directions(a.stage4_directions_path, a.stage4_manifest_path, a.stage0_path)
    selection = load_locked_selection(a.stage5_path, stage4_sha)
    verify_stage6_outputs(a.stage6_path, CFG.stage6_gen_path, CFG.table1_csv_path, CFG.table1_md_path,
                          CFG.table4_csv_path, stage5_path=a.stage5_path)
    stage6 = read_json(a.stage6_path)
    groups = load_popqa_groups(a.popqa_path, read_json(a.stage2_path))
    test_pairs, b_ctrl = load_and_validate_test_and_b_ctrl_datasets()
    print(f"[Stage 8] {len(groups['pair_ids'])} matched pairs, {len(groups['abstain'])} abstained items", flush=True)
    raw_model, tokenizer, _ = load_stage_model(contract, a.device, a.dtype)
    result = run_stage8_popqa_uncertainty(raw_model, tokenizer, contract, directions, selection, groups, test_pairs,
                                          b_ctrl, stage6, a.batch_size, stage4_sha, sha256_file(a.stage6_path))
    save_stage8_artifacts(result, a.output_path, a.table3_csv_path, a.table3_md_path)
    d = verify_stage8_outputs(a.output_path, a.table3_csv_path, a.table3_md_path, a.stage5_path)
    print(f"[Stage 8] AUROC_proj={d['auroc_proj']:.4f} CI={d['auroc_proj_ci']} AUROC_maxp={d['auroc_maxp']:.4f} "
          f"AUROC_ent={d['auroc_ent']:.4f}")
    for row in d["holm_bonferroni_table"]:
        print(f"[Stage 8] {row['test_id']}: p={row['raw_p_value']:.4g} thr={row['holm_threshold']:.4f} "
              f"holm={row['passed_holm_bonferroni']} confirmed={row['confirmed']}")
    print(f"[Stage 8] interpretation: {d['interpretation']['overall']}")
    print(a.table3_md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
