#!/usr/bin/env python3
"""Build and verify Stage 4 candidate directions and control vectors."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.activation_cache import verify_stage3_outputs
from src.config import CFG
from src.direction_extraction import (extract_all_directions_and_controls,
                                      save_stage4_artifacts, verify_stage4_outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage3-cache-path", type=Path, default=CFG.stage3_out_path)
    parser.add_argument("--stage3-manifest-path", type=Path, default=CFG.stage3_manifest_path)
    parser.add_argument("--pairs-path", type=Path, default=CFG.data_dir / "pairs.jsonl")
    parser.add_argument("--stage0-path", type=Path, default=CFG.stage0_out_path)
    parser.add_argument("--output-path", type=Path, default=CFG.stage4_out_path)
    parser.add_argument("--manifest-path", type=Path, default=CFG.stage4_manifest_path)
    args = parser.parse_args(argv)

    if not args.force and args.output_path.exists() and args.manifest_path.exists():
        try:
            verify_stage4_outputs(args.output_path, args.manifest_path, args.stage0_path)
        except (AssertionError, KeyError, ValueError, RuntimeError):
            pass
        else:
            print(f"[SKIP] Stage 4 already verified in {args.output_path}")
            return 0

    if not args.stage3_cache_path.exists():
        repo_id = os.environ.get("HF_TENSOR_REPO_ID", "")
        if not repo_id:
            raise FileNotFoundError(f"Missing {args.stage3_cache_path}; set HF_TENSOR_REPO_ID to restore it")
        from scripts.kaggle_sync import ensure_tensor_cached_from_hf
        ensure_tensor_cached_from_hf("stage3_caches.pt", repo_id, args.stage3_cache_path.parent.parent)
    import json
    stage3_metadata = json.loads(args.stage3_manifest_path.read_text(encoding="utf-8"))
    stage3_manifest = verify_stage3_outputs(args.stage3_cache_path, args.stage3_manifest_path,
                                             expected_n_pairs=int(stage3_metadata.get("n_train_pairs", CFG.n_pairs_train)))
    cache = torch.load(args.stage3_cache_path, map_location="cpu", weights_only=False)
    directions, manifest = extract_all_directions_and_controls(cache, args.pairs_path)
    manifest["source_stage3_sha256"] = stage3_manifest["cache_sha256"]
    save_stage4_artifacts(directions, manifest, args.output_path, args.manifest_path)
    verify_stage4_outputs(args.output_path, args.manifest_path, args.stage0_path)
    print(f"[Stage 4] Verified directions at {args.output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
