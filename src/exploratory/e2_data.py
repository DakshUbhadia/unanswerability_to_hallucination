"""Validated E2 data accessors; all external inputs pass the exploratory guards."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CFG
from src.eval_common import ensure_stage4_directions, sha256_file
from src.exploratory import assert_not_fresh, require_tier1_complete
from src.exploratory.e1_data import load_e1_frame

ROOT = Path(__file__).resolve().parents[2]
STAR = (11, 1)
R_NORM_AT_STAR = 3.807962656


def load_e1_cache(features_dir=ROOT / "results/exploratory/e1_features", dry_run=False):
    """Load and checksum E1's feature cache without materializing activations."""
    features_dir = assert_not_fresh(features_dir)
    if dry_run:
        if "results/exploratory/dryrun" not in str(features_dir):
            raise ValueError("Dry-run cache must live under results/exploratory/dryrun/")
    else:
        require_tier1_complete(ROOT)
    manifest_path = assert_not_fresh(Path(features_dir) / "features_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = {}
    for key in ("suf.npy", "ans.npy"):
        path = assert_not_fresh(Path(features_dir) / key)
        if not path.exists():
            repo = os.environ.get("HF_TENSOR_REPO_ID")
            if not repo:
                raise FileNotFoundError(f"Missing E1 tensor {path}; restore it from HF_TENSOR_REPO_ID")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo_id=repo, filename=f"exploratory/e1_features/{key}", repo_type="dataset")
            path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(downloaded, path)
        if sha256_file(path) != manifest["sha256"][key]:
            raise ValueError(f"E1 feature checksum mismatch: {path}")
        arrays[key[:-4]] = np.load(path, mmap_mode="r")
    meta_path = assert_not_fresh(Path(features_dir) / "meta.parquet")
    if "meta.parquet" in manifest.get("sha256", {}) and sha256_file(meta_path) != manifest["sha256"]["meta.parquet"]:
        raise ValueError("E1 metadata checksum mismatch")
    meta = pd.read_parquet(meta_path)
    if len(meta) != arrays["suf"].shape[0] or meta["popqa_id"].astype(str).tolist() != manifest["row_order"]:
        raise ValueError("E1 metadata row count/order does not match features_manifest.json")
    if list(arrays["suf"].shape) != manifest["shapes"]["suf"] or list(arrays["ans"].shape) != manifest["shapes"]["ans"]:
        raise ValueError("E1 feature shapes do not match features_manifest.json")
    if not dry_run:
        from src.popqa_uncertainty import _verify_popqa_input_identity
        from src.eval_common import load_runtime_contract
        from transformers import AutoTokenizer
        contract = load_runtime_contract()
        tok = AutoTokenizer.from_pretrained(contract["model_name"])
        _verify_popqa_input_identity(tok, meta.to_dict("records"), contract["system_prompt_used"],
                                     contract["model_name"], contract["suffix_token_ids"])
    return {**arrays, "meta": meta, "manifest": manifest, "dry_run": bool(dry_run)}


def load_rhat(contract=None):
    """Load hash-verified Stage 4 vectors and assert the registered Stage 5 coordinate."""
    stage4, _ = ensure_stage4_directions(CFG.stage4_out_path, CFG.stage4_manifest_path, CFG.stage0_out_path)
    selection = json.loads(assert_not_fresh(ROOT / "results/stage5_selection.json").read_text(encoding="utf-8"))
    if (int(selection["l_star"]), int(selection["pos_idx"])) != STAR:
        raise RuntimeError("Stage 5 locked coordinate differs from E2 preregistration")
    units = stage4["unit_directions"].detach().cpu().numpy().astype(np.float64)
    diff = stage4["diff_in_means"].detach().cpu().numpy().astype(np.float64)
    norms = stage4["norms"].detach().cpu().numpy().astype(np.float64)
    if abs(float(norms[11, 1]) - R_NORM_AT_STAR) >= 1e-4:
        raise RuntimeError(f"Stage 4 norm mismatch at (11,1): {norms[11,1]}")
    out = {"unit": units, "diff": diff, "norms": norms,
           "random_unit": stage4["random_controls_unit"].detach().cpu().numpy().astype(np.float64),
           "concept_unit": stage4["r_concept_unit"].detach().cpu().numpy().astype(np.float64),
           "concept_scaled": stage4["r_concept_ortho_scaled"].detach().cpu().numpy().astype(np.float64)}
    return out


def population(meta, name):
    """Return the exact E1 row frame and grouped-CV identifiers for P1 or P2."""
    frame = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)
    from scripts.exploratory.e1_analyze import _population
    rows, y, groups, units, base = _population(frame, name)
    if name == "P1" and len(np.unique(groups)) < 100:
        raise RuntimeError(f"P1 has fewer than 100 matched pairs ({len(np.unique(groups))})")
    return rows, y, groups, units, base


def projection_sign_check(meta, proj, stage8):
    """Align E2's frozen projection sign to Stage 8's registered wrong-like score."""
    from sklearn.metrics import roc_auc_score
    rows, y, _groups, _units, _base = population(meta, "P1")
    indices = rows.index.to_numpy(dtype=int)
    score = np.asarray(proj, dtype=float)[indices]
    target = float(stage8["auroc_proj"])
    auc = float(roc_auc_score(y, score))
    if abs(auc - target) > 1e-6:
        flipped = float(roc_auc_score(y, -score))
        if abs(flipped - target) > 1e-6:
            raise RuntimeError(f"Frozen projection AUROC {auc:.9f} (flipped {flipped:.9f}) != Stage 8 {target:.9f}")
        return -np.asarray(proj), {"flipped": True, "auroc": flipped}
    return np.asarray(proj), {"flipped": False, "auroc": auc}
