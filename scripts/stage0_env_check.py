import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/stage0_env_check.py
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Dict
import torch

from src.config import CONFIG
from src.model_utils import load_model_and_verify_runtime

REQUIRED_STAGE0_KEYS = {
    "model_name": str,
    "n_layers": int,
    "d_model": int,
    "dtype": str,
    "is_layer_output_tuple": bool,
    "unknown_token_id_nospace": int,
    "unknown_token_id_space": int,
    "k_suffix": int,
    "suffix_token_ids": list,
    "all_layer_activations_finite": bool,
    "zero_intervention_max_abs_diff": float,
    "fp32_logit_cosine_sim": float,
    "fp32_logit_mean_kl_nats": float,
    "passed_all_checks": bool,
}


def validate_existing_stage0_output(output_path: Path) -> bool:
    """Checks whether results/stage0_env.json exists and passes all acceptance criteria."""
    if not output_path.exists():
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, expected_type in REQUIRED_STAGE0_KEYS.items():
            if key not in data or not isinstance(data[key], expected_type):
                return False
        if not data["passed_all_checks"] or not data["all_layer_activations_finite"]:
            return False
        if data["zero_intervention_max_abs_diff"] > CONFIG.atol_zero:
            return False
        if data["fp32_logit_cosine_sim"] < CONFIG.min_fp16_fp32_logit_cos:
            return False
        if data["fp32_logit_mean_kl_nats"] > CONFIG.max_fp16_fp32_kl:
            return False
        if data["k_suffix"] < 1 or len(data["suffix_token_ids"]) != data["k_suffix"]:
            return False
        return True
    except Exception:
        return False


def verify_installed_packages(allow_kaggle_env: bool = False) -> Dict[str, str]:
    """
    Verifies installed package versions. In --allow-kaggle-env mode (DEV-002),
    strictly enforces nnsight==0.3.7 and transformers==4.46.3 while logging
    Kaggle's native CUDA torch/scipy/numpy versions.
    """
    strict_kaggle_keys = {"nnsight", "transformers"}
    observed = {}
    mismatches = []
    for pkg, expected_ver in CONFIG.pinned_versions.items():
        try:
            actual_ver = importlib.metadata.version(pkg)
            clean_actual = actual_ver.split("+")[0]
            observed[pkg] = actual_ver
            if clean_actual != expected_ver:
                if not allow_kaggle_env or pkg in strict_kaggle_keys:
                    mismatches.append(f"{pkg}: expected {expected_ver}, found {actual_ver}")
        except importlib.metadata.PackageNotFoundError:
            observed[pkg] = "MISSING"
            mismatches.append(f"{pkg}: MISSING (expected {expected_ver})")

    if mismatches:
        raise RuntimeError(
            "Critical package version mismatch:\n  - " + "\n  - ".join(mismatches)
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0: Environment & Startup Verification")
    parser.add_argument(
        "--model-name",
        type=str,
        default=CONFIG.primary_model_name,
        help="Target Hugging Face model ID (primary or fallback).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Execution device (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model precision ('auto' uses float16 on Kaggle T4/P100 per DEV-001).",
    )
    parser.add_argument(
        "--allow-kaggle-env",
        action="store_true",
        help="Enforce strict pins on nnsight/transformers while logging Kaggle host packages (DEV-002).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution even if results/stage0_env.json already passes checks.",
    )
    args = parser.parse_args()

    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.results_dir / "stage0_env.json"

    if not args.force and validate_existing_stage0_output(output_path):
        print(f"[SKIP] Stage 0 already verified at {output_path}")
        return 0

    print("[Stage 0] Checking installed packages...")
    observed_pkgs = verify_installed_packages(allow_kaggle_env=args.allow_kaggle_env)

    print(f"[Stage 0] Loading {args.model_name} on {args.device} (dtype={args.dtype})...")
    _, env_record = load_model_and_verify_runtime(
        model_name=args.model_name,
        device=args.device,
        requested_dtype=args.dtype,
    )
    env_record["installed_packages"] = observed_pkgs

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, indent=2)

    assert validate_existing_stage0_output(output_path), "Stage 0 output failed validation gate!"
    print(f"[PASS] Stage 0 complete. Saved verified contract to {output_path}:")
    print(json.dumps(env_record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
