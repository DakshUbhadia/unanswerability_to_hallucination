"""Runtime contracts shared by stages that follow the Stage 2 gate."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import CFG


def load_stage2_model_contract(stage2_path: Path = CFG.stage2_out_path) -> Dict[str, Any]:
    path = Path(stage2_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Stage 2 before any downstream stage")
    with path.open(encoding="utf-8") as f:
        record = json.load(f)
    if record.get("passed_pilot_and_power_gate") is not True:
        raise RuntimeError(f"Stage 2 gate did not pass in {path}; downstream stages are blocked")
    model_name = record.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise RuntimeError(f"Stage 2 result in {path} is missing a valid model_name")
    prompt = record.get("system_prompt_used")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError(f"Stage 2 result in {path} is missing a valid system_prompt_used")
    return record


def resolve_downstream_model_name(requested_model_name: Optional[str] = None,
                                  stage2_path: Path = CFG.stage2_out_path) -> str:
    """Use Stage 2's locked model and fail if a stage requests another model."""
    locked = load_stage2_model_contract(stage2_path)["model_name"]
    if requested_model_name is not None and requested_model_name != locked:
        raise RuntimeError(
            f"Requested model {requested_model_name!r} differs from Stage 2 locked model {locked!r}"
        )
    return locked
