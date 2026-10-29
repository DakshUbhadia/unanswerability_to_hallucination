"""Post-registration exploratory analyses, kept separate from confirmatory outputs."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS = "exploratory_post_registration"


def assert_not_fresh(path: str | Path) -> Path:
    """Reject any exploratory input or output whose path names the Tier 2 fresh sample."""
    path = Path(path)
    if "popqa_fresh" in str(path).lower():
        raise ValueError(f"Exploratory code cannot access the Tier 2 fresh sample: {path}")
    return path


def require_tier1_complete(root: str | Path | None = None) -> None:
    """Fail closed unless judged Stage 2 and both Tier 1 validation gates pass."""
    root = Path(root or Path(__file__).resolve().parents[2])
    requirements = (
        (root / "results/stage2_pilot.json", "label_source", "canonical_v1_batch1_judged"),
        (root / "results/judge_validation.json", "gate_passed", True),
        (root / "results/grader_validation.json", "gate_passed", True),
    )
    for path, key, expected in requirements:
        if not path.exists():
            raise RuntimeError(f"Tier 1 E0 is incomplete: required file is missing: {path}")
        try:
            actual = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Tier 1 E0 state cannot be read from {path}: {exc}") from exc
        if actual != expected:
            raise RuntimeError(f"Tier 1 E0 is incomplete: {path} has {key}={actual!r}; expected {expected!r}")


def exploratory_header(extra: dict[str, Any] | None = None, root: str | Path | None = None) -> dict[str, Any]:
    """Build required provenance fields for exploratory artifacts."""
    return {"status": STATUS, "git_commit": git_commit(root),
            "created_utc": datetime.now(timezone.utc).isoformat(), **(extra or {})}


def git_commit(root: str | Path | None = None) -> str:
    root = Path(root or Path(__file__).resolve().parents[2])
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def exploratory_payload(data: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    return {**data, "status": STATUS, "git_commit": git_commit(root)}


def write_exploratory_json(path: str | Path, data: dict[str, Any], root: str | Path | None = None) -> Path:
    path = assert_not_fresh(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(exploratory_payload(data, root), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
