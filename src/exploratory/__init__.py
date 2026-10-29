"""Post-registration exploratory analyses, kept separate from confirmatory outputs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

STATUS = "exploratory_post_registration"


def assert_not_fresh(path: str | Path) -> Path:
    """Reject any exploratory input or output whose path names the Tier 2 fresh sample."""
    path = Path(path)
    if "popqa_fresh" in str(path).lower():
        raise ValueError(f"Exploratory code cannot access the Tier 2 fresh sample: {path}")
    return path


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
