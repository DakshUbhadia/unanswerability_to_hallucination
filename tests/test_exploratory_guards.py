import json
from pathlib import Path

import pytest

from src.exploratory import assert_not_fresh, require_tier1_complete


def test_fresh_path_is_rejected():
    with pytest.raises(ValueError, match="Tier 2 fresh sample"):
        assert_not_fresh("data/popqa_fresh.jsonl")


def test_tier1_gate_fails_in_unjudged_checkout():
    with pytest.raises(RuntimeError, match="Tier 1 E0 is incomplete"):
        require_tier1_complete()


def test_exploratory_source_has_no_forbidden_sample_reference():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "src/exploratory").rglob("*.py")) + list((root / "scripts/exploratory").rglob("*.py"))
    occurrences = [(path, line_no) for path in paths for line_no, line in enumerate(path.read_text().splitlines(), 1)
                   if "popqa_fresh" in line.lower() and path.name != "__init__.py"]
    assert not occurrences, occurrences
