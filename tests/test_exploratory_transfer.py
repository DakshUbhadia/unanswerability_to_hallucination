import numpy as np
import pytest

from src.exploratory import assert_not_fresh
from src.exploratory.transfer import transfer_matrix


def test_fresh_sample_guard_rejects_path():
    with pytest.raises(ValueError, match="Tier 2 fresh sample"):
        assert_not_fresh("data/popqa_fresh.jsonl")


def test_transfer_matrix_with_tiny_arrays():
    rng = np.random.default_rng(7)
    train_y = np.array([0, 1] * 20)
    train_x = rng.normal(size=(40, 4)) + train_y[:, None] * 0.8
    test_y = np.array([0, 1] * 10)
    test_x = rng.normal(size=(20, 4)) + test_y[:, None] * 0.8
    baseline = test_y.astype(float) * 0.1

    result = transfer_matrix("tiny_train", "tiny_test", 11, -4,
                             train_arrays=(train_x, train_y),
                             test_arrays=(test_x, test_y, baseline))

    assert result["n_train"] == 40
    assert result["n_test"] == 20
    assert 0.0 <= result["auroc"] <= 1.0
    assert result["auroc_minus_logit_baseline"] == pytest.approx(
        result["auroc"] - result["logit_baseline_auroc"])


def test_exploratory_source_files_do_not_reference_fresh_sample():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for directory in (root / "src" / "exploratory", root / "scripts" / "exploratory"):
        for path in directory.rglob("*.py"):
            if path.name == "__init__.py":
                continue
            assert "popqa_fresh" not in path.read_text(encoding="utf-8").lower(), str(path)
