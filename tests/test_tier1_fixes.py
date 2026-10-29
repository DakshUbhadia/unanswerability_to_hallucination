import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import export_judge_labeling_sheet as exporter


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_pool_rows_accepts_null_classifier(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    write_jsonl(data_dir / "popqa_sample.jsonl", [{
        "popqa_id": "p1", "question": "Q?", "baseline_response": "A", "classifier": None,
    }])
    monkeypatch.setattr(exporter, "CFG", SimpleNamespace(data_dir=data_dir))

    rows = exporter.pool_rows(("popqa",))

    assert rows["popqa:p1"]["question"] == "Q?"
    assert rows["popqa:p1"]["final_label"] is None


def test_stage6_bctrl_joins_by_bctrl_id(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    write_jsonl(data_dir / "b_ctrl.jsonl", [{"b_ctrl_id": "bctrl_001", "prompt": "B-control question"}])
    write_jsonl(data_dir / "pairs.jsonl", [])
    stage6_path = tmp_path / "stage6.jsonl"
    write_jsonl(stage6_path, [{
        "condition": "baseline", "item_id": "bctrl_001", "dataset_part": "bctrl", "generated_text": "Reply",
    }])
    monkeypatch.setattr(exporter, "CFG", SimpleNamespace(data_dir=data_dir, stage6_gen_path=stage6_path))

    rows = exporter.pool_rows(("stage6",))

    assert rows["stage6:baseline:bctrl_001"]["question"] == "B-control question"


def test_default_sources_exclude_stage10():
    assert exporter.DEFAULT_SOURCES == ("popqa", "stage6")
    assert "stage10" not in exporter.DEFAULT_SOURCES


def test_stage10_validation_uses_fresh_generation_jsonl(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    write_jsonl(data_dir / "popqa_sample.jsonl", [{"popqa_id": "p1", "question": "Fresh question?"}])
    generation_path = tmp_path / "stage10_generations.jsonl"
    write_jsonl(generation_path, [{"direction": "learned", "alpha": 2.0, "popqa_id": "p1",
                                  "generated_text": "Fresh response", "final_label": "ANSWER"}])
    monkeypatch.setattr(exporter, "CFG", SimpleNamespace(data_dir=data_dir, stage10_generation_path=generation_path))

    rows = exporter.pool_rows(("stage10",))

    row = rows["stage10:learned:2.0:p1"]
    assert row["question"] == "Fresh question?"
    assert row["response"] == "Fresh response"


def test_offline_export_raises_cache_miss_without_loading_model(tmp_path, monkeypatch):
    import src.abstention_classifier as classifier

    data_dir = tmp_path / "data"
    write_jsonl(data_dir / "popqa_sample.jsonl", [{
        "popqa_id": "p1", "question": "Q?", "baseline_response": "A", "classifier": None,
    }])
    cache = tmp_path / "judge_cache.jsonl"
    monkeypatch.setattr(exporter, "CFG", SimpleNamespace(data_dir=data_dir))
    monkeypatch.setattr(classifier, "CFG", SimpleNamespace(
        judge_backend="local", judge_model_id="pinned-model", judge_model_revision="a" * 40,
        judge_model_pin_path=tmp_path / "missing-pin.json", judge_quantization="nf4", judge_batch_size=1,
        judge_cache_path=cache,
    ))
    monkeypatch.setattr(classifier, "_revision", lambda: "a" * 40)
    monkeypatch.setattr(classifier, "_load_local_judge", lambda: pytest.fail("offline export tried to load a model"))

    with pytest.raises(RuntimeError, match="Offline judge cache miss"):
        exporter.export_sheets(tmp_path / "out", "sheet", ("popqa",), 1, 0, offline=True)


def test_local_judge_rejects_mixed_quantization(tmp_path, monkeypatch):
    import src.abstention_classifier as classifier

    cache_path = tmp_path / "judge_cache.jsonl"
    cache_path.write_text(json.dumps({"cache_key": "existing", "judge_quantization": "fp16"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(classifier, "CFG", SimpleNamespace(
        judge_backend="local", judge_model_id="pinned-model", judge_model_revision="a" * 40,
        judge_model_pin_path=tmp_path / "missing-pin.json", judge_quantization="nf4", judge_batch_size=1,
    ))
    monkeypatch.setattr(classifier, "_revision", lambda: "a" * 40)
    monkeypatch.setattr(classifier, "_load_local_judge", lambda: (None, None, "nf4"))
    monkeypatch.setattr(classifier, "_score_local_batch", lambda *_args: pytest.fail("mixed cache was scored"))

    with pytest.raises(RuntimeError, match="Set CFG.judge_quantization explicitly"):
        classifier.classify_texts([{"question": "Q?", "response": "A"}], cache_path=cache_path)


def test_answer_then_unknown_diagnostic_regex():
    from src.abstention_classifier import is_answer_then_unknown

    assert is_answer_then_unknown("Paul\n\nUnknown")
    assert not is_answer_then_unknown("Unknown")
    assert not is_answer_then_unknown("Paul Unknown Smith")
