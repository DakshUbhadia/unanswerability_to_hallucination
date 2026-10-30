import csv
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import import_grader_labels, judge_pass


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_judge_collect_uses_neutral_rows_own_question(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    _jsonl(data / "pairs.jsonl", [])
    _jsonl(data / "b_ctrl.jsonl", [])
    _jsonl(data / "popqa_sample.jsonl", [{"popqa_id": "p1", "question": "registered question", "baseline_response": "registered answer"}])
    neutral = tmp_path / "neutral.jsonl"
    _jsonl(neutral, [{"popqa_id": "p1", "question": "neutral question", "response": "neutral response"}])
    monkeypatch.setattr(judge_pass, "CFG", SimpleNamespace(data_dir=data))

    items, _, _ = judge_pass.collect([neutral])

    assert items == [{"uid": "p1", "question": "neutral question", "response": "neutral response",
                      "conditions": ["baseline"], "source_paths": [str(neutral)]}]


def test_grader_import_accepts_custom_key_and_keeps_default_paths(tmp_path, monkeypatch):
    csv_path, key_path, output = tmp_path / "labels.csv", tmp_path / "key.json", tmp_path / "out.json"
    csv_path.write_text("uid,human_label\nu1,CORRECT\n", encoding="utf-8")
    key_path.write_text(json.dumps({"u1": {"grade": "correct", "grade_reason": "alias"}}), encoding="utf-8")
    monkeypatch.setattr(import_grader_labels, "CFG", SimpleNamespace(
        results_dir=tmp_path, grader_validation_seed=20260409))
    monkeypatch.setattr("sys.argv", ["import_grader_labels", "--csv", str(csv_path), "--key", str(key_path), "--output", str(output)])

    import_grader_labels.main()

    assert json.loads(output.read_text(encoding="utf-8"))["precision_correct"] == 1.0
    assert import_grader_labels.CFG.results_dir / "grader_validation_key.json" == tmp_path / "grader_validation_key.json"
