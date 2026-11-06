import csv
import json

import pytest

from scripts import rescore_judge_validation_sheet as rescore


FIELDS = ["uid", "question", "response", "human_label", "notes"]


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path):
    test = {"uid": "Jtest", "question": "Who?", "response": "A malformed name", "human_label": "ANSWER", "notes": ""}
    dev = {"uid": "Jdev", "question": "Where?", "response": "Somewhere", "human_label": "ANSWER", "notes": ""}
    _write_csv(tmp_path / "judge_validation_test.csv", [test])
    _write_csv(tmp_path / "judge_validation_dev.csv", [dev])
    key = {
        "Jtest": {"final_label": "DEGENERATE", "regex_label": "ANSWER", "judge_margin": 2.0, "source": "popqa"},
        "Jdev": {"final_label": "DEGENERATE", "regex_label": "ANSWER", "judge_margin": 2.0, "source": "popqa"},
    }
    manifest = {
        "judge_prompt_version": "v1",
        "judge_prompt_sha256": "old-hash",
        "judge_model_revision": "old-revision",
        "judge_quantization": "nf4",
    }
    (tmp_path / "judge_validation_key.json").write_text(json.dumps(key), encoding="utf-8")
    (tmp_path / "judge_validation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("judge_validation.json", "judge_validation_dev.json"):
        (tmp_path / name).write_text(
            json.dumps({"judge_prompt_version": "v1", "cohen_kappa": 0.0}), encoding="utf-8"
        )
    return [test, dev]


def test_rescore_updates_only_predictions_and_archives_v1(tmp_path, monkeypatch):
    rows = _fixture(tmp_path)

    def fake_classify(items, offline, cache_path):
        assert offline is True
        assert [item["uid"] for item in items] == ["Jtest", "Jdev"]
        return [
            {
                **item,
                "judge_label": "ANSWER",
                "regex_label": "ANSWER",
                "judge_margin": 3.0,
                "judge_quantization": "nf4",
                "judge_prompt_version": rescore.JUDGE_PROMPT_VERSION,
                "judge_prompt_sha256": rescore.JUDGE_PROMPT_SHA256,
                "final_label_source": "judge",
            }
            for item in items
        ]

    monkeypatch.setattr(rescore, "classify_texts", fake_classify)
    monkeypatch.setattr(rescore, "_revision", lambda: "new-revision")
    result = rescore.rescore_frozen_sheets(tmp_path, cache_path=tmp_path / "cache.jsonl")

    assert result["rows"] == 2
    key = json.loads((tmp_path / "judge_validation_key.json").read_text(encoding="utf-8"))
    assert all(row["final_label"] == "ANSWER" and row["judge_margin"] == 3.0 for row in key.values())
    manifest = json.loads((tmp_path / "judge_validation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_prompt_version"] == "v2"
    assert manifest["judge_model_revision"] == "new-revision"
    assert manifest["judge_rescore_history"][0]["judge_prompt_version"] == "v1"
    assert json.loads((tmp_path / "judge_validation_v1.json").read_text())["cohen_kappa"] == 0.0
    assert json.loads((tmp_path / "judge_validation_dev_v1.json").read_text())["cohen_kappa"] == 0.0
    with (tmp_path / "judge_validation_test.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [rows[0]]


def test_rescore_rejects_human_cache_override(tmp_path, monkeypatch):
    _fixture(tmp_path)

    def fake_classify(items, offline, cache_path):
        return [
            {
                **item,
                "judge_label": "ANSWER",
                "regex_label": "ANSWER",
                "judge_margin": 0.2,
                "judge_quantization": "nf4",
                "judge_prompt_version": rescore.JUDGE_PROMPT_VERSION,
                "judge_prompt_sha256": rescore.JUDGE_PROMPT_SHA256,
                "final_label_source": "human" if item["uid"] == "Jtest" else "judge",
            }
            for item in items
        ]

    monkeypatch.setattr(rescore, "classify_texts", fake_classify)
    with pytest.raises(ValueError, match="human cache override"):
        rescore.rescore_frozen_sheets(tmp_path, cache_path=tmp_path / "cache.jsonl")
