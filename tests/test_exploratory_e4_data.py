import hashlib
import json

from scripts.exploratory.e4_prepare_data import (
    _prepare_squad_rows,
    _rows_selfaware,
    _sample_title_capped,
    prepare,
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()


def _squad_fixture():
    rows = []
    for y in (0, 1):
        for title_ix in range(60):
            for item_ix in range(7):
                rows.append({"id": f"{y}-{title_ix}-{item_ix}", "title": f"article-{title_ix}",
                             "context": "short context", "question": "question",
                             "answers": {"text": [] if y else ["answer"]}})
    return rows


def test_squad_sampling_is_balanced_and_title_capped():
    prepared, _ = _prepare_squad_rows(_squad_fixture(), TinyTokenizer())
    sampled, counts = _sample_title_capped(prepared, 300, 20260415, cap=6)
    assert counts == {"0": 300, "1": 300}
    per_class_title = {}
    for row in sampled:
        key = (row["y"], row["title"])
        per_class_title[key] = per_class_title.get(key, 0) + 1
    assert max(per_class_title.values()) <= 6


def test_squad_context_length_filter_uses_tokenizer_count():
    rows = _squad_fixture()[:2]
    rows[0]["context"] = "one two three four"
    rows[1]["context"] = "one two"
    kept, dropped = _prepare_squad_rows(rows, TinyTokenizer(), max_context_tokens=3)
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["id"] == rows[1]["id"]


def test_sampling_is_deterministic_for_fixed_seed():
    prepared, _ = _prepare_squad_rows(_squad_fixture(), TinyTokenizer())
    a, _ = _sample_title_capped(prepared, 300, 20260415, cap=6)
    b, _ = _sample_title_capped(prepared, 300, 20260415, cap=6)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_selfaware_json_envelope_and_binary_labels_are_parsed():
    parsed = _rows_selfaware({"dataset_name": "SelfAware", "example": [
        {"question_id": 2, "question": "Answerable?", "answer": ["yes"],
         "answerable": True, "source": "squadqa_train"},
        {"question_id": 9, "question": "No consensus?", "answer": None,
         "answerable": False, "source": "SelfAware"},
    ]})
    assert [row["y"] for row in parsed] == [0, 1]
    assert parsed[0]["answer"] == ["yes"]
    assert parsed[1]["answer"] is None


def test_manifest_hash_matches_written_syn_fixture(tmp_path):
    manifest = prepare(["syn"], out_dir=tmp_path)
    path = tmp_path / "syn.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert manifest["sources"]["syn"]["sha256"] == digest
    assert saved["sources"]["syn"]["row_count"] == 768
    assert saved["sources"]["syn"]["sha256"] == digest
