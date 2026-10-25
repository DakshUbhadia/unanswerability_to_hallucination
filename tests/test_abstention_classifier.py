import json

import pytest

from src.abstention_classifier import (ABSTAIN, ANSWER, DEGENERATE, JUDGE_PROMPT_SHA256,
                                       classify_texts, is_abstention, regex_label)


@pytest.mark.parametrize("text, expected", [
    ("Unknown", ABSTAIN),
    ("There is no information available about the director", ABSTAIN),
    ("There is no director associated with this film", ABSTAIN),
    ("There is no well-known author by that name", ABSTAIN),
    ("The answer is: Unknown", ABSTAIN),
    ("Mara Quinn", ANSWER),
    ("Thriller", ANSWER),
    ("None", ANSWER),
    ("Verification note: Verified!", ANSWER),
])
def test_expanded_regex_regressions(text, expected):
    assert regex_label(text) == expected


def test_judge_parsing_cache_retry_and_offline(tmp_path):
    cache = tmp_path / "judge_cache.jsonl"
    calls = []
    outputs = iter(["not a label", "  abstain  "])
    def judge(prompt):
        calls.append(prompt)
        return next(outputs)
    item = {"uid": "x1", "question": "Who directed it?", "response": "No information available."}
    first = classify_texts([item], backend="api", judge_fn=judge, cache_path=cache, rate_limit_seconds=0)
    assert first[0]["final_label"] == ABSTAIN and first[0]["parse_attempts"] == 2
    assert len(calls) == 2 and "Who directed it?" in calls[0] and "No information available." in calls[0]
    assert first[0]["judge_prompt_sha256"] == JUDGE_PROMPT_SHA256
    second = classify_texts([item], backend="api", judge_fn=lambda _: pytest.fail("cache should prevent a call"),
                            cache_path=cache, offline=True, rate_limit_seconds=0)
    assert second[0]["final_label"] == ABSTAIN
    with pytest.raises(RuntimeError, match="cache miss"):
        classify_texts([{**item, "response": "different"}], backend="api", cache_path=cache, offline=True)
    cached = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert len(cached) == 1 and cached[0]["judge_label"] == ABSTAIN


def test_invalid_judge_is_unparseable_after_one_retry(tmp_path):
    calls = []
    def judge(_):
        calls.append(1)
        return "Maybe"
    out = classify_texts([{"uid": "x", "question": "Q?", "response": "R"}], backend="api",
                         judge_fn=judge, cache_path=tmp_path / "cache.jsonl", rate_limit_seconds=0)
    assert len(calls) == 2 and out[0]["final_label"] == "UNPARSEABLE"
    assert out[0]["agree"] is None


def test_regex_only_is_nonfinal_and_rejected(tmp_path):
    row = classify_texts([{"uid": "x", "question": "Q?", "response": "Unknown"}],
                         backend="regex_only", cache_path=tmp_path / "cache.jsonl")[0]
    assert row["final_label"] == ABSTAIN and row["classifier_final"] is False
    with pytest.raises(ValueError, match="not final"):
        is_abstention(row)


def test_direct_legacy_calls_are_forbidden_outside_wrapper():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    hits = []
    for base in (root / "src", root / "scripts"):
        for path in base.rglob("*.py"):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "classify_abstention_response(" in line and path.name != "pilot_check.py":
                    hits.append(f"{path}:{line_no}:{line.strip()}")
    assert not hits, "Direct legacy classifier calls remain:\n" + "\n".join(hits)
