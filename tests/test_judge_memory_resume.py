import json

import pytest
import torch

from tests._tiny_model import StubTokenizer, tiny_model
from src import abstention_classifier as classifier


def _old_full_logits_scores(model, tokenizer, prompt):
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
    prompt_ids = classifier._ids(tokenizer, chat)
    labels = [classifier._ids(tokenizer, label) for label in classifier.LABELS]
    seqs = [prompt_ids + ids for ids in labels]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float(), -1)
    return {label: float(logp[i, torch.arange(len(prompt_ids)-1, len(prompt_ids)-1+len(label_ids)),
                            torch.tensor(label_ids)].sum())
            for i, (label, label_ids) in enumerate(zip(classifier.LABELS, labels))}


def test_hidden_state_scorer_matches_full_logits_for_single_and_padded_batch():
    tokenizer = StubTokenizer()
    model = tiny_model(n_layers=2, d_model=16, seed=41)
    prompts = ["Short question", "A much longer question with different prompt length"]
    for prompt in prompts:
        expected = _old_full_logits_scores(model, tokenizer, prompt)
        actual = classifier.score_labels(model, tokenizer, prompt)["judge_logprobs"]
        assert actual == pytest.approx(expected, abs=1e-5)

    prepared_prompts = []
    labels_by_prompt = []
    for prompt in prompts:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        prepared_prompts.append(classifier._ids(tokenizer, chat))
        labels_by_prompt.append([classifier._ids(tokenizer, label) for label in classifier.LABELS])
    actual_rows = classifier._score_prepared_labels(model, tokenizer, prepared_prompts, labels_by_prompt)
    for prompt, row in zip(prompts, actual_rows):
        assert row["judge_logprobs"] == pytest.approx(_old_full_logits_scores(model, tokenizer, prompt), abs=1e-5)


def test_local_classification_appends_each_completed_chunk_and_resumes(monkeypatch, tmp_path):
    cache = tmp_path / "cache.jsonl"
    monkeypatch.setattr(classifier, "_revision", lambda: "a" * 40)
    monkeypatch.setattr(classifier, "_load_local_judge", lambda: (object(), object(), "nf4"))
    calls = []
    def scorer(items, _batch_size):
        calls.append([item["uid"] for item in items])
        if len(calls) == 3:
            raise RuntimeError("injected chunk failure")
        return [{"judge_label": "ANSWER", "judge_logprobs": {"ABSTAIN": -2, "ANSWER": -1,
                "DEGENERATE": -3}, "judge_margin": 1.0, "judge_quantization": "nf4"} for _ in items]
    monkeypatch.setattr(classifier, "_score_local_batch", scorer)
    items = [{"uid": str(i), "question": f"q{i}", "response": f"r{i}"} for i in range(5)]
    with pytest.raises(RuntimeError, match="injected chunk failure"):
        classifier.classify_texts(items, backend="local", cache_path=cache, batch_size=2)
    assert len(cache.read_text().splitlines()) == 4
    calls.clear()
    result = classifier.classify_texts(items, backend="local", cache_path=cache, batch_size=2)
    assert calls == [["4"]]
    assert len(result) == 5
    assert len(cache.read_text().splitlines()) == 5
    assert all(json.loads(line)["judge_quantization"] == "nf4" for line in cache.read_text().splitlines())
