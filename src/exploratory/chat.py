"""Prompt-aware chat rendering for post-registration exploratory conditions."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Sequence

from src.tokenization_contract import canonical_input_ids, input_ids_sha256


def _ids(value: Any) -> list[int]:
    if isinstance(value, dict) or hasattr(value, "input_ids"):
        value = value["input_ids"]
    return [int(x) for x in value]


def discover_suffix(tokenizer, system_prompt: str) -> tuple[int, list[int]]:
    """Find the common generation suffix while varying only the user message."""
    def render(user):
        return _ids(tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True))
    a, b = render(" AlphaUniqueMarker"), render(" BetaUniqueMarkerWithDifferentLength")
    k = 0
    while k < min(len(a), len(b)) and a[-k - 1] == b[-k - 1]:
        k += 1
    if k < 1:
        raise AssertionError("No common post-instruction suffix found across chat prompts")
    return k, a[-k:]


def render_ids(tokenizer, user_prompt: str, system_prompt: str,
               expected_suffix_ids: Sequence[int] | None, date_string: str | None = None) -> list[int]:
    """Render canonical IDs, optionally supplying a chat-template date."""
    if date_string is None:
        return canonical_input_ids(tokenizer, user_prompt, system_prompt, expected_suffix_ids)
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True, date_string=date_string)
    ids = [int(x) for x in tokenizer.encode(rendered, add_special_tokens=False)]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer has no BOS token ID")
    assert ids and ids[0] == int(bos), "Canonical chat input must start with BOS"
    assert ids.count(int(bos)) == 1, "Canonical chat input must contain exactly one BOS"
    if expected_suffix_ids is not None:
        suffix = [int(x) for x in expected_suffix_ids]
        assert suffix and ids[-len(suffix):] == suffix, "Post-instruction suffix differs from Stage 0 contract"
    return ids


def template_date_probe(tokenizer, system_prompt: str) -> dict:
    """Compare ordinary template rendering with an explicitly supplied date."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "E3DateProbeUniqueQuestion"}]
    normal = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    dated = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                          date_string="01 Jan 2000")
    return {"dynamic": normal != dated, "normal_render": normal, "fixed_date_render": dated,
            "fixed_date": "01 Jan 2000"}


def recover_render_date(tokenizer, rows: Sequence[dict], system_prompt: str, model_name: str,
                        suffix_ids: Sequence[int], start_date: date | str,
                        end_date: date | str) -> str | None:
    """Find a date whose canonical input hashes match every supplied archived row."""
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if not rows:
        raise ValueError("At least one archived row is required for date recovery")
    template_digest = __import__("src.tokenization_contract", fromlist=["chat_template_sha256"]).chat_template_sha256(tokenizer)
    for offset in range((end_date - start_date).days + 1):
        candidate = (start_date + timedelta(days=offset)).strftime("%d %b %Y")
        if all(row.get("input_ids_sha256") == input_ids_sha256(
                render_ids(tokenizer, row["question"], system_prompt, suffix_ids, candidate),
                model_name, template_digest) for row in rows):
            return candidate
    return None
