"""Canonical chat tokenization and stable input identity for measurement pipelines."""
import hashlib
import json
from typing import Any, Optional, Sequence

CONTRACT_VERSION = "chat-template/no-extra-bos/mask-position-ids/v1"

def chat_template_sha256(tokenizer: Any) -> str:
    """Hash the exact active chat-template source used by the tokenizer."""
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        getter = getattr(tokenizer, "get_chat_template", None)
        template = getter() if callable(getter) else None
    if not isinstance(template, str) or not template:
        raise ValueError("Tokenizer does not expose a non-empty chat template")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()

def canonical_input_ids(tokenizer: Any, user_prompt: str, system_prompt: str,
                        expected_suffix_token_ids: Optional[Sequence[int]] = None) -> list[int]:
    """Format one chat and tokenize it without adding tokenizer special tokens.

    The row is unpadded. When supplied, the suffix is checked against the locked
    Stage 0 contract. Exactly one BOS must occur, at index zero.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    ids = [int(token_id) for token_id in ids]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        raise ValueError("Tokenizer has no BOS token ID")
    assert ids and ids[0] == int(bos), "Canonical chat input must start with BOS"
    assert ids.count(int(bos)) == 1, "Canonical chat input must contain exactly one BOS"
    if expected_suffix_token_ids is not None:
        suffix = [int(x) for x in expected_suffix_token_ids]
        assert suffix and ids[-len(suffix):] == suffix, "Post-instruction suffix differs from Stage 0 contract"
    return ids

def input_ids_sha256(ids: Sequence[int], tokenizer_name: str, chat_template_digest: str) -> str:
    payload = {"ids": [int(x) for x in ids], "tok": tokenizer_name,
               "tpl": chat_template_digest, "v": CONTRACT_VERSION}
    encoded = json.dumps(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
