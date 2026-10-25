import hashlib
import json

import torch

from src.activation_cache import format_and_verify_batch_inputs
from src.config import CFG
from src.eval_common import read_jsonl
from src.tokenization_contract import (CONTRACT_VERSION, canonical_input_ids, chat_template_sha256,
                                      input_ids_sha256)
from tests._tiny_model import SUFFIX, StubTokenizer


def test_single_bos_and_stage3_fixture_equality():
    tokenizer = StubTokenizer()
    prompts = ["alpha", "Mara Quinn", "What is the result?", "No details."]
    expected = [
        [2, 86, 61, 118, 124, 118, 119, 104, 112, 35, 115, 117, 114, 112, 115, 127, 88, 61, 100, 111, 115, 107, 100, 63, 100, 118, 65, 13],
        [2, 86, 61, 118, 124, 118, 119, 104, 112, 35, 115, 117, 114, 112, 115, 127, 88, 61, 80, 100, 117, 100, 35, 84, 120, 108, 113, 113, 63, 100, 118, 65, 13],
        [2, 86, 61, 118, 124, 118, 119, 104, 112, 35, 115, 117, 114, 112, 115, 127, 88, 61, 90, 107, 100, 119, 35, 108, 118, 35, 119, 107, 104, 35, 117, 104, 118, 120, 111, 119, 66, 63, 100, 118, 65, 13],
        [2, 86, 61, 118, 124, 118, 119, 104, 112, 35, 115, 117, 114, 112, 115, 127, 88, 61, 81, 114, 35, 103, 104, 119, 100, 108, 111, 118, 49, 63, 100, 118, 65, 13],
    ]
    got = [canonical_input_ids(tokenizer, p, "system prompt here") for p in prompts]
    assert got == expected
    assert all(row[0] == tokenizer.bos_token_id and row.count(tokenizer.bos_token_id) == 1 for row in got)
    assert all(row[-len(tokenizer.encode(SUFFIX)):] == tokenizer.encode(SUFFIX) for row in got)


def test_twenty_popqa_prompts_match_left_padded_stage3_rows():
    tokenizer = StubTokenizer()
    prompts = [row["question"] for row in read_jsonl(CFG.data_dir / "popqa_sample.jsonl")[:20]]
    suffix_ids = tokenizer.encode(SUFFIX)
    expected = [canonical_input_ids(tokenizer, p, CFG.system_prompt, suffix_ids) for p in prompts]
    batch = format_and_verify_batch_inputs(tokenizer, prompts, CFG.system_prompt, suffix_ids, "cpu")
    mask = batch["attention_mask"].bool()
    got = [batch["input_ids"][i][mask[i]].tolist() for i in range(len(prompts))]
    assert got == expected


def test_hash_stability_and_version_binding():
    ids = [2, 10, 11]
    digest = input_ids_sha256(ids, "model-x", "template-hash")
    payload = {"ids": ids, "tok": "model-x", "tpl": "template-hash", "v": CONTRACT_VERSION}
    assert digest == hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()
    assert digest == input_ids_sha256(ids, "model-x", "template-hash")
    assert digest != input_ids_sha256(ids + [12], "model-x", "template-hash")
    assert len(chat_template_sha256(StubTokenizer())) == 64
