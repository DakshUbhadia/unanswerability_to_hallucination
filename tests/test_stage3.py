import json
from pathlib import Path

import pytest
import torch

from src.activation_cache import (format_and_verify_batch_inputs, load_and_validate_train_pairs,
                                  save_stage3_artifacts, verify_stage3_outputs,
                                  _check_batch_single_parity, _position_ids_from_attention_mask)
from src.config import CFG


class DummyTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    chat_template = "dummy-chat-template-v1"
    def __init__(self, right=False, corrupt=False):
        self.padding_side = "right" if right else "left"
        self.corrupt = corrupt

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 7, 9 if self.corrupt else 8]

    def pad(self, features, return_tensors="pt", padding=True):
        rows = [list(row) for row in features["input_ids"]]
        if self.corrupt:
            rows[-1][-1] = 9
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.ones(len(rows), 4, dtype=torch.long)}


def _synthetic(n=8, layers=4, k=5, d=16):
    a = torch.randn(n, layers, k, d) + torch.arange(n).view(n, 1, 1, 1) * 0.02
    u = torch.randn(n, layers, k, d) + 2 + torch.arange(n).view(n, 1, 1, 1) * 0.02
    pairs = [f"pair_{i + 1:04d}" for i in range(n)]
    cache = {
        "train_ans": a.double().mean(0).float(), "train_unans": u.double().mean(0).float(),
        "train_pair_acts_ans": a, "train_pair_acts_unans": u,
        "pair_ids": pairs, "template_ids": [f"template_{i}" for i in range(n)],
        "families": ["A"] * (n // 2) + ["B"] * (n // 2),
        "model_name": "synthetic", "system_prompt_used": "system", "n_layers": layers,
        "k_suffix": k, "d_model": d, "suffix_indices": list(range(-k, 0)),
        "suffix_token_ids": [1, 2, 3, 4, 5],
    }
    diff = torch.linalg.vector_norm(cache["train_unans"] - cache["train_ans"], dim=-1)
    manifest = {
        "stage": 3, "model_name": "synthetic", "runtime_dtype": "float32", "cache_dtype": "float32",
        "n_train_pairs": n, "n_family_a_pairs": n // 2, "n_family_b_pairs": n // 2,
        "n_layers": layers, "k_suffix": k, "d_model": d, "suffix_indices": list(range(-k, 0)),
        "suffix_token_ids": [1, 2, 3, 4, 5], "train_ans_shape": [layers, k, d],
        "train_pair_acts_shape": [n, layers, k, d], "batch_vs_single_min_cosine": 1.0,
        "batch_vs_single_max_rel_l2_diff": 0.0, "batch_vs_single_max_abs_diff": 0.0,
        "batch_vs_single_allclose_passed": True, "batch_vs_single_allclose_mismatch_count": 0,
        "min_prompt_std_ans": float(a.std(0).min()),
        "min_prompt_std_unans": float(u.std(0).min()), "min_diff_in_means_norm": float(diff.min()),
        "max_diff_in_means_norm": float(diff.max()), "elapsed_sec": 0.1, "passed_acceptance_gate": True,
    }
    return cache, manifest


def _write(tmp_path, cache=None, manifest=None):
    if cache is None or manifest is None:
        cache, manifest = _synthetic()
    cp, mp = tmp_path / "cache.pt", tmp_path / "manifest.json"
    save_stage3_artifacts(cache, manifest, cp, mp)
    return cp, mp


def test_load_and_validate_train_pairs_real_repo_data():
    pairs = load_and_validate_train_pairs()
    splits = json.loads((Path(__file__).parents[1] / "data/splits.json").read_text())
    assert len(pairs) == 192
    assert [p["pair_id"] for p in pairs] == sorted(p["pair_id"] for p in pairs)
    assert sum(p["family"] == "A" for p in pairs) == sum(p["family"] == "B" for p in pairs) == 96
    assert not ({p["template_id"] for p in pairs} & (set(splits["val_template_ids"]) | set(splits["test_template_ids"])))


def test_format_and_verify_batch_inputs_suffix_alignment():
    got = format_and_verify_batch_inputs(DummyTokenizer(), ["a", "b"], "sys", [7, 8], "cpu")
    assert got["input_ids"].shape == (2, 4)
    assert torch.equal(got["position_ids"], torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]))
    with pytest.raises(AssertionError):
        format_and_verify_batch_inputs(DummyTokenizer(right=True), ["a"], "sys", [7, 8], "cpu")
    with pytest.raises(AssertionError):
        format_and_verify_batch_inputs(DummyTokenizer(corrupt=True), ["a", "b"], "sys", [7, 8], "cpu")


def test_position_ids_ignore_left_padding():
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    expected = torch.tensor([[0, 0, 0, 1, 2], [0, 0, 1, 2, 3]])
    assert torch.equal(_position_ids_from_attention_mask(mask), expected)


def test_verify_stage3_outputs_valid_synthetic_cache(tmp_path):
    cp, mp = _write(tmp_path)
    assert verify_stage3_outputs(cp, mp, expected_n_pairs=8)["passed_acceptance_gate"] is True


def test_float16_ulp_parity_gate_accepts_single_ulp_shift():
    single = torch.full((1, 1, 3072), 40.0, dtype=torch.float32)
    batched = single.clone()
    batched[0, 0, 0] += 0.03125  # One float16 ULP at magnitude 40.
    min_cos, max_rel_l2, max_diff, mismatch_count = _check_batch_single_parity(batched, single, "float16")
    assert min_cos >= CFG.stage3_min_batch_single_cosine
    assert max_rel_l2 <= CFG.stage3_max_batch_single_rel_l2
    assert torch.allclose(batched, single, rtol=CFG.stage3_allclose_rtol, atol=CFG.stage3_allclose_atol)
    assert max_diff <= CFG.stage3_max_batch_single_abs_fp16
    assert mismatch_count == 0

    corrupted = single.clone()
    corrupted[0, 0, 0] += 0.5
    with pytest.raises(AssertionError, match="coordinates outside allclose"):
        _check_batch_single_parity(corrupted, single, "float16")


@pytest.mark.parametrize("mutation", ["nan", "constant", "mean"])
def test_verify_stage3_outputs_rejects_bad_numerics(tmp_path, mutation):
    cache, manifest = _synthetic()
    if mutation == "nan":
        cache["train_pair_acts_ans"][0, 0, 0, 0] = float("nan")
    elif mutation == "constant":
        cache["train_pair_acts_ans"][:] = 1
    else:
        cache["train_ans"] += 1
    cp, mp = _write(tmp_path, cache, manifest)
    with pytest.raises(AssertionError):
        verify_stage3_outputs(cp, mp, expected_n_pairs=8)


@pytest.mark.parametrize("mutation", ["ids", "families"])
def test_verify_stage3_outputs_rejects_bad_metadata(tmp_path, mutation):
    cache, manifest = _synthetic()
    if mutation == "ids":
        cache["pair_ids"][0], cache["pair_ids"][1] = cache["pair_ids"][1], cache["pair_ids"][0]
    else:
        cache["families"] = ["A"] * 7 + ["B"]
    cp, mp = _write(tmp_path, cache, manifest)
    with pytest.raises(AssertionError):
        verify_stage3_outputs(cp, mp, expected_n_pairs=8)


def test_stage3_cli_skips_when_already_verified(tmp_path, capsys):
    cp, mp = _write(tmp_path)
    from scripts.stage3_cache_activations import main
    assert main(["--output-path", str(cp), "--manifest-path", str(mp)]) == 0
    assert "[SKIP]" in capsys.readouterr().out


@pytest.mark.integration
def test_stage3_real_artifacts_verified():
    from src.config import CFG
    if not CFG.stage3_manifest_path.exists():
        pytest.skip("Stage 3 manifest has not been generated")
    manifest = json.loads(CFG.stage3_manifest_path.read_text())
    assert manifest["passed_acceptance_gate"] is True
    assert manifest["n_train_pairs"] == 192 and manifest["n_layers"] == 28
    assert manifest["d_model"] == 3072 and manifest["k_suffix"] == 5
    assert manifest["batch_vs_single_min_cosine"] >= CFG.stage3_min_batch_single_cosine
    assert manifest["batch_vs_single_max_rel_l2_diff"] <= CFG.stage3_max_batch_single_rel_l2
    assert manifest["batch_vs_single_allclose_passed"] is True
    assert manifest["batch_vs_single_allclose_mismatch_count"] == 0
    if CFG.stage3_out_path.exists():
        verify_stage3_outputs()
