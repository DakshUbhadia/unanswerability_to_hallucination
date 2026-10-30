import json

from src.config import CFG
from src.tokenization_contract import (canonical_input_ids, chat_template_sha256,
                                      input_ids_sha256, render_chat)


class DateLlamaStub:
    bos_token_id = 2
    chat_template = "llama-date-stub-v1"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            date_string=None, **kwargs):
        text = f"S:{date_string}|{messages[-1]['content']}!"
        return [2, *[b + 3 for b in text.encode()]] if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return [2, *[b + 3 for b in text.encode()]]


class DateQwenStub:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return f"Q:{messages[-1]['content']}"


def test_shared_renderer_uses_pinned_and_explicit_dates():
    tok = DateLlamaStub()
    msgs = [{"role": "user", "content": "q"}]
    assert "27 Oct 2026" in render_chat(tok, msgs, tokenize=False)
    assert "03 Jan 2000" in render_chat(tok, msgs, tokenize=False, date_string="03 Jan 2000")


def test_canonical_ids_have_fixture_hash_for_pinned_date():
    tok = DateLlamaStub()
    ids = canonical_input_ids(tok, "question", "system")
    got = input_ids_sha256(ids, "fixture-llama", chat_template_sha256(tok))
    assert got == "d1c1f7dbb9d36c58ca880c4b70d1ab4e1184cbf4c0139de2913bf7a722285507"


def test_qwen_template_ignores_pinned_date_keyword():
    tok = DateQwenStub()
    messages = [{"role": "user", "content": "question"}]
    assert render_chat(tok, messages, date_string=None) == render_chat(tok, messages)


def test_preflight_date_hash_gate_passes_or_fails(tmp_path):
    from scripts.exploratory.e3_preflight import _apply_pinned_date_preflight, LLAMA
    tok = DateLlamaStub()
    rows = [{"question": f"q{i}"} for i in range(2)]
    for row in rows:
        ids = canonical_input_ids(tok, row["question"], CFG.system_prompt, [36], CFG.chat_template_date)
        row["input_ids_sha256"] = input_ids_sha256(ids, LLAMA, chat_template_sha256(tok))
    passing = {"template_date": "pinned"}
    assert _apply_pinned_date_preflight(passing, tok, rows, [36], tmp_path / "pass.json") == 0
    assert passing["template_date"] == "pinned" and passing["template_date_value"] == CFG.chat_template_date
    rows[0]["input_ids_sha256"] = "wrong"
    failing = {"template_date": "pinned"}
    failure_path = tmp_path / "fail.json"
    assert _apply_pinned_date_preflight(failing, tok, rows, [36], failure_path) == 3
    assert json.loads(failure_path.read_text())["stop_reason"]


def test_e4_parity_forwards_historical_dates(monkeypatch, tmp_path):
    import json
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from scripts.exploratory import e4_extract_features as e4
    from src import tokenization_contract

    dates_cfg = SimpleNamespace(chat_template_date=CFG.chat_template_date,
                                chat_template_date_stage3_4=CFG.chat_template_date_stage3_4,
                                chat_template_date_stage5_10=CFG.chat_template_date_stage5_10,
                                stage4_out_path=tmp_path / "stage4.pt",
                                stage4_manifest_path=tmp_path / "stage4.json",
                                stage0_out_path=tmp_path / "stage0.json",
                                stage5_out_path=tmp_path / "stage5.json")
    monkeypatch.setattr(e4, "CFG", dates_cfg)
    monkeypatch.setattr("src.eval_common.ensure_stage4_directions", lambda *_: (
        {"diff_in_means": torch.zeros(12, 5, 32).index_put(
            (torch.tensor([11]), torch.tensor([1])), torch.tensor([[1.0] + [0.0] * 31]))}, None))
    (tmp_path / "stage5.json").write_text(json.dumps({"val_unans_baseline_log_odds": 6.779,
                                                         "val_ans_baseline_log_odds": -11.317}))
    dates = []
    original = tokenization_contract.render_chat
    def spy(tokenizer, messages, *, date_string=CFG.chat_template_date, **kw):
        dates.append(date_string)
        return original(DateLlamaStub(), messages, date_string=date_string, **kw)
    monkeypatch.setattr(tokenization_contract, "render_chat", spy)
    contract = {"model_name": "fixture", "system_prompt_used": "system", "suffix_token_ids": [1],
                "n_layers": 12, "d_model": 32, "is_layer_output_tuple": True, "unknown_token_ids": [1]}
    rows = pd.DataFrame({"split": ["train"] * 4 + ["val"] * 4,
                         "y": [0, 0, 1, 1, 0, 0, 1, 1],
                         "log_odds": [0] * 8})
    def fake_capture(_model, _tokenizer, c, prompts, positions, batch_size):
        for p in prompts:
            tokenization_contract.render_chat(None, [{"role": "user", "content": p}], tokenize=False,
                                              date_string=c["date_string"])
        if c["date_string"] == CFG.chat_template_date_stage3_4:
            residual = np.zeros((4, 5, 12, 32), dtype=np.float16)
            residual[2:, 1, 11, 0] = 1
            return {"resid": residual}
        return {"log_odds": np.array([-11.317, -11.317, 6.779, 6.779])}
    monkeypatch.setattr("src.exploratory.capture.capture_suffix_all_layers", fake_capture)
    e4._check_syn_parity(rows, None, None, contract, [str(i) for i in range(8)], 2)
    assert dates == [CFG.chat_template_date_stage3_4] * 4 + [CFG.chat_template_date_stage5_10] * 4

