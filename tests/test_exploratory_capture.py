import numpy as np
import torch

from src.eval_common import build_chat_batches, capture_residuals_and_logits
from src.exploratory.capture import capture_answer_positions, capture_suffix_all_layers, make_condition_contract
from src.tokenization_contract import canonical_input_ids
from tests._tiny_model import StubTokenizer, tiny_contract, tiny_model


def _setup():
    tok = StubTokenizer()
    model = tiny_model(n_layers=3, d_model=32, seed=19)
    contract = tiny_contract(tok, n_layers=3, d_model=32)
    return model, tok, contract


def test_suffix_capture_matches_native_capture_for_every_layer_and_position():
    model, tok, contract = _setup()
    prompts = ["Who wrote a book?", "What color is the sky?"]
    got = capture_suffix_all_layers(model, tok, contract, prompts, batch_size=2)
    batches = build_chat_batches(tok, prompts, contract["system_prompt_used"], contract["suffix_token_ids"], 2)
    for j, pos in enumerate((-5, -4, -3, -2, -1)):
        expected, _ = capture_residuals_and_logits(model, batches, contract, [0, 1, 2], pos)
        np.testing.assert_allclose(got["resid"][:, j].astype(np.float32),
                                   expected.numpy().astype(np.float32), atol=0.01, rtol=0.001)


def test_answer_pass_prompt_logits_match_suffix_and_empty_answer_is_nan():
    model, tok, contract = _setup()
    prompt = "A short prompt"
    pids = canonical_input_ids(tok, prompt, contract["system_prompt_used"], contract["suffix_token_ids"])
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([pids]), attention_mask=torch.ones((1, len(pids)), dtype=torch.long),
                       position_ids=torch.arange(len(pids)).unsqueeze(0)).logits[0, -1].numpy()
    row = {"question": prompt, "baseline_response": "abc"}
    answer = capture_answer_positions(model, tok, contract, [row], return_prompt_logits=True)
    np.testing.assert_allclose(answer["prompt_last_logits"][0], logits, atol=1e-5, rtol=1e-5)
    empty = capture_answer_positions(model, tok, contract, [{**row, "baseline_response": ""}])
    assert np.isnan(empty["resid_ans"]).all()
    assert np.isnan(empty["seq_logprob_mean"]).all()


def test_greedy_consistency_tracks_teacher_forced_tokens():
    tok = StubTokenizer()
    model = tiny_model(n_layers=3, d_model=32, seed=0)
    contract = tiny_contract(tok, n_layers=3, d_model=32)
    prompt, generated = None, None
    # Find a one-token greedy completion that round-trips through the byte fixture's UTF-8 decoder.
    for candidate in (f"Generate {i}" for i in range(100)):
        pids = canonical_input_ids(tok, candidate, contract["system_prompt_used"], contract["suffix_token_ids"])
        with torch.no_grad():
            token = int(model(input_ids=torch.tensor([pids]),
                              attention_mask=torch.ones((1, len(pids)), dtype=torch.long),
                              position_ids=torch.arange(len(pids)).unsqueeze(0)).logits[0, -1].argmax())
        if 35 <= token <= 126:
            prompt, generated = candidate, [token]
            break
    assert generated is not None
    text = tok.decode(generated)
    row = {"question": prompt, "baseline_response": text}
    result = capture_answer_positions(model, tok, contract, [row])
    if result["n_answer_tokens"][0]:
        assert result["greedy_consistent"][0]
        corrupt = {**row, "baseline_response": text + "z"}
        changed = capture_answer_positions(model, tok, contract, [corrupt])
        assert not changed["greedy_consistent"][0]


def test_neutral_condition_contract_changes_early_positions_and_preserves_suffix():
    model, tok, registered = _setup()
    neutral = make_condition_contract(registered, model_name="tiny-llama",
                                      system_prompt="neutral system prompt", suffix_ids=registered["suffix_token_ids"],
                                      n_layers=3, d_model=32, is_layer_output_tuple=True,
                                      unknown_token_ids=registered["unknown_token_ids"])
    original = dict(registered)
    registered_capture = capture_suffix_all_layers(model, tok, registered, ["same question"])
    neutral_capture = capture_suffix_all_layers(model, tok, neutral, ["same question"])
    assert registered["system_prompt_used"] == original["system_prompt_used"]
    assert neutral["suffix_token_ids"] == registered["suffix_token_ids"]
    assert not np.allclose(registered_capture["resid"][:, 0], neutral_capture["resid"][:, 0])
