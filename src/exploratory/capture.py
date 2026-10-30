"""Exploratory residual and teacher-forced answer feature capture."""
from __future__ import annotations

from contextlib import ExitStack
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.eval_common import build_chat_batches, log_odds_and_p_abs, model_device, to_device
from src.config import CFG
from src.exploratory.chat import render_ids


def make_condition_contract(base_contract, *, model_name, system_prompt, suffix_ids, n_layers,
                            d_model, is_layer_output_tuple, unknown_token_ids, date_string=CFG.chat_template_date):
    """Return an independent runtime contract for one exploratory prompt/model condition."""
    contract = dict(base_contract)
    contract.update({"model_name": str(model_name), "system_prompt_used": str(system_prompt),
                     "suffix_token_ids": [int(x) for x in suffix_ids], "k_suffix": len(suffix_ids),
                     "n_layers": int(n_layers), "d_model": int(d_model),
                     "is_layer_output_tuple": bool(is_layer_output_tuple),
                     "unknown_token_ids": [int(x) for x in unknown_token_ids]})
    if date_string is None:
        contract.pop("date_string", None)
    else:
        contract["date_string"] = str(date_string)
    return contract


def _contract_batches(tokenizer, prompts, contract, batch_size):
    if not contract.get("date_string"):
        return build_chat_batches(tokenizer, list(prompts), contract["system_prompt_used"],
                                  list(contract["suffix_token_ids"]), batch_size)
    if tokenizer.padding_side != "left":
        raise AssertionError("Exploratory capture requires left padding")
    import torch
    from src.activation_cache import _position_ids_from_attention_mask
    batches = []
    for start in range(0, len(prompts), batch_size):
        rows = [render_ids(tokenizer, p, contract["system_prompt_used"], contract["suffix_token_ids"],
                           contract["date_string"]) for p in prompts[start:start + batch_size]]
        encoded = tokenizer.pad({"input_ids": rows}, return_tensors="pt", padding=True)
        encoded["position_ids"] = _position_ids_from_attention_mask(encoded["attention_mask"])
        tails = encoded["input_ids"][:, -len(contract["suffix_token_ids"]):].cpu()
        expected = torch.tensor(contract["suffix_token_ids"], dtype=tails.dtype)
        assert torch.equal(tails, expected.expand_as(tails)), "Post-instruction suffix token IDs mismatch"
        batches.append(encoded)
    return batches


@torch.no_grad()
def capture_suffix_all_layers(raw_model, tokenizer, contract, prompts: Sequence[str],
                              positions=(-5, -4, -3, -2, -1), batch_size=16):
    """Capture every layer's post-block residual at each suffix position."""
    positions = tuple(int(p) for p in positions)
    n_layers, d_model = int(contract["n_layers"]), int(contract["d_model"])
    batches = _contract_batches(tokenizer, prompts, contract, batch_size)
    layers = raw_model.model.layers
    if len(layers) != n_layers:
        raise ValueError(f"Contract expects {n_layers} layers, found {len(layers)}")
    residual_batches, logits_batches = [], []
    device = model_device(raw_model)
    for batch in batches:
        captured = {}
        handles = []
        for layer_idx, layer in enumerate(layers):
            def hook(_module, _inputs, output, idx=layer_idx):
                hidden = output[0] if contract["is_layer_output_tuple"] else output
                captured[idx] = hidden[:, list(positions), :].float().detach()
                return None
            handles.append(layer.register_forward_hook(hook))
        try:
            out = raw_model(**to_device(batch, device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(n_layers)):
            raise RuntimeError("Suffix capture hook did not fire for every layer")
        # [batch, position, layer, hidden]
        acts = torch.stack([captured[i] for i in range(n_layers)], dim=2)
        residual_batches.append(acts.cpu().numpy().astype(np.float16, copy=False))
        logits = out.logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        log_odds, _ = log_odds_and_p_abs(logits, contract["unknown_token_ids"])
        logits_batches.append({"logits": logits.cpu().numpy(), "maxp": probs.max(-1).values.cpu().numpy(),
                               "entropy": (-(probs * probs.clamp(min=1e-12).log()).sum(-1)).cpu().numpy(),
                               "log_odds": log_odds.cpu().numpy()})
    if not residual_batches:
        resid = np.empty((0, len(positions), n_layers, d_model), dtype=np.float16)
        conf = {k: np.empty((0,), dtype=np.float32) for k in ("logits", "maxp", "entropy", "log_odds")}
    else:
        resid = np.concatenate(residual_batches, axis=0)
        conf = {key: np.concatenate([b[key] for b in logits_batches], axis=0) for key in logits_batches[0]}
    return {"resid": resid, **conf}


@torch.no_grad()
def capture_answer_positions(raw_model, tokenizer, contract, rows, return_prompt_logits=False):
    """Teacher-force saved responses and capture residuals after first/last answer token."""
    n, n_layers, d_model = len(rows), int(contract["n_layers"]), int(contract["d_model"])
    residuals = np.full((n, 2, n_layers, d_model), np.nan, dtype=np.float16)
    means = np.full(n, np.nan, dtype=np.float32)
    minima = np.full(n, np.nan, dtype=np.float32)
    greedy = np.zeros(n, dtype=bool)
    first_match = np.zeros(n, dtype=bool)
    token_counts = np.zeros(n, dtype=np.int32)
    prompt_logits = None
    device = model_device(raw_model)
    layers = raw_model.model.layers
    for row_idx, row in enumerate(rows):
        prompt_ids = render_ids(tokenizer, row["question"], contract["system_prompt_used"],
                                contract["suffix_token_ids"], contract.get("date_string"))
        answer_ids = tokenizer.encode(row.get("baseline_response", ""), add_special_tokens=False)
        answer_ids = [int(x) for x in answer_ids]
        token_counts[row_idx] = len(answer_ids)
        if not answer_ids:
            continue
        all_ids = prompt_ids + answer_ids
        input_ids = torch.tensor([all_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        pos_ids = torch.arange(len(all_ids), dtype=torch.long, device=device).unsqueeze(0)
        captured = {}
        handles = []
        for layer_idx, layer in enumerate(layers):
            def hook(_module, _inputs, output, idx=layer_idx):
                hidden = output[0] if contract["is_layer_output_tuple"] else output
                captured[idx] = hidden[0, len(prompt_ids):, :].float().detach()
                return None
            handles.append(layer.register_forward_hook(hook))
        try:
            out = raw_model(input_ids=input_ids, attention_mask=attn, position_ids=pos_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(range(n_layers)):
            raise RuntimeError("Answer capture hook did not fire for every layer")
        residuals[row_idx, 0] = torch.stack([captured[l][0] for l in range(n_layers)]).cpu().numpy().astype(np.float16)
        residuals[row_idx, 1] = torch.stack([captured[l][-1] for l in range(n_layers)]).cpu().numpy().astype(np.float16)
        logits = out.logits[0].float()
        if return_prompt_logits:
            if prompt_logits is None:
                prompt_logits = np.full((n, logits.shape[-1]), np.nan, dtype=np.float32)
            prompt_logits[row_idx] = logits[len(prompt_ids) - 1].cpu().numpy()
        prompt_last = len(prompt_ids) - 1
        answer_logits = logits[prompt_last:prompt_last + len(answer_ids)]
        targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(answer_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
        means[row_idx], minima[row_idx] = float(log_probs.mean().cpu()), float(log_probs.min().cpu())
        predictions = answer_logits.argmax(dim=-1)
        greedy[row_idx] = bool(torch.equal(predictions, targets))
        first_match[row_idx] = bool(int(predictions[0]) == answer_ids[0])
    result = {"resid_ans": residuals, "seq_logprob_mean": means, "seq_logprob_min": minima,
              "greedy_consistent": greedy, "first_token_match": first_match, "n_answer_tokens": token_counts}
    if return_prompt_logits:
        result["prompt_last_logits"] = prompt_logits
    return result
