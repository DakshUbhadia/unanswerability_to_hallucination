"""Stage 7: capability retention (WikiText-2 token-level KL for all directions, chat MMLU accuracy)."""
import hashlib
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import CFG
from src.eval_common import (ConditionCheckpoint, atomic_write_json, build_chat_batches, control_specs,
                             evaluate_condition, model_device, read_json, slice_vectors_at_star, to_device)
from src.model_utils import native_intervention_hooks
from src.stats_utils import item_bootstrap_mean, summarize_null

OPTION_LETTERS = ("A", "B", "C", "D")
STAGE7_KEYS = {"stage", "model_name", "l_star", "i_star", "pos_idx", "n_wikitext_passages", "wikitext_tokens",
               "n_kl_positions", "wikitext_passage_rule_used", "wikitext_baseline_nll", "mmlu_evaluated", "mmlu_n_questions", "mmlu_baseline_acc",
               "mmlu_neutral_system_prompt",
               "mmlu_baseline_ci", "learned_direction_capability", "controls_wikitext_kl", "concept_capability",
               "source_stage4_sha256", "source_stage5_sha256", "elapsed_sec", "passed_acceptance_gate"}


# ----------------------------- WikiText-2 ------------------------------------
def _wikitext_rows() -> Iterable[str]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return (row["text"] for row in ds)


def _is_top_level_article_header(text: str) -> bool:
    return bool(re.match(r"^=\s+[^=].*?\s+=$", text.strip()))


def load_wikitext2_passages(tokenizer: Any, n_passages: int = CFG.n_wikitext_kl, max_tokens: int = CFG.len_wikitext_tokens,
                            min_words: int = CFG.min_wikitext_words, rows: Optional[Iterable[str]] = None,
                            passage_rule: Optional[str] = None, return_rule: bool = False):
    """Load full 256-token paragraphs, falling back deterministically to article chunks.

    Smaller custom max_tokens retain the legacy truncation behavior for compact test fixtures.
    """
    rule = passage_rule or CFG.wikitext_passage_rule
    source = list(rows if rows is not None else _wikitext_rows())
    if max_tokens != CFG.len_wikitext_tokens and passage_rule is None:
        passages = []
        for text in source:
            stripped = text.strip()
            if not stripped or stripped.startswith("=") or len(stripped.split()) < min_words:
                continue
            ids = tokenizer.encode(stripped, add_special_tokens=False)[:max_tokens]
            if len(ids) >= 2:
                passages.append([int(x) for x in ids])
            if len(passages) == n_passages:
                break
        assert len(passages) == n_passages, f"Only {len(passages)} WikiText-2 passages found (need {n_passages})"
        return (passages, "legacy_custom") if return_rule else passages
    if rule != "full_length_256":
        raise ValueError(f"Unsupported WikiText passage rule: {rule}")
    passages = []
    for text in source:
        stripped = text.strip()
        if not stripped or stripped.startswith("=") or len(stripped.split()) < min_words:
            continue
        ids = [int(x) for x in tokenizer.encode(stripped, add_special_tokens=False)]
        if len(ids) >= max_tokens:
            passages.append(ids[:max_tokens])
            if len(passages) == n_passages:
                break
    used = "full_length_256"
    if len(passages) < n_passages:
        passages = []
        article_parts, current = [], []
        for text in source:
            stripped = text.strip()
            if _is_top_level_article_header(stripped):
                if current:
                    article_parts.append(current)
                current = []
                continue
            if not stripped:
                continue
            current.append(stripped)
        if current:
            article_parts.append(current)
        chunks = []
        for paragraphs in article_parts:
            text = " ".join(paragraphs)
            ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
            chunks.extend(ids[i:i + max_tokens] for i in range(0, len(ids) - max_tokens + 1, max_tokens))
        if len(chunks) < n_passages:
            raise AssertionError(f"Only {len(chunks)} contiguous WikiText chunks found (need {n_passages})")
        passages = chunks[:n_passages]
        used = "contiguous_chunks_256"
    assert len(passages) == n_passages and all(len(p) == max_tokens for p in passages)
    return (passages, used) if return_rule else passages


def build_wikitext_batches(passages: List[List[int]], pad_token_id: int, batch_size: int) -> List[Dict[str, torch.Tensor]]:
    """Right-padded raw-text batches; `kl_mask[b, t]` marks next-token predictions t -> t+1 inside the passage."""
    batches = []
    for s in range(0, len(passages), batch_size):
        chunk = passages[s:s + batch_size]
        width = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), width), int(pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for i, p in enumerate(chunk):
            ids[i, :len(p)] = torch.tensor(p)
            mask[i, :len(p)] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        batches.append({"input_ids": ids, "attention_mask": mask, "position_ids": pos,
                        "kl_mask": (mask[:, :-1] * mask[:, 1:]).bool()})
    return batches


def _model_inputs(b: Dict[str, torch.Tensor], dev: Any) -> Dict[str, torch.Tensor]:
    return {k: b[k].to(dev) for k in ("input_ids", "attention_mask", "position_ids")}


@torch.no_grad()
def _log_probs_chunks(logits: torch.Tensor, chunk: int) -> List[torch.Tensor]:
    return [F.log_softmax(logits[:, s:s + chunk].float(), dim=-1) for s in range(0, logits.shape[1], chunk)]


@torch.no_grad()
def kl_per_passage(base_chunks: List[torch.Tensor], logits_int: torch.Tensor, kl_mask: torch.Tensor,
                   chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum over valid positions of KL(P_base || P_int) per passage, and the number of valid positions."""
    sums = torch.zeros(logits_int.shape[0], dtype=torch.float64, device=logits_int.device)
    for c, base_lp in enumerate(base_chunks):
        s = c * chunk
        lp_int = F.log_softmax(logits_int[:, s:s + chunk].float(), dim=-1)
        kl = (base_lp.exp() * (base_lp - lp_int)).sum(-1).clamp(min=0.0)
        sums += (kl * kl_mask[:, s:s + chunk]).double().sum(-1)
        del lp_int, kl
    return sums.cpu(), kl_mask.sum(-1).double().cpu()


def evaluate_wikitext_kl_for_intervention(raw_model, passage_batches, base_log_probs, is_layer_output_tuple, mode,
                                          r_vec, target_layer=None, alpha=0.0, chunk: int = 64) -> Tuple[float, float]:
    """Mean and SD of per-passage mean KL for one intervention; `base_log_probs[b]` holds chunked baseline log-probs."""
    dev = model_device(raw_model)
    sums, counts = [], []
    for b, base in zip(passage_batches, base_log_probs):
        with native_intervention_hooks(raw_model, is_layer_output_tuple, mode, r_vec, target_layer, alpha):
            logits = raw_model(**_model_inputs(b, dev), use_cache=False).logits[:, :-1]
        s, n = kl_per_passage([x.to(dev) for x in base], logits, b["kl_mask"].to(dev), chunk)
        sums.append(s); counts.append(n)
    per = (torch.cat(sums) / torch.cat(counts)).numpy()
    return float(torch.cat(sums).sum() / torch.cat(counts).sum()), float(per.std(ddof=1)) if len(per) > 1 else 0.0


def ratio_bootstrap(sums: np.ndarray, counts: np.ndarray, n_boot: int = CFG.n_bootstrap,
                    seed: int = CFG.seed_bootstrap) -> List[float]:
    """Passage-level bootstrap CI of a token-weighted mean (sum KL / number of positions)."""
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(sums), size=(int(n_boot), len(sums)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def wikitext_conditions(vectors: Dict[str, Any], l_star: int) -> List[Tuple[str, str, Any, Optional[int], float]]:
    conds = [("learned|ablate_all|0|+0.00", "ablate_all", vectors["learned_unit"], None, 0.0)]
    alphas = sorted({a for a in CFG.grid_alpha_nec + CFG.grid_alpha_suf if a != 0.0})
    conds += [(f"learned|steer_layer|0|{a:+.2f}", "steer_layer", vectors["learned_diff"], l_star, a) for a in alphas]
    for tier, m, unit, steer in control_specs(vectors):
        conds.append((f"{tier}|ablate_all|{m}|+0.00", "ablate_all", unit, None, 0.0))
        for a in CFG.ctrl_operating_alphas:
            conds.append((f"{tier}|steer_layer|{m}|{a:+.2f}", "steer_layer", steer, l_star, a))
    return conds


@torch.no_grad()
def run_wikitext_kl_sweep(raw_model, passage_batches, conditions, is_layer_output_tuple, checkpoint: ConditionCheckpoint,
                          chunk: int = 64, log_every: int = 1) -> Dict[str, Any]:
    """Batch-outer / condition-inner sweep so each batch's baseline log-probs are computed once."""
    dev = model_device(raw_model)
    n_items = sum(b["input_ids"].shape[0] for b in passage_batches)
    started = time.perf_counter()
    for bi, b in enumerate(passage_batches):
        key = f"wt_batch|{bi}"
        if key in checkpoint:
            continue
        inputs = _model_inputs(b, dev)
        kl_mask = b["kl_mask"].to(dev)
        base_logits = raw_model(**inputs, use_cache=False).logits[:, :-1]
        base_chunks = _log_probs_chunks(base_logits, chunk)
        targets = inputs["input_ids"][:, 1:]
        nll = torch.zeros(targets.shape[0], dtype=torch.float64, device=dev)
        for c, lp in enumerate(base_chunks):
            s = c * chunk
            tok_lp = lp.gather(-1, targets[:, s:s + chunk].unsqueeze(-1)).squeeze(-1)
            nll += (-tok_lp * kl_mask[:, s:s + chunk]).double().sum(-1)
        del base_logits
        batch_res = {"nll_sum": nll.cpu().numpy(), "count": kl_mask.sum(-1).double().cpu().numpy(), "kl": {}}
        for name, mode, vec, layer, alpha in conditions:
            with native_intervention_hooks(raw_model, is_layer_output_tuple, mode, vec, layer, alpha):
                logits = raw_model(**inputs, use_cache=False).logits[:, :-1]
            sums, _ = kl_per_passage(base_chunks, logits, kl_mask, chunk)
            batch_res["kl"][name] = sums.numpy()
            del logits
        checkpoint.put(key, batch_res)
        checkpoint.flush()
        del base_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (bi + 1) % log_every == 0:
            print(f"[Stage 7] WikiText batch {bi + 1}/{len(passage_batches)} x {len(conditions)} conditions "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    parts = [checkpoint.get(f"wt_batch|{bi}") for bi in range(len(passage_batches))]
    counts = np.concatenate([p["count"] for p in parts])
    assert len(counts) == n_items and (counts > 0).all()
    kl_sums = {name: np.concatenate([p["kl"][name] for p in parts]) for name, *_ in conditions}
    return {"counts": counts, "nll_sums": np.concatenate([p["nll_sum"] for p in parts]), "kl_sums": kl_sums}


# ----------------------------- MMLU ------------------------------------------
def _mmlu_rows() -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    return [{"question": r["question"], "subject": r["subject"], "choices": list(r["choices"]),
             "answer": int(r["answer"]), "hf_row_idx": i} for i, r in enumerate(ds)]


def load_stratified_mmlu_questions(n_questions: int = CFG.n_mmlu, seed: int = CFG.seed_mmlu,
                                   rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Stratified across all subjects: floor(n/S) each, +1 for the first n mod S subjects in seeded order."""
    rows = rows if rows is not None else _mmlu_rows()
    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        by_subject.setdefault(r["subject"], []).append({**r, "hf_row_idx": r.get("hf_row_idx", i)})
    subjects = sorted(by_subject)
    rng = random.Random(int(seed))
    order = subjects[:]
    rng.shuffle(order)
    base, extra = divmod(n_questions, len(subjects))
    picked = []
    for rank, subj in enumerate(order):
        k = base + (1 if rank < extra else 0)
        pool = sorted(by_subject[subj], key=lambda r: r["hf_row_idx"])
        assert len(pool) >= k, f"Subject {subj} has only {len(pool)} questions"
        picked.extend(rng.sample(pool, k))
    picked.sort(key=lambda r: (r["subject"], r["hf_row_idx"]))
    assert len(picked) == n_questions
    return picked


def format_mmlu_prompt(q: Dict[str, Any]) -> str:
    opts = "\n".join(f"{letter}. {choice}" for letter, choice in zip(OPTION_LETTERS, q["choices"]))
    return f"{q['question']}\n{opts}\nAnswer with a single letter (A, B, C, or D):"


def mmlu_option_token_ids(tokenizer: Any) -> List[List[int]]:
    return [sorted({int(tokenizer.encode(k, add_special_tokens=False)[-1]),
                    int(tokenizer.encode(" " + k, add_special_tokens=False)[-1])}) for k in OPTION_LETTERS]


def score_mmlu_logits(logits: torch.Tensor, gold: Sequence[int], option_ids: List[List[int]],
                      unknown_ids: Sequence[int]) -> Dict[str, Any]:
    """4-way accuracy from summed option-token probability mass; false abstention = P(Unknown) > best option."""
    probs = F.softmax(logits.float(), dim=-1)
    mass = torch.stack([probs[:, ids].sum(-1) for ids in option_ids], dim=-1)
    pred = mass.argmax(-1).cpu().numpy()
    correct = (pred == np.asarray(gold)).astype(np.float64)
    p_unk = probs[:, list(dict.fromkeys(map(int, unknown_ids)))].sum(-1)
    false_abs = (p_unk > mass.max(-1).values).double().cpu().numpy()
    return {"correct": correct, "false_abstention": false_abs, "pred": pred,
            "option_mass_mean": float(mass.sum(-1).mean())}


def evaluate_mmlu_accuracy_for_intervention(raw_model, tokenizer, mmlu_batches, gold_indices, option_token_ids,
                                            unknown_token_ids, is_layer_output_tuple, mode="none", r_vec=None,
                                            target_layer=None, alpha=0.0) -> Dict[str, Any]:
    contract = {"is_layer_output_tuple": is_layer_output_tuple, "unknown_token_ids": list(unknown_token_ids)}
    res = evaluate_condition(raw_model, mmlu_batches, contract, mode, r_vec, target_layer, alpha, return_logits=True)
    scored = score_mmlu_logits(res["logits"], gold_indices, option_token_ids, unknown_token_ids)
    return {"accuracy": float(scored["correct"].mean()), "false_abstention_rate": float(scored["false_abstention"].mean()),
            "correct": scored["correct"], "false_abstention": scored["false_abstention"],
            "option_mass_mean": scored["option_mass_mean"]}


def mmlu_conditions(vectors: Dict[str, Any], l_star: int) -> List[Tuple[str, str, Any, Optional[int], float]]:
    conds = [("learned|none|0|+0.00", "none", None, None, 0.0),
             ("learned|ablate_all|0|+0.00", "ablate_all", vectors["learned_unit"], None, 0.0)]
    alphas = sorted({a for a in CFG.grid_alpha_nec + CFG.grid_alpha_suf if a != 0.0})
    conds += [(f"learned|steer_layer|0|{a:+.2f}", "steer_layer", vectors["learned_diff"], l_star, a) for a in alphas]
    conds.append(("concept|ablate_all|0|+0.00", "ablate_all", vectors["concept_unit"], None, 0.0))
    conds += [(f"concept|steer_layer|0|{a:+.2f}", "steer_layer", vectors["concept_scaled"], l_star, a)
              for a in CFG.ctrl_operating_alphas]
    return conds


# ----------------------------- orchestration ---------------------------------
def summarize_stage7(wt: Dict[str, Any], mmlu: Optional[Dict[str, Dict[str, Any]]], conditions_wt, n_boot=CFG.n_bootstrap,
                     seed=CFG.seed_bootstrap) -> Dict[str, Any]:
    counts = wt["counts"]
    kl = {name: float(wt["kl_sums"][name].sum() / counts.sum()) for name, *_ in conditions_wt}
    kl_ci = {name: ratio_bootstrap(wt["kl_sums"][name], counts, n_boot, seed) for name, *_ in conditions_wt}

    def mm(name):
        if not mmlu or name not in mmlu:
            return {"acc": None, "ci": None, "false_abs": None, "non_abstained_acc": None,
                    "secondary_acc": None, "secondary_ci": None, "secondary_false_abs": None,
                    "secondary_non_abstained_acc": None}
        primary = mmlu[name]["neutral"]
        c = primary["correct"]
        b = item_bootstrap_mean(c, n_boot, seed)
        fa = primary["false_abstention"]
        non_abs = c[fa == 0]
        secondary = mmlu[name]["abstention_prompt"]
        sc = secondary["correct"]
        sb = item_bootstrap_mean(sc, n_boot, seed)
        sfa = secondary["false_abstention"]
        snon_abs = sc[sfa == 0]
        return {"acc": float(c.mean()), "ci": [b["ci95_low"], b["ci95_high"]],
                "false_abs": float(np.mean(fa)), "non_abstained_acc": float(non_abs.mean()) if len(non_abs) else None,
                "secondary_acc": float(sc.mean()), "secondary_ci": [sb["ci95_low"], sb["ci95_high"]],
                "secondary_false_abs": float(np.mean(sfa)),
                "secondary_non_abstained_acc": float(snon_abs.mean()) if len(snon_abs) else None}

    learned = []
    base = mm("learned|none|0|+0.00")
    learned.append({"mode": "none", "alpha": 0.0, "wikitext_kl": 0.0, "wikitext_kl_ci": [0.0, 0.0],
                    "mmlu_acc": base["acc"], "mmlu_ci": base["ci"],
                    "mmlu_false_abstention_rate": base["false_abs"],
                    "mmlu_non_abstained_acc": base["non_abstained_acc"],
                    "mmlu_abstention_prompt_acc": base["secondary_acc"],
                    "mmlu_abstention_prompt_ci": base["secondary_ci"],
                    "mmlu_abstention_prompt_false_abstention_rate": base["secondary_false_abs"],
                    "mmlu_abstention_prompt_non_abstained_acc": base["secondary_non_abstained_acc"]})
    for name, mode, _, _, alpha in conditions_wt:
        if not name.startswith("learned|"):
            continue
        metrics = mm(name)
        learned.append({"mode": mode, "alpha": float(alpha), "wikitext_kl": kl[name], "wikitext_kl_ci": kl_ci[name],
                        "mmlu_acc": metrics["acc"], "mmlu_ci": metrics["ci"],
                        "mmlu_false_abstention_rate": metrics["false_abs"],
                        "mmlu_non_abstained_acc": metrics["non_abstained_acc"],
                        "mmlu_abstention_prompt_acc": metrics["secondary_acc"],
                        "mmlu_abstention_prompt_ci": metrics["secondary_ci"],
                        "mmlu_abstention_prompt_false_abstention_rate": metrics["secondary_false_abs"],
                        "mmlu_abstention_prompt_non_abstained_acc": metrics["secondary_non_abstained_acc"]})
    controls = {}
    learned_by_op = {"ablate_all": kl["learned|ablate_all|0|+0.00"]}
    for a in CFG.ctrl_operating_alphas:
        learned_by_op[f"alpha={a:+.1f}"] = kl[f"learned|steer_layer|0|{a:+.2f}"]
    for op, (mode, alpha) in {"ablate_all": ("ablate_all", 0.0),
                              **{f"alpha={a:+.1f}": ("steer_layer", a) for a in CFG.ctrl_operating_alphas}}.items():
        entry = {}
        for tier, n in (("random", CFG.n_ctrl_random), ("shuffled", CFG.n_ctrl_shuffled), ("concept", 1)):
            vals = [kl[f"{tier}|{mode}|{m}|{alpha:+.2f}"] for m in range(n)]
            entry[tier] = vals
            if n > 1:
                entry[f"{tier}_summary"] = summarize_null(learned_by_op[op], vals)
        entry["learned"] = learned_by_op[op]
        controls[op] = entry
    concept = []
    for name in ("concept|ablate_all|0|+0.00", *[f"concept|steer_layer|0|{a:+.2f}" for a in CFG.ctrl_operating_alphas]):
        _, mode, _, alpha = name.split("|")
        metrics = mm(name)
        concept.append({"mode": mode, "alpha": float(alpha), "wikitext_kl": kl[name], "wikitext_kl_ci": kl_ci[name],
                        "mmlu_acc": metrics["acc"], "mmlu_ci": metrics["ci"],
                        "mmlu_false_abstention_rate": metrics["false_abs"],
                        "mmlu_non_abstained_acc": metrics["non_abstained_acc"],
                        "mmlu_abstention_prompt_acc": metrics["secondary_acc"],
                        "mmlu_abstention_prompt_ci": metrics["secondary_ci"],
                        "mmlu_abstention_prompt_false_abstention_rate": metrics["secondary_false_abs"],
                        "mmlu_abstention_prompt_non_abstained_acc": metrics["secondary_non_abstained_acc"]})
    return {"wikitext_baseline_nll": float(wt["nll_sums"].sum() / counts.sum()), "n_kl_positions": int(counts.sum()),
            "mmlu_baseline_acc": base["acc"], "mmlu_baseline_ci": base["ci"],
            "mmlu_baseline_false_abstention_rate": base["false_abs"],
            "mmlu_baseline_non_abstained_acc": base["non_abstained_acc"],
            "mmlu_abstention_prompt_baseline_acc": base["secondary_acc"],
            "mmlu_abstention_prompt_baseline_ci": base["secondary_ci"],
            "mmlu_abstention_prompt_baseline_false_abstention_rate": base["secondary_false_abs"],
            "mmlu_abstention_prompt_baseline_non_abstained_acc": base["secondary_non_abstained_acc"],
            "learned_direction_capability": learned, "controls_wikitext_kl": controls, "concept_capability": concept}


def run_stage7_capability_evaluation(raw_model, tokenizer, contract, directions_dict, selection_dict,
                                     batch_size=CFG.batch_size_forward, checkpoint_path=CFG.stage7_checkpoint_path,
                                     force=False, wikitext_passages=None, mmlu_questions=None, run_mmlu=True,
                                     wikitext_batch_size=CFG.batch_size_wikitext, source_stage4_sha256="",
                                     source_stage5_sha256="", kl_chunk=64) -> Dict[str, Any]:
    started = time.perf_counter()
    l_star, i_star, pos_idx = int(selection_dict["l_star"]), int(selection_dict["i_star"]), int(selection_dict["pos_idx"])
    vectors = slice_vectors_at_star(directions_dict, l_star, pos_idx)
    is_tuple = bool(contract["is_layer_output_tuple"])
    if wikitext_passages is not None:
        passages, passage_rule_used = wikitext_passages, "provided"
    else:
        passages, passage_rule_used = load_wikitext2_passages(tokenizer, return_rule=True)
    expected_positions = len(passages) * (int(CFG.len_wikitext_tokens) - 1)
    if passage_rule_used in {"full_length_256", "contiguous_chunks_256"}:
        assert len(passages) == CFG.n_wikitext_kl and expected_positions == 51000
    wt_batches = build_wikitext_batches(passages, tokenizer.pad_token_id, wikitext_batch_size)
    conds_wt = wikitext_conditions(vectors, l_star)
    fp = hashlib.sha256(f"{source_stage4_sha256}|{source_stage5_sha256}|{l_star}|{pos_idx}|{len(passages)}|"
                        f"{sum(map(len, passages))}|{wikitext_batch_size}|{len(conds_wt)}".encode()).hexdigest()
    ckpt = ConditionCheckpoint(checkpoint_path, fp, force=force, every=1)
    print(f"[Stage 7] WikiText-2: {len(passages)} passages, {len(wt_batches)} batches, {len(conds_wt)} conditions", flush=True)
    wt = run_wikitext_kl_sweep(raw_model, wt_batches, conds_wt, is_tuple, ckpt, kl_chunk)

    mmlu_results = None
    n_mmlu = 0
    if run_mmlu:
        questions = mmlu_questions if mmlu_questions is not None else load_stratified_mmlu_questions()
        n_mmlu = len(questions)
        mmlu_prompts = [format_mmlu_prompt(q) for q in questions]
        batches_by_prompt = {
            "neutral": build_chat_batches(tokenizer, mmlu_prompts, CFG.mmlu_neutral_system_prompt,
                                           list(contract["suffix_token_ids"]), batch_size),
            "abstention_prompt": build_chat_batches(tokenizer, mmlu_prompts, contract["system_prompt_used"],
                                                     list(contract["suffix_token_ids"]), batch_size),
        }
        gold = [int(q["answer"]) for q in questions]
        opt_ids = mmlu_option_token_ids(tokenizer)
        mmlu_results = {}
        for name, mode, vec, layer, alpha in mmlu_conditions(vectors, l_star):
            mmlu_results[name] = {}
            for prompt_key, batches in batches_by_prompt.items():
                key = f"mmlu|{prompt_key}|{name}"
                if key not in ckpt:
                    r = evaluate_mmlu_accuracy_for_intervention(raw_model, tokenizer, batches, gold, opt_ids,
                                                                contract["unknown_token_ids"], is_tuple, mode, vec, layer, alpha)
                    ckpt.put(key, {"correct": r["correct"], "false_abstention": r["false_abstention"]})
                got = ckpt.get(key)
                mmlu_results[name][prompt_key] = {"correct": np.asarray(got["correct"]),
                                                  "false_abstention": np.asarray(got["false_abstention"])}
        ckpt.flush()
        print(f"[Stage 7] neutral MMLU baseline acc = {mmlu_results['learned|none|0|+0.00']['neutral']['correct'].mean():.3f}", flush=True)

    summary = summarize_stage7(wt, mmlu_results, conds_wt)
    return {"stage": 7, "model_name": contract["model_name"], "l_star": l_star, "i_star": i_star, "pos_idx": pos_idx,
            "n_wikitext_passages": len(passages), "wikitext_tokens": CFG.len_wikitext_tokens,
            "wikitext_passage_rule_used": passage_rule_used, "mmlu_neutral_system_prompt": CFG.mmlu_neutral_system_prompt,
            "wikitext_batch_size": wikitext_batch_size, "mmlu_evaluated": bool(run_mmlu), "mmlu_n_questions": n_mmlu,
            "mmlu_sampling": "stratified across all subjects (floor(n/S) each, +1 for first n mod S in seeded order)",
            **summary, "n_bootstrap": CFG.n_bootstrap, "source_stage4_sha256": source_stage4_sha256,
            "source_stage5_sha256": source_stage5_sha256, "elapsed_sec": time.perf_counter() - started,
            "passed_acceptance_gate": True}


def save_stage7_artifacts(capability_dict, output_path=CFG.stage7_out_path, table1_csv_path=CFG.table1_csv_path,
                          table1_md_path=CFG.table1_md_path, checkpoint_path=None) -> None:
    from src.results_tables import write_table1
    atomic_write_json(output_path, capability_dict)
    write_table1(table1_csv_path, table1_md_path, stage7=capability_dict)
    if checkpoint_path and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()


def verify_stage7_outputs(output_path=CFG.stage7_out_path, stage5_path=None) -> Dict[str, Any]:
    d = read_json(output_path)
    missing = STAGE7_KEYS - d.keys()
    assert not missing, f"Stage 7 JSON missing keys: {sorted(missing)}"
    assert d["stage"] == 7 and d["passed_acceptance_gate"] is True
    assert d["mmlu_neutral_system_prompt"] == CFG.mmlu_neutral_system_prompt
    assert d["wikitext_passage_rule_used"] in {"full_length_256", "contiguous_chunks_256", "provided", "legacy_custom"}
    if d["wikitext_passage_rule_used"] in {"full_length_256", "contiguous_chunks_256"}:
        assert d["n_kl_positions"] == 51000
    assert d["n_wikitext_passages"] > 0 and d["n_kl_positions"] > 0 and d["wikitext_baseline_nll"] > 0
    expected_learned = 1 + 1 + len({a for a in CFG.grid_alpha_nec + CFG.grid_alpha_suf if a != 0.0})
    assert len(d["learned_direction_capability"]) == expected_learned
    for row in d["learned_direction_capability"] + d["concept_capability"]:
        assert row["wikitext_kl"] >= 0.0 and row["wikitext_kl_ci"][0] <= row["wikitext_kl_ci"][1] + 1e-12
        if d["mmlu_evaluated"]:
            assert 0.0 <= row["mmlu_acc"] <= 1.0
    for op in ("ablate_all", *[f"alpha={a:+.1f}" for a in CFG.ctrl_operating_alphas]):
        entry = d["controls_wikitext_kl"][op]
        assert (len(entry["random"]), len(entry["shuffled"]), len(entry["concept"])) == (CFG.n_ctrl_random, CFG.n_ctrl_shuffled, 1)
        assert min(entry["random"] + entry["shuffled"] + entry["concept"]) >= 0.0
    if d["mmlu_evaluated"]:
        assert d["mmlu_n_questions"] > 0 and 0.0 <= d["mmlu_baseline_acc"] <= 1.0
    if stage5_path and Path(stage5_path).exists():
        sel = read_json(stage5_path)
        assert (d["l_star"], d["i_star"], d["pos_idx"]) == (sel["l_star"], sel["i_star"], sel["pos_idx"])
    return d
