"""Stage 2 pilot feasibility and PopQA power gate."""
import gc
import json
import math
import random
import re
import string
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
except ImportError:  # CPU-only helpers remain importable for unit tests.
    torch = None

from src.config import CFG
from src.abstention_classifier import classify_texts, regex_label as judged_regex_label

_ABSTAIN_PATTERNS = [re.compile(p, re.I) for p in (
    r"\bi do not know\b", r"\bi don['’]?t know\b", r"\bcannot be determined\b",
    r"\bcannot be answered\b", r"\bnot enough information\b",
    r"\binsufficient information\b", r"\bnot provided in the context\b",
    r"\bnot mentioned\b", r"^unspecified\b")]
_STRICT_UNKNOWN = re.compile(r"^unknown(?:[\s\.,!?;:\-\)\]\"']*$|(?:\s*[:\-—]\s*|\s+(?:the\s+|this\s+|because\b|as\b|since\b|from\b|based\b|cannot\b|is\s+not\b)))", re.I)
_GENERIC_ALIAS_TAILS = {"film", "movie", "music", "series", "genre", "church", "game"}


def build_calibrated_system_prompt(base_prompt: str = CFG.system_prompt, failure_mode: str = "none") -> str:
    clauses = {"none": "", "under_abstention": CFG.under_abstention_clause,
               "over_abstention": CFG.over_abstention_clause,
               "narrow_gap": CFG.under_abstention_clause + " " + CFG.over_abstention_clause}
    if failure_mode not in clauses:
        raise ValueError(f"Unknown failure mode: {failure_mode}")
    return base_prompt + clauses[failure_mode]


def legacy_v4_abstention_flags(generated_text: str) -> Tuple[bool, bool]:
    """Return the frozen v4 strict and regex rates for secondary reporting only."""
    clean = str(generated_text).strip().lower()
    strict = bool(_STRICT_UNKNOWN.search(clean)) if clean else False
    return strict, strict or any(p.search(clean) for p in _ABSTAIN_PATTERNS)


def classify_abstention_response(generated_text: str) -> Tuple[bool, bool]:
    """Deprecated v4 regex wrapper; use src.abstention_classifier.classify_texts."""
    warnings.warn("classify_abstention_response is deprecated; use classify_texts", DeprecationWarning, stacklevel=2)
    return legacy_v4_abstention_flags(generated_text)


def normalize_popqa_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = re.sub(r"[-–—/]", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def grade_popqa(response: str, possible_answers, obj, abstention_record: dict) -> dict:
    """Grade a response and retain the deterministic matching rule as an audit code."""
    label = abstention_record.get("final_label")
    if abstention_record.get("classifier_final") is False:
        raise ValueError("Final PopQA grading rejects regex_only classifier records")
    if label == "UNPARSEABLE":
        return {"grade": "ambiguous", "reason": "no_match", "matched_alias": None}
    if label == "ABSTAIN":
        return {"grade": "abstain", "reason": "abstain_judged", "matched_alias": None}
    if label == "DEGENERATE":
        return {"grade": "wrong", "reason": "degenerate", "matched_alias": None}
    if label != "ANSWER":
        raise ValueError(f"Unknown classifier final_label: {label!r}")
    if isinstance(possible_answers, str):
        try:
            parsed = json.loads(possible_answers)
            aliases = parsed if isinstance(parsed, list) else [possible_answers]
        except (ValueError, TypeError):
            aliases = [possible_answers]
    else:
        aliases = list(possible_answers or [])
    obj = str(obj or "")
    if obj:
        aliases.append(obj)
    normalized_response = normalize_popqa_text(response)
    if not normalized_response:
        return {"grade": "wrong", "reason": "no_match", "matched_alias": None}
    ambiguous = False
    response_tokens = normalized_response.split()
    for alias in aliases:
        normalized = normalize_popqa_text(alias)
        if not normalized:
            continue
        forms = [(normalized, False)]
        tail_tokens = normalized.split()
        while len(tail_tokens) > 1 and tail_tokens[-1] in _GENERIC_ALIAS_TAILS:
            tail_tokens.pop()
            forms.append((" ".join(tail_tokens), True))
        for form, stripped in forms:
            if re.search(r"(?:^|\s)" + re.escape(form) + r"(?:$|\s)", normalized_response):
                reason = "generic_tail_stripped" if stripped else (
                    "exact_prefix" if normalized_response.startswith(form) else "token_boundary_substring")
                return {"grade": "correct", "reason": reason, "matched_alias": str(alias)}
        parts = normalized.split()
        if len(parts) > 1:
            for begin in range(len(parts)):
                for finish in range(begin + 1, len(parts) + 1):
                    piece = parts[begin:finish]
                    if len(piece) != len(parts) and re.search(
                            r"(?:^|\s)" + re.escape(" ".join(piece)) + r"(?:$|\s)", normalized_response):
                        ambiguous = True
    if ambiguous:
        return {"grade": "ambiguous", "reason": "partial_compound_alias", "matched_alias": None}
    return {"grade": "wrong", "reason": "no_match", "matched_alias": None}


def normalize_and_grade_popqa_response(generated_text: str, possible_answers: Union[List[str], str], obj_fallback: str = "") -> str:
    """Compatibility helper for legacy unit fixtures; production callers use grade_popqa."""
    record = {"final_label": "ABSTAIN" if judged_regex_label(generated_text) == "ABSTAIN" else "ANSWER",
              "classifier_final": True}
    return grade_popqa(generated_text, possible_answers, obj_fallback, record)["grade"]


def determine_pilot_gate_status(unans_rate: float, ans_rate: float) -> Tuple[bool, float, str]:
    gap = round(unans_rate - ans_rate, 6)
    passed = CFG.pilot_unans_min <= unans_rate <= CFG.pilot_unans_max and ans_rate <= CFG.pilot_ans_max and gap >= CFG.pilot_gap_min
    if passed:
        mode = "none"
    elif unans_rate < CFG.pilot_unans_min and ans_rate <= CFG.pilot_ans_max:
        mode = "under_abstention"
    elif ans_rate > CFG.pilot_ans_max and unans_rate >= CFG.pilot_unans_min:
        mode = "over_abstention"
    else:
        mode = "narrow_gap"
    return passed, gap, mode


def compute_log10_s_pop(raw_s_pop: float) -> float:
    """Return log10(max(raw monthly pageviews, 0) + 1)."""
    return math.log10(max(float(raw_s_pop), 0.0) + 1.0)


def _format_prompts(tokenizer: Any, prompts: List[str], system_prompt: str) -> List[str]:
    from src.tokenization_contract import render_chat
    return [render_chat(tokenizer, [{"role":"system","content":system_prompt},{"role":"user","content":p}], tokenize=False, add_generation_prompt=True) for p in prompts]


def evaluate_prompts_log_odds_and_generations(raw_model: Any, tokenizer: Any, user_prompts: List[str], system_prompt: str,
        unknown_token_ids: List[int], batch_size_forward: int = CFG.batch_size_forward,
        batch_size_gen: int = CFG.batch_size_gen, max_new_tokens: int = CFG.max_new_tokens,
        run_log_odds: bool = True, offline: bool = False,
        suffix_token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("Stage 2 inference requires PyTorch")
    from src.tokenization_contract import canonical_input_ids
    canonical_rows = [canonical_input_ids(tokenizer, prompt, system_prompt, suffix_token_ids) for prompt in user_prompts]
    device = next(raw_model.parameters()).device
    out: Dict[str, Any] = {"responses": [], "abstained_strict": [], "abstained_regex": [], "classifier_records": []}
    if run_log_odds:
        odds, probs = [], []
        for start in range(0, len(chats), batch_size_forward):
            enc = tokenizer.pad({"input_ids": canonical_rows[start:start+batch_size_forward]}, return_tensors="pt").to(device)
            enc["position_ids"] = enc["attention_mask"].to(torch.long).cumsum(-1).sub(1).masked_fill(enc["attention_mask"] == 0, 0)
            with torch.no_grad():
                logits = raw_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, -1, :].float()
                p = torch.softmax(logits, dim=-1)
                abs_p = p[:, unknown_token_ids].sum(-1).clamp(1e-12, 1-1e-12)
                odds.extend(torch.log(abs_p / (1-abs_p)).cpu().tolist())
                probs.extend(abs_p.cpu().tolist())
        out.update(log_odds=odds, abstention_probabilities=probs)
    for start in range(0, len(canonical_rows), batch_size_gen):
        enc = tokenizer.pad({"input_ids": canonical_rows[start:start+batch_size_gen]}, return_tensors="pt").to(device)
        enc["position_ids"] = enc["attention_mask"].to(torch.long).cumsum(-1).sub(1).masked_fill(enc["attention_mask"] == 0, 0)
        with torch.no_grad():
            ids = raw_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = ids[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        out["responses"].extend(texts)
        legacy = []
        for text in texts:
            clean = str(text).strip().lower()
            strict = bool(_STRICT_UNKNOWN.search(clean)) if clean else False
            legacy.append((strict, strict or any(p.search(clean) for p in _ABSTAIN_PATTERNS)))
        out["abstained_strict"].extend(x[0] for x in legacy)
        out["abstained_regex"].extend(x[1] for x in legacy)
    records = classify_texts([{"uid": str(i), "question": q, "response": r}
                              for i, (q, r) in enumerate(zip(user_prompts, out["responses"]))], offline=offline)
    out["classifier_records"] = records
    out["abstained_judged"] = [r["final_label"] == "ABSTAIN" for r in records]
    out["degenerate"] = [r["final_label"] == "DEGENERATE" for r in records]
    out["regex_judge_agreement"] = [r["agree"] for r in records]
    return out


def run_validation_pilot_gate_state_machine(raw_model: Any, tokenizer: Any, env_record: Dict[str, Any], val_pairs: List[Dict[str, Any]], device: str = "cuda:0", requested_dtype: str = "auto"):
    from src.model_utils import load_model_and_verify_runtime
    history = []
    active_env = dict(env_record)
    prompt = CFG.system_prompt
    model_name = active_env["model_name"]
    for attempt in (1, 2, 3):
        if attempt == 3:
            # The caller may still hold a reference to the primary model. Move it
            # off the GPU before dropping local references so the fallback fits.
            try:
                raw_model.to("cpu")
            except (AttributeError, RuntimeError):
                pass
            del raw_model, tokenizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            try:
                wrapper, active_env = load_model_and_verify_runtime(CFG.fallback_model_name, device, requested_dtype)
            except Exception as exc:
                message = str(exc)
                gated = any(term in message.lower() for term in ("gated repo", "403 client error", "not in the authorized list", "access to model"))
                if not gated:
                    raise
                # There is no valid Attempt 3 result until Hugging Face grants
                # this account access. Preserve Attempt 2 diagnostics for the CLI.
                summary["fallback_access_error"] = (
                    f"Cannot load {CFG.fallback_model_name}: Hugging Face denied access. "
                    "Request/accept access for the Kaggle account used by HF_TOKEN, "
                    "then rerun Stage 2."
                )
                return None, None, active_env, summary
            raw_model, tokenizer = wrapper._model, wrapper.tokenizer
            model_name, prompt = CFG.fallback_model_name, CFG.system_prompt
        ans = [p["ans_prompt"] for p in val_pairs]
        unans = [p["unans_prompt"] for p in val_pairs]
        ids = [int(active_env["unknown_token_id_nospace"]), int(active_env["unknown_token_id_space"])]
        suffix_ids = active_env.get("suffix_token_ids")
        a = evaluate_prompts_log_odds_and_generations(raw_model, tokenizer, ans, prompt, ids, suffix_token_ids=suffix_ids)
        u = evaluate_prompts_log_odds_and_generations(raw_model, tokenizer, unans, prompt, ids, suffix_token_ids=suffix_ids)
        ar = sum(u["abstained_judged"])/len(unans); nr = sum(a["abstained_judged"])/len(ans)
        passed, gap, mode = determine_pilot_gate_status(ar, nr)
        history.append({"attempt": attempt, "model_name": model_name, "unans_rate": ar, "ans_rate": nr, "rate_gap": gap, "passed": passed})
        summary = {"attempt_number": attempt, "model_name": model_name, "system_prompt_used": prompt,
            "val_unans_abstention_rate": ar, "val_ans_abstention_rate": nr, "val_rate_gap": gap,
            "val_unans_abstention_rate_strict": sum(u["abstained_strict"])/len(unans),
            "val_ans_abstention_rate_strict": sum(a["abstained_strict"])/len(ans),
            "val_unans_abstention_rate_regex_v4": sum(u["abstained_regex"])/len(unans),
            "val_ans_abstention_rate_regex_v4": sum(a["abstained_regex"])/len(ans),
            "val_unans_abstention_rate_judged": ar, "val_ans_abstention_rate_judged": nr,
            "val_unans_degenerate_rate": sum(u["degenerate"])/len(unans),
            "val_ans_degenerate_rate": sum(a["degenerate"])/len(ans),
            "regex_judge_agreement": sum(x is True for x in u["regex_judge_agreement"] + a["regex_judge_agreement"]) / (len(unans)+len(ans)),
            "val_unans_mean_log_odds": sum(u["log_odds"])/len(unans), "val_ans_mean_log_odds": sum(a["log_odds"])/len(ans),
            "attempt_history": history, "passed_pilot_gate": passed}
        if passed or attempt == 3: return raw_model, tokenizer, active_env, summary
        if attempt == 1: prompt = build_calibrated_system_prompt(CFG.system_prompt, mode)
    return raw_model, tokenizer, active_env, summary


def match_popqa_by_popularity_caliper(popqa_records: List[Dict[str, Any]], caliper_sd_mult: float = CFG.caliper_sd_default, seed: int = CFG.seed_popqa):
    log_s_pops = [compute_log10_s_pop(r["s_pop"]) for r in popqa_records]
    mean = sum(log_s_pops)/len(log_s_pops) if log_s_pops else 0.0
    sd = math.sqrt(sum((x-mean)**2 for x in log_s_pops)/len(log_s_pops)) if log_s_pops else 1.0
    sd = sd or 1.0
    width = caliper_sd_mult * sd
    for r in popqa_records: r["is_popularity_matched"], r["matched_pair_id"] = False, None
    order = sorted(range(len(popqa_records)), key=lambda i: popqa_records[i]["popqa_id"])
    rng = random.Random(seed); ties = {i:rng.random() for i in order}
    wrong = [i for i,r in enumerate(popqa_records) if r["baseline_partition"] == "wrong"]
    correct = [i for i,r in enumerate(popqa_records) if r["baseline_partition"] == "correct"]
    edges = []
    for w in wrong:
        for c in correct:
            d = abs(log_s_pops[w]-log_s_pops[c])
            if d <= width: edges.append((round(d,10),ties[w],ties[c],w,c))
    used=set(); pairs=[]
    for *_,w,c in sorted(edges):
        if w not in used and c not in used: used.update((w,c)); pairs.append((w,c))
    pairs.sort(key=lambda x: popqa_records[x[0]]["popqa_id"])
    for k,(w,c) in enumerate(pairs,1):
        for i in (w,c): popqa_records[i]["is_popularity_matched"],popqa_records[i]["matched_pair_id"] = True,f"m_{k:03d}"
    return popqa_records,len(pairs),width


def _atomic_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    tmp.replace(path)


def run_popqa_grading_and_power_gate(raw_model: Any, tokenizer: Any, system_prompt: str, unknown_token_ids: List[int], popqa_path: Path, force_regrade: bool = False, offline: bool = False, suffix_token_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    popqa_path=Path(popqa_path)
    with popqa_path.open(encoding="utf-8") as f: rows=[json.loads(line) for line in f if line.strip()]
    if force_regrade: rows=rows[:CFG.n_popqa_init]
    active_model = getattr(getattr(raw_model, "config", None), "_name_or_path", "")
    for start in range(0,len(rows),CFG.batch_size_gen):
        batch=rows[start:start+CFG.batch_size_gen]
        if not force_regrade:
            batch=[r for r in batch if (r.get("baseline_partition") not in ("correct","wrong","abstain","ambiguous") or not r.get("baseline_response") or
                r.get("_stage2_model_name") != active_model or r.get("_stage2_system_prompt") != system_prompt)]
        if not batch: continue
        result=evaluate_prompts_log_odds_and_generations(raw_model,tokenizer,[r["question"] for r in batch],system_prompt,unknown_token_ids,run_log_odds=False,offline=offline,suffix_token_ids=suffix_token_ids)
        for r,response, classifier_record in zip(batch,result["responses"],result["classifier_records"]):
            r["baseline_response"]=response
            r["classifier"] = classifier_record
            grade = grade_popqa(response, r.get("possible_answers",[]), r.get("obj",""), classifier_record)
            r["baseline_partition"], r["grade_reason"] = grade["grade"], grade["reason"]
            r["_stage2_model_name"],r["_stage2_system_prompt"] = active_model,system_prompt
    rows,matched,width=match_popqa_by_popularity_caliper(rows)
    _atomic_jsonl(popqa_path,rows)
    widened=False
    # Matching expansion is performed from the full HF dataset when required.
    if matched < CFG.n_popqa_matched_min and len(rows)<CFG.n_popqa_max:
        from src.dataset_builder import load_popqa_rows
        full=load_popqa_rows(); ordered=sorted(full,key=lambda x:(float(x["s_pop"]),int(x["hf_row_idx"]))); n=len(ordered)
        dec={i+1:[] for i in range(10)}
        for idx,row in enumerate(ordered): dec[min(9,idx*10//n)+1].append(row)
        for round_idx in range(1,5):
            if matched>=CFG.n_popqa_matched_min or len(rows)>=CFG.n_popqa_max: break
            used={int(r["hf_row_idx"]) for r in rows}; rng=random.Random(CFG.seed_popqa+round_idx); new=[]
            for d in range(6,11):
                available=sorted((r for r in dec[d] if int(r["hf_row_idx"]) not in used),key=lambda r:r["hf_row_idx"])
                take=min(50,len(available)); new.extend({**r,"s_pop_decile":d} for r in rng.sample(available,take))
            new.sort(key=lambda r:(r["s_pop_decile"],r["hf_row_idx"]))
            new=new[:min(CFG.n_popqa_step,CFG.n_popqa_max-len(rows))]
            result=evaluate_prompts_log_odds_and_generations(raw_model,tokenizer,[r["question"] for r in new],system_prompt,unknown_token_ids,run_log_odds=False,offline=offline,suffix_token_ids=suffix_token_ids)
            for i,(r,response,classifier_record) in enumerate(zip(new,result["responses"],result["classifier_records"])):
                grade = grade_popqa(response,r["possible_answers"],r["obj"],classifier_record)
                r.update(popqa_id=f"popqa_{len(rows)+i+1:04d}",baseline_response=response,baseline_partition=grade["grade"],
                         grade_reason=grade["reason"],classifier=classifier_record,is_popularity_matched=False,matched_pair_id=None,
                         _stage2_model_name=active_model,_stage2_system_prompt=system_prompt)
            rows.extend(new); rows,matched,width=match_popqa_by_popularity_caliper(rows); _atomic_jsonl(popqa_path,rows)
    if len(rows)>=CFG.n_popqa_max and matched<CFG.n_popqa_matched_min:
        rows,matched,width=match_popqa_by_popularity_caliper(rows,CFG.caliper_sd_fallback); widened=True
    _atomic_jsonl(popqa_path,rows)
    counts={k:sum(r["baseline_partition"]==k for r in rows) for k in ("correct","wrong","abstain","ambiguous")}
    return {"popqa_evaluated_count":len(rows),"popqa_correct_count":counts["correct"],"popqa_wrong_count":counts["wrong"],"popqa_abstain_count":counts["abstain"],"popqa_ambiguous_count":counts["ambiguous"],"popqa_matched_pair_count":matched,"caliper_widened":widened,"caliper_width_s_pop":width,"power_gate_passed":matched>=CFG.n_popqa_matched_min or (len(rows)==CFG.n_popqa_max and widened and matched>0)}
