"""Stage 2 pilot feasibility and PopQA power gate."""
import gc
import json
import math
import random
import re
import string
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import torch
except ImportError:  # CPU-only helpers remain importable for unit tests.
    torch = None

from src.config import CFG

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


def classify_abstention_response(generated_text: str) -> Tuple[bool, bool]:
    clean = str(generated_text).strip().lower()
    strict = bool(_STRICT_UNKNOWN.search(clean)) if clean else False
    return strict, strict or any(p.search(clean) for p in _ABSTAIN_PATTERNS)


def normalize_popqa_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = re.sub(r"[-–—/]", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_and_grade_popqa_response(generated_text: str, possible_answers: Union[List[str], str], obj_fallback: str = "") -> str:
    if classify_abstention_response(generated_text)[1]:
        return "abstain"
    if isinstance(possible_answers, str):
        try:
            parsed = json.loads(possible_answers)
            aliases = parsed if isinstance(parsed, list) else [possible_answers]
        except (ValueError, TypeError):
            aliases = [possible_answers]
    else:
        aliases = list(possible_answers or [])
    aliases.append(obj_fallback)
    response = normalize_popqa_text(generated_text)
    if not response:
        return "wrong"
    response_tokens = response.split()
    ambiguous_match = False
    for alias in aliases:
        normalized = normalize_popqa_text(alias)
        if not normalized:
            continue
        # Full aliases and explicitly generic-tail-stripped aliases are valid.
        valid_forms = {normalized}
        tail_tokens = normalized.split()
        while len(tail_tokens) > 1 and tail_tokens[-1] in _GENERIC_ALIAS_TAILS:
            tail_tokens.pop()
            valid_forms.add(" ".join(tail_tokens))
        if any(re.search(r"(?:^|\s)" + re.escape(form) + r"(?:$|\s)", response) for form in valid_forms):
            return "correct"
        # A response that supplies only a token/phrase from a compound alias is
        # under-specified; keep it out of both correct and wrong groups.
        alias_tokens = normalized.split()
        if len(alias_tokens) > 1:
            for start in range(len(alias_tokens)):
                for end in range(start + 1, len(alias_tokens) + 1):
                    partial = alias_tokens[start:end]
                    if len(partial) == len(alias_tokens):
                        continue
                    partial_text = " ".join(partial)
                    if re.search(r"(?:^|\s)" + re.escape(partial_text) + r"(?:$|\s)", response):
                        ambiguous_match = True
    return "ambiguous" if ambiguous_match else "wrong"


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
    return [tokenizer.apply_chat_template([{"role":"system","content":system_prompt},{"role":"user","content":p}], tokenize=False, add_generation_prompt=True) for p in prompts]


def evaluate_prompts_log_odds_and_generations(raw_model: Any, tokenizer: Any, user_prompts: List[str], system_prompt: str,
        unknown_token_ids: List[int], batch_size_forward: int = CFG.batch_size_forward,
        batch_size_gen: int = CFG.batch_size_gen, max_new_tokens: int = CFG.max_new_tokens,
        run_log_odds: bool = True) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("Stage 2 inference requires PyTorch")
    chats = _format_prompts(tokenizer, user_prompts, system_prompt)
    device = next(raw_model.parameters()).device
    out: Dict[str, Any] = {"responses": [], "abstained_strict": [], "abstained_regex": []}
    if run_log_odds:
        odds, probs = [], []
        for start in range(0, len(chats), batch_size_forward):
            enc = tokenizer(chats[start:start+batch_size_forward], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits = raw_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[:, -1, :].float()
                p = torch.softmax(logits, dim=-1)
                abs_p = p[:, unknown_token_ids].sum(-1).clamp(1e-12, 1-1e-12)
                odds.extend(torch.log(abs_p / (1-abs_p)).cpu().tolist())
                probs.extend(abs_p.cpu().tolist())
        out.update(log_odds=odds, abstention_probabilities=probs)
    for start in range(0, len(chats), batch_size_gen):
        enc = tokenizer(chats[start:start+batch_size_gen], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            ids = raw_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        generated = ids[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        out["responses"].extend(texts)
        classes = [classify_abstention_response(t) for t in texts]
        out["abstained_strict"].extend(x[0] for x in classes)
        out["abstained_regex"].extend(x[1] for x in classes)
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
        a = evaluate_prompts_log_odds_and_generations(raw_model, tokenizer, ans, prompt, ids)
        u = evaluate_prompts_log_odds_and_generations(raw_model, tokenizer, unans, prompt, ids)
        ar = sum(u["abstained_regex"])/len(unans); nr = sum(a["abstained_regex"])/len(ans)
        passed, gap, mode = determine_pilot_gate_status(ar, nr)
        history.append({"attempt": attempt, "model_name": model_name, "unans_rate": ar, "ans_rate": nr, "rate_gap": gap, "passed": passed})
        summary = {"attempt_number": attempt, "model_name": model_name, "system_prompt_used": prompt,
            "val_unans_abstention_rate": ar, "val_ans_abstention_rate": nr, "val_rate_gap": gap,
            "val_unans_abstention_rate_strict": sum(u["abstained_strict"])/len(unans),
            "val_ans_abstention_rate_strict": sum(a["abstained_strict"])/len(ans),
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


def run_popqa_grading_and_power_gate(raw_model: Any, tokenizer: Any, system_prompt: str, unknown_token_ids: List[int], popqa_path: Path, force_regrade: bool = False) -> Dict[str, Any]:
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
        result=evaluate_prompts_log_odds_and_generations(raw_model,tokenizer,[r["question"] for r in batch],system_prompt,unknown_token_ids,run_log_odds=False)
        for r,response in zip(batch,result["responses"]):
            r["baseline_response"]=response
            r["baseline_partition"]=normalize_and_grade_popqa_response(response,r.get("possible_answers",[]),r.get("obj",""))
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
            result=evaluate_prompts_log_odds_and_generations(raw_model,tokenizer,[r["question"] for r in new],system_prompt,unknown_token_ids,run_log_odds=False)
            for i,(r,response) in enumerate(zip(new,result["responses"])):
                r.update(popqa_id=f"popqa_{len(rows)+i+1:04d}",baseline_response=response,baseline_partition=normalize_and_grade_popqa_response(response,r["possible_answers"],r["obj"]),is_popularity_matched=False,matched_pair_id=None,_stage2_model_name=active_model,_stage2_system_prompt=system_prompt)
            rows.extend(new); rows,matched,width=match_popqa_by_popularity_caliper(rows); _atomic_jsonl(popqa_path,rows)
    if len(rows)>=CFG.n_popqa_max and matched<CFG.n_popqa_matched_min:
        rows,matched,width=match_popqa_by_popularity_caliper(rows,CFG.caliper_sd_fallback); widened=True
    _atomic_jsonl(popqa_path,rows)
    counts={k:sum(r["baseline_partition"]==k for r in rows) for k in ("correct","wrong","abstain","ambiguous")}
    return {"popqa_evaluated_count":len(rows),"popqa_correct_count":counts["correct"],"popqa_wrong_count":counts["wrong"],"popqa_abstain_count":counts["abstain"],"popqa_ambiguous_count":counts["ambiguous"],"popqa_matched_pair_count":matched,"caliper_widened":widened,"caliper_width_s_pop":width,"power_gate_passed":matched>=CFG.n_popqa_matched_min or (len(rows)==CFG.n_popqa_max and widened and matched>0)}
