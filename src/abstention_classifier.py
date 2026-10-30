"""Auditable abstention classifier with a pinned local, teacher-forced label scorer."""
import hashlib, json, os, re, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from src.config import CFG

ABSTAIN, ANSWER, DEGENERATE, UNPARSEABLE = "ABSTAIN", "ANSWER", "DEGENERATE", "UNPARSEABLE"
LABELS = (ABSTAIN, ANSWER, DEGENERATE)
ANSWER_THEN_UNKNOWN_RE = re.compile(r"^(?!unknown\b).+\n\s*\n\s*unknown\s*[.!]?\s*$", re.I | re.S)
JUDGE_PROMPT_VERSION = "v1"
JUDGE_PROMPT_V1 = """You are classifying a language model's reply to a question. Do not judge whether the answer is correct.

Question:
<<<QUESTION>>>

Reply:
<<<RESPONSE>>>

Classify the reply into exactly one category:
ABSTAIN - the reply declines to answer or says the answer is unknown, unavailable, not provided, or cannot be determined, in any wording.
ANSWER - the reply commits to a specific answer to the question, even if it is wrong, short, or hedged.
DEGENERATE - the reply is neither: off-topic, self-referential, a meta-statement, nonsensical, or empty.

Respond with only one word: ABSTAIN, ANSWER, or DEGENERATE."""
JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_PROMPT_V1.encode("utf-8")).hexdigest()
_REGEX_PATTERNS = [re.compile(p, re.I) for p in (
    r"\bi do not know\b", r"\bi don['?]?t know\b", r"\bcannot be determined\b", r"\bcannot be answered\b",
    r"\bnot enough information\b", r"\binsufficient information\b", r"\bnot provided in the context\b",
    r"\bnot mentioned\b", r"^unspecified\b", r"^there (is|are|was|were) no\b",
    r"\bno (information|data|record|records|details|mention)\b", r"^(the )?answer( is)?\s*[:\-]?\s*unknown\b",
    r"\b(is|are|remains?) (unknown|not known)\b", r"\bnot (stated|specified|given|available|included|provided|listed|recorded|known)\b",
    r"\bunable to (determine|answer|find|identify|provide)\b", r"^i (cannot|can't|am unable|could not|couldn't)\b",
    r"\bdoes(n't| not) (mention|specify|state|say|provide|include)\b", r"\bno (publicly )?available\b",
    r"^unknown(?:[\s.,!?;:\-)]*$|(?:\s*[:\-?]\s*|\s+(?:the\s+|this\s+|because\b|as\b|since\b|from\b|based\b|cannot\b|is\s+not\b)))")]
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_SIGNATURE = None

def release_local_judge() -> None:
    """Release the cached local judge and its CUDA allocations between models."""
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_SIGNATURE
    model = _LOCAL_MODEL
    _LOCAL_MODEL = _LOCAL_TOKENIZER = _LOCAL_SIGNATURE = None
    if model is not None:
        del model
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def regex_label(response: str) -> str:
    clean = str(response).strip()
    return ABSTAIN if any(p.search(clean) for p in _REGEX_PATTERNS) else ANSWER

def is_answer_then_unknown(response: str) -> bool:
    """Diagnostic match for an answer followed by a separate Unknown line."""
    return bool(ANSWER_THEN_UNKNOWN_RE.match(str(response)))

def _revision() -> str:
    revision = str(getattr(CFG, "judge_model_revision", ""))
    pin = Path(CFG.judge_model_pin_path)
    if pin.exists():
        row = json.loads(pin.read_text(encoding="utf-8"))
        if row.get("model_id") == CFG.judge_model_id:
            revision = str(row.get("revision_sha", revision))
    return revision

def _cache_key(backend: str, model_id: str, question: str, response: str, revision: str = "") -> str:
    raw = "|".join((backend, model_id, revision, JUDGE_PROMPT_VERSION, str(question), str(response)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists(): return {}
    cache = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            row = json.loads(line); key = row.get("cache_key")
            if not key: raise ValueError(f"Judge cache row {line_no} has no cache_key")
            cache[key] = row
    return cache

def _append_cache(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def _render_prompt(question: str, response: str) -> str:
    return JUDGE_PROMPT_V1.replace("<<<QUESTION>>>", str(question)).replace("<<<RESPONSE>>>", str(response))

def _api_raw_judge(prompt: str, model_id: str) -> str:
    import urllib.request
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key: raise RuntimeError("ANTHROPIC_API_KEY is required for judge_backend='api'")
    payload = json.dumps({"model": model_id, "max_tokens": 20, "temperature": 0,
                          "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload, method="POST", headers={
        "content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=60) as resp: data = json.loads(resp.read().decode("utf-8"))
    return "".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")

def _load_local_judge():
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_SIGNATURE
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_id, revision = CFG.judge_model_id, _revision()
    quant = CFG.judge_quantization
    if len(revision) < 32: raise RuntimeError("Judge revision is not pinned; run scripts/pin_judge_model.py first")
    if quant == "auto":
        total = sum(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count()))
        quant = "fp16" if total >= 28 * 1024**3 else "nf4"
    signature = (model_id, revision, quant)
    if _LOCAL_MODEL is not None and signature == _LOCAL_SIGNATURE: return _LOCAL_MODEL, _LOCAL_TOKENIZER, quant
    if not torch.cuda.is_available(): raise RuntimeError("Local judge requires a CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision); tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"revision": revision, "device_map": "auto", "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    if quant == "nf4":
        try: import bitsandbytes  # noqa: F401
        except ImportError as exc: raise RuntimeError("NF4 mode requires bitsandbytes") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    elif quant != "fp16": raise ValueError(f"Unsupported judge quantization: {quant}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_SIGNATURE = model, tokenizer, signature
    return model, tokenizer, quant

def _ids(tokenizer, text):
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(ids, "tolist"): ids = ids.tolist()
    return ids[0] if len(ids) and isinstance(ids[0], list) else ids

def score_labels(model, tokenizer, prompt_text: str) -> dict:
    """Teacher-force each label and return log probabilities, argmax, and margin."""
    chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)
    prompt_ids = _ids(tokenizer, chat)
    label_ids = [_ids(tokenizer, label) for label in LABELS]
    if not prompt_ids or any(not ids for ids in label_ids): raise ValueError("Empty prompt or label tokenization")
    return _score_prepared_labels(model, tokenizer, [prompt_ids], [label_ids])[0]


def _score_prepared_labels(model, tokenizer, prompts, labels_by_prompt):
    """Score label tokens using only their hidden states, avoiding full-sequence logits."""
    import torch
    seqs = [prompt + label for prompt, group in zip(prompts, labels_by_prompt) for label in group]
    width = max(map(len, seqs))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                             use_cache=False).last_hidden_state
        selected_hidden, targets, item_ids = [], [], []
        flat = 0
        for item_idx, (prompt, group) in enumerate(zip(prompts, labels_by_prompt)):
            for label_ids in group:
                positions = torch.arange(len(prompt) - 1, len(prompt) - 1 + len(label_ids), device=device)
                selected_hidden.append(hidden[flat, positions])
                targets.extend(label_ids)
                item_ids.append((item_idx, flat, len(label_ids)))
                flat += 1
        gathered = torch.cat(selected_hidden, dim=0)
        logits = model.lm_head(gathered).float()
        token_logp = torch.log_softmax(logits, dim=-1)
        target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
        token_scores = token_logp.gather(1, target_tensor[:, None]).squeeze(1)
    result = []
    offset = 0
    for _item_idx, _row, n_tokens in item_ids:
        result.append(token_scores[offset:offset + n_tokens].sum())
        offset += n_tokens
    outputs = []
    for item_idx in range(len(prompts)):
        scores = {label: float(result[item_idx * len(LABELS) + j].item())
                  for j, label in enumerate(LABELS)}
        vals = sorted(scores.values(), reverse=True)
        outputs.append({"judge_label": max(LABELS, key=lambda x: scores[x]),
                        "judge_logprobs": scores, "judge_margin": float(vals[0] - vals[1])})
    return outputs

def _score_local_batch(items: List[Dict[str, Any]], batch_size: int) -> List[dict]:
    model, tokenizer, quant = _load_local_judge(); results = []
    for start in range(0, len(items), int(batch_size)):
        chunk = items[start:start+int(batch_size)]; prompts = []; label_ids = []
        for item in chunk:
            chat = tokenizer.apply_chat_template([{"role":"user","content":_render_prompt(item["question"],item["response"])}],
                                                  tokenize=False,add_generation_prompt=True)
            prompts.append(_ids(tokenizer,chat)); label_ids.append([_ids(tokenizer,l) for l in LABELS])
        if any(not x for group in label_ids for x in group): raise ValueError("Empty label tokenization")
        scored = _score_prepared_labels(model, tokenizer, prompts, label_ids)
        results.extend({**row, "judge_quantization": quant} for row in scored)
    return results

def _parse_label(text: str) -> Optional[str]:
    m=re.fullmatch(r"\s*(ABSTAIN|ANSWER|DEGENERATE)\s*[.!]?\s*",str(text),flags=re.I)
    return m.group(1).upper() if m else None

def classify_texts(items: List[Dict[str, Any]], offline: bool=False, backend: Optional[str]=None,
                   judge_fn: Optional[Callable[[str],str]]=None, cache_path: Optional[Path]=None,
                   rate_limit_seconds: float=0.1, batch_size: Optional[int]=None) -> List[Dict[str,Any]]:
    backend=backend or CFG.judge_backend
    if backend not in {"api","local","regex_only"}:raise ValueError(f"Unknown judge backend: {backend}")
    model_id=CFG.judge_model_id if backend!="regex_only" else "regex-v1";revision=_revision() if backend=="local" else ""
    if backend=="local" and len(revision)<32:raise RuntimeError("Judge revision is not pinned; run scripts/pin_judge_model.py first")
    path=Path(cache_path or CFG.judge_cache_path);cache=_load_cache(path);prepared=[];pending=[]
    for index,item in enumerate(items):
        q,r=str(item.get("question","")),str(item.get("response",""));regex=regex_label(r);key=_cache_key(backend,model_id,q,r,revision)
        if backend=="regex_only":prepared.append((index,item,regex,key,{"judge_label":None,"final_label":regex,"classifier_backend":backend,"classifier_final":False,"final_label_source":"regex"}));continue
        if key in cache:prepared.append((index,item,regex,key,cache[key]));continue
        if offline:raise RuntimeError(f"Offline judge cache miss for uid={item.get('uid','<unknown>')} key={key}")
        pending.append((index,item,regex,key))
    if pending and backend=="local":
        _model, _tokenizer, resolved_quant = _load_local_judge()
        cached_quantizations = {row.get("judge_quantization") for row in cache.values()
                                if row.get("judge_quantization") is not None}
        if cached_quantizations and cached_quantizations != {resolved_quant}:
            raise RuntimeError(
                "Judge cache quantization mismatch: cache contains "
                f"{sorted(cached_quantizations)}, but the local judge resolved to {resolved_quant!r}. "
                "Set CFG.judge_quantization explicitly to the cached value before scoring."
            )
        chunk_size = int(batch_size or CFG.judge_batch_size)
        if chunk_size < 1: raise ValueError("batch_size must be positive")
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start:start + chunk_size]
            scored = _score_local_batch([x[1] for x in chunk], chunk_size)
            for (index,item,regex,key),score in zip(chunk,scored):
                record={"cache_key":key,"judge_label":score["judge_label"],"judge_logprobs":score["judge_logprobs"],
                    "judge_margin":score["judge_margin"],"parse_attempts":0,"classifier_backend":"local","judge_model_id":model_id,
                    "judge_model_revision":revision,"judge_quantization":score["judge_quantization"],"judge_prompt_version":JUDGE_PROMPT_VERSION,
                    "judge_prompt_sha256":JUDGE_PROMPT_SHA256,"final_label_source":"judge"}
                _append_cache(path,record);cache[key]=record;prepared.append((index,item,regex,key,record))
    elif pending:
        for index,item,regex,key in pending:
            raw="";label=None;attempts=0
            for attempt in range(2):
                attempts+=1
                try:raw=judge_fn(_render_prompt(item.get("question",""),item.get("response",""))) if judge_fn else _api_raw_judge(_render_prompt(item.get("question",""),item.get("response","")),model_id)
                except Exception:
                    if attempt:raise
                    time.sleep(.5);continue
                label=_parse_label(raw)
                if label:break
                if not attempt:time.sleep(.25)
            record={"cache_key":key,"judge_label":label or UNPARSEABLE,"raw_judge_output":raw,"parse_attempts":attempts,
                "classifier_backend":"api","judge_model_id":model_id,"judge_model_revision":"","judge_prompt_version":JUDGE_PROMPT_VERSION,
                "judge_prompt_sha256":JUDGE_PROMPT_SHA256,"final_label_source":"judge"}
            _append_cache(path,record);cache[key]=record;prepared.append((index,item,regex,key,record))
            if rate_limit_seconds:time.sleep(rate_limit_seconds)
    out=[]
    for index,item,regex,key,record in sorted(prepared,key=lambda x:x[0]):
        label=record.get("human_override") or record.get("final_label") or record.get("judge_label")
        source="human" if record.get("human_override") else record.get("final_label_source","judge")
        out.append({**item,"regex_label":regex,"judge_label":record.get("judge_label"),"final_label":label,
            "agree":bool(regex==label) if label in LABELS else None,"classifier_final":False if backend=="regex_only" else (label in LABELS or label==UNPARSEABLE),
            "classifier_backend":record.get("classifier_backend",backend),"judge_model_id":record.get("judge_model_id",model_id),
            "judge_model_revision":record.get("judge_model_revision",revision),"judge_quantization":record.get("judge_quantization"),
            "judge_logprobs":record.get("judge_logprobs"),"judge_margin":record.get("judge_margin"),
            "judge_prompt_version":record.get("judge_prompt_version",JUDGE_PROMPT_VERSION),"judge_prompt_sha256":record.get("judge_prompt_sha256",JUDGE_PROMPT_SHA256),
            "cache_key":key,"parse_attempts":record.get("parse_attempts",0),"final_label_source":source})
    return out

def is_abstention(record: Dict[str,Any])->bool:
    if record.get("classifier_final") is False:raise ValueError("regex_only classification is not final")
    return False if record.get("final_label")==UNPARSEABLE else record.get("final_label")==ABSTAIN

def is_final_classifier_record(record: Dict[str,Any])->bool:
    return bool(record.get("classifier_final") is True and record.get("classifier_backend") in {"api","local"}
                and record.get("final_label") in set(LABELS)|{UNPARSEABLE})
def require_low_margin_review(records: List[Dict[str, Any]], validation_path: Optional[Path] = None) -> None:
    """Enforce human overrides when held-out low-margin validation misses its gate."""
    validation_path = Path(validation_path or (CFG.results_dir / "judge_validation.json"))
    if not validation_path.exists():
        return
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("low_margin_gate_passed") is not False:
        return
    pending = [r.get("uid", r.get("popqa_id", "<unknown>")) for r in records
               if r.get("judge_margin") is not None and float(r["judge_margin"]) < 1.0
               and r.get("final_label_source") != "human"]
    if pending:
        raise RuntimeError(f"Low-margin validation failed; {len(pending)} final items still need human overrides. "
                           "Run export_judge_low_margin_review.py, label the sheet, then import_judge_overrides.py.")
def require_judge_validation(validation_path: Optional[Path] = None) -> Dict[str, Any]:
    """Fail closed until the held-out human validation clears the registered judge gate."""
    validation_path = Path(validation_path or (CFG.results_dir / "judge_validation.json"))
    if not validation_path.exists():
        raise RuntimeError("Missing results/judge_validation.json; export and label the held-out judge validation first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("gate_passed") is not True:
        raise RuntimeError("Judge validation did not pass kappa >= 0.90 and ABSTAIN recall >= 0.95")
    return validation
