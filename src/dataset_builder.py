# src/dataset_builder.py
import csv
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from src.config import CONFIG

REVIEW_FLAGS = ("check_answer_unambiguous", "check_zero_unans_support", "check_distinct_predicate", "check_b_ctrl_valid")
FORBIDDEN_UNANS_PHRASES = (
    "at this point in time", "at present", "at the moment", "the profile subject",
    "described in the profile", "during the same period", "during the period",
)


def find_unanswerable_style_confounds(templates: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Return template IDs with unanswerable-only -ly words or banned filler phrases."""
    failures: Dict[str, List[str]] = {}
    for tmpl in templates:
        issues: List[str] = []
        for slot_index, slot in enumerate(tmpl["slots"]):
            ans = tmpl["ans_template"].format(**slot).casefold()
            unans = tmpl["unans_template"].format(**slot).casefold()
            ans_words = set(re.findall(r"\b[\w'-]+\b", ans))
            unans_words = set(re.findall(r"\b[\w'-]+\b", unans))
            leaked_adverbs = sorted(w for w in unans_words - ans_words if w.endswith("ly"))
            for word in leaked_adverbs:
                issues.append(f"slot{slot_index}:unans_only_-ly:{word}")
            for phrase in FORBIDDEN_UNANS_PHRASES:
                if phrase in unans:
                    issues.append(f"slot{slot_index}:forbidden_phrase:{phrase}")
        if issues:
            failures[tmpl["template_id"]] = issues
    return failures


_INTERROGATIVE_STEMS = (
    "offshore of which", "in what", "in which", "on which", "at which",
    "from which", "across which", "through which", "for which", "near which",
    "how many", "what", "which", "where",
)


def _interrogative_stem(prompt: str) -> str:
    question = prompt.split("\nQuestion: ", 1)[-1].strip().casefold()
    prefix = "according to the profile,"
    if question.startswith(prefix):
        question = question[len(prefix):].strip()
    return next((stem for stem in _INTERROGATIVE_STEMS if question.startswith(stem)), "")


def find_family_b_interrogative_stem_mismatches(
    templates: List[Dict[str, Any]],
) -> Dict[str, Tuple[str, str]]:
    """Return Family B templates whose paired questions start with different wh stems."""
    failures: Dict[str, Tuple[str, str]] = {}
    for tmpl in templates:
        if tmpl.get("family") != "B":
            continue
        ans_stem = _interrogative_stem(tmpl["ans_template"])
        unans_stem = _interrogative_stem(tmpl["unans_template"])
        if not ans_stem or ans_stem != unans_stem:
            failures[tmpl["template_id"]] = (ans_stem or "<unrecognized>", unans_stem or "<unrecognized>")
    return failures


def _extract_reviewed_slots(template: str, rendered: str) -> Dict[str, str] | None:
    """Extract slot values only when the reviewed prompt has the exact current wording."""
    pieces = re.split(r"(\{(?:entity|attr1|attr2)\})", template)
    pattern, names = [], []
    for piece in pieces:
        match = re.fullmatch(r"\{(entity|attr1|attr2)\}", piece)
        if match:
            name = match.group(1)
            names.append(name)
            pattern.append(f"(?P<{name}>.+?)" if names.count(name) == 1 else f"(?P={name})")
        else:
            pattern.append(re.escape(piece))
    match = re.fullmatch("".join(pattern), rendered)
    return match.groupdict() if match else None


def _adopt_reviewed_slot_values(tmpl: Dict[str, Any], row: Dict[str, str]) -> bool:
    """Reuse names from the reviewed example while requiring all template wording to match."""
    family = tmpl["family"]
    parsed = _extract_reviewed_slots(tmpl["ans_template"], row.get("ans_prompt", "").strip())
    unans = _extract_reviewed_slots(tmpl["unans_template"], row.get("unans_prompt", "").strip())
    bctrl = ({"entity": "NA"} if family == "A" else
             _extract_reviewed_slots(tmpl["b_ctrl_template"], row.get("b_ctrl_prompt", "").strip()))
    if parsed is None or unans is None or bctrl is None:
        return False
    values = {**parsed}
    if any(unans.get(k) != v for k, v in parsed.items() if k == "entity"):
        return False
    values.update(unans)
    if family == "B":
        if bctrl.get("entity") != parsed.get("entity"):
            return False
        if any(bctrl.get(k) != v for k, v in values.items() if k in bctrl):
            return False
    slot = tmpl["slots"][0]
    slot.update(values)
    slot["ground_truth"] = values["attr1"]
    return True


def parse_review_value(value: str, family: str, flag: str) -> bool:
    value = value.strip().upper()
    if value in {"Y", "YES", "TRUE"}:
        return True
    if value in {"N", "NO", "FALSE", ""}:
        return False
    if value == "NA" and family == "A" and flag == "check_b_ctrl_valid":
        return True
    raise ValueError(f"Invalid review value {value!r} for {family}/{flag}; use Y, N, or NA")


def validate_review_sheet(templates: List[Dict[str, Any]], path: Path) -> Dict[str, List[str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["template_id"]: r for r in csv.DictReader(f)}
    failures: Dict[str, List[str]] = {}
    for tmpl in templates:
        row = rows.get(tmpl["template_id"])
        reasons: List[str] = []
        if row is None:
            reasons.append("missing_row")
        else:
            # Earlier review exports may have used different natural synthetic names.
            # Preserve those reviewed slot values if (and only if) literal wording matches.
            if not _adopt_reviewed_slot_values(tmpl, row):
                reasons.append("prompt_mismatch")
            slot = tmpl["slots"][0]
            expected_prompts = {
                "ans_prompt": tmpl["ans_template"].format(**slot),
                "unans_prompt": tmpl["unans_template"].format(**slot),
                "b_ctrl_prompt": "NA" if tmpl["family"] == "A" else tmpl["b_ctrl_template"].format(**slot),
                "ground_truth": str(slot["ground_truth"]),
            }
            if any(row.get(key, "").strip() != value for key, value in expected_prompts.items()):
                if "prompt_mismatch" not in reasons:
                    reasons.append("prompt_mismatch")
        for flag in REVIEW_FLAGS:
            val = "" if row is None else row.get(flag, "")
            passed = parse_review_value(val, tmpl["family"], flag)
            tmpl[flag] = passed
            if not passed:
                reasons.append(f"manual_check:{flag}")
        if reasons:
            failures[tmpl["template_id"]] = reasons
    return failures


def apply_review_sheet(templates: List[Dict[str, Any]], path: Path) -> List[str]:
    """Compatibility helper returning the IDs of failed or stale review rows."""
    return list(validate_review_sheet(templates, path))


def normalize_aliases(aliases: List[str]) -> List[str]:
    cleaned, seen = [], set()
    for alias in aliases:
        text = "".join(ch for ch in str(alias) if unicodedata.category(ch) != "Cf" and ch not in "\u034f\u061c\u180e\u200b\u200c\u200d\u2060\ufeff")
        text = " ".join(text.split())
        key = text.casefold()
        if len(text) <= 2 or not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    if not cleaned:
        short_seen = set()
        cleaned = []
        for alias in aliases:
            text = "".join(ch for ch in str(alias) if unicodedata.category(ch) != "Cf" and ch not in "\u034f\u061c\u180e\u200b\u200c\u200d\u2060\ufeff")
            text = " ".join(text.split())
            if text and text.casefold() not in short_seen:
                short_seen.add(text.casefold())
                cleaned.append(text)
    return cleaned


def create_template_splits(
    templates: List[Dict[str, Any]],
    seed_split: int = CONFIG.seed_split,
) -> Dict[str, Any]:
    """
    Splits templates strictly at the template_id level into train, val, and test
    with exact 50% / 25% / 25% proportions within each family (A and B).
    """
    fam_a_ids = sorted([t["template_id"] for t in templates if t["family"] == "A"])
    fam_b_ids = sorted([t["template_id"] for t in templates if t["family"] == "B"])
    assert len(fam_a_ids) == len(fam_b_ids), "Family A and Family B must have equal template counts!"

    n_fam = len(fam_a_ids)
    n_train_fam = n_fam // 2
    n_val_fam = n_fam // 4

    rng = random.Random(seed_split)
    rng.shuffle(fam_a_ids)
    rng.shuffle(fam_b_ids)

    train_ids = sorted(fam_a_ids[:n_train_fam] + fam_b_ids[:n_train_fam])
    val_ids = sorted(
        fam_a_ids[n_train_fam : n_train_fam + n_val_fam]
        + fam_b_ids[n_train_fam : n_train_fam + n_val_fam]
    )
    test_ids = sorted(fam_a_ids[n_train_fam + n_val_fam :] + fam_b_ids[n_train_fam + n_val_fam :])

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    return {
        "seed_split": seed_split,
        "train_template_ids": train_ids,
        "val_template_ids": val_ids,
        "test_template_ids": test_ids,
    }


def instantiate_pairs_and_b_ctrl(
    templates: List[Dict[str, Any]],
    splits: Dict[str, Any],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Instantiates all 384 minimal pairs (`data/pairs.jsonl`) and 48 Family B
    Private-Detail Answerable Controls (`data/b_ctrl.jsonl` for Family B test templates).
    """
    split_lookup: Dict[str, str] = {}
    for tid in splits["train_template_ids"]:
        split_lookup[tid] = "train"
    for tid in splits["val_template_ids"]:
        split_lookup[tid] = "val"
    for tid in splits["test_template_ids"]:
        split_lookup[tid] = "test"

    pairs: List[Dict[str, Any]] = []
    b_ctrl_rows: List[Dict[str, Any]] = []

    pair_counter = 1
    bctrl_counter = 1

    for tmpl in templates:
        tid = tmpl["template_id"]
        fam = tmpl["family"]
        split_name = split_lookup[tid]

        # Enforce all 4 human review checklist booleans
        for flag in REVIEW_FLAGS:
            assert tmpl.get(flag) is True, f"Template {tid} failed human checklist flag {flag}!"

        for idx, slot in enumerate(tmpl["slots"]):
            ans_prompt = tmpl["ans_template"].format(**slot)
            unans_prompt = tmpl["unans_template"].format(**slot)

            ans_len = len(tokenizer.encode(ans_prompt, add_special_tokens=False))
            unans_len = len(tokenizer.encode(unans_prompt, add_special_tokens=False))
            len_diff = ans_len - unans_len
            assert abs(len_diff) <= CONFIG.max_token_diff_pair, (
                f"Token length parity violation in {tid} instance {idx}: "
                f"ans_len={ans_len}, unans_len={unans_len}, diff={len_diff}"
            )

            pair_id = f"pair_{pair_counter:04d}"
            pair_counter += 1

            pairs.append(
                {
                    "pair_id": pair_id,
                    "template_id": tid,
                    "split": split_name,
                    "family": fam,
                    "instance_idx": idx,
                    "ans_prompt": ans_prompt,
                    "unans_prompt": unans_prompt,
                    "ground_truth": str(slot["ground_truth"]),
                    "ans_token_len": int(ans_len),
                    "unans_token_len": int(unans_len),
                    "token_len_diff": int(len_diff),
                }
            )

            if fam == "B" and split_name == "test":
                assert tmpl["b_ctrl_template"] is not None, f"Missing b_ctrl_template on {tid}"
                b_ctrl_prompt = tmpl["b_ctrl_template"].format(**slot)
                b_ctrl_len = len(tokenizer.encode(b_ctrl_prompt, add_special_tokens=False))
                b_ctrl_gt = str(slot.get("b_ctrl_ground_truth", slot["attr2"]))
                b_ctrl_rows.append(
                    {
                        "b_ctrl_id": f"bctrl_{bctrl_counter:03d}",
                        "pair_id": pair_id,
                        "template_id": tid,
                        "instance_idx": idx,
                        "prompt": b_ctrl_prompt,
                        "ground_truth": b_ctrl_gt,
                        "token_len": int(b_ctrl_len),
                    }
                )
                bctrl_counter += 1

    return pairs, b_ctrl_rows


def _extract_word_ngrams(text: str, max_n: int = 3) -> Set[str]:
    """Extracts unique lowercased word unigrams, bigrams, and trigrams from prompt text."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    ngrams: Set[str] = set()
    n_tok = len(tokens)
    for n in range(1, max_n + 1):
        for i in range(n_tok - n + 1):
            ngrams.add(" ".join(tokens[i : i + n]))
    return ngrams


def compute_max_ngram_phi(
    pairs: List[Dict[str, Any]],
    max_n: int = 3,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Computes the exact 2x2 Phi coefficient (Pearson correlation for binary variables)
    between n-gram presence (n in {1, 2, 3}) and prompt label (unans=1 vs ans=0).
    Returns (max_abs_phi, worst_ngram, top_10_dict).
    """
    n1 = len(pairs)  # unans count
    n0 = len(pairs)  # ans count
    total_n = n1 + n0

    unans_counts: Dict[str, int] = defaultdict(int)
    ans_counts: Dict[str, int] = defaultdict(int)

    for p in pairs:
        for g in _extract_word_ngrams(p["unans_prompt"], max_n=max_n):
            unans_counts[g] += 1
        for g in _extract_word_ngrams(p["ans_prompt"], max_n=max_n):
            ans_counts[g] += 1

    all_ngrams = set(unans_counts.keys()) | set(ans_counts.keys())
    phi_scores: Dict[str, float] = {}
    max_abs_phi = 0.0
    worst_ngram = ""

    # Sorted iteration makes ties (and hence worst_ngram / top_10) independent of PYTHONHASHSEED
    for g in sorted(all_ngrams):
        n11 = unans_counts[g]
        n01 = ans_counts[g]
        n_pos = n11 + n01
        if n_pos < 2 or n_pos >= total_n - 1:
            continue
        n10 = n1 - n11
        n00 = n0 - n01
        denom = math.sqrt(float(n1) * float(n0) * float(n_pos) * float(total_n - n_pos))
        if denom <= 0.0:
            continue
        phi = (float(n11 * n00) - float(n10 * n01)) / denom
        abs_phi = abs(phi)
        phi_scores[g] = phi
        if abs_phi > max_abs_phi:
            max_abs_phi = abs_phi
            worst_ngram = g

    top_10 = dict(sorted(phi_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10])
    return max_abs_phi, worst_ngram, top_10


def load_popqa_rows() -> List[Dict[str, Any]]:
    """Loads `akariasai/PopQA` (test split) into plain dicts with parsed `possible_answers`."""
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    all_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        raw_ans = row["possible_answers"]
        if isinstance(raw_ans, str):
            try:
                parsed_ans = json.loads(raw_ans)
            except Exception:
                parsed_ans = [raw_ans]
        else:
            parsed_ans = list(raw_ans)

        all_rows.append(
            {
                "hf_row_idx": int(idx),
                "question": str(row["question"]).strip(),
                "prop": str(row.get("prop", "")),
                "subj": str(row.get("subj", "")),
                "obj": str(row.get("obj", "")),
                "possible_answers": [str(a) for a in parsed_ans],
                "possible_answers_clean": normalize_aliases([str(a) for a in parsed_ans]),
                "s_pop": float(row["s_pop"]),
                "log_s_pop": math.log10(float(row["s_pop"]) + 1.0),
            }
        )
    return all_rows


def stratify_and_sample_popqa(
    all_rows: List[Dict[str, Any]],
    n_total: int = CONFIG.n_popqa_init,
    w_bottom5: float = CONFIG.w_popqa_bottom5,
    w_top5: float = CONFIG.w_popqa_top5,
    seed_popqa: int = CONFIG.seed_popqa,
) -> List[Dict[str, Any]]:
    """
    Partitions rows into 10 equal-sized s_pop deciles (1..10) and samples without
    replacement using exact weights:
      - Deciles 1..5: 0.06 per decile (60 items/decile for n_total=1000)
      - Deciles 6..10: 0.14 per decile (140 items/decile for n_total=1000)
    """
    # Sort deterministically by (s_pop, hf_row_idx) to assign equal-sized deciles 1..10
    # Normalize at sampling time too, so callers with in-memory or cached raw rows
    # receive the same schema as rows loaded directly from PopQA.
    normalized_rows = []
    for row in all_rows:
        normalized = dict(row)
        normalized.setdefault("possible_answers_clean", normalize_aliases(list(row["possible_answers"])))
        normalized.setdefault("log_s_pop", math.log10(float(row["s_pop"]) + 1.0))
        normalized_rows.append(normalized)
    all_rows = sorted(normalized_rows, key=lambda r: (r["s_pop"], r["hf_row_idx"]))
    n_all = len(all_rows)
    decile_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for rank_idx, item in enumerate(all_rows):
        decile = min(10, int((rank_idx * 10) // n_all) + 1)
        decile_buckets[decile].append({**item, "s_pop_decile": decile})

    rng = random.Random(seed_popqa)
    sampled_rows: List[Dict[str, Any]] = []

    for decile in range(1, 11):
        weight = w_bottom5 if decile <= 5 else w_top5
        n_target = int(round(n_total * weight))
        bucket = list(decile_buckets[decile])
        rng.shuffle(bucket)
        assert len(bucket) >= n_target, f"Decile {decile} has {len(bucket)} rows < {n_target}"
        sampled_rows.extend(bucket[:n_target])

    assert len(sampled_rows) == n_total, f"Expected {n_total} PopQA rows, got {len(sampled_rows)}"

    # Assign sequential popqa_id and Stage 1 initial placeholders for Stage 2 grading
    final_records: List[Dict[str, Any]] = []
    for i, item in enumerate(sampled_rows):
        final_records.append(
            {
                "popqa_id": f"popqa_{i + 1:04d}",
                "hf_row_idx": item["hf_row_idx"],
                "question": item["question"],
                "prop": item["prop"],
                "subj": item["subj"],
                "obj": item["obj"],
                "possible_answers": item["possible_answers"],
                "possible_answers_clean": item["possible_answers_clean"],
                "s_pop": item["s_pop"],
                "log_s_pop": item["log_s_pop"],
                "s_pop_decile": item["s_pop_decile"],
                "baseline_response": "",
                "baseline_partition": "ungraded",
                "is_popularity_matched": False,
                "matched_pair_id": None,
            }
        )
    return final_records


def sample_stratified_popqa(
    n_total: int = CONFIG.n_popqa_init,
    w_bottom5: float = CONFIG.w_popqa_bottom5,
    w_top5: float = CONFIG.w_popqa_top5,
    seed_popqa: int = CONFIG.seed_popqa,
) -> List[Dict[str, Any]]:
    """Loads `akariasai/PopQA` (test split) and draws the decile-stratified sample."""
    return stratify_and_sample_popqa(
        load_popqa_rows(),
        n_total=n_total,
        w_bottom5=w_bottom5,
        w_top5=w_top5,
        seed_popqa=seed_popqa,
    )


def write_audit_log(audit_path: Path, audit_rows: List[Dict[str, Any]]) -> None:
    """Writes or appends audit rows to `data/audit_log.csv`."""
    fieldnames = [
        "timestamp_utc",
        "template_id",
        "failure_reason",
        "offending_detail",
        "old_text",
        "new_text",
        "post_fix_passed",
    ]
    write_header = not audit_path.exists()
    with open(audit_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

