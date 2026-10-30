#!/usr/bin/env python3
"""Acquire and reproducibly sample the non-GPU E4 source datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import CFG
from src.eval_common import sha256_file
from src.exploratory import assert_not_fresh, exploratory_header, git_commit, require_tier1_complete

SEED = 20260415
OUT = ROOT / "data/exploratory/e4"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class PreparationError(RuntimeError):
    def __init__(self, message, source_info=None):
        super().__init__(message)
        self.source_info = source_info or {}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_title_capped(rows, n_per_class, seed, cap=6):
    """Deterministic per-class sample with no title exceeding cap."""
    rng = random.Random(seed)
    selected = []
    counts = Counter()
    for label in (0, 1):
        candidates = [r for r in rows if int(r["y"]) == label]
        rng.shuffle(candidates)
        for row in candidates:
            title = str(row["title"])
            if counts[(label, title)] >= cap:
                continue
            selected.append(row)
            counts[(label, title)] += 1
            if sum(1 for r in selected if int(r["y"]) == label) == n_per_class:
                break
    counts_by_class = {str(label): sum(int(r["y"]) == label for r in selected) for label in (0, 1)}
    return selected, counts_by_class


def _prepare_squad_rows(dataset_rows, tokenizer, max_context_tokens=350):
    rows, dropped = [], 0
    for item in dataset_rows:
        y = int(not bool(item.get("answers", {}).get("text", [])))
        context = str(item.get("context", ""))
        if len(tokenizer.encode(context, add_special_tokens=False)) > max_context_tokens:
            dropped += 1
            continue
        rows.append({"id": str(item["id"]), "title": str(item["title"]), "context": context,
                     "question": str(item["question"]), "y": y,
                     "gold_answers": list(item.get("answers", {}).get("text", []))})
    return rows, dropped


def _github_file_and_revision():
    import requests
    headers = {"Accept": "application/vnd.github+json"}
    if __import__("os").environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {__import__('os').environ['GITHUB_TOKEN']}"
    repo = "yinzhangyue/SelfAware"
    repo_info = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=30)
    repo_info.raise_for_status()
    revision = repo_info.json()["default_branch"]
    commit = requests.get(f"https://api.github.com/repos/{repo}/commits/{revision}", headers=headers, timeout=30)
    commit.raise_for_status()
    sha = commit.json()["sha"]
    tree = requests.get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", headers=headers, timeout=30)
    tree.raise_for_status()
    matches = [item["path"] for item in tree.json().get("tree", [])
               if Path(item["path"]).name.lower() == "selfaware.json"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one SelfAware.json in {repo}, found {matches}")
    raw = requests.get(f"https://raw.githubusercontent.com/{repo}/{sha}/{matches[0]}", timeout=30)
    raw.raise_for_status()
    return json.loads(raw.text), f"https://github.com/{repo}/blob/{sha}/{matches[0]}", sha


def _rows_selfaware(payload):
    if isinstance(payload, dict):
        for key in ("data", "questions", "examples", "example"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            if payload and all(isinstance(value, dict) for value in payload.values()):
                payload = [{"question_id": key, **value} for key, value in payload.items()]
    if not isinstance(payload, list):
        raise ValueError("SelfAware.json format was not a list or a known list wrapper")
    rows = []
    for ix, item in enumerate(payload):
        qid = item.get("question_id", item.get("id", item.get("qid", ix)))
        answerable = item.get("answerable")
        if answerable is None:
            raise ValueError(f"SelfAware row {qid} has no answerable field")
        if isinstance(answerable, str):
            value = answerable.strip().lower()
            if value not in {"true", "false", "1", "0"}:
                raise ValueError(f"SelfAware row {qid} has non-binary answerable={answerable!r}")
            answerable = value in {"true", "1"}
        question = item.get("question", item.get("query"))
        if not question:
            raise ValueError(f"SelfAware row {qid} has no question")
        rows.append({"question_id": str(qid), "question": str(question), "y": int(not bool(answerable)),
                     "source": str(item.get("source", "unknown")),
                     "answer": item.get("answer", item.get("reference"))})
    return rows


def prepare_squad(seed=SEED, tokenizer=None, tokenizer_name=MODEL):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id = "rajpurkar/squad_v2"
    try:
        revision = HfApi().dataset_info(dataset_id).sha
        dataset = load_dataset(dataset_id, split="validation", revision=revision)
    except Exception as exc:
        raise RuntimeError(f"SQuAD dataset ID/split unavailable: {dataset_id}:validation") from exc
    source_info = {"source_url": f"https://huggingface.co/datasets/{dataset_id}",
                   "revision": revision, "licence": "CC BY-SA 4.0"}
    if tokenizer is None:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=__import__("os").environ.get("HF_TOKEN"))
        except Exception as exc:
            raise PreparationError(f"Llama tokenizer unavailable; context-length filter cannot be applied: {exc}",
                                   source_info) from exc
    rows, dropped = _prepare_squad_rows(dataset, tokenizer)
    selected, counts = _sample_title_capped(rows, 300, seed, cap=6)
    if min(counts.values(), default=0) < 250:
        print(f"squad: title and token filters leave fewer than 250 items in a class: {counts}", flush=True)
    return selected, {**source_info, "dropped_contexts_over_350_tokens": dropped,
                      "class_counts": counts, "title_cap": 6}


def prepare_selfaware(seed=SEED):
    payload, url, revision = _github_file_and_revision()
    rows = _rows_selfaware(payload)
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        group = [r for r in rows if r["y"] == label]
        rng.shuffle(group)
        selected.extend(group[:300])
    distribution = {str(y): dict(Counter(r["source"] for r in selected if r["y"] == y)) for y in (0, 1)}
    return selected, {"source_url": url, "revision": revision, "licence": "CC BY-SA 4.0",
                      "class_counts": {str(y): sum(r["y"] == y for r in selected) for y in (0, 1)},
                      "source_distribution_by_class": distribution}


def prepare_triviaqa(seed=SEED):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    dataset_id, config = "mandarjoshi/trivia_qa", "rc.nocontext"
    try:
        info = HfApi().dataset_info(dataset_id)
        revision = info.sha
        dataset = load_dataset(dataset_id, config, split="validation", revision=revision)
    except Exception as exc:
        raise RuntimeError(f"TriviaQA dataset/config/split unavailable: {dataset_id}/{config}:validation") from exc
    rows = []
    for item in dataset:
        ans = item.get("answer", {})
        rows.append({"question_id": str(item.get("question_id", item.get("id"))),
                     "question": str(item["question"]), "aliases": list(ans.get("aliases", [])),
                     "normalized_aliases": list(ans.get("normalized_aliases", [])),
                     "value": str(ans.get("value", ""))})
    random.Random(seed).shuffle(rows)
    card = getattr(info, "card_data", None)
    license_name = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    if not license_name:
        raise RuntimeError("TriviaQA dataset card does not declare a licence; stop before redistribution")
    return rows[:1000], {"source_url": f"https://huggingface.co/datasets/{dataset_id}",
                         "revision": revision, "licence": str(license_name),
                         "class_counts": None}


def _syn_rows():
    pairs_path = assert_not_fresh(CFG.data_dir / "pairs.jsonl")
    rows = []
    with pairs_path.open(encoding="utf-8") as stream:
        for line in stream:
            pair = json.loads(line)
            for prompt_key, y in (("ans_prompt", 0), ("unans_prompt", 1)):
                rows.append({"pair_id": str(pair["pair_id"]), "family": pair.get("family"),
                             "split": pair["split"], "user_prompt": pair[prompt_key], "y": y,
                             "source_id": str(pair["pair_id"])})
    return rows


def prepare(sources, out_dir=OUT, seed=SEED, tokenizer_name=MODEL):
    out_dir = assert_not_fresh(out_dir)
    manifest = {"status": "exploratory_post_registration", "sampling_seed": seed, "sources": {}}
    failures = {}
    for source in sources:
        try:
            if source == "syn":
                rows = _syn_rows()
                info = {"source_url": "repository:data/pairs.jsonl", "revision": git_commit(ROOT),
                        "licence": "repository dataset terms", "class_counts": {"0": sum(r["y"] == 0 for r in rows),
                                                                                "1": sum(r["y"] == 1 for r in rows)}}
            elif source == "squad":
                rows, info = prepare_squad(seed, tokenizer_name=tokenizer_name)
                for row in rows:
                    row["user_prompt"] = f"Passage: {row['context']}\n\nQuestion: {row['question']}"
                    row["group"] = row["title"]
            elif source == "selfaware":
                rows, info = prepare_selfaware(seed)
                for row in rows:
                    row["user_prompt"] = row["question"]
                    row["group"] = row["question_id"]
            elif source == "triviaqa":
                rows, info = prepare_triviaqa(seed)
                for row in rows:
                    row["user_prompt"] = row["question"]
                    row["group"] = row["question_id"]
            elif source == "popqa":
                require_tier1_complete(ROOT)
                from src.exploratory.e1_data import load_e1_frame
                frame = load_e1_frame()
                rows = []
                for row in frame.to_dict("records"):
                    grade = str(row.get("grade", row.get("baseline_partition", ""))).lower()
                    if grade not in {"correct", "wrong"}:
                        continue
                    pid = str(row.get("matched_pair_id") or row.get("popqa_id"))
                    rows.append({**row, "question": str(row["question"]), "user_prompt": str(row["question"]),
                                 "y": int(grade == "wrong"), "group": pid, "source_id": str(row["popqa_id"])})
                info = {"source_url": "repository:data/popqa_sample.jsonl plus E0 labels",
                        "revision": git_commit(ROOT), "licence": "PopQA dataset card",
                        "class_counts": {"0": sum(r["y"] == 0 for r in rows), "1": sum(r["y"] == 1 for r in rows)}}
            else:
                raise ValueError(f"Unknown source {source!r}")
        except Exception as exc:
            failures[source] = {**getattr(exc, "source_info", {}), "sampling_seed": seed,
                                "row_count": 0, "sha256": None,
                                "error": f"{type(exc).__name__}: {exc}"}
            print(f"{source}: stopped: {failures[source]['error']}", flush=True)
            continue
        path = assert_not_fresh(out_dir / f"{source}.jsonl")
        _write_jsonl(path, rows)
        manifest["sources"][source] = {**info, "row_count": len(rows), "sha256": sha256_file(path),
                                       "path": path.name}
        print(f"{source}: {len(rows)} rows -> {path}")
    manifest["failed_sources"] = failures
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ATTRIBUTION.md").write_text(
        "# E4 dataset attribution\n\n"
        "- SQuAD 2.0: Rajpurkar et al. (2018), [*Know What You Don't Know: Unanswerable Questions for SQuAD*](https://aclanthology.org/P18-2124/); dataset licence: CC BY-SA 4.0. Source: [Hugging Face dataset](https://huggingface.co/datasets/rajpurkar/squad_v2).\n"
        "- SelfAware: Yin et al. (2023), [*Do Large Language Models Know What They Don't Know?*](https://aclanthology.org/2023.findings-acl.551/); dataset licence: CC BY-SA 4.0. Source: [authors' repository](https://github.com/yinzhangyue/SelfAware).\n"
        "- TriviaQA: optional; see its pinned dataset card and the licence recorded in `MANIFEST.json` before redistribution.\n"
        "- Synthetic and PopQA records are derived from this repository's registered data and E0 labels.\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="syn,squad,selfaware")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tokenizer", default=MODEL,
                        help="Llama tokenizer repository or local path (use HF_TOKEN for gated access)")
    args = parser.parse_args(argv)
    prepare([x.strip() for x in args.sources.split(",") if x.strip()], args.out_dir, args.seed, args.tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
