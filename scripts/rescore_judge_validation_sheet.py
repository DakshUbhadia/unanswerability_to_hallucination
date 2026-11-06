#!/usr/bin/env python3
"""Rescore frozen judge-validation sheets from the current offline judge cache.

The human labels and blinded UIDs remain unchanged.  This helper updates only the
hidden prediction key and prompt metadata after an audited judge-prompt revision.
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.abstention_classifier import (  # noqa: E402
    JUDGE_PROMPT_SHA256,
    JUDGE_PROMPT_VERSION,
    LABELS,
    _revision,
    classify_texts,
)
from src.config import CFG  # noqa: E402


def _read_sheet(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"uid", "question", "response", "human_label", "notes"}
    if not rows:
        raise ValueError(f"Frozen validation sheet is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    uids = [row["uid"] for row in rows]
    if any(not uid for uid in uids) or len(uids) != len(set(uids)):
        raise ValueError(f"{path} contains missing or duplicate blinded UIDs")
    invalid = [row["uid"] for row in rows if row["human_label"] not in LABELS]
    if invalid:
        raise ValueError(f"Every frozen row must retain a valid human_label; invalid UIDs: {invalid[:5]}")
    return rows


def _write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def _archive_prior_result(path: Path) -> Path | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("judge_prompt_version")
    if not version or version == JUDGE_PROMPT_VERSION:
        return None
    archive = path.with_name(f"{path.stem}_{version}{path.suffix}")
    if archive.exists() and json.loads(archive.read_text(encoding="utf-8")) != payload:
        raise FileExistsError(f"Refusing to overwrite a different archived validation result: {archive}")
    if not archive.exists():
        _write_json_atomic(archive, payload)
    return archive


def rescore_frozen_sheets(
    results_dir: Path,
    prefix: str = "judge_validation",
    splits=("test", "dev"),
    cache_path: Path | None = None,
):
    results_dir = Path(results_dir)
    sheet_rows = []
    seen = set()
    for split in splits:
        path = results_dir / f"{prefix}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen validation sheet: {path}")
        rows = _read_sheet(path)
        overlap = seen.intersection(row["uid"] for row in rows)
        if overlap:
            raise ValueError(f"Blinded UIDs occur in multiple splits: {sorted(overlap)[:5]}")
        seen.update(row["uid"] for row in rows)
        sheet_rows.extend((split, row) for row in rows)

    key_path = results_dir / f"{prefix}_key.json"
    manifest_path = results_dir / f"{prefix}_manifest.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(key) != seen:
        missing = sorted(seen - set(key))[:5]
        extra = sorted(set(key) - seen)[:5]
        raise ValueError(f"Frozen sheets/key UID mismatch; missing={missing}, extra={extra}")

    items = [
        {"uid": row["uid"], "question": row["question"], "response": row["response"]}
        for _split, row in sheet_rows
    ]
    records = classify_texts(items, offline=True, cache_path=cache_path)
    if len(records) != len(items):
        raise RuntimeError(f"Expected {len(items)} rescored rows, received {len(records)}")

    quantizations = set()
    for (_split, row), record in zip(sheet_rows, records):
        if record.get("final_label_source") == "human":
            raise ValueError(f"Validation UID {row['uid']} has a human cache override; held-out scoring must use the judge")
        prediction = record.get("judge_label")
        if prediction not in LABELS:
            raise ValueError(f"Validation UID {row['uid']} has invalid judge label {prediction!r}")
        if record.get("judge_prompt_version") != JUDGE_PROMPT_VERSION:
            raise ValueError(f"Validation UID {row['uid']} was not scored with {JUDGE_PROMPT_VERSION}")
        if record.get("judge_prompt_sha256") != JUDGE_PROMPT_SHA256:
            raise ValueError(f"Validation UID {row['uid']} has the wrong prompt hash")
        quant = record.get("judge_quantization")
        if quant:
            quantizations.add(quant)
        key[row["uid"]].update(
            final_label=prediction,
            regex_label=record.get("regex_label"),
            judge_margin=record.get("judge_margin"),
        )
    if len(quantizations) != 1:
        raise ValueError(f"Expected exactly one judge quantization, found {sorted(quantizations)}")

    prior = {
        "judge_prompt_version": manifest.get("judge_prompt_version"),
        "judge_prompt_sha256": manifest.get("judge_prompt_sha256"),
        "judge_model_revision": manifest.get("judge_model_revision"),
        "judge_quantization": manifest.get("judge_quantization"),
    }
    history = list(manifest.get("judge_rescore_history", []))
    if prior.get("judge_prompt_version") != JUDGE_PROMPT_VERSION and prior not in history:
        history.append(prior)
    manifest.update(
        judge_model_revision=_revision(),
        judge_quantization=next(iter(quantizations)),
        judge_prompt_version=JUDGE_PROMPT_VERSION,
        judge_prompt_sha256=JUDGE_PROMPT_SHA256,
        judge_rescore_history=history,
    )

    archives = []
    for split in splits:
        archive = _archive_prior_result(
            results_dir / (f"{prefix}.json" if split == "test" else f"{prefix}_{split}.json")
        )
        if archive:
            archives.append(str(archive))
    _write_json_atomic(key_path, key)
    _write_json_atomic(manifest_path, manifest)
    return {"rows": len(records), "splits": list(splits), "archives": archives}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=CFG.results_dir)
    parser.add_argument("--prefix", default="judge_validation")
    parser.add_argument("--splits", default="test,dev")
    parser.add_argument("--cache", type=Path, default=CFG.judge_cache_path)
    args = parser.parse_args()
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits or len(splits) != len(set(splits)):
        raise ValueError("--splits must contain one or more unique comma-separated names")
    result = rescore_frozen_sheets(args.results_dir, args.prefix, splits, args.cache)
    print(
        f"Rescored {result['rows']} frozen rows across {','.join(result['splits'])} with "
        f"judge prompt {JUDGE_PROMPT_VERSION}; archived={result['archives']}"
    )


if __name__ == "__main__":
    main()
