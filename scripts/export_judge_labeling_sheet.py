#!/usr/bin/env python3
"""Create frozen, blinded dev/test sheets for response-judge validation."""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import CFG

SOURCES = {"popqa", "stage6", "stage10"}
DEFAULT_SOURCES = ("popqa", "stage6")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool_rows(sources=("popqa", "stage6")):
    sources = set(sources)
    unknown = sources - SOURCES
    if unknown:
        raise ValueError(f"Unknown validation sources: {sorted(unknown)}")
    pool = {}
    if "stage6" in sources:
        pairs = {r["pair_id"]: r for r in read_jsonl(CFG.data_dir / "pairs.jsonl")}
        bctrl = {}
        for row in read_jsonl(CFG.data_dir / "b_ctrl.jsonl"):
            key = row.get("b_ctrl_id") or row.get("pair_id") or row.get("item_id")
            if key is not None:
                bctrl[str(key)] = row
        for row in read_jsonl(CFG.stage6_gen_path):
            uid = f"stage6:{row.get('condition')}:{row.get('item_id')}"
            if row.get("dataset_part") == "bctrl":
                prompt = bctrl.get(str(row.get("item_id")), {}).get("prompt")
            else:
                pair = pairs.get(row.get("pair_id"), {})
                prompt = pair.get("unans_prompt" if row.get("dataset_part") == "unans" else "ans_prompt")
            if not prompt:
                raise ValueError(f"Cannot join Stage 6 prompt for {uid}")
            pool[uid] = {
                "uid": uid,
                "source": "stage6",
                "stratum": f"stage6:{row.get('condition')}",
                "question": prompt,
                "response": row.get("generated_text", ""),
                "final_label": row.get("final_label"),
                "regex_label": row.get("regex_label"),
                "judge_margin": row.get("judge_margin"),
                "judge_quantization": row.get("judge_quantization"),
            }
    if "stage10" in sources:
        generations_path = Path(CFG.stage10_generation_path)
        popqa = {r["popqa_id"]: r for r in read_jsonl(CFG.data_dir / "popqa_sample.jsonl")}
        for row in read_jsonl(generations_path):
            if row.get("alpha") is None:
                continue
            source = popqa.get(row.get("popqa_id"), {})
            uid = f"stage10:{row.get('direction', 'learned')}:{row.get('alpha')}:{row.get('popqa_id')}"
            pool[uid] = {
                "uid": uid,
                "source": "stage10",
                "stratum": f"stage10:{row.get('alpha')}:{row.get('direction', 'learned')}",
                "question": row.get("question") or source.get("question", ""),
                "response": row.get("generated_text", ""),
                "final_label": row.get("final_label"),
                "regex_label": row.get("regex_label"),
                "judge_margin": row.get("judge_margin"),
                "judge_quantization": row.get("judge_quantization"),
            }
        if not generations_path.exists():
            raise FileNotFoundError(f"Stage 10 generations not found: {generations_path}")
    if "popqa" in sources:
        for row in read_jsonl(CFG.data_dir / "popqa_sample.jsonl"):
            uid = f"popqa:{row['popqa_id']}"
            classifier = row.get("classifier") or {}
            pool[uid] = {
                "uid": uid,
                "source": "popqa",
                "stratum": "popqa:baseline",
                "question": row["question"],
                "response": row.get("baseline_response", ""),
                "final_label": classifier.get("final_label"),
                "regex_label": classifier.get("regex_label"),
                "judge_margin": classifier.get("judge_margin"),
                "judge_quantization": classifier.get("judge_quantization"),
            }
    return pool


def allocate(groups, n, rng, floor):
    if len(groups) * floor > n:
        raise ValueError("Sample size is too small for the per-stratum floor")
    quotas = {key: min(len(rows), floor) for key, rows in groups.items()}
    left = n - sum(quotas.values())
    while left:
        eligible = [key for key, rows in groups.items() if quotas[key] < len(rows)]
        if not eligible:
            break
        weights = {key: len(groups[key]) - quotas[key] for key in eligible}
        denominator = sum(weights.values())
        exact = {key: left * weights[key] / denominator for key in eligible}
        additions = {key: min(len(groups[key]) - quotas[key], int(exact[key])) for key in eligible}
        for key, count in additions.items():
            quotas[key] += count
        used = sum(additions.values())
        left -= used
        if left:
            order = sorted(
                (key for key in eligible if quotas[key] < len(groups[key])),
                key=lambda key: (-(exact[key] - int(exact[key])), key),
            )
            added = min(left, len(order))
            for key in order[:added]:
                quotas[key] += 1
            left -= added
    out = []
    for key, rows in groups.items():
        out.extend(rng.sample(rows, quotas[key]))
    return out


def export_sheets(output_dir, prefix, sources, n_test, n_dev, offline=True):
    from src.abstention_classifier import classify_texts

    pool = pool_rows(sources)
    pending = [row for row in pool.values() if not row.get("final_label")]
    if pending:
        classified = classify_texts(
            [{"uid": row["uid"], "question": row["question"], "response": row["response"]} for row in pending],
            offline=offline,
        )
        for row, record in zip(pending, classified):
            row["final_label"] = record["final_label"]
            row["regex_label"] = record["regex_label"]
            row["judge_margin"] = record.get("judge_margin")
            row["judge_quantization"] = record.get("judge_quantization")
    groups = defaultdict(list)
    for row in pool.values():
        groups[row["stratum"]].append(row)
    if not groups:
        raise ValueError("No rows found for selected validation sources")
    bad = {key: sum(row["final_label"] == "UNPARSEABLE" for row in rows) for key, rows in groups.items()}
    over = {key: f"{count}/{len(groups[key])}" for key, count in bad.items() if count / len(groups[key]) > 0.01}
    if over:
        raise RuntimeError(f"Judge validation pool exceeds the 1% UNPARSEABLE limit: {over}")
    rng = random.Random(CFG.judge_validation_seed)
    test = allocate(groups, n_test, rng, 4)
    selected = {row["uid"] for row in test}
    remaining = {key: [row for row in rows if row["uid"] not in selected] for key, rows in groups.items()}
    ranked = []
    for rows in remaining.values():
        for row in rows:
            alpha_extreme = row["source"] == "stage10" and (":-2.0:" in row["stratum"] or ":2.0:" in row["stratum"])
            ranked.append((0 if row["final_label"] != row["regex_label"] else 1, 0 if alpha_extreme else 1, row))
    ranked.sort(key=lambda entry: (entry[0], entry[1], entry[2]["uid"]))
    dev = [entry[2] for entry in ranked[:n_dev]]
    output_dir.mkdir(parents=True, exist_ok=True)
    blinded = {row["uid"]: "J" + hashlib.sha256(row["uid"].encode()).hexdigest()[:18] for row in test + dev}
    for split, rows in (("test", test), ("dev", dev)):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        with (output_dir / f"{prefix}_{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["uid", "question", "response", "human_label", "notes"])
            writer.writeheader()
            for row in shuffled:
                writer.writerow({"uid": blinded[row["uid"]], "question": row["question"], "response": row["response"],
                                 "human_label": "", "notes": ""})
    key = {
        blinded[row["uid"]]: {"final_label": row["final_label"], "regex_label": row["regex_label"],
                              "judge_margin": row.get("judge_margin"), "stratum": row["stratum"], "source": row["source"]}
        for row in test + dev
    }
    (output_dir / f"{prefix}_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from src.abstention_classifier import JUDGE_PROMPT_SHA256, JUDGE_PROMPT_VERSION, _revision
    quantizations = sorted({row.get("judge_quantization") for row in pool.values() if row.get("judge_quantization")})
    if len(quantizations) > 1:
        raise RuntimeError(f"Mixed judge quantizations in validation pool: {quantizations}")
    manifest = {
        "seed": CFG.judge_validation_seed,
        "sources": sorted(sources),
        "judge_model_id": CFG.judge_model_id,
        "judge_model_revision": _revision(),
        "judge_quantization": quantizations[0] if quantizations else CFG.judge_quantization,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "test_uids": sorted(blinded[row["uid"]] for row in test),
        "dev_uids": sorted(blinded[row["uid"]] for row in dev),
        "strata": {key: len(rows) for key, rows in groups.items()},
    }
    (output_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {prefix}: {len(test)} held-out, {len(dev)} dev; sources={','.join(sorted(sources))}; offline={offline}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=CFG.results_dir)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--prefix", default="judge_validation")
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-dev", type=int, default=100)
    parser.add_argument("--allow-online-judge", action="store_true")
    args = parser.parse_args()
    sources = tuple(part.strip() for part in args.sources.split(",") if part.strip())
    if not sources or len(set(sources)) != len(sources) or set(sources) - SOURCES:
        raise ValueError(f"--sources must be a comma-separated subset of {sorted(SOURCES)}")
    if args.n_test <= 0 or args.n_dev < 0:
        raise ValueError("--n-test must be positive and --n-dev must not be negative")
    export_sheets(args.output_dir, args.prefix, sources, args.n_test, args.n_dev, offline=not args.allow_online_judge)


if __name__ == "__main__":
    main()
