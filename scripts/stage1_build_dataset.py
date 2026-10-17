# scripts/stage1_build_dataset.py
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from src.config import CONFIG
from src.template_catalog import (
    generate_raw_template_catalog,
    calibrate_template_token_parity,
)
from src.dataset_builder import (
    create_template_splits,
    instantiate_pairs_and_b_ctrl,
    compute_max_ngram_phi,
    sample_stratified_popqa,
    write_audit_log,
)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validate_existing_stage1_outputs(data_dir: Path, cut3: bool = False) -> bool:
    """
    Deterministic Stage 1 Acceptance Gate (Section 9, Stage 1):
    Verifies all 6 files in `data/` exist and satisfy all schema, count,
    disjoint split, token-length parity, and n-gram leakage invariants.
    """
    expected_n_templates = CONFIG.n_templates_cut3 if cut3 else CONFIG.n_templates_total
    expected_n_pairs = expected_n_templates * CONFIG.n_pairs_per_template
    expected_n_b_ctrl = (expected_n_templates // 8) * CONFIG.n_pairs_per_template

    templates_path = data_dir / "templates.jsonl"
    splits_path = data_dir / "splits.json"
    pairs_path = data_dir / "pairs.jsonl"
    b_ctrl_path = data_dir / "b_ctrl.jsonl"
    popqa_path = data_dir / "popqa_sample.jsonl"
    audit_path = data_dir / "audit_log.csv"

    for p in (templates_path, splits_path, pairs_path, b_ctrl_path, popqa_path, audit_path):
        if not p.exists():
            return False

    try:
        templates = read_jsonl(templates_path)
        if len(templates) != expected_n_templates:
            return False
        for t in templates:
            if not (
                t.get("check_answer_unambiguous") is True
                and t.get("check_zero_unans_support") is True
                and t.get("check_distinct_predicate") is True
                and t.get("check_b_ctrl_valid") is True
            ):
                return False

        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        tr_set = set(splits["train_template_ids"])
        va_set = set(splits["val_template_ids"])
        te_set = set(splits["test_template_ids"])
        if not (tr_set.isdisjoint(va_set) and tr_set.isdisjoint(te_set) and va_set.isdisjoint(te_set)):
            return False
        if len(tr_set) + len(va_set) + len(te_set) != expected_n_templates:
            return False

        pairs = read_jsonl(pairs_path)
        if len(pairs) != expected_n_pairs:
            return False
        for pair in pairs:
            if abs(int(pair["token_len_diff"])) > CONFIG.max_token_diff_pair:
                return False

        max_phi, _, _ = compute_max_ngram_phi(pairs, max_n=3)
        if max_phi > CONFIG.max_ngram_phi:
            return False

        b_ctrl = read_jsonl(b_ctrl_path)
        if len(b_ctrl) != expected_n_b_ctrl:
            return False

        popqa = read_jsonl(popqa_path)
        if len(popqa) < CONFIG.n_popqa_init:
            return False

        with open(audit_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) < 1:
                return False

        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1: Build & Validate Datasets on CPU")
    parser.add_argument(
        "--model-name",
        type=str,
        default=CONFIG.primary_model_name,
        help="Tokenizer model name (default: Qwen/Qwen2.5-3B-Instruct).",
    )
    parser.add_argument(
        "--cut3",
        action="store_true",
        help="Use Cut 3 reduced template count (64 templates instead of 96).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all Stage 1 files.",
    )
    args = parser.parse_args()

    CONFIG.data_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and validate_existing_stage1_outputs(CONFIG.data_dir, cut3=args.cut3):
        print(f"[SKIP] Stage 1 already verified in {CONFIG.data_dir}")
        return 0

    n_per_fam = (CONFIG.n_templates_cut3 // 2) if args.cut3 else (CONFIG.n_templates_total // 2)
    print(f"[Stage 1] Loading tokenizer for {args.model_name} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[Stage 1] Generating {2 * n_per_fam} templates and running token-parity calibration...")
    raw_catalog = generate_raw_template_catalog(n_per_family=n_per_fam)
    calibrated_catalog, audit_rows = calibrate_template_token_parity(
        raw_catalog,
        tokenizer=tokenizer,
        max_diff=CONFIG.max_token_diff_pair,
    )

    templates_path = CONFIG.data_dir / "templates.jsonl"
    write_jsonl(templates_path, calibrated_catalog)

    audit_path = CONFIG.data_dir / "audit_log.csv"
    if args.force and audit_path.exists():
        audit_path.unlink()
    write_audit_log(audit_path, audit_rows)
    print(f"[Stage 1] Saved {len(calibrated_catalog)} templates and {len(audit_rows)} audit log entry(s).")

    splits = create_template_splits(calibrated_catalog, seed_split=CONFIG.seed_split)
    splits_path = CONFIG.data_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    pairs, b_ctrl_rows = instantiate_pairs_and_b_ctrl(calibrated_catalog, splits, tokenizer)
    write_jsonl(CONFIG.data_dir / "pairs.jsonl", pairs)
    write_jsonl(CONFIG.data_dir / "b_ctrl.jsonl", b_ctrl_rows)

    max_phi, worst_ngram, top_10_phi = compute_max_ngram_phi(pairs, max_n=3)
    print(
        f"[Stage 1 Quality Gate 2] Max N-Gram |phi| = {max_phi:.4f} "
        f"(threshold <= {CONFIG.max_ngram_phi:.2f}) | Top n-gram: '{worst_ngram}'"
    )
    if max_phi > CONFIG.max_ngram_phi:
        # Section 4.3: every initial gate failure is logged permanently; templates must then be
        # rewritten by hand in src/template_catalog.py and Stage 1 re-run.
        write_audit_log(
            audit_path,
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "template_id": "ALL",
                    "failure_reason": "ngram_leakage",
                    "offending_detail": f"Top n-grams by phi: {top_10_phi}",
                    "old_text": worst_ngram,
                    "new_text": "",
                    "post_fix_passed": False,
                }
            ],
        )
        print(
            f"[FAIL] N-Gram leakage gate failed! '{worst_ngram}' has "
            f"|phi|={max_phi:.4f} > {CONFIG.max_ngram_phi}. Logged to {audit_path}."
        )
        return 1

    print(f"[Stage 1] Sampling {CONFIG.n_popqa_init} stratified questions from akariasai/PopQA...")
    popqa_rows = sample_stratified_popqa(
        n_total=CONFIG.n_popqa_init,
        w_bottom5=CONFIG.w_popqa_bottom5,
        w_top5=CONFIG.w_popqa_top5,
        seed_popqa=CONFIG.seed_popqa,
    )
    write_jsonl(CONFIG.data_dir / "popqa_sample.jsonl", popqa_rows)

    assert validate_existing_stage1_outputs(CONFIG.data_dir, cut3=args.cut3), (
        "Stage 1 outputs failed final acceptance gate!"
    )
    print(
        f"[PASS] Stage 1 complete: {len(calibrated_catalog)} templates, "
        f"{len(pairs)} pairs, {len(b_ctrl_rows)} B-ctrl items, "
        f"{len(popqa_rows)} PopQA items verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

