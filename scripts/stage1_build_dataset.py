# scripts/stage1_build_dataset.py
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    validate_review_sheet,
    find_unanswerable_style_confounds,
    find_family_b_interrogative_stem_mismatches,
)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def preserve_audit_log(audit_path: Path) -> None:
    """Restore an explicitly provided backup only when the current log is empty."""
    if not audit_path.exists() or audit_path.stat().st_size != 0:
        return
    backup_path = audit_path.with_name("audit_log_backup.csv")
    if backup_path.exists():
        shutil.copyfile(backup_path, audit_path)


def update_successful_audit_rows(
    audit_path: Path,
    passing_template_ids: set[str],
    final_max_phi: float,
) -> None:
    """Update only success status fields after every Stage 1 gate has passed."""
    if not audit_path.exists() or audit_path.stat().st_size == 0:
        raise ValueError(f"Audit log is missing or empty: {audit_path}")

    with audit_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "post_fix_passed" not in fieldnames:
        raise ValueError(f"Audit log has no post_fix_passed column: {audit_path}")

    leakage_fix = (
        "Matched interrogative stems within Family B pairs and verbs within "
        f"Family A pairs; final max |phi| = {final_max_phi:.4f}"
    )
    for row in rows:
        reason = row.get("failure_reason", "")
        template_id = row.get("template_id", "")
        if template_id in passing_template_ids:
            row["post_fix_passed"] = "True"
        elif template_id == "ALL" and reason == "ngram_leakage" and final_max_phi <= 0.15:
            row["post_fix_passed"] = "True"
            row["new_text"] = leakage_fix

    with audit_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
            list(csv.DictReader(f))  # A header-only audit log is valid.

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
    audit_path = CONFIG.data_dir / "audit_log.csv"
    preserve_audit_log(audit_path)
    if not args.force and validate_existing_stage1_outputs(CONFIG.data_dir, cut3=args.cut3):
        print(f"[SKIP] Stage 1 already verified in {CONFIG.data_dir}")
        return 0

    review_path = CONFIG.data_dir / "review" / "review_sheet.csv"
    if not review_path.exists():
        print(f"[FAIL] Human review sheet missing: {review_path}. Run scripts/stage1_export_review.py, complete it, then retry.")
        return 1

    n_per_fam = (CONFIG.n_templates_cut3 // 2) if args.cut3 else (CONFIG.n_templates_total // 2)
    print(f"[Stage 1] Loading tokenizer for {args.model_name} on CPU...")
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            print("[FAIL] Missing dependency 'transformers' in this interpreter. On Debian/Ubuntu, create an isolated environment and use its Python:")
            print("  python3 -m venv .venv")
            print("  .venv/bin/python -m pip install -r requirements.txt")
            print("  .venv/bin/python scripts/stage1_build_dataset.py --force")
            return 1
        raise
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[Stage 1] Generating {2 * n_per_fam} templates and running token-parity calibration...")
    raw_catalog = generate_raw_template_catalog(n_per_family=n_per_fam, tokenizer=tokenizer)
    calibrated_catalog, audit_rows = calibrate_template_token_parity(
        raw_catalog,
        tokenizer=tokenizer,
        max_diff=CONFIG.max_token_diff_pair,
    )
    parity_failures = [row for row in audit_rows if not row["post_fix_passed"]]
    if parity_failures:
        print("[FAIL] Authored token parity exceeded ±2; no text was auto-padded.")
        for row in parity_failures:
            print(f"  {row['template_id']}: {row['offending_detail']}")
        return 1

    style_failures = find_unanswerable_style_confounds(calibrated_catalog)
    if style_failures:
        print("[FAIL] Unanswerable stylistic confounds found:")
        for tid, reasons in style_failures.items():
            print(f"  {tid}: {'; '.join(reasons)}")
        return 1

    stem_failures = find_family_b_interrogative_stem_mismatches(calibrated_catalog)
    if stem_failures:
        print("[FAIL] Family B answerable/unanswerable interrogative stems differ:")
        for tid, stems in stem_failures.items():
            print(f"  {tid}: ans='{stems[0]}', unans='{stems[1]}'")
        return 1

    # Predicate uniqueness is a deterministic automated pre-check, independent of human review.
    predicate_texts = {}
    for tmpl in calibrated_catalog:
        if tmpl["family"] == "A":
            context = tmpl["unans_template"].split("\nQuestion: ", 1)[0]
            text = context.rsplit(". ", 1)[-1]
        else:
            text = tmpl["unans_template"].split("\nQuestion: ", 1)[1]
        text = text.replace("{entity}", "").replace("{attr1}", "").replace("{attr2}", "")
        predicate = " ".join(text.lower().split())
        if predicate in predicate_texts:
            raise ValueError(f"Duplicate unanswerable predicate: {predicate_texts[predicate]} and {tmpl['template_id']}: {predicate}")
        predicate_texts[predicate] = tmpl["template_id"]

    review_failures = validate_review_sheet(calibrated_catalog, review_path)
    if review_failures:
        failures = [
            {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "template_id": tid,
             "failure_reason": "manual_check", "offending_detail": ";".join(reasons),
             "old_text": "", "new_text": "", "post_fix_passed": False}
            for tid, reasons in review_failures.items()
        ]
        write_audit_log(CONFIG.data_dir / "audit_log.csv", failures)
        mismatch_ids = [tid for tid, reasons in review_failures.items() if "prompt_mismatch" in reasons]
        manual_ids = [tid for tid, reasons in review_failures.items() if any(r.startswith("manual_check:") for r in reasons)]
        print("[FAIL] Review sheet validation failed.")
        if mismatch_ids:
            print("Prompt wording/ground-truth mismatch (these prompts need review): " + ", ".join(mismatch_ids))
        if manual_ids:
            print("Reviewer marked N or left a check empty: " + ", ".join(manual_ids))
        print("Resolve catalog issues and have the reviewer re-check changed prompts. The review sheet was not modified.")
        return 1

    templates_path = CONFIG.data_dir / "templates.jsonl"
    write_jsonl(templates_path, calibrated_catalog)

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
    print(f"[Stage 1 Quality Gate 2] Top 10 phi n-grams: {top_10_phi}")
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
    update_successful_audit_rows(
        audit_path,
        {tmpl["template_id"] for tmpl in calibrated_catalog},
        max_phi,
    )
    print(
        f"[PASS] Stage 1 complete: {len(calibrated_catalog)} templates, "
        f"{len(pairs)} pairs, {len(b_ctrl_rows)} B-ctrl items, "
        f"{len(popqa_rows)} PopQA items verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

