"""Export the human review worksheet for the current template catalog."""
import csv
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG
from src.template_catalog import generate_raw_template_catalog, calibrate_template_token_parity
from src.dataset_builder import (
    _adopt_reviewed_slot_values,
    find_unanswerable_style_confounds,
    find_family_b_interrogative_stem_mismatches,
)

FLAGS = ("check_answer_unambiguous", "check_zero_unans_support", "check_distinct_predicate", "check_b_ctrl_valid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=CONFIG.primary_model_name)
    args = parser.parse_args()
    output = Path("data/review/review_sheet.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["template_id", "family", "ans_prompt", "unans_prompt", "b_ctrl_prompt", "ground_truth", *FLAGS]
    old_rows = {}
    if output.exists():
        with output.open(encoding="utf-8-sig", newline="") as f:
            old_rows = {r["template_id"]: r for r in csv.DictReader(f)}
    backup_path = output.with_name("review_sheet.before_calibration_refresh.csv")
    backup_rows = {}
    if backup_path.exists():
        with backup_path.open(encoding="utf-8-sig", newline="") as f:
            backup_rows = {r["template_id"]: r for r in csv.DictReader(f)}
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            print("[FAIL] Missing dependency 'transformers' in this interpreter. On Debian/Ubuntu, create an isolated environment and use its Python:")
            print("  python3 -m venv .venv")
            print("  .venv/bin/python -m pip install -r requirements.txt")
            print("  .venv/bin/python scripts/stage1_export_review.py")
            return 1
        raise
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    catalog, audit_rows = calibrate_template_token_parity(
        generate_raw_template_catalog(tokenizer=tokenizer), tokenizer, max_diff=CONFIG.max_token_diff_pair)
    parity_failures = [r["template_id"] for r in audit_rows if not r["post_fix_passed"]]
    if parity_failures:
        print("[FAIL] Token parity calibration needs catalog edits for: " + ", ".join(parity_failures))
        return 1
    style_failures = find_unanswerable_style_confounds(catalog)
    if style_failures:
        print("[FAIL] Unanswerable stylistic confounds need catalog edits: " + ", ".join(style_failures))
        return 1
    stem_failures = find_family_b_interrogative_stem_mismatches(catalog)
    if stem_failures:
        print("[FAIL] Family B interrogative stems differ: " + ", ".join(stem_failures))
        return 1
    preserved_ids = []
    reset_ids = []
    pending_review_ids = []
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for tmpl in catalog:
            tid = tmpl["template_id"]
            previous = old_rows.get(tid, {})
            backup = backup_rows.get(tid, {})
            matching_sources = []
            for source in (previous, backup):
                if not source:
                    continue
                # Slot values can differ across exports. Adopt them only when the
                # literal template wording matches, then compare the three prompts.
                if not _adopt_reviewed_slot_values(tmpl, source):
                    continue
                candidate_slot = tmpl["slots"][0]
                candidate_prompts = {
                    "ans_prompt": tmpl["ans_template"].format(**candidate_slot),
                    "unans_prompt": tmpl["unans_template"].format(**candidate_slot),
                    "b_ctrl_prompt": "NA" if tmpl["family"] == "A" else tmpl["b_ctrl_template"].format(**candidate_slot),
                }
                if all(source.get(key, "").strip() == value for key, value in candidate_prompts.items()):
                    matching_sources.append(source)

            slot = tmpl["slots"][0]
            prompts = {
                "ans_prompt": tmpl["ans_template"].format(**slot),
                "unans_prompt": tmpl["unans_template"].format(**slot),
                "b_ctrl_prompt": "NA" if tmpl["family"] == "A" else tmpl["b_ctrl_template"].format(**slot),
            }
            if matching_sources:
                preserved_ids.append(tid)
                restored_flags = {}
                for flag in FLAGS:
                    value = next((src.get(flag, "") for src in matching_sources if src.get(flag, "").strip()), "")
                    restored_flags[flag] = value
            else:
                reset_ids.append(tid)
                restored_flags = {flag: "" for flag in FLAGS}
            for flag in FLAGS:
                value = restored_flags[flag].strip().upper()
                accepted = value in {"Y", "YES", "TRUE"}
                accepted = accepted or (value == "NA" and tmpl["family"] == "A" and flag == "check_b_ctrl_valid")
                if not accepted:
                    pending_review_ids.append(tid)
                    break
            writer.writerow({
                "template_id": tmpl["template_id"], "family": tmpl["family"],
                **prompts,
                "ground_truth": slot["ground_truth"],
                **restored_flags,
            })
    print(f"[PASS] Wrote {output} with {len(catalog)} templates; preserved {len(preserved_ids)} rows, reset {len(reset_ids)} rows.")
    print(f"Prompt rows reset in this export ({len(reset_ids)}): " + (", ".join(reset_ids) if reset_ids else "none"))
    print(f"Templates still needing review ({len(pending_review_ids)}): " + (", ".join(pending_review_ids) if pending_review_ids else "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
