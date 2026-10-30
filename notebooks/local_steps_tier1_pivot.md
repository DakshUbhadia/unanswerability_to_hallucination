# Local WSL steps: Tier 1 refresh and exploratory pivot

Run every step from the repository root in the existing WSL virtual environment. `HF_TOKEN` is needed for gated model artifacts; `HF_TENSOR_REPO_ID` is needed when restoring feature tensors. Keep the judge cache closed until the held-out judge sheet has been labeled. Never open, print, or diff any `*_key.json` file.

## L1 — Judge validation (after Kaggle G1)

```bash
git pull --ff-only
export PYTHONPATH=.
python scripts/export_judge_labeling_sheet.py
```

Label only `results/judge_validation_test.csv` with `ANSWER`, `ABSTAIN`, or `DEGENERATE`. The dev sheet is used only if the held-out gate fails and the owner is instructed to label it.

```bash
python scripts/import_judge_labels.py --csv results/judge_validation_test.csv
python -c 'import json; d=json.load(open("results/judge_validation.json")); print({"gate_passed":d.get("gate_passed"),"kappa":d.get("cohen_kappa"),"abstain_recall":d.get("recall",{}).get("ABSTAIN"),"low_margin_gate_passed":d.get("low_margin_gate_passed")})'
```

If `low_margin_gate_passed` is false, export and label the low-margin review, then import it:

```bash
python scripts/export_judge_low_margin_review.py
# Human labels results/judge_low_margin_review.csv.
python scripts/import_judge_overrides.py
```

After a successful import, retain the blinded audit key with the labeled artifacts:

```bash
git add -f results/judge_validation_test.csv results/judge_validation.json results/judge_validation_manifest.json results/judge_validation_key.json results/judge_low_margin_review.csv
git commit -m "Tier 1: judge validation (held-out)"
git pull --rebase origin main
git push origin main
```

Do not continue to the judge-dependent local steps unless `gate_passed` is true.

## L2 — Grader gate, rematch, and E3 spot-checks (after G2)

```bash
git pull --ff-only
export PYTHONPATH=.
python scripts/stage6_rescore_generations.py --offline
python scripts/stage2_pilot_gate.py --regrade-saved-popqa-only --offline
```

Read the command summary and stop if there are fewer than 100 matched pairs. Do not widen the caliper or use `popqa_fresh`.

```bash
python scripts/export_grader_labeling_sheet.py
python scripts/exploratory/e3_export_validation.py
```

Label `results/grader_validation_test.csv`, `results/exploratory/e3_judge_validation_test.csv`, and `results/exploratory/e3_grader_validation_test.csv`. The test CSVs are labeled first; dev sheets are only labeled if a gate fails and the owner is instructed to do so. E3 grader rule: mark CORRECT only if the answer commits to an accepted answer; hedged lists such as “France or Spain” are INCORRECT.

```bash
python scripts/import_grader_labels.py --csv results/grader_validation_test.csv
python -c 'import json; assert json.load(open("results/grader_validation.json"))["gate_passed"]'
python scripts/stage2_pilot_gate.py --rematch-only --offline
python scripts/verify_tier1_popqa_regeneration.py
python scripts/import_judge_labels.py --prefix e3_judge_validation --csv results/exploratory/e3_judge_validation_test.csv --output results/exploratory/e3_judge_validation.json
python scripts/import_grader_labels.py --key results/exploratory/e3_grader_validation_key.json --csv results/exploratory/e3_grader_validation_test.csv --output results/exploratory/e3_grader_validation.json
python scripts/exploratory/e4_analyze.py --condition registered
```

The E3 reliability checks use κ ≥ 0.85 and ABSTAIN recall ≥ 0.90 for the judge, and CORRECT precision ≥ 0.90 for the grader. If these are missed, E3 PopQA-dependent conclusions remain provisional. Do not change prompts or thresholds.

```bash
git add -f results/grader_validation_test.csv results/grader_validation.json results/grader_validation_manifest.json results/grader_validation_key.json results/exploratory/e3_judge_validation_test.csv results/exploratory/e3_judge_validation.json results/exploratory/e3_judge_validation_manifest.json results/exploratory/e3_judge_validation_key.json results/exploratory/e3_grader_validation_test.csv results/exploratory/e3_grader_validation.json results/exploratory/e3_grader_validation_manifest.json results/exploratory/e3_grader_validation_key.json results/stage2_pilot.json data/popqa_sample.jsonl
git add results/*.json results/*.csv data/popqa_sample.jsonl tables figures
# Include the E2/E3/E4 exploratory outputs, but never add tensor files or keys except the imported audit keys above.
git add -f results/exploratory/e3_judge_validation_test.csv results/exploratory/e3_judge_validation.json results/exploratory/e3_judge_validation_manifest.json results/exploratory/e3_judge_validation_key.json results/exploratory/e3_grader_validation_test.csv results/exploratory/e3_grader_validation.json results/exploratory/e3_grader_validation_manifest.json results/exploratory/e3_grader_validation_key.json results/exploratory/e4_features/registered/*/manifest.json results/exploratory/e4_features/registered/*/meta.parquet data/exploratory/e4
git commit -m "Tier 1: grader validation, rematch; E3 spot-check; E4 non-PopQA"
git pull --rebase origin main
git push origin main
```

## L3 — Stage 10, E1, and E2 (after Kaggle G3)

```bash
git pull --ff-only
export PYTHONPATH=.
python scripts/export_judge_labeling_sheet.py --sources stage10 --prefix judge_validation_stage10 --n-test 100 --n-dev 50
```

Label `results/judge_validation_stage10_test.csv`, then run:

```bash
python scripts/import_judge_labels.py --prefix judge_validation_stage10 --csv results/judge_validation_stage10_test.csv
python scripts/stage10_popqa_steering.py --analyze-only --offline
python scripts/stage9_generate_figures_and_tables.py
python scripts/build_archive_manifest.py --name archive_v43 --files 'data/popqa_sample.jsonl' 'data/judge_cache.jsonl' 'results/stage2*.json*' 'results/stage6*.json*' 'results/stage8*.json*' 'results/stage9*.json*' 'results/stage10*.json*' 'results/judge_validation*.json' 'results/grader_validation.json' 'tables' 'figures'
python scripts/verify_archive_v43.py
python scripts/exploratory/e1_analyze.py --population all --n-jobs 4
python scripts/exploratory/e2_analyze.py --parts R,G,L,A --n-jobs 4
```

E1 extraction ran only after the forced, judged Stage 8 output in G3. Confirm its `stage8_confidence_parity_max_abs_diff` and `greedy_consistency_rate` before using answer-position results. The E2 run writes `results/exploratory/e2_dom_11_suf-4.npy`; its checksum is stored in `e2_results.json`.

```bash
git add -f results/judge_validation_stage10_test.csv results/judge_validation_stage10.json results/judge_validation_stage10_manifest.json results/judge_validation_stage10_key.json results/archive_v43 results/exploratory/e1_results.json results/exploratory/e1_summary.md results/exploratory/e2_results.json results/exploratory/e2_summary.md results/exploratory/e2_dom_11_suf-4.npy figures/exploratory
git commit -m "Tier 1 complete; E1; E2 R/G/L/A"
git pull --rebase origin main
git push origin main
```

## L4 — E2 Part C, E3, E4, and report (after Kaggle G4)

```bash
git pull --ff-only
export PYTHONPATH=.
python scripts/exploratory/e2_analyze.py --parts C
python scripts/exploratory/e3_label_and_match.py --model llama32_3b --offline
```

Run the Qwen label step only if `results/exploratory/e3_pilot_qwen25_3b.json` records `arm_included: true`:

```bash
python scripts/exploratory/e3_label_and_match.py --model qwen25_3b --offline
```

Then run CPU analyses and reporting:

```bash
python scripts/exploratory/e3_analyze.py
python scripts/exploratory/e4_analyze.py --condition both
python scripts/exploratory/build_report.py
```

Review the outputs and their provisional flags, then commit and push:

```bash
git add -f results/exploratory/e2_results.json results/exploratory/e2_summary.md results/exploratory/e3_labels_*.json results/exploratory/e3_results.json results/exploratory/e3_summary.md results/exploratory/e4_results.json results/exploratory/e4_summary.md results/exploratory/REPORT.md figures/exploratory data/exploratory/popqa_neutral_*.jsonl
# Include the five E3 PNGs and E4 figures if generated.
git add -f figures/exploratory/e3_*.png
git commit -m "E2 Part C; E3; E4 complete"
git pull --rebase origin main
git push origin main
```
