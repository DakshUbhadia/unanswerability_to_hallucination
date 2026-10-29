# Implementation Plan: Finish Tier 1 (E0) and Enable the Exploratory Pivot

**Repository:** `DakshUbhadia/unanswerability_to_hallucination` (plan written against commit `32de864`, "Run Kaggle stages 2")
**Audience:** an AI coding agent with full read/write access to the repository, working with a human owner (Daksh) who controls Kaggle GPU sessions and supplies all human labels.
**Goal:**
1. Complete the Tier 1 measurement correction (Addendum B, protocol v4.2). The PopQA correctness labels become judge-based and tokenization-consistent, and Stages 6, 8, 9 and 10 are rerun on them.
2. Build the scaffolding for the exploratory pivot (experiments E1–E4) without contaminating any confirmatory artifact.

---

## 0. How to use this plan

- Execute phases **in order**. Every task lists **who runs it**, **exact changes**, and **acceptance checks**. Do not start a task until the previous task's acceptance checks pass.
- Role tags:
  - `[AGENT-CPU]`: you implement and run it locally, with no GPU.
  - `[GPU-OWNER]`: you prepare everything; the human runs it on Kaggle (T4) with the command given.
  - `[HUMAN]`: the human must do it personally. **You must stop and hand off.**
- After each phase, commit with the message prefix given in that phase. Never squash history; the audit trail matters.
- If anything in "Stop conditions" (Section 9) happens, stop and report to the owner. Do not improvise around it.

---

## 1. Non-negotiable ground rules

1. **Pre-registration integrity.**
   - Do not change any value in `src/config.py` that already exists: seeds, thresholds, sample sizes, prompts, caliper, `l*`, `i*`.
   - Do not rerun Stages 0, 1, 3, 4 or 5. Addendum B states the corrected H4/H5 are *corrected post-hoc* results, not confirmatory. Preserve that framing in every output.
2. **Human labels are human-only.**
   - Never fill `human_label` columns yourself, and never use any LLM (including the local judge) to pre-fill or "suggest" them.
   - Never open `results/judge_validation_key.json` or `results/grader_validation_key.json` while preparing sheets for the human. They are the blinding keys.
3. **Dev-only revision rule (Addendum B).**
   - Judge prompt or grader rule revisions may use **only** the development split, with **at most three** revisions.
   - The held-out split is scored **once** per revision chain.
   - If the gate still fails, keep the best version, record the failure, and mark generation-rate claims provisional. Do not iterate on the held-out set.
4. **Fresh PopQA is off-limits.**
   - `data/popqa_fresh.jsonl` (Tier 2 confirmatory sample) must not be created, read, plotted or used by any Tier 1 or exploratory code.
   - Add the guard in Task F0.
5. **Fail closed.** Existing gates (`require_judge_validation`, `require_low_margin_review`, offline cache-miss errors, the Stage 10 α=0 label-equality check) stay in force. Do not weaken, bypass or catch-and-ignore them.
6. **Every implementation choice that departs from `documentation.md` gets a `DEV-xxx` entry** in `DEVIATIONS.md` (Section 8 lists them).
   - IDs `DEV-016/017` and `DEV-019`–`DEV-022` are unused. Do **not** reuse them for unrelated changes.
   - Start new entries at **`DEV-032`**.
7. **Archives are immutable.**
   - Never modify anything under `results/archive_v4/` or `results/archive_v42/`.
   - Run `python scripts/verify_archive_v42.py` and `pytest tests/test_archive_v4.py` before and after your changes.

---

## 2. Verified current state (do not re-derive; re-check quickly in A0)

| Item | State at `32de864` | Meaning |
|---|---|---|
| `data/popqa_sample.jsonl` | 1,000 rows; all have `input_ids_sha256`, `tokenization_contract_version`, `_generation_batch_size == 1`, and `_v4_*` backups | Canonical batch-size-1 regeneration **already ran** on GPU |
| Same file: `baseline_partition`, `grade`, `classifier`, `final_label` | `None` for all 1,000 rows; `label_source == "canonical_v1_batch1_unjudged"` | Pipeline stopped **before** the judge pass |
| Responses changed vs v4 | **321 / 1,000** changed (179 were v4 "wrong", 97 v4 "correct") | Tier 1 will materially change H4/H5 and the matched set |
| v4 matched-wrong items whose new response is a regex abstention | ≈45 of 147 (regex estimate only) | `verify_tier1_popqa_regeneration.py` (requires ≥20) will likely pass; the judge decides |
| New "answer then Unknown" outputs | e.g. `'Paul\n\nUnknown'` | Hard cases for the judge; must be surfaced (Task A6) |
| `results/judge_model.json` | Qwen2.5-7B-Instruct pinned at `a09a3545…` | Judge pinned (DEV-018) |
| `data/judge_cache.jsonl` | 0 lines | Judge pass never ran |
| `results/judge_validation*.json`, `results/grader_validation*.json` | absent | No human validation yet |
| `results/stage2_pilot.json` | still holds **v4** counts (196/253/517/34, 147 pairs); no `label_source` | Inconsistent with rows until rematch |
| `results/stage10_generations.jsonl` | absent | Stage 10 must regenerate (generation-only) |
| `results/archive_v42/` | snapshot of the pre-fix v4.1 state | Leave untouched |

---

## 3. Execution overview

```
Phase A [AGENT-CPU]   fix blocking bugs + guards + tests                     (commit "tier1-A")
Phase B [GPU-OWNER]   judge pass over PopQA + Stage 6 generations            (Kaggle)
Phase C [HUMAN]       judge validation labels → import → gate (+ low-margin)
Phase D [AGENT-CPU]   Stage 6 rescore → Stage 2 regrade+match →
        [HUMAN]       grader validation labels → import → gate →
        [AGENT-CPU]   Stage 2 rematch (gated) → Tier 1 verifier               (commit "tier1-D")
Phase E [GPU-OWNER]   Stage 8 (FORCE) + Stage 10 generation-only → judge pass on Stage 10
        [HUMAN]       Stage 10 supplementary judge validation
        [AGENT-CPU]   Stage 10 analyze-only → Stage 9 report → archive_v43   (commit "tier1-E")
Phase F [AGENT-CPU + GPU-OWNER]  exploratory pivot scaffolding E1–E4         (commit "pivot-F")
```

Phase D cannot start before C passes. E needs D, because Stage 8 and Stage 10 consume matched pairs that only exist after the gated rematch. F needs E.

---

## 4. Phase A: code fixes, guards and tests `[AGENT-CPU]`

### A0. Baseline checks

1. Create a virtualenv. `torch` is needed by many tests; a CPU wheel is fine:
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
2. Run the CPU suite and **save the output** to `results/tier1_preflight_tests.txt`:
   ```bash
   PYTHONPATH=. .venv/bin/python -m pytest tests/ -m "not integration and not slow" -q
   PYTHONPATH=. .venv/bin/python scripts/verify_archive_v42.py
   ```
3. Re-verify every row of Section 2 with a short script, and print the counts into the commit message.

**Acceptance:** the archive verifies. Note any pre-existing test failures; do not fix unrelated failures in this phase.

### A1. Crash: `export_judge_labeling_sheet.py` assumes `classifier` is a dict

- **File:** `scripts/export_judge_labeling_sheet.py`, inside `pool_rows()`, at the PopQA loop (currently line 36).
- **Current:**
  ```python
  rec=r.get("classifier",{})
  ```
- **Replace with:**
  ```python
  rec=r.get("classifier") or {}
  ```
- **Why:** canonical regeneration sets `classifier: null`. `dict.get` returns `None` for a present-but-null key, so `rec.get(...)` raises `AttributeError`.

### A2. Crash: B-control join uses a non-existent key

- **File:** `scripts/export_judge_labeling_sheet.py`, `pool_rows()` (currently line 16).
- **Current:**
  ```python
  bctrl={r["item_id"]:r for r in read_jsonl(CFG.data_dir/"b_ctrl.jsonl")}
  ```
- **Replace with** the same convention as `stage6_rescore_generations.py`:
  ```python
  bctrl={r.get("b_ctrl_id", r["pair_id"]):r for r in read_jsonl(CFG.data_dir/"b_ctrl.jsonl")}
  ```
- **Why:** `data/b_ctrl.jsonl` rows have `b_ctrl_id` (e.g. `bctrl_001`), not `item_id`. Stage 6 B-control rows carry `item_id == b_ctrl_id`, so the existing lookup `bctrl.get(r.get("item_id"))` then works.

### A3. Export must be offline by default and source-selectable

**File:** `scripts/export_judge_labeling_sheet.py`, `main()` and `pool_rows()`.

1. Add CLI flags:
   - `--allow-online-judge` (default off). Without it, call `classify_texts(..., offline=True)` so a cache miss raises instead of silently loading the judge on a CPU machine.
   - `--sources` (comma list from `{popqa, stage6, stage10}`, default `popqa,stage6`). `pool_rows(sources)` includes only the selected sources.
   - `--prefix` (default `judge_validation`). Output files become `{prefix}_test.csv`, `{prefix}_dev.csv`, `{prefix}_key.json` and `{prefix}_manifest.json`.
   - `--n-test` (default 200) and `--n-dev` (default 100). Keep the per-stratum floor of 4 for the main sheet.
2. **Why `stage10` is excluded by default:**
   - `results/stage10_popqa_steering.json` currently holds **stale v4** Stage 10 items.
   - New Stage 10 generations can only exist after the gated rematch (Phase D), so they cannot be in the main validation sheet. See A5 and DEV-032.
3. Record `sources` in the manifest.
4. **File:** `scripts/import_judge_labels.py`.
   - Add a `--prefix` flag so key and manifest paths derive from it (`{prefix}_key.json`, `{prefix}_manifest.json`).
   - Default the `--output` to `results/{prefix}.json`.
   - With the default prefix the current behaviour is unchanged.

### A4. Judge quantization consistency guard

**Problem:**
- The local judge auto-selects `fp16` when total CUDA memory is ≥ 28 GB (two T4s) and `nf4` otherwise.
- `scripts/kaggle_run.sh` exports `CUDA_VISIBLE_DEVICES=0`, so runs through it use **nf4**, but a direct run on 2×T4 would use fp16.
- The cache key does not include quantization, so labels from different precisions could mix silently.

**Changes:**
1. In `src/abstention_classifier.py::classify_texts`, before scoring `pending` items with `backend == "local"`:
   - If the loaded cache contains any record with `judge_quantization` not equal to the quantization about to be used, raise `RuntimeError`.
   - The message should tell the user to set `CFG.judge_quantization` explicitly to the cached value.
   - Check against the resolved value: call `_load_local_judge()` first and use its returned `quant`.
2. In `scripts/import_judge_labels.py`, reject the sheet if `manifest["judge_quantization"]` differs from the set of `judge_quantization` values in `data/judge_cache.jsonl`, or if that set has more than one value.
3. In `scripts/judge_pass.py`, add to the summary:
   - `quantizations_in_cache`: a sorted list of distinct values;
   - `cuda_visible_devices`: from the environment.
4. **Policy:** all Tier 1 judging runs through `kaggle_run.sh` (single GPU → `nf4`). Record this in DEV-033.

### A5. Stage 10 supplementary judge validation (two-sheet design)

**Problem:** Addendum B wants one stratified judge-validation sheet across conditions, including α = ±2. But new Stage 10 generations depend on matched pairs, which depend on a passed judge validation. That's circular.

**Design (DEV-032):**
- The **main** sheet covers PopQA baseline + Stage 6. It gates Stage 2, Stage 6 and Stage 8.
- A **supplementary** sheet covers new Stage 10 generations only. It gates the Stage 10 analysis.

**Changes:**
1. Export the supplementary sheet with:
   ```
   --sources stage10 --prefix judge_validation_stage10 --n-test 100 --n-dev 50
   ```
   - Keep the floor of 4 per stratum.
   - Oversample α = ±2 and regex/judge disagreements in dev, as the existing ranking already does.
   - `pool_rows` must read Stage 10 items from `results/stage10_generations.jsonl` (the new generations), **not** from the stale `stage10_popqa_steering.json`. Join `question` via `popqa_id`.
2. Gate thresholds are identical: κ ≥ 0.90 and ABSTAIN recall ≥ 0.95.
3. In the Stage 10 **analyze** path (`src/popqa_steering.py`, wherever `require_judge_validation()` is called for analysis), add a second call:
   ```python
   require_judge_validation(CFG.results_dir / "judge_validation_stage10.json")
   ```
   - The `--generation-only` path must **not** require it.

### A6. Surface "answer-then-Unknown" responses (diagnostic only)

- **File:** `scripts/judge_pass.py`.
- Add `n_answer_then_unknown` to the summary: the count of unique responses matching `^(?!unknown\b).+\n\s*\n\s*unknown\s*[.!]?\s*$` (case-insensitive, DOTALL).
- Also add their judge-label distribution.
- **Do not change grading.** Addendum B fixes `DEGENERATE → wrong` for PopQA; that rule stays.
- These cases feed the sensitivity analysis in F2. Also include these items in the low-margin review export when their margin is < 1 nat (no change needed; just confirm).

### A7. Tier 1 verifier hygiene

- **File:** `scripts/verify_tier1_popqa_regeneration.py`.
- Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` so it runs without `PYTHONPATH`.
- Print the full v4→v4.2 transition table and the "answer-then-Unknown" count.
- **Do not change the `moved < 20` stop rule.** It is a registered sanity check. If it fires, it is a Stop condition (Section 9).

### A8. Runner support

**File:** `scripts/kaggle_run.sh`.
1. Add a `stage10gen` alias that runs Stage 10 with `--generation-only` (equivalent to `STAGE10_GENERATION_ONLY=1`).
2. Leave the existing `judge`, `rescore6` and env-var modes as they are.
3. Document in `notebooks/kaggle_runner.md` that **Stage 8 must be run with `FORCE=1`**. Otherwise the old v4 `stage8_popqa_core.json` may pass `verify_stage8_outputs` and the stage is skipped.

### A9. Tests

Create `tests/test_tier1_fixes.py`, CPU-only, with no model loading. Cover:

1. **A1:** `pool_rows` on a temp dir whose PopQA row has `"classifier": null` returns rows without raising (monkeypatch `CFG` paths).
2. **A2:** a B-control Stage 6 row with `item_id="bctrl_001"` joins to the prompt of the `b_ctrl.jsonl` row with `b_ctrl_id="bctrl_001"`.
3. **A3:** the default `--sources` excludes `stage10`. With `offline=True` and an empty cache, export raises a cache-miss error rather than loading a model.
4. **A4:** a cache containing `judge_quantization="fp16"` plus a pending local scoring under `"nf4"` raises. Monkeypatch `_load_local_judge` to return `(None, None, "nf4")` and `_score_local_batch` to a stub.
5. **A6:** the regex matches `"Paul\n\nUnknown"` and does not match `"Unknown"` or `"Paul Unknown Smith"`.

**Acceptance for Phase A:**
- All new tests pass.
- The pre-existing suite is no worse than A0.
- The archive still verifies.
- DEV-032 and DEV-033 are drafted in `DEVIATIONS.md`.
- Commit `tier1-A: unblock judge validation export; quantization guard; stage10 supplementary validation`.

---

## 5. Phase B: judge pass `[GPU-OWNER]`

**Kaggle setup:** T4 ×1, Internet on, secrets `GITHUB_TOKEN` and `HF_TOKEN`. Then:

```bash
bash scripts/kaggle_run.sh judge
```

This scores the deduplicated `(question, response)` pairs from `data/popqa_sample.jsonl` and `results/stage6_generation_samples.jsonl`. `stage10_generations.jsonl` does not exist yet and is skipped automatically. The script appends to `data/judge_cache.jsonl` and writes `results/judge_pass_summary.json`. The runner commits and pushes.

**Acceptance (agent checks after pull):**
- `judge_pass_summary.json` has:
  - `n_unique_scored == n_unique_available`;
  - `quantizations_in_cache == ["nf4"]`;
  - `revision` equal to the pinned SHA;
  - `prompt_version == "v1"`.
- No condition exceeds 1% `UNPARSEABLE`. The local backend cannot emit UNPARSEABLE, but assert it anyway.
- Re-running `classify_texts(..., offline=True)` over the same inputs on CPU produces **zero** cache misses.
- Report the following to the owner:
  - `n_answer_then_unknown`;
  - per-condition low-margin counts (margin < 1 nat).

---

## 6. Phase C: judge validation `[HUMAN]`, with agent prep and import

### C1. Prepare sheets `[AGENT-CPU]`

```bash
PYTHONPATH=. python scripts/export_judge_labeling_sheet.py            # defaults: popqa,stage6; offline
```

- **Outputs:**
  - `results/judge_validation_test.csv` (200 rows) and `results/judge_validation_dev.csv` (100 rows);
  - `results/judge_validation_key.json` and `results/judge_validation_manifest.json`.
- Commit the CSVs and manifest **but not the key**. Add `results/*_key.json` to `.gitignore` until labels are imported, then commit the key afterwards for auditability. Record this in DEV-034.

### C2. Hand-off message to the human (copy verbatim)

> Please label `results/judge_validation_test.csv` **first**, then `results/judge_validation_dev.csv`.
>
> For each row, fill `human_label` with exactly one of `ANSWER`, `ABSTAIN`, `DEGENERATE`:
> - **ANSWER** means the reply commits to a specific answer, even if it is wrong or hedged.
> - **ABSTAIN** means it declines, or says the answer is unknown, unavailable or not provided, in any wording.
> - **DEGENERATE** means off-topic, meta, nonsensical, repeated or empty.
>
> Judge the reply only. Don't look up whether it is correct.
>
> For replies like "Paul\n\nUnknown" (an answer followed by "Unknown"), use your own judgment and explain it in `notes`.
>
> Do not open any `*_key.json` file. Do not use any AI tool to label. Commit the filled CSVs when done.

### C3. Import and gate `[AGENT-CPU]`

```bash
PYTHONPATH=. python scripts/import_judge_labels.py --csv results/judge_validation_test.csv
```

**Branches:**
- **`gate_passed == true` and `low_margin_gate_passed` is `true` or `null`:** proceed to Phase D.
- **`gate_passed == true` and `low_margin_gate_passed == false`:**
  1. Run `PYTHONPATH=. python scripts/export_judge_low_margin_review.py`.
  2. Hand `results/judge_low_margin_review.csv` to the human, with the same labeling instructions.
  3. Then run `PYTHONPATH=. python scripts/import_judge_overrides.py`.
  4. Downstream `require_low_margin_review` will then pass.
- **`gate_passed == false`:** follow the dev-only revision procedure.
  1. Score the **dev** CSV with `import_judge_labels.py --csv results/judge_validation_dev.csv --output results/judge_validation_dev.json`, and inspect its confusion matrix.
  2. Propose **one** prompt revision as `JUDGE_PROMPT_V2`:
     - bump `JUDGE_PROMPT_VERSION` to `"v2"`;
     - recompute `JUDGE_PROMPT_SHA256`;
     - keep `JUDGE_PROMPT_V1` in the file for the record.
  3. Write it into a new DEV entry **before** rerunning.
  4. The owner reruns Phase B; the cache key changes automatically with the version.
  5. Re-import the **same** held-out test labels with the new judge outputs. Do **not** re-export new sheets; the held-out items are frozen. You need a small helper `scripts/rescore_judge_validation_sheet.py` that rebuilds the key's `final_label` and `judge_margin` from the new cache for the frozen UIDs.
  6. Maximum three revisions. If it still fails:
     - keep the best version;
     - set `"generation_rate_claims": "provisional"` in `results/judge_validation.json`;
     - **stop and report.** Downstream stages require `gate_passed == true`, so the owner must decide.
- **Never** change the thresholds (κ ≥ 0.90, ABSTAIN recall ≥ 0.95).

---

## 7. Phases D and E: regrade, grader validation, rematch, reruns

### D1. Stage 6 judged rescore `[AGENT-CPU]`

```bash
PYTHONPATH=. python scripts/stage6_rescore_generations.py --offline
```

- **Acceptance:** zero cache misses, and Table 4 (`tables/table4_generation_audit.*`) is rewritten with judged rates alongside the retained v4 regex columns.
- **Record** the H3 B-ctrl judged abstention rate. The criterion is ≤ 0.15. Report whether it changed versus the regex rate of 0.083. **Do not reinterpret H3 if it flips; just report.**

### D2. Stage 2 regrade and match `[AGENT-CPU]`

```bash
PYTHONPATH=. python scripts/stage2_pilot_gate.py --regrade-saved-popqa-only --offline
```

This grades every row via `grade_popqa` using the judged record, runs log10-popularity caliper matching, and updates `results/stage2_pilot.json`:
- the `v4_*` copies are preserved;
- `label_source = "canonical_v1_batch1_judged"`.

**Acceptance and branch:**
- Print the counts `{correct, wrong, abstain, ambiguous}` and the matched pair count.
- **If matched pairs < 100** (`CFG.n_popqa_matched_min`), the function raises "Branch Rule 5 requires top-up generation". **Stop.**
  - Do **not** widen the caliper or loosen grading.
  - Report to the owner. Top-up is a GPU job and a protocol decision.
- Otherwise continue. **Note:** a drop from 147 pairs is expected, because about 45 former wrong items now abstain.

### D3. Grader validation `[HUMAN]`, with agent prep and import

1. Run `PYTHONPATH=. python scripts/export_grader_labeling_sheet.py`. It produces 60 dev and 140 held-out rows, including up to 40 ambiguous items.
2. Hand-off message:

   > Label `results/grader_validation_test.csv`, then `results/grader_validation_dev.csv`.
   >
   > `human_label` must be one of `CORRECT`, `INCORRECT`, `ABSTAIN`, `UNGRADEABLE`.
   > - Use `possible_answers` as the reference.
   > - Mark `CORRECT` only if the response clearly states an accepted answer or an obvious equivalent.
   > - Mark `UNGRADEABLE` if you genuinely cannot decide.
   >
   > No AI tools. Do not open `*_key.json`.

3. Import:
   ```bash
   PYTHONPATH=. python scripts/import_grader_labels.py --csv results/grader_validation_test.csv
   ```
   - The gate is precision(correct) ≥ 0.95 and recall(correct) ≥ 0.90.
4. **If the gate fails:** follow the same dev-only rule.
   - Revise matching rules in `src/pilot_check.py` (`normalize_popqa_text`, `_GENERIC_ALIAS_TAILS`, the ambiguity logic) using dev labels only, with at most three revisions.
   - Each revision gets a DEV entry.
   - Re-run D2, then re-score the **same frozen** held-out UIDs once.
   - If it still fails: stop and report.

### D4. Gated rematch and verification `[AGENT-CPU]`

```bash
PYTHONPATH=. python scripts/stage2_pilot_gate.py --rematch-only --offline
PYTHONPATH=. python scripts/verify_tier1_popqa_regeneration.py
PYTHONPATH=. python -c "from scripts.stage2_pilot_gate import verify_stage2_outputs as v; from src.config import CFG; v(CFG.stage2_out_path, CFG.data_dir/'popqa_sample.jsonl')"
```

**Acceptance:**
- All three commands succeed.
- The verifier prints the transition table and "Previously matched wrong items now abstaining: N" with N ≥ 20. If N < 20, that's a Stop condition.
- **Ambiguous sensitivity (Addendum B):** write `results/tier1_ambiguous_sensitivity.json`. Re-run the matched AUROC computation in Phase E with ambiguous items resolved by available human labels. This is reported only; it never replaces the primary analysis.

Commit `tier1-D: judged PopQA labels, validated grader, rematched pairs`.

### E1. Stage 8 rerun `[GPU-OWNER]`

```bash
FORCE=1 bash scripts/kaggle_run.sh 8
```

- `load_popqa_groups` now enforces `label_source == canonical_v1_batch1_judged` and Stage 2 count consistency.
- `_verify_popqa_input_identity` re-checks the input hashes.
- **Acceptance:**
  - `results/stage8_popqa_core.json` has `label_source == "canonical_v1_batch1_judged"` and `verified_input_hash_count` equal to the number of feature rows;
  - `v4_stage8` is embedded;
  - the Stage 8 unit and integration tests pass.

### E2. Stage 10 generation `[GPU-OWNER]`

```bash
bash scripts/kaggle_run.sh stage10gen          # alias from A8; or STAGE10_GENERATION_ONLY=1 bash scripts/kaggle_run.sh 10
bash scripts/kaggle_run.sh judge               # now also scores results/stage10_generations.jsonl
```

- **Acceptance:** `results/stage10_generations.jsonl` exists, there are zero cache misses for its items, and `quantizations_in_cache == ["nf4"]`.

### E3. Stage 10 supplementary validation `[HUMAN]`, with agent prep and import

```bash
PYTHONPATH=. python scripts/export_judge_labeling_sheet.py --sources stage10 --prefix judge_validation_stage10 --n-test 100 --n-dev 50
# human labels (same instructions as C2)
PYTHONPATH=. python scripts/import_judge_labels.py --prefix judge_validation_stage10 --csv results/judge_validation_stage10_test.csv
```

- Same gate and same dev-only revision rule.
- If it fails after revisions: Stage 10 stays unanalysed and the README marks Stage 10 "not re-analysed (judge gate failed)". **Stage 10 is exploratory; this does not block E5.**

### E4. Stage 10 analysis `[AGENT-CPU]`

```bash
STAGE10_ANALYZE_ONLY=1 bash scripts/kaggle_run.sh 10     # runs --analyze-only --offline (CPU is fine)
```

- The registered α=0 equality check must pass: the regenerated α=0 grade equals the stored Stage 2/8 label for every matched item.
- **If it fails:** stop. The analysis records both responses and input hashes; report them.

### E5. Report and archive `[AGENT-CPU]`

1. Run Stage 9: `bash scripts/kaggle_run.sh 9`. It is CPU-only, as it regenerates tables, figures and the README auto block.
2. Update the **hand-written** part of `README.md` (outside the `BEGIN/END_STAGE9_RESULTS` markers):
   - **Summary of findings:** the corrected H4/H5 numbers, explicitly labelled *corrected post-hoc (Addendum B), not confirmatory*, alongside the v4 numbers.
   - **Open measurement issues:** delete the two Tier 1 bullets and replace them with the judge and grader validation results (κ, ABSTAIN recall, grader precision/recall), the quantization (`nf4`), and the ambiguous-sensitivity result.
3. Create `scripts/build_archive_manifest.py`. It takes a generic `--name` and `--files` list and writes `results/<name>/MANIFEST.json` with `sha256`, `size_bytes` and `source_git_commit`, in the same format as `archive_v4`.
4. Snapshot the Tier 1 final state to `results/archive_v43/`:
   - files: `data/popqa_sample.jsonl`, `data/judge_cache.jsonl`, `results/stage{2,6,8,9,10}*.json*`, `results/judge_validation*.json`, `results/grader_validation.json`, all `tables/`, all `figures/`.
5. Add `scripts/verify_archive_v43.py`, mirroring `verify_archive_v42.py`.

Commit `tier1-E: Stage 8/10 on judged labels; report; archive_v43`.

**Definition of done for Tier 1:**
- Every gate file exists with `gate_passed: true`, or the documented provisional status for Stage 10 only.
- README Tier 1 bullets are resolved.
- `archive_v43` verifies.
- The full CPU test suite passes.

---

## 8. Phase F: exploratory pivot scaffolding (E1–E4)

All of this is **exploratory, post-registration**. It never writes to `results/stage*.json`, `tables/`, `figures/` (the confirmatory ones) or `data/popqa_sample.jsonl`.

- New code goes in `src/exploratory/` and `scripts/exploratory/`.
- Outputs go in `results/exploratory/` and `figures/exploratory/`.

### F0. Guardrails

1. `src/exploratory/__init__.py` defines `assert_not_fresh(path)`. It raises if `"popqa_fresh"` appears in the path. Every exploratory loader calls it.
2. Every exploratory JSON output includes `"status": "exploratory_post_registration"` and the git commit.
3. Add a CPU test asserting that no file under `src/exploratory/` or `scripts/exploratory/` references `popqa_fresh`, apart from the guard itself.
4. **Data source:** the judged v4.2 `data/popqa_sample.jsonl`, all graded rows (not only matched pairs).

### F1. Multi-position, all-layer feature cache `[GPU-OWNER runs; AGENT-CPU writes]`

- **Script:** `scripts/exploratory/cache_popqa_features.py`.
- **Reuse:**
  - `load_runtime_contract` and `load_stage_model` for the model;
  - `build_chat_batches` for left-padded batches with mask-derived position IDs;
  - `capture_residuals_and_logits` for residuals and logits.
- Run one pass per suffix position `pos ∈ {-5, -4, -3, -2, -1}`. Position −1 is the residual that produces the first answer token.
- **Inputs:** every PopQA row with a final grade, the 192 train, 96 val and 96 test pairs (ans and unans), and the 48 B-ctrl prompts.
- **Verify input identity** with `_verify_popqa_input_identity` for the PopQA rows.
- **Outputs** (fp16, memory-mapped `.npy`, ≈0.9 GB for PopQA):
  - `features/popqa_resid.npy`, shape `[N, 5, 28, 3072]`;
  - `features/popqa_meta.parquet`: `popqa_id`, grade, `matched_pair_id`, `prop` (relation), `log_s_pop`, `maxp`, `entropy`, `log_odds`, `input_ids_sha256`, `final_label`, `judge_margin`, and an `answer_then_unknown` flag;
  - the same layout for the pair and B-ctrl sets.
- **Storage:** push the tensors with `kaggle_sync.py` to the HF tensor repo, not git. Commit only `results/exploratory/features_manifest.json` with hashes, shapes and dtypes.
- **Acceptance:**
  - finite values everywhere;
  - at layer 11, pos −4, the projections on the cached test sets match `results/stage6_test_h1_h2_h3.json` projections within the Stage 8 tolerance (≤ 0.05 × scale).

### F2. E1: supervised upper bound `[AGENT-CPU]`

- **Script:** `scripts/exploratory/e1_supervised_probe.py`.
- **Task:** wrong (y = 1) vs correct (y = 0). Exclude `ambiguous`.
- **Primary population:** the rematched pairs.
- **Secondary population:** all graded correct/wrong rows, with relation-grouped folds.
- **Folds:** `StratifiedGroupKFold`, grouped by `matched_pair_id` (primary) or by `prop` (secondary), with 5 folds × 20 repeats. Reuse `src/incremental_value.cv_oof_auroc`.
- **Per (layer, position):** standardized L2 logistic regression. Choose `C ∈ {0.01, 0.1, 1}` by **inner** CV only, inside each outer training fold.
- **Report:** out-of-fold AUROC with bootstrap CI over pairs.
- **Baselines on identical folds:**
  - `maxp`, `entropy`, `log_odds`;
  - **popularity only** (`log_s_pop`): essential, because matching controls it only coarsely;
  - relation one-hot;
  - the frozen-direction projection `proj` (r̂ at each layer).
- **Incremental value:** build out-of-fold probe scores (nested, so the probe score is never trained on its own test fold). Then call `incremental_value_test` with `FULL_FEATURES = BASE_FEATURES + ["probe_oof"]`.
  - Add a parameter for the feature list rather than editing the registered constants.
- **Sensitivity re-runs:**
  1. `answer_then_unknown` items excluded;
  2. DEGENERATE excluded rather than counted as wrong;
  3. ambiguous resolved with human labels where available.
- **Outputs:**
  - `results/exploratory/e1_probe.json`;
  - `figures/exploratory/e1_layer_position_auroc.png`: a heatmap of probe AUROC minus maxp AUROC.

### F3. E2: decomposition `[AGENT-CPU]`

- **Script:** `scripts/exploratory/e2_decompose.py`.
- **Similarity:** cosine(probe weight in original activation units, r̂) per layer, at pos −4 and −1. Use probes refit on all data **only for this geometric analysis**, never for scores.
- **Logit lens of r̂:**
  - Load only `model.norm.weight` and `lm_head.weight` (or the tied embedding) from the pinned HF checkpoint on CPU with `safetensors`, about 0.8 GB.
  - Compute `logits = W_U @ (norm_weight ⊙ r̂)`, approximating RMSNorm on direction (document the approximation).
  - Report the top-20 promoted and demoted tokens, and the rank of the Unknown token IDs from `contract["unknown_token_ids"]`.
- **Residualized signal:**
  - Within each outer fold, regress `proj` (layer 11, pos −4) on `log_odds` on the training fold.
  - Compute AUROC of the residual on the test fold.
  - Also report the partial Spearman correlation of `proj` with y, controlling for `log_odds` and `maxp`.
- **Group-contrast table per layer:** AUROC for abstain-vs-correct, wrong-vs-correct and abstain-vs-wrong, with CIs.
- **Interpretation rule** (write into the output and fix it **before** running):

  | Residual AUROC CI | Probe vs maxp | Label |
  |---|---|---|
  | includes 0.5 | — | **"abstention-intent readout"** |
  | excludes 0.5 | probe beats maxp | **"knowledge-state signal beyond output confidence"** |
  | anything else | — | **"inconclusive"** |

- **Output:** `results/exploratory/e2_decomposition.json`.

### F4. E3: instruction ablation `[GPU-OWNER + HUMAN-light]`

- **Config:** add a **new** field `exploratory_neutral_system_prompt: str = "You are a helpful assistant. Answer the question concisely."` to `MasterConfig`.
  - Add it at the end; never edit `system_prompt`.
  - It reuses the existing `mmlu_neutral_system_prompt` style.
  - Add DEV-035.
- **Script:** `scripts/exploratory/e3_neutral_prompt.py`.
  1. Generate canonically at batch size 1 for the same questions as `data/popqa_sample.jsonl`, using `canonical_input_ids` with the neutral prompt and `max_new_tokens=16`.
  2. Write `data/exploratory/popqa_neutral.jsonl`.
  3. Judge with the same pinned judge and quantization.
  4. Grade with `grade_popqa`.
- **Judge on a new distribution:** the judge is validated on the abstention-instructed distribution only. Export 50 random neutral-prompt items with `--prefix judge_validation_neutral` for the human, and report agreement **descriptively**. It is not a gate, because this is exploratory.
- **Features:** cache them with F1 under the neutral prompt.
- **Directions:** re-derive r̂_neutral as the difference of means on the train pairs under the neutral prompt (layer 11, pos −4). This is CPU-cheap once the pair features are cached.
- **Evaluation:** run E1 and E2 metrics for both the frozen r̂ and r̂_neutral.
- **Retry note:** with this prompt, re-attempt Qwen2.5-3B-Instruct as a **second model**. It previously failed only because of 100% instructed abstention. Mark it exploratory and log it in DEV-036.

### F5. E4: transfer matrix (interface only in this pass)

- **Module:** `src/exploratory/transfer.py`.
- **API:**
  ```python
  load_source(name) -> (prompts, labels, groups)
  ```
  with sources:
  - `synthetic_pairs` (the existing data);
  - `squad_v2` (HF `rajpurkar/squad_v2`, unanswerable vs answerable; pass the context in the prompt);
  - `popqa_v42` (the judged sample).
- **Function:**
  ```python
  transfer_matrix(train_source, test_source, layer, pos)
  ```
  It returns AUROC, and AUROC minus the logit baseline on the test source.
- Write unit tests with tiny synthetic arrays.
- **Do not** download or run SQuAD in this pass. Leave a `TODO(owner)` noting the dataset licence check and the GPU run.

### F6. Reporting

- `scripts/exploratory/build_report.py` writes `results/exploratory/REPORT.md` covering E1–E3:
  - one table per experiment;
  - every number with its CI;
  - the pre-declared interpretation label from F3.
- Add a hand-written README section **"Exploratory analyses (post-registration)"**, placed below the Stage 9 auto block, linking to the report. Never place exploratory numbers inside the Stage 9 markers.

Commit `pivot-F: exploratory feature cache, E1/E2 analyses, E3 neutral-prompt pipeline, E4 interface`.

### Deviation entries to add (`DEVIATIONS.md`)

| ID | Summary |
|---|---|
| DEV-032 | Judge validation split into main (PopQA + Stage 6) and supplementary Stage 10 sheets, because new Stage 10 generations depend on matched pairs that require a passed judge gate; same thresholds; Stage 10 analysis gated on the supplementary sheet. |
| DEV-033 | All Tier 1 judging runs single-GPU via `kaggle_run.sh` ⇒ `nf4` quantization; a cache/manifest consistency guard prevents mixed precision. |
| DEV-034 | Blinding keys git-ignored until human labels are imported, then committed for audit. |
| DEV-035 | Exploratory neutral system prompt added as a new config field; the registered `system_prompt` is unchanged. |
| DEV-036 | Exploratory second-model attempt (Qwen2.5-3B-Instruct) under the neutral prompt; not part of the registered pipeline. |
| DEV-037+ | One entry per judge-prompt or grader-rule revision, if any (C3/D3), each stating the dev-split evidence. |

---

## 9. Stop conditions (report to owner; do not work around)

| Condition | Where | Why stop |
|---|---|---|
| Archive hash mismatch | A0, E5 | Provenance broken |
| Judge held-out gate fails after 3 dev revisions | C3 | Protocol requires owner decision; downstream is gated |
| Grader held-out gate fails after 3 dev revisions | D3 | Same |
| Matched pairs < 100 after regrade | D2 | Branch Rule 5 top-up is a GPU and protocol decision; never widen the caliper |
| `verify_tier1_popqa_regeneration.py` reports < 20 moved | D4 | Registered sanity check failed |
| Stage 10 α=0 label-equality mismatch | E4 | Registered stop rule |
| Any offline cache miss in a step documented as offline | B–E | Indicates a key or input mismatch; investigate rather than rescoring online |
| Mixed judge quantization detected | any | Labels not comparable |
| Any code path touches `popqa_fresh` | F | Would contaminate Tier 2 confirmatory data |

---

## 10. Command cheat sheet

```bash
# Phase A (CPU)
PYTHONPATH=. .venv/bin/python -m pytest tests/ -m "not integration and not slow" -q
PYTHONPATH=. .venv/bin/python scripts/verify_archive_v42.py

# Phase B (Kaggle)
bash scripts/kaggle_run.sh judge

# Phase C (CPU, then HUMAN, then CPU)
PYTHONPATH=. python scripts/export_judge_labeling_sheet.py
PYTHONPATH=. python scripts/import_judge_labels.py --csv results/judge_validation_test.csv
PYTHONPATH=. python scripts/export_judge_low_margin_review.py && PYTHONPATH=. python scripts/import_judge_overrides.py   # only if low-margin gate fails

# Phase D (CPU, HUMAN for grader labels)
PYTHONPATH=. python scripts/stage6_rescore_generations.py --offline
PYTHONPATH=. python scripts/stage2_pilot_gate.py --regrade-saved-popqa-only --offline
PYTHONPATH=. python scripts/export_grader_labeling_sheet.py
PYTHONPATH=. python scripts/import_grader_labels.py --csv results/grader_validation_test.csv
PYTHONPATH=. python scripts/stage2_pilot_gate.py --rematch-only --offline
PYTHONPATH=. python scripts/verify_tier1_popqa_regeneration.py

# Phase E (Kaggle, then HUMAN, then CPU)
FORCE=1 bash scripts/kaggle_run.sh 8
bash scripts/kaggle_run.sh stage10gen && bash scripts/kaggle_run.sh judge
PYTHONPATH=. python scripts/export_judge_labeling_sheet.py --sources stage10 --prefix judge_validation_stage10 --n-test 100 --n-dev 50
PYTHONPATH=. python scripts/import_judge_labels.py --prefix judge_validation_stage10 --csv results/judge_validation_stage10_test.csv
STAGE10_ANALYZE_ONLY=1 bash scripts/kaggle_run.sh 10
bash scripts/kaggle_run.sh 9
```

## 11. File change summary

| File | Change |
|---|---|
| `scripts/export_judge_labeling_sheet.py` | A1, A2, A3 (offline default, `--sources`, `--prefix`, `--n-test/--n-dev`, Stage 10 from `stage10_generations.jsonl`) |
| `scripts/import_judge_labels.py` | A3 `--prefix`; A4 quantization check |
| `src/abstention_classifier.py` | A4 mixed-quantization guard in `classify_texts` |
| `scripts/judge_pass.py` | A4 summary fields; A6 answer-then-Unknown diagnostic |
| `src/popqa_steering.py` | A5 supplementary-validation gate on the analyze path only |
| `scripts/verify_tier1_popqa_regeneration.py` | A7 sys.path + diagnostics (threshold unchanged) |
| `scripts/kaggle_run.sh`, `notebooks/kaggle_runner.md` | A8 `stage10gen` alias; `FORCE=1` note for Stage 8 |
| `tests/test_tier1_fixes.py` | A9 new |
| `scripts/rescore_judge_validation_sheet.py` | C3 new (only if a dev revision is needed) |
| `scripts/build_archive_manifest.py`, `scripts/verify_archive_v43.py` | E5 new |
| `src/config.py` | F4: append `exploratory_neutral_system_prompt` only |
| `src/exploratory/*`, `scripts/exploratory/*`, `tests/test_exploratory_*.py` | F0–F6 new |
| `DEVIATIONS.md`, `README.md`, `.gitignore` | DEV-032+; README hand-written sections; key files ignored until import |
