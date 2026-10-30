# Execution Timeline and Kaggle Notebook Cells: Tier 1 (E0) and Pivot (E1–E4)

**Audience:** an AI coding agent with full access to `DakshUbhadia/unanswerability_to_hallucination`, plus the human owner (Daksh), who runs Kaggle T4 sessions and fills in the labelling CSVs.

**Snapshot analysed:** commit `a47f09a` (2026-10-30), plus the owner's Kaggle notebook `notebook8fe744be94.ipynb` (cells 0–32).

**What the agent must deliver:**

1. Apply the blocking fixes in **Part B** to the repo, with tests, and push.
2. Write `notebooks/kaggle_cells_tier1_pivot.md`. It must contain **copy-paste-ready Kaggle Python cells**, one block per session (Part D), each with purpose, expected runtime, expected outputs and failure handling.
3. Write `notebooks/local_steps_tier1_pivot.md`. It must contain the exact **local WSL** commands for every CPU and labelling step (Part E).
4. Report back to the owner with the timeline in **Part C**, filled in with any adjustments.

**The agent must not:**
- run anything on GPU;
- fill or pre-fill any `human_label` column;
- open any `*_key.json` file;
- change any pre-registration file;
- change registered thresholds, prompts or seeds.

---

## Part A: Verified current state (do not re-derive; re-check quickly)

### A.1 What has already run on Kaggle (from the notebook)

| Cell | What ran | Outcome |
|---|---|---|
| 0–25 | Stages 0–10 (v4 labels) | Done. Results committed (H1–H4 confirmed, H5 not). Stage 3/4 tensors on HF (`IndivisibleTornado/unanswerability-stage-caches`) |
| 27 | `pin-judge` | Done: Qwen2.5-7B-Instruct @ `a09a3545…` pinned in `results/judge_model.json` |
| 29 | `STAGE2_POPQA_REGENERATE=1` Stage 2 | **Generation succeeded and was pushed** (commit `32de864`: 1,000 rows with `input_ids_sha256`, batch size 1, `_v4_*` backups, `label_source="canonical_v1_batch1_unjudged"`). The trailing full pytest then failed (4 tests; since fixed except one, see B1), so `kaggle_run.sh` exited non-zero |
| 30 | `STAGE2_POPQA_REGRADE=1` | Failed: "Choose only one Stage 2 special mode". Cell 29's exception skipped its `os.environ.pop`, so `REGENERATE` was still set. **Lesson: env flags must be set in `try/finally`** |
| 31 | Regrade, retried | Failed correctly: `judge_validation.json` missing |
| 32 | Judge sheet export | Failed: `KeyError: 'item_id'`. **Fixed** in `a47f09a` (A2 of the Tier 1 plan; covered by `tests/test_tier1_fixes.py`) |

Other notebook lessons:
- A `git push` was rejected because the remote was ahead (cell 6).
- `git commit` failed for lack of an author identity (cells 9 and 18).
- Several checkout paths were used (`/kaggle/working/repo`, `…/unanswerability_to_hallucination`, `…_stage4`, `stage5-run-…`).
- **The new cells must use one path (`/kaggle/working/repo`), a fresh `fetch + reset --hard` at session start, a git identity, pull-rebase-retry before every push, and token-free remote URLs.**

### A.2 Data state now

| Item | State |
|---|---|
| `data/popqa_sample.jsonl` | 1,000 canonical rows; all labels `None`. 321 responses differ from v4 |
| `data/judge_cache.jsonl` | Empty. **The judge pass has never run** |
| `results/judge_validation*.json`, `results/grader_validation*.json` | Absent |
| `results/stage2_pilot.json` | Still holds v4 counts; no `label_source` |
| `results/exploratory/e{1,2,3,4}_preregistration.json` | Committed 10-29, **before** any data use; unchanged since. Good |
| E4 data | `syn.jsonl` (768 rows) and `selfaware.jsonl` (600 rows) prepared. Hashes verify on a Linux checkout. **SQuAD preparation failed** locally (gated Llama tokenizer: the run had no HF token). It must be prepared on Kaggle |
| `results/exploratory/e4_results.json` | A "pending_features" placeholder. Will be overwritten |
| Exploratory feature caches | None exist |

### A.3 Tests (CPU, Python 3.12, torch 2.4.1 CPU, transformers 4.46.3)

- 217 passed, 3 skipped, **1 failed**: `tests/test_tier2_cpu.py::test_archive_v42_manifest_verifies` (see B1).
- The same test passes on the owner's WSL checkout ("verified 55 archived files") only because that working tree still holds the old CRLF bytes.

---

## Part B: Blocking fixes the agent must make before any GPU session

Each fix needs a test (where testable) and a DEV entry if it changes registered-adjacent code. Use the next free DEV IDs. Commit message prefix: `pre-gpu:`.

### B1. Archive v42 verification fails on every Linux checkout (line endings)

- **Diagnosis (verified):** three files inside `results/archive_v42/` were hashed when the working copy had CRLF line endings:
  - `results/archive_v4/MANIFEST.json` (3,994 bytes in the manifest; 3,893 on disk);
  - `tables/table1_main_results.csv`;
  - `tables/table3_popqa_auroc.csv`.

  `.gitattributes` (`* text=auto eol=lf`) stores them with LF. Converting the LF bytes back to CRLF reproduces the recorded sha256 exactly for all three.
- **Consequence:**
  - `kaggle_run.sh` ends with a full `pytest`, so **every Kaggle run exits non-zero even after a successful push**;
  - notebook cells using `check=True` raise;
  - env flags leak (this is what broke cell 30).
- **Fix:** in `scripts/verify_archive_v42.py`, accept a file if its raw bytes **or** its LF→CRLF conversion match the recorded sha256 and size.
  - Record which files matched via EOL normalisation and return that list.
  - Do **not** edit `MANIFEST.json` or any archived file.
  - Apply the same tolerance in `scripts/verify_archive_v43.py`, and make `scripts/build_archive_manifest.py` hash **LF-normalised** bytes for text files from now on.
- **Test:** a temp archive whose manifest holds CRLF hashes and whose files are LF must verify. A genuinely altered file must still fail.

### B2. Exploratory outputs are never committed by `kaggle_run.sh`

- `kaggle_run.sh` only adds `results/*.json`, `data/*.jsonl` and similar top-level paths. `results/exploratory/**`, `data/exploratory/**` and `figures/exploratory/**` are never committed. Kaggle disks are wiped at session end.
- **Fix:** add `scripts/exploratory/commit_outputs.py --paths <globs…> --message <msg>`. It:
  - `git add -f` the matches;
  - **refuses** any path matching `*_key.json`, `*.npy`, `*.npz` or `*checkpoint*`;
  - commits if anything is staged;
  - runs `git pull --rebase origin main` and then `git push`, retrying 3 times.
- **Test:** the refusal list.

### B3. E3 and E4 feature tensors are never uploaded, so GPU work would be lost

- `e1_extract_features.py` has `--push-hf`. `e3_generate_and_capture.py` and `e4_extract_features.py` do not, and `e4_transfer.load_source` / the E3 loaders cannot download from HF.
- **Fix:**
  - Add `--push-hf` to both scripts. Upload with `huggingface_hub.upload_folder` to `$HF_TENSOR_REPO_ID` (repo type **dataset**), under:
    - `exploratory/e3_features/{slug}_neut/`;
    - `exploratory/e4_features/{condition}/{source}/`.
    - Exclude checkpoints.
  - Add an HF-download fallback to every exploratory loader that reads `.npy` (E3 loaders, `e4_transfer.load_source`), mirroring `e1_analyze.py` lines 115–124. Verify the sha256 against the manifest after download.
  - Note that `e1_extract_features.py` calls `upload_folder` **without `repo_type="dataset"`**. Fix it (the Stage 3/4 caches live in a dataset repo).
- **Test:** mock `hf_hub_download` and `upload_folder`; assert the paths and the sha check.

### B4. Blinding keys: keep them off GitHub and off Kaggle

- `results/*_key.json` is git-ignored, but `results/exploratory/*_key.json` (E3 sheets) is **not**. Add `results/exploratory/**/*_key.json` to `.gitignore`.
- **Workflow rule:** export and import of every labelling sheet run **locally in WSL**, never on Kaggle. A key created on Kaggle would vanish with the session (it is git-ignored), and the import would then fail. All exports work offline from the committed `data/judge_cache.jsonl`.
- After a successful import, the key may be committed for auditability with `git add -f`, as the Tier 1 plan's DEV entry says.

### B5. E3 pilot loads the 7B judge while the subject model is still on the GPU

- In `_pilot()`, `classify_texts` loads Qwen2.5-7B (nf4, about 5.5 GB) while Llama-3B fp16 (about 6.5 GB) is resident. That is tight on a 15 GB T4, and an **out-of-memory error is certain** if Qwen runs in the fp32 fallback (about 12 GB).
- **Fix:**
  1. After generating the pilot responses, move the subject model to CPU and call `torch.cuda.empty_cache()`.
  2. Judge.
  3. Release the judge with a new `src/abstention_classifier.release_local_judge()` that clears the module-level cache and empties the CUDA cache.
  4. Move the subject model back.
- Also make the **pilot a separate process invocation** from the other steps in the notebook cells.

### B6. One canonical pipeline: retire the duplicate Phase F scripts

- Two parallel implementations exist. The canonical ones, per the E1–E4 plans, are:
  - `e1_extract_features.py`, `e1_analyze.py`;
  - `e2_analyze.py`, `e2_causal_vocab.py`;
  - `e3_preflight.py`, `e3_generate_and_capture.py`, `e3_export_validation.py`, `e3_label_and_match.py`, and the new `e3_analyze.py` (B7);
  - `e4_prepare_data.py`, `e4_extract_features.py`, `e4_analyze.py`.
- The legacy Phase F ones are:
  - `cache_popqa_features.py`, `e1_supervised_probe.py`, `e2_decompose.py`, `e3_neutral_prompt.py`;
  - `export_neutral_validation.py`, `score_neutral_validation.py`;
  - `src/exploratory/transfer.py`;
  - the Phase F parts of `build_report.py`, and the `kaggle_sync.py --exploratory-*` flags.
- **Fix:**
  - Add a guard at the top of each legacy script: exit with code 2 and the message "superseded by <canonical>; pass --legacy to run". Do not delete code.
  - Rewrite the "Tier 1 refresh and exploratory pivot" section of `notebooks/kaggle_runner.md` to list only canonical commands, pointing to the two new files.
  - If `e2_analyze.py` Part L needs the unembedding loader that lives in `e2_decompose.py`, move that loader into `src/exploratory/e2_analysis.py`. A legacy guard must never break a canonical path.

### B7. Missing E3 analysis runner

- `src/exploratory/e3_analysis.py` only has building blocks (incremental value, independent bootstrap, concordant subset, transitions, verdict). The runner `scripts/exploratory/e3_analyze.py` from the E3 plan (Task 5.7: analyses A–F, optional G; outputs `e3_results.json`, `e3_summary.md`, five figures) does not exist.
- **Implement it**, reusing:
  - `e1_cv` (for analysis D);
  - `e2_analysis` (`crossfit_residual`, `cosine_spaces`, `geometry_nulls`, `fit_dom`);
  - `incremental_value_test` (via column mapping).
- **Timing:** this is needed only after labels (Part C, local step L4), so it **may be done during the owner's labelling window**. It must exist before step L4.

### B8. The E2 Part C input vector must survive between sessions

- `e2_causal_vocab.py` needs `--dom-vector results/exploratory/e2_dom_11_suf-4.npy`, produced on CPU by `e2_analyze.py` Part G. But `results/exploratory/**/*.npy` is git-ignored, so the file won't reach Kaggle.
- **Fix:**
  - Make `e2_analyze.py` write it (verify that it does).
  - Add the `.gitignore` exception `!results/exploratory/e2_dom_11_suf-4.npy`. The file is 3,072 float32 values, about 12 KB.
  - Record its sha256 in `e2_results.json`.
  - Make `e2_causal_vocab.py` check that sha256.

### B9. Small correctness items

1. **E1 extraction compares against Stage 8.** `e1_extract_features.py` checks recomputed `maxp`/`entropy`/`log_odds` against `results/stage8_popqa_core.json["matched_items"]`. The current file is the **v4** Stage 8 output.
   - E1 extraction must therefore run **after** `FORCE=1` Stage 8 in the post-label session (Part D, session G3). Keep it gated as it is; just enforce the order in the cells.
   - Confirm that `matched_items` carries `popqa_id`, `maxp`, `entropy` and `log_odds`. If not, fix the parity code.
2. **Optional dev sheets.** `import_judge_labels.py` scores the held-out test CSV only; dev sheets are needed **only if a gate fails**. Update the hand-off text in `notebooks/local_steps_tier1_pivot.md` so the owner labels **test first, and dev only if instructed**.
3. **Judge quantization.** Every GPU cell must export `CUDA_VISIBLE_DEVICES=0` before **any** Python process that may load the judge (`judge_pass.py`, E3 pilot). With two T4s visible, the judge auto-selects fp16. The mixed-quantization guard would then abort, and nf4 is the Tier 1 policy.
4. **`bitsandbytes` on Kaggle.** Kaggle runs Python 3.13 with torch 2.11, and the judge's nf4 path needs `bitsandbytes`. Add a 5-item judge smoke test (`classify_texts` on 5 synthetic items with `offline=False`) at the start of session G1. A failure is then caught in 2 minutes instead of after a long run.

**Acceptance for Part B:**
- the full CPU suite passes on a fresh **Linux** clone (`git clone` into a new directory, then `pytest -m "not integration and not slow"`);
- the new tests pass;
- DEV entries are added;
- everything is pushed to `main`.

---

## Part C: Timeline (what runs when, and who does it)

Legend: **[GPU]** = Kaggle session (owner pastes cells); **[LOCAL]** = owner's WSL machine, CPU; **[HUMAN]** = owner labels CSVs; **[AGENT]** = coding agent.

| # | When (suggested) | Who | What | Needs | Time |
|---|---|---|---|---|---|
| 0 | Day 0 (Oct 30–31) | AGENT | Part B fixes B1–B6 and B8–B9; write both notebook/step files | none | — |
| 1 | Day 1 | GPU **G1** | E3 preflight (template-date check) → judge smoke test → **E0 judge pass** → E4 SQuAD preparation → E4 feature extraction (registered: syn, squad, selfaware) | Part B pushed | ~1.5–2.5 h |
| 2 | Day 1, right after G1 | LOCAL + **HUMAN** | Pull → export judge sheet → **label `judge_validation_test.csv` (200 rows)** → import → gate. If the low-margin gate fails: label the low-margin review (40–200 rows) | G1 pushed | 45–60 min (+10–50) |
| 3 | Day 1–2, in parallel with step 2 | GPU **G2** | E3 Llama: pilot → popqa → pairs → features. E3 Qwen: pilot → (if included) popqa, pairs, features. Judge pass on the neutral files. *Optional:* E4 neutral-condition extraction | G1 preflight shows no dynamic date | ~2–3 h |
| 4 | Day 2 | LOCAL | Stage 6 judged rescore → Stage 2 regrade + match (**check matched pairs ≥ 100**) → export grader sheet and E3 sheets | Judge gate passed; G2 pushed | ~10 min CPU |
| 5 | Day 2 | **HUMAN** | **Label `grader_validation_test.csv` (140)** and the **E3 judge (100) and grader (60) sheets** | step 4 | 1.5–2 h total |
| 6 | Day 2 | LOCAL | Import grader → gate → **rematch** → `verify_tier1_popqa_regeneration.py` → import E3 sheets. E4 non-PopQA cells (`e4_analyze.py`; PopQA cells show pending) | step 5 | ~30 min CPU |
| 7 | Day 3 | GPU **G3** | `FORCE=1` Stage 8 → **E1 feature extraction** (`--push-hf`) → `stage10gen` → judge pass (now includes Stage 10 generations) | Rematch pushed | ~1–1.5 h |
| 8 | Day 3 | LOCAL + **HUMAN** | Export the Stage 10 supplementary sheet → **label (100)** → import → Stage 10 analyze-only → Stage 9 → `archive_v43` → **E1 analyze** → **E2 analyze R, G, L, A** (writes the dom vector) | G3 pushed | 25–35 min human; 1–2 h CPU |
| 9 | Day 4 | GPU **G4** | **E2 Part C** (`e2_causal_vocab.py`, 99 random controls) | dom vector committed (B8) | ~30–45 min |
| 10 | Day 4 | LOCAL | E2 Part C merge → `e3_label_and_match.py` → `e3_analyze.py` (B7) → `e4_analyze.py --condition both` → reports, README | G4 pushed | ~1–2 h CPU |

**What runs on the GPU before any CSV is filled:**
- G1 (preflight, judge pass, all E4 non-PopQA extraction);
- G2 (all E3 generation, capture and judging).

These need **no** human labels.

**After labelling, the remaining GPU work is unavoidably label-dependent:**
- Stage 8 and Stage 10 use the re-matched pairs;
- E1 extraction checks parity against the re-run Stage 8;
- E2 Part C's pre-registered prompt set and PopQA direction need labels.

**Totals:**
- human labelling: about 2.5–3.5 hours across three sittings (steps 2, 5 and 8);
- GPU: about 5–8 hours across four sessions, well within Kaggle's weekly quota.

### Contingency branches (stop and report; never work around)

| Trigger | Where | Action |
|---|---|---|
| `e3_preflight.py` exits 3 (dynamic chat-template date) | G1 | **Stop all sessions.** This affects the input-hash checks for Stage 8, E1, E2 and E4. The agent runs the date-recovery procedure from the E3 plan; the owner approves pinning |
| Judge smoke test fails (`bitsandbytes`/CUDA) | G1 | Stop; the agent fixes the install pin. Do not switch to fp16 silently |
| Judge held-out gate fails | step 2 | Label the **dev** sheet; the agent proposes ≤ 3 dev-only prompt revisions (Tier 1 plan C3); rerun the judge pass; re-score the frozen test UIDs |
| Matched pairs < 100 after regrade | step 4 | Stop. Branch Rule 5 top-up is a protocol decision. **Do not widen the caliper. Do not use `popqa_fresh`** |
| Grader gate fails | step 6 | Dev-only revisions (≤ 3); otherwise stop |
| Qwen pilot fails | G2 | Expected-possible. Record it and continue with Llama only. **Do not change the neutral prompt** |
| E1 greedy consistency < 90% | G3 | Stop the answer-position analyses; suffix analyses proceed |
| CUDA out of memory | any | Rerun the same cell. Checkpoints resume. Reduce `--batch-size` only for capture (not generation) |

---

## Part D: Kaggle cells the agent must write (`notebooks/kaggle_cells_tier1_pivot.md`)

**Kaggle settings for every session:**
- Accelerator **GPU T4 ×2** (only GPU 0 is used, which keeps the judge in nf4); Internet **on**; Persistence off is fine.
- Secrets: `GITHUB_TOKEN`, `HF_TOKEN`, `HF_TENSOR_REPO_ID`.

**Rules for every cell the agent writes:**
- Start every session with **Cell 0** below.
- Never put a token in a URL or print one.
- Set env flags only inside `try/finally`.
- Commit through `commit_push()` (which calls `scripts/exploratory/commit_outputs.py` from B2, or git directly for registered paths).
- Print a short pass/fail summary from the produced JSON at the end of each cell.
- Every cell must be **idempotent and resumable**: rerunning after a crash or timeout continues from checkpoints.

The agent must check every script flag below against the repo **after** the Part B changes, and correct the cells if names differ.

### Cell 0: session setup (identical in every session)

```python
# CELL 0 — session setup. Run first in every Kaggle session.
import os, json, subprocess, base64, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

S = UserSecretsClient()
for k in ("GITHUB_TOKEN", "HF_TOKEN", "HF_TENSOR_REPO_ID"):
    os.environ[k] = S.get_secret(k).strip()
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "0",            # one GPU => judge stays nf4 (Tier 1 policy)
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "PYTHONUNBUFFERED": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
    "GIT_TERMINAL_PROMPT": "0",
})
_auth = base64.b64encode(f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()).decode()
os.environ.update({"GIT_CONFIG_COUNT": "1",
                   "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                   "GIT_CONFIG_VALUE_0": f"Authorization: Basic {_auth}"})
_SECRETS = [os.environ["GITHUB_TOKEN"], os.environ["HF_TOKEN"], _auth]

REPO = Path("/kaggle/working/repo")
URL = "https://github.com/DakshUbhadia/unanswerability_to_hallucination.git"

def sh(cmd, cwd=REPO, check=True):
    """Stream a shell command; redact secrets; raise on failure if check."""
    p = subprocess.Popen(cmd, shell=True, cwd=str(cwd), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        for s in _SECRETS:
            line = line.replace(s, "***")
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f"Command failed ({rc}): {cmd}")
    return rc

def load(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))

def run_stage(stage, **flags):
    """kaggle_run.sh with env flags that can never leak into the next cell."""
    old = {k: os.environ.get(k) for k in flags}
    try:
        os.environ.update({k: str(v) for k, v in flags.items()})
        sh(f"bash scripts/kaggle_run.sh {stage}")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

def commit_push(paths, message):
    """Commit exploratory/registered outputs (never keys/tensors) with pull-rebase-retry."""
    globs = " ".join(f"'{p}'" for p in paths)
    sh(f"python scripts/exploratory/commit_outputs.py --message '{message}' --paths {globs}")

if not (REPO / ".git").is_dir():
    sh(f"git clone -q {URL} {REPO}", cwd="/kaggle/working")
sh("git config user.name 'Kaggle Runner' && git config user.email kaggle-runner@users.noreply.github.com")
sh("git fetch -q origin main && git checkout -q main && git reset -q --hard origin/main")
os.environ["PYTHONPATH"] = str(REPO)
sh("pip install -q -r requirements-kaggle.txt pyarrow")
sh("git log --oneline -1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
for k in ("STAGE2_POPQA_REGENERATE", "STAGE2_POPQA_REGRADE", "STAGE2_POPQA_REMATCH",
          "STAGE6_GENERATION_ONLY", "STAGE10_GENERATION_ONLY", "STAGE10_ANALYZE_ONLY",
          "FORCE", "JUDGE_INPUTS"):
    os.environ.pop(k, None)
print("Session ready.")
```

### Session G1 (pre-label): preflight, judge pass, E4 extraction

**Cell G1-1. E3 preflight.** This is the template-date check, and it **gates everything else**.
- Run `python -u scripts/exploratory/e3_preflight.py`.
- Then `commit_push(["results/exploratory/e3_preflight.json", "results/exploratory/e3_qwen_contract.json"], "E3: preflight")`.
- Print `template_date`, the Llama suffix equality check, the Qwen precision chosen, and `l_Q`.
- **If the exit code is 3: print a red STOP banner and do not run further cells.**
- About 10 minutes, including the Qwen-3B download.

**Cell G1-2. Judge smoke test.** Run 5 synthetic items through `classify_texts(offline=False)`. Assert `judge_quantization == "nf4"` and the revision equals `results/judge_model.json`. Do **not** write these items to the study cache: use a temp cache path, or `uid`s prefixed `smoke:`, and delete them afterwards.

**Cell G1-3. E0 judge pass.**
- Run `run_stage("judge")`.
- Print `results/judge_pass_summary.json`:
  - `n_unique_scored == n_unique_available` (expected about 2,488 inputs before deduplication: 1,000 PopQA + 1,488 Stage 6);
  - `quantizations_in_cache == ["nf4"]`;
  - `n_answer_then_unknown`;
  - low-margin counts.
- `kaggle_run.sh` commits `data/judge_cache.jsonl` itself.
- About 20–45 minutes.

**Cell G1-4. E4 SQuAD preparation.**
- Run `python -u scripts/exploratory/e4_prepare_data.py --sources squad`. `HF_TOKEN` gives tokenizer access.
- Assert that the MANIFEST lists `squad` with 300 + 300 rows (or report the shortfall per the E4 plan) and that `failed_sources` no longer contains `squad`.
- Then `commit_push(["data/exploratory/e4/*"], "E4: prepare SQuAD")`.

**Cell G1-5. E4 extraction (registered condition).**
- Run `python -u scripts/exploratory/e4_extract_features.py --sources syn,squad,selfaware --condition registered --push-hf`.
- Print the `syn` parity results: direction cosine must be ≥ 0.9999, and the validation log-odds error ≤ 0.05.
- Then `commit_push(["results/exploratory/e4_features/**/manifest.json", "results/exploratory/e4_features/**/meta.parquet"], "E4: registered features")`.
- About 15–25 minutes.

### Session G2 (pre-label, can overlap with the owner's labelling): E3 generation

**Cell G2-1. Llama pilot** (separate process, B5):
- `python -u scripts/exploratory/e3_generate_and_capture.py --model llama --steps pilot`.
- Print the judged `unans`/`ans` rates and `behaviourally_anchored`.

**Cell G2-2. Llama main steps:**
- `… --model llama --steps popqa,pairs,features --push-hf`.
- Then `commit_push(["data/exploratory/popqa_neutral_llama32_3b.jsonl", "results/exploratory/e3_pilot_llama32_3b.json", "results/exploratory/e3_features/**/manifest*.json", "results/exploratory/e3_features/**/meta.parquet"], "E3: Llama neutral generation and capture")`.
- About 40–70 minutes.

**Cell G2-3. Qwen pilot:**
- `… --model qwen --steps pilot`.
- Read `results/exploratory/e3_pilot_qwen25_3b.json`. If `arm_included` is false, print "Qwen arm excluded (pre-registered rule)" and skip G2-4.

**Cell G2-4. Qwen main steps** (only if included): as G2-2 with `--model qwen` and the Qwen file names. About 40–70 minutes.

**Cell G2-5. Judge pass on the neutral files:**
- `run_stage("judge", JUDGE_INPUTS="data/exploratory/popqa_neutral_llama32_3b.jsonl data/exploratory/popqa_neutral_qwen25_3b.jsonl")`. Drop the Qwen file if excluded.
- Assert that the cache quantization is still `["nf4"]`.

**Cell G2-6 (optional). E4 neutral condition:**
- `e4_extract_features.py --sources syn,squad,selfaware --condition neutral --push-hf`, then commit the manifests.
- Run only if `e3_preflight.json` passed.

### Session G3 (post-label): requires the rematch to be pushed from local step L2

**Cell G3-0. Precondition check.**
- Assert:
  - `load("results/stage2_pilot.json")["label_source"] == "canonical_v1_batch1_judged"`;
  - `judge_validation.json["gate_passed"]`;
  - `grader_validation.json["gate_passed"]`.
- Print the matched-pair count. Abort otherwise.

**Cell G3-1.** `run_stage(8, FORCE=1)`. Print H4/H5 and `label_source` from `stage8_popqa_core.json`. About 10–15 minutes.

**Cell G3-2. E1 extraction:**
- `python -u scripts/exploratory/e1_extract_features.py --push-hf`.
- Print the parity results, `greedy_consistency_rate` and the Stage 8 confidence parity.
- Then `commit_push(["results/exploratory/e1_features/features_manifest.json", "results/exploratory/e1_features/meta.parquet"], "E1: features")`.
- About 10–20 minutes.

**Cell G3-3.** `run_stage("stage10gen")`, then `run_stage("judge")`. The default judge inputs now include `results/stage10_generations.jsonl`. About 30–50 minutes.

### Session G4 (post-label): E2 Part C

**Cell G4-1.**
- Assert that `results/exploratory/e2_dom_11_suf-4.npy` exists and its sha256 matches `e2_results.json` (B8).
- Run `python -u scripts/exploratory/e2_causal_vocab.py --dom-vector results/exploratory/e2_dom_11_suf-4.npy --push-hf`.
- Then `commit_push(["results/exploratory/e2_causal_vocab.json"], "E2: Part C causal vocabulary")`.
- Print the `learned_alpha_1` and `popqa_dom` labels.
- About 30–45 minutes (resumable via checkpoint).

---

## Part E: Local WSL steps the agent must write (`notebooks/local_steps_tier1_pivot.md`)

All of these run in the owner's WSL clone with the repo's virtualenv (`requirements.txt`) and `export PYTHONPATH=.`. Set `HF_TOKEN` and `HF_TENSOR_REPO_ID` in the shell, because several analyses download tensors.

**Blinding rules:**
- Begin every step with `git pull --ff-only`.
- Never `cat`, open or `git diff` a `*_key.json`.
- Don't look at `data/judge_cache.jsonl` before labelling.

**L1 (after G1): judge sheet**

```bash
python scripts/export_judge_labeling_sheet.py                     # offline; popqa+stage6; writes test+dev CSVs and key
# HUMAN: label results/judge_validation_test.csv ONLY (ANSWER/ABSTAIN/DEGENERATE). Dev sheet only if told.
python scripts/import_judge_labels.py --csv results/judge_validation_test.csv
python -c "import json;d=json.load(open('results/judge_validation.json'));print({k:d.get(k) for k in ('gate_passed','kappa','abstain_recall','low_margin_gate_passed')})"
# if low_margin_gate_passed is False:
python scripts/export_judge_low_margin_review.py   # HUMAN labels results/judge_low_margin_review.csv
python scripts/import_judge_overrides.py
git add -f results/judge_validation*.csv results/judge_validation.json results/judge_validation_manifest.json \
           results/judge_validation_key.json results/judge_low_margin_review.csv 2>/dev/null
git commit -m "Tier 1: judge validation (held-out)" && git push
```

The agent must confirm the exact field names in `judge_validation.json` and fix the one-liner accordingly.

**L2 (after the judge gate passes and G2 is pushed)**

```bash
python scripts/stage6_rescore_generations.py --offline
python scripts/stage2_pilot_gate.py --regrade-saved-popqa-only --offline     # prints counts + matched pairs; STOP if < 100
python scripts/export_grader_labeling_sheet.py
python scripts/exploratory/e3_export_validation.py                            # E3 judge(100)+grader(60) sheets
# HUMAN: label results/grader_validation_test.csv and the two E3 *_test.csv sheets
python scripts/import_grader_labels.py --csv results/grader_validation_test.csv
python scripts/stage2_pilot_gate.py --rematch-only --offline
python scripts/verify_tier1_popqa_regeneration.py
# import E3 sheets (agent: exact commands for judge --prefix and grader --key per E3 plan Task 5.5)
python scripts/exploratory/e4_analyze.py --condition registered               # non-PopQA cells now; PopQA pending
git add -f results/*.json results/*.csv data/popqa_sample.jsonl tables figures results/exploratory data/exploratory
git commit -m "Tier 1: grader validation, rematch; E3 spot-check; E4 non-PopQA" && git push
```

**L3 (after G3)**

```bash
python scripts/export_judge_labeling_sheet.py --sources stage10 --prefix judge_validation_stage10 --n-test 100 --n-dev 50
# HUMAN: label results/judge_validation_stage10_test.csv
python scripts/import_judge_labels.py --prefix judge_validation_stage10 --csv results/judge_validation_stage10_test.csv
python scripts/stage10_popqa_steering.py --analyze-only --offline
python scripts/stage9_generate_figures_and_tables.py
python scripts/build_archive_manifest.py --name archive_v43 ...   # per Tier 1 plan E5; then verify_archive_v43.py
python scripts/exploratory/e1_analyze.py --population all --n-jobs 4
python scripts/exploratory/e2_analyze.py --parts R,G,L,A --n-jobs 4  # writes e2_dom_11_suf-4.npy (B8)
git add ... && git commit -m "Tier 1 complete; E1; E2 R/G/L/A" && git push
```

**L4 (after G4)**

```bash
python scripts/exploratory/e2_analyze.py --parts C
python scripts/exploratory/e3_label_and_match.py --model llama32_3b --offline   # and qwen25_3b if included
python scripts/exploratory/e3_analyze.py                                         # B7
python scripts/exploratory/e4_analyze.py --condition both
python scripts/exploratory/build_report.py   # canonical-only after B6
git add ... && git commit -m "E2 Part C; E3; E4 complete" && git push
```

---

## Part F: Definition of done for the agent

- [ ] Part B fixes B1–B9 committed with tests and DEV entries; the CPU suite is green on a **fresh Linux clone**
- [ ] `notebooks/kaggle_cells_tier1_pivot.md` contains Cell 0 plus G1–G4 cells, each with purpose, runtime, expected outputs, assertions and failure handling, using the verified script flags
- [ ] `notebooks/local_steps_tier1_pivot.md` contains L1–L4, with exact commands and verified field names
- [ ] `notebooks/kaggle_runner.md` lists canonical commands only (B6)
- [ ] A short report to the owner: the Part C timeline with any changes, and anything the agent could not verify
