# Kaggle cells: Tier 1 refresh and exploratory pivot

Run one Kaggle session per group (G1–G4). Configure GPU **T4 ×2**, Internet on, with Kaggle secrets `GITHUB_TOKEN`, `HF_TOKEN`, and `HF_TENSOR_REPO_ID`. The code exposes only GPU 0, preserving the Tier 1 NF4 judge policy. Do not run any cell after a stated STOP condition. Rerunning a cell resumes from script checkpoints and skips outputs already present.

Every cell below assumes Cell 0 has run in the current session. It fetches `origin/main`, resets the single checkout at `/kaggle/working/repo`, sets a Git identity, and installs the Kaggle requirements. Tokens stay in environment variables and are redacted from streamed output.

## Cell 0 — Session setup (run first in every session)

**Purpose:** establish the one canonical checkout, credentials, runtime settings, and safe push helpers. **Runtime:** about 2–5 minutes. **Expected output:** latest commit, GPU details, and `Session ready.` **Failure:** fix the reported secret/network/install issue, rerun this cell, and do not run later cells until it completes.

```python
import os, json, subprocess, base64, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

S = UserSecretsClient()
for k in ("GITHUB_TOKEN", "HF_TOKEN", "HF_TENSOR_REPO_ID"):
    os.environ[k] = S.get_secret(k).strip()
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "0",
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

def sh(cmd, cwd=REPO, check=True):
    """Stream output with credentials redacted; raise on failure when check=True."""
    p = subprocess.Popen(cmd, shell=True, cwd=str(cwd), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        for secret in _SECRETS:
            line = line.replace(secret, "***")
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc:
        raise RuntimeError(f"Command failed ({rc}): {cmd}")
    return rc

def load(rel):
    return json.loads((REPO / rel).read_text(encoding="utf-8"))

def run_stage(stage, **flags):
    """Set stage flags only for this invocation, restoring the prior environment."""
    old = {k: os.environ.get(k) for k in flags}
    try:
        os.environ.update({k: str(v) for k, v in flags.items()})
        sh(f"bash scripts/kaggle_run.sh {stage}")
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

def commit_push(paths, message):
    quoted = " ".join("'" + p.replace("'", "'\\''") + "'" for p in paths)
    sh(f"python scripts/exploratory/commit_outputs.py --message '{message}' --paths {quoted}")

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

## Session G1 — Preflight, judge, and registered E4 features

### Cell G1-1 — E3 tokenizer/runtime preflight

**Purpose:** gate all later work on the Llama chat-template date and suffix checks; probe Qwen's precision and position contract. **Runtime:** about 10 minutes, including Qwen download. **Expected outputs:** `e3_preflight.json`, `e3_qwen_contract.json`; report `template_date`, suffix equality, Qwen precision, and `qwen_depth_matched_layer`. **Failure:** exit 3 means print STOP and end G1. Do not try another date or pin one. A nonzero failure means stop and report.

```python
rc = sh("python -u scripts/exploratory/e3_preflight.py", check=False)
if rc == 3:
    print("\n🛑 STOP: dynamic chat-template date. Do not run any later cell. Report this before changing registered input rendering.\n")
    raise RuntimeError("G1 stopped by preflight")
if rc:
    raise RuntimeError(f"Preflight failed with exit code {rc}")
p = load("results/exploratory/e3_preflight.json")
q = load("results/exploratory/e3_qwen_contract.json")
print({"template_date": p["template_date"],
       "llama_suffix_matches_stage0": p["llama_neutral_suffix_token_ids"] == load("results/stage0_env.json")["suffix_token_ids"],
       "qwen_precision": q["dtype"], "qwen_layer": p["qwen_depth_matched_layer"],
       "qwen_passed": p["qwen_passed"]})
commit_push(["results/exploratory/e3_preflight.json", "results/exploratory/e3_qwen_contract.json"], "E3: preflight")
```

### Cell G1-2 — Five-item local judge smoke test

**Purpose:** catch missing CUDA/NF4/bitsandbytes or a revision mismatch before the long judge pass. **Runtime:** about 2 minutes. **Expected output:** five scored items, all NF4, pinned revision match. **Failure:** stop G1 and report the import/CUDA failure; do not switch to fp16. The cache path is temporary and no smoke rows enter the study cache.

```python
import tempfile
from dataclasses import replace
import src.abstention_classifier as judge
pin = load("results/judge_model.json")
old_cfg = judge.CFG
try:
    with tempfile.TemporaryDirectory(prefix="judge-smoke-") as tmp:
        judge.CFG = replace(judge.CFG, judge_cache_path=Path(tmp) / "smoke_cache.jsonl")
        items = [{"uid": f"smoke:{i}", "question": f"Synthetic smoke question {i}?",
                  "response": "I do not know." if i % 2 else "A synthetic answer."} for i in range(5)]
        scored = judge.classify_texts(items, offline=False)
        revisions = {r.get("judge_model_revision") for r in scored}
        quants = {r.get("judge_quantization") for r in scored}
        assert len(scored) == 5 and quants == {"nf4"}, (len(scored), quants)
        assert revisions == {pin["revision_sha"]}, revisions
        print({"smoke_items": len(scored), "quantization": sorted(quants), "revision_matches_pin": True})
finally:
    judge.release_local_judge()
    judge.CFG = old_cfg
```

### Cell G1-3 — Tier 1 judge pass

**Purpose:** score registered PopQA and Stage 6 responses. **Runtime:** 20–45 minutes. **Expected output:** all available unique rows scored, cache quantization `['nf4']`, summary includes `n_answer_then_unknown` and low-margin counts. **Failure:** inspect missing inputs/cache status and rerun; it is resumable. Do not proceed to local judge labeling unless the pass completed.

```python
run_stage("judge")
s = load("results/judge_pass_summary.json")
print({"n_unique_scored": s["n_unique_scored"], "n_unique_available": s["n_unique_available"],
       "quantizations_in_cache": s["quantizations_in_cache"],
       "n_answer_then_unknown": s["n_answer_then_unknown"],
       "low_margin": s["margin_summary_by_condition"]})
assert s["n_unique_scored"] == s["n_unique_available"]
assert s["quantizations_in_cache"] == ["nf4"]
```

### Cell G1-4 — Prepare gated SQuAD data

**Purpose:** prepare the registered E4 SQuAD source using the Kaggle HF token. **Runtime:** about 10–20 minutes. **Expected output:** manifest has `squad`, 300 rows in each class or the documented source-filter shortfall, and no failed SQuAD source. **Failure:** stop if tokenizer access or the dataset revision fails; report a shortfall without changing the sampling rules.

```python
sh("python -u scripts/exploratory/e4_prepare_data.py --sources squad")
m = load("data/exploratory/e4/MANIFEST.json")
squad = m.get("sources", {}).get("squad", {})
print({"squad_row_count": squad.get("row_count"), "class_counts": squad.get("class_counts"),
       "failed_sources": m.get("failed_sources", [])})
assert "squad" not in m.get("failed_sources", [])
commit_push(["data/exploratory/e4/*"], "E4: prepare SQuAD")
```

### Cell G1-5 — Registered E4 feature extraction

**Purpose:** capture `syn`, `squad`, and `selfaware` under the registered prompt and upload feature tensors to the HF dataset. **Runtime:** 15–25 minutes. **Expected output:** `syn` direction cosine ≥ 0.9999, validation log-odds error ≤ 0.05, and source manifests. **Failure:** stop on parity failure; rerun after OOM with the same command and batch size unless capture alone must resume with a lower batch size.

```python
sh("python -u scripts/exploratory/e4_extract_features.py --sources syn,squad,selfaware --condition registered --push-hf")
for source in ("syn", "squad", "selfaware"):
    m = load(f"results/exploratory/e4_features/registered/{source}/manifest.json")
    print(source, {"n_rows": m.get("n_rows"), "parity": m.get("parity")})
syn = load("results/exploratory/e4_features/registered/syn/manifest.json")["parity"]
assert syn["direction_cosine"] >= 0.9999 and max(syn["val_log_odds_abs_error"].values()) <= 0.05
commit_push(["results/exploratory/e4_features/registered/**/manifest.json", "results/exploratory/e4_features/registered/**/meta.parquet"], "E4: registered features")
```

## Session G2 — E3 neutral generation and judging

### Cell G2-1 — Llama pilot (separate process)

**Purpose:** generate and judge the 96 validation pairs, after the G1 preflight. **Runtime:** about 5 minutes. **Expected output:** judged neutral answerable/unanswerable rates and `behaviourally_anchored`. **Failure:** rerun the same command; if judge/CUDA fails, stop and report.

```python
sh("python -u scripts/exploratory/e3_generate_and_capture.py --model llama --steps pilot")
p = load("results/exploratory/e3_pilot_llama32_3b.json")
print({"unans": p["neutral_unans_abstention_rate"], "ans": p["neutral_ans_abstention_rate"],
       "behaviourally_anchored": p["behaviourally_anchored"]})
```

### Cell G2-2 — Llama PopQA, pairs, and features

**Purpose:** generate all 1,000 fixed PopQA questions, capture train/validation/test pairs and feature tensors, and upload tensors. **Runtime:** 40–70 minutes. **Expected output:** PopQA JSONL, pilot JSON, feature manifests, and HF tensor files. **Failure:** rerun the same command; checkpoints resume generation/capture.

```python
sh("python -u scripts/exploratory/e3_generate_and_capture.py --model llama --steps popqa,pairs,features --push-hf")
commit_push(["data/exploratory/popqa_neutral_llama32_3b.jsonl", "results/exploratory/e3_pilot_llama32_3b.json",
            "results/exploratory/e3_features/llama32_3b_neut/pairs_manifest.json",
            "results/exploratory/e3_features/llama32_3b_neut/features_manifest.json",
            "results/exploratory/e3_features/llama32_3b_neut/meta.parquet"], "E3: Llama neutral generation and capture")
print({"popqa_rows": len((REPO / "data/exploratory/popqa_neutral_llama32_3b.jsonl").read_text().splitlines()),
       "features": load("results/exploratory/e3_features/llama32_3b_neut/features_manifest.json")["n_rows"]})
```

### Cell G2-3 — Qwen pilot (separate process)

**Purpose:** apply the preregistered neutral pilot and precision inclusion rule. **Runtime:** about 5 minutes. **Expected output:** Qwen rates and `arm_included`. **Failure:** if Qwen fails, record the output and continue with Llama only; do not alter the neutral prompt.

```python
sh("python -u scripts/exploratory/e3_generate_and_capture.py --model qwen --steps pilot")
p = load("results/exploratory/e3_pilot_qwen25_3b.json")
print({"unans": p["neutral_unans_abstention_rate"], "ans": p["neutral_ans_abstention_rate"],
       "arm_included": p["arm_included"]})
```

### Cell G2-4 — Qwen main steps (conditional)

**Purpose:** capture the Qwen neutral condition only after its pilot and precision checks pass. **Runtime:** 40–70 minutes. **Expected output:** Qwen PopQA JSONL and HF tensors. **Failure:** rerun the same command; if the pilot excluded Qwen, this cell is skipped.

```python
if load("results/exploratory/e3_pilot_qwen25_3b.json").get("arm_included"):
    sh("python -u scripts/exploratory/e3_generate_and_capture.py --model qwen --steps popqa,pairs,features --push-hf")
    commit_push(["data/exploratory/popqa_neutral_qwen25_3b.jsonl",
                "results/exploratory/e3_pilot_qwen25_3b.json",
                "results/exploratory/e3_features/qwen25_3b_neut/pairs_manifest.json",
                "results/exploratory/e3_features/qwen25_3b_neut/features_manifest.json",
                "results/exploratory/e3_features/qwen25_3b_neut/meta.parquet"], "E3: Qwen neutral generation and capture")
    print({"arm_included": True, "features": load("results/exploratory/e3_features/qwen25_3b_neut/features_manifest.json")["n_rows"]})
else:
    print("Qwen arm excluded by the preregistered pilot; continuing with Llama only.")
```

### Cell G2-5 — Judge the neutral PopQA files

**Purpose:** score each included model's neutral generations into the shared cache. **Runtime:** 15–30 minutes per 1,000-row model. **Expected output:** complete available rows and NF4-only cache. **Failure:** rerun the same invocation; cache hits make it resumable.

```python
inputs = "data/exploratory/popqa_neutral_llama32_3b.jsonl"
if load("results/exploratory/e3_pilot_qwen25_3b.json").get("arm_included"):
    inputs += " data/exploratory/popqa_neutral_qwen25_3b.jsonl"
run_stage("judge", JUDGE_INPUTS=inputs)
s = load("results/judge_pass_summary.json")
assert s["quantizations_in_cache"] == ["nf4"]
print({"scored": s["n_unique_scored"], "available": s["n_unique_available"],
       "quantizations": s["quantizations_in_cache"], "margins": s["margin_summary_by_condition"]})
```

### Cell G2-6 — Optional neutral E4 extraction

**Purpose:** capture the E4 neutral condition when E3 preflight passed. **Runtime:** 15–25 minutes. **Expected output:** neutral manifests and HF tensors. **Failure:** stop on E3 date/preflight failure; otherwise rerun for checkpoint recovery.

```python
if load("results/exploratory/e3_preflight.json").get("template_date") == "static":
    sh("python -u scripts/exploratory/e4_extract_features.py --sources syn,squad,selfaware --condition neutral --push-hf")
    commit_push(["results/exploratory/e4_features/neutral/**/manifest.json",
                 "results/exploratory/e4_features/neutral/**/meta.parquet"], "E4: neutral features")
    print({s: load(f"results/exploratory/e4_features/neutral/{s}/manifest.json").get("n_rows")
           for s in ("syn", "squad", "selfaware")})
else:
    print("Skipped: E3 preflight did not establish a static template date.")
```

## Session G3 — Post-label Stage 8, E1, and Stage 10 generation

### Cell G3-0 — Gate and rematch precondition

**Purpose:** prevent post-label GPU work before judge/grader gates and the canonical regrade pass. **Runtime:** under 1 minute. **Expected output:** `canonical_v1_batch1_judged`, both gates true, and matched-pair count. **Failure:** stop and return to the WSL gate/rematch steps.

```python
s2 = load("results/stage2_pilot.json")
j = load("results/judge_validation.json")
g = load("results/grader_validation.json")
assert s2.get("label_source") == "canonical_v1_batch1_judged"
assert j.get("gate_passed") and g.get("gate_passed")
print({"label_source": s2["label_source"], "judge_gate": j["gate_passed"], "grader_gate": g["gate_passed"],
       "matched_pairs": s2.get("matched_pair_count", s2.get("n_matched_pairs"))})
```

### Cell G3-1 — Forced Stage 8 after rematch

**Purpose:** regenerate Stage 8 features and statistics using the judged canonical rematch. **Runtime:** 10–15 minutes. **Expected output:** H4/H5 and current `label_source`. **Failure:** rerun the same cell; do not proceed to E1 if Stage 8 fails.

```python
run_stage(8, FORCE=1)
s = load("results/stage8_popqa_core.json")
print({"H4": s.get("h4"), "H5": s.get("h5"), "label_source": s.get("label_source"),
       "matched_items": len(s.get("matched_items", []))})
assert all(k in s["matched_items"][0] for k in ("popqa_id", "maxp", "entropy", "log_odds"))
```

### Cell G3-2 — E1 features (after Stage 8)

**Purpose:** capture the E1 feature tensors and verify Stage 8 confidence parity. **Runtime:** 10–20 minutes. **Expected output:** greedy consistency ≥ 0.90, all confidence errors ≤ 1e-3, HF upload complete. **Failure:** stop answer-position analyses if greedy consistency is below 90%; rerun same command for OOM/checkpoint recovery.

```python
sh("python -u scripts/exploratory/e1_extract_features.py --push-hf")
m = load("results/exploratory/e1_features/features_manifest.json")
print({"greedy_consistency_rate": m["greedy_consistency_rate"],
       "stage8_confidence_parity_max_abs_diff": m["stage8_confidence_parity_max_abs_diff"],
       "parity": m["parity"]})
assert m["greedy_consistency_rate"] >= 0.90
assert max(m["stage8_confidence_parity_max_abs_diff"].values()) <= 1e-3
commit_push(["results/exploratory/e1_features/features_manifest.json", "results/exploratory/e1_features/meta.parquet"], "E1: features")
```

### Cell G3-3 — Stage 10 generation and judge pass

**Purpose:** generate supplementary Stage 10 outputs, then add them to the NF4 judge cache. **Runtime:** 30–50 minutes. **Expected output:** Stage 10 generation file and judge summary. **Failure:** rerun the same commands; generation checkpoints and judge cache resume.

```python
run_stage("stage10gen")
run_stage("judge")
s = load("results/judge_pass_summary.json")
print({"scored": s["n_unique_scored"], "available": s["n_unique_available"],
       "quantizations": s["quantizations_in_cache"]})
assert s["quantizations_in_cache"] == ["nf4"]
```

## Session G4 — E2 Part C

### Cell G4-1 — Validate and run Part C

**Purpose:** use the exact checksummed E2 Part G vector to run 99 random controls and the learned steering condition. **Runtime:** 30–45 minutes. **Expected output:** `e2_causal_vocab.json`, learned α=1 and PopQA-DOM labels, HF upload. **Failure:** stop on checksum mismatch; rerun the same command after OOM because checkpoints resume.

```python
vector = REPO / "results/exploratory/e2_dom_11_suf-4.npy"
e2 = load("results/exploratory/e2_results.json")
expected = e2["parts"]["G"]["dom_vector_sha256"]
actual = __import__("hashlib").sha256(vector.read_bytes()).hexdigest()
assert vector.exists() and actual == expected, {"expected": expected, "actual": actual}
sh("python -u scripts/exploratory/e2_causal_vocab.py --dom-vector results/exploratory/e2_dom_11_suf-4.npy --push-hf")
r = load("results/exploratory/e2_causal_vocab.json")
labels = {x["name"]: x.get("label") for x in r["conditions"] if x["name"] in ("learned_alpha_1", "popqa_dom")}
print({"labels": labels, "C_dom_label": r.get("C_dom_label")})
commit_push(["results/exploratory/e2_causal_vocab.json"], "E2: Part C causal vocabulary")
```
