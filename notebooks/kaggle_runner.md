# Kaggle runner

Set Accelerator = GPU T4 (single GPU is fine), Internet = On, and configure Kaggle secrets `GITHUB_TOKEN` and `HF_TOKEN`. For Stage 3/4, also create `HF_TENSOR_REPO_ID` with the private Hugging Face dataset repo ID in `owner/repo` form. Then run this cell:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["GITHUB_TOKEN"] = s.get_secret("GITHUB_TOKEN").strip()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN").strip()
try:
    os.environ["HF_TENSOR_REPO_ID"] = s.get_secret("HF_TENSOR_REPO_ID").strip()
except Exception:
    pass  # Required only when running Stages 3 or 4.
subprocess.run("rm -rf /tmp/repo && git clone -q https://x-access-token:$GITHUB_TOKEN@github.com/DakshUbhadia/unanswerability_to_hallucination.git /tmp/repo && cd /tmp/repo && bash scripts/kaggle_run.sh 0 2", shell=True, check=True)
```

Stage 3 runs activation caching after Stage 2. It can be run with `bash scripts/kaggle_run.sh 3`; the runner restores tensor artifacts from the private HF dataset before execution and syncs the Stage 3 cache and manifest after completion. To run Stage 3 directly:

```bash
python scripts/stage3_cache_activations.py
```

Stage 4 extracts candidate abstention directions and the random, shuffled-label, and orthogonalized concept controls from the verified Stage 3 cache. The Kaggle runner restores `results/stage3_caches.pt` from the private HF dataset when needed, then saves and syncs the Stage 4 manifest and tensor artifact:

```bash
bash scripts/kaggle_run.sh 4
```

To run Stage 4 directly after restoring the cache, use `python -u scripts/stage4_extract_directions.py`. The local CPU suite includes `tests/test_stage4.py`.

Stage 2 can also be run independently after Stage 0 and Stage 1 with:

```bash
python scripts/stage2_pilot_gate.py
```

After Stage 2, measure token-length parity with the locked model tokenizer without changing the frozen prompt files:

```bash
python scripts/measure_stage2_tokenizer_parity.py
```

This writes `results/stage2_llama_parity.json` and updates the measured summary in `DEVIATIONS.md`. The Stage 2 runner also invokes this check after Stage 2 completes.

Stage 5 selects the validation coordinate, writes the selection JSON and Table 2 in CSV and Markdown, runs its CPU unit tests, and syncs the outputs. The runner restores Stage 4 directions from the private HF dataset when absent:

```bash
bash scripts/kaggle_run.sh 5
```

This requires `HF_TENSOR_REPO_ID` and `HF_TOKEN` as for the earlier tensor stages.

## Stages 5-10 (notebook)

Import `notebooks/kaggle_stages_5_to_10.ipynb` into Kaggle (File -> Import Notebook), then run the setup cell and each stage cell in order. Each cell calls `bash scripts/kaggle_run.sh <stage>`. The runner restores Stage 4 directions from HF, runs the stage, runs its CPU unit tests and artifact check, and commits and pushes the outputs to GitHub `main`. Stages that already verified print `[SKIP]`.

The runner also accepts ranges and `all`:

```bash
bash scripts/kaggle_run.sh 5-9   # Stages 5, 6, 7, 8, 9 with a sync after each
bash scripts/kaggle_run.sh all   # Stages 5-10
```

| Stage | Script | Main outputs |
|---|---|---|
| 6 | `scripts/stage6_test_evaluation.py` | `results/stage6_test_h1_h2_h3.json`, `results/stage6_generation_samples.jsonl`, Tables 1 and 4 |
| 7 | `scripts/stage7_capability_eval.py` | `results/stage7_capability.json`, Table 1 capability columns (`SKIP_MMLU=1` applies Cut 2) |
| 8 | `scripts/stage8_popqa_uncertainty.py` | `results/stage8_popqa_core.json`, Table 3, Holm-Bonferroni H1-H5 in Table 1 |
| 9 | `scripts/stage9_generate_figures_and_tables.py` | Figures 1, 2A, S1; `results/stage9_summary.json`; README results block |
| 10 | `scripts/stage10_popqa_steering.py` | `results/stage10_popqa_steering.json`, Figure 2B (optional stretch) |

After the Tier 1 judged-label rematch, rerun Stage 8 with `FORCE=1` so the existing v4 output cannot cause a skip:

```bash
FORCE=1 bash scripts/kaggle_run.sh 8
```

To create fresh unjudged Stage 10 generations before the supplementary judge pass, use `bash scripts/kaggle_run.sh stage10gen`.

## Tier 1 refresh and exploratory pivot

Use the owner-facing runbooks for the ordered GPU and local steps:

- [`kaggle_cells_tier1_pivot.md`](kaggle_cells_tier1_pivot.md) contains the resumable Kaggle cells for G1–G4.
- [`local_steps_tier1_pivot.md`](local_steps_tier1_pivot.md) contains the WSL CPU and labeling steps L1–L4.

Canonical GPU commands, in dependency order:

```bash
python scripts/exploratory/e3_preflight.py
python scripts/judge_pass.py                         # smoke test precedes this in G1
python scripts/exploratory/e4_prepare_data.py --sources squad
python scripts/exploratory/e4_extract_features.py --sources syn,squad,selfaware --condition registered --push-hf
python scripts/exploratory/e3_generate_and_capture.py --model llama --steps pilot
python scripts/exploratory/e3_generate_and_capture.py --model llama --steps popqa,pairs,features --push-hf
python scripts/exploratory/e3_generate_and_capture.py --model qwen --steps pilot
python scripts/exploratory/e3_generate_and_capture.py --model qwen --steps popqa,pairs,features --push-hf
FORCE=1 bash scripts/kaggle_run.sh 8
python scripts/exploratory/e1_extract_features.py --push-hf
bash scripts/kaggle_run.sh stage10gen
bash scripts/kaggle_run.sh judge
python scripts/exploratory/e2_causal_vocab.py --dom-vector results/exploratory/e2_dom_11_suf-4.npy --push-hf
```

Run the Qwen main steps only when its neutral pilot and precision preflight pass. Run `e4_extract_features.py --condition neutral` only after E3 preflight. `JUDGE_INPUTS` is passed by `kaggle_run.sh` when judging neutral generations. Do not use the retired Phase F scripts; they remain available only with `--legacy` where applicable.
