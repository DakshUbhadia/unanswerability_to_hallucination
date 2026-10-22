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
