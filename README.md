# Unanswerability to Hallucination

Does the unanswerability direction in Qwen2.5-3B-Instruct predict closed-book hallucinations? A replication and extension of Lavi, Milo & Geva (EACL 2026).

The frozen research protocol is in [documentation.md](documentation.md). Implementation deviations are tracked in [DEVIATIONS.md](DEVIATIONS.md).

## Status

| Stage | Status |
|---|---|
| Stages 0-4 | Passed (Stage 2 fallback to `meta-llama/Llama-3.2-3B-Instruct`, see DEV-004) |
| Stages 5-10 | Implemented; run on Kaggle with `notebooks/kaggle_stages_5_to_10.ipynb` |

## Kaggle setup

Use Accelerator = GPU T4 (single GPU is sufficient), Internet = On, and add Kaggle secrets `GITHUB_TOKEN` and `HF_TOKEN`. Run this cell:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["GITHUB_TOKEN"] = s.get_secret("GITHUB_TOKEN").strip()
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN").strip()
subprocess.run("rm -rf /tmp/repo && git clone -q https://x-access-token:$GITHUB_TOKEN@github.com/DakshUbhadia/unanswerability_to_hallucination.git /tmp/repo && cd /tmp/repo && bash scripts/kaggle_run.sh 0", shell=True, check=True)
```

Set `FORCE=1` in the environment to pass `--force` to stages that support it. To build Stage 1, first export and complete `data/review/review_sheet.csv`, then run `bash scripts/kaggle_run.sh 1`.

## Local validation

On Debian/Ubuntu/WSL, install project packages in a virtual environment. The system Python is externally managed, so running `pip install` directly may be rejected (PEP 668). From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/test_stage0.py tests/test_stage1.py tests/test_stage2.py -m "not integration" -v
```

If creating the virtual environment reports that `venv` or `ensurepip` is unavailable, install the OS support package once with `sudo apt install python3-full`, then rerun the commands above. Always invoke pytest through `.venv/bin/python` so it uses the environment where the dependencies were installed.

To run Stage 2 on Kaggle after Stage 0 and Stage 1:

```bash
python scripts/stage2_pilot_gate.py
```

The full local setup and Stage 1 preparation commands are:

```bash
.venv/bin/python scripts/stage1_export_review.py
.venv/bin/python scripts/stage1_build_dataset.py --force  # after completing data/review/review_sheet.csv
```
