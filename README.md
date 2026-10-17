# Unanswerability to Hallucination

Does the unanswerability direction in Qwen2.5-3B-Instruct predict closed-book hallucinations? A replication and extension of Lavi, Milo & Geva (EACL 2026).

The frozen research protocol is in [documentation.md](documentation.md). Implementation deviations are tracked in [DEVIATIONS.md](DEVIATIONS.md).

## Status

| Stage | Status |
|---|---|
| Stage 0 | Passed |
| Stage 1 | Pending human review |

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

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/stage1_export_review.py
.venv/bin/python scripts/stage1_build_dataset.py --force  # after completing data/review/review_sheet.csv
.venv/bin/python -m pytest -q -m "not slow"
```
