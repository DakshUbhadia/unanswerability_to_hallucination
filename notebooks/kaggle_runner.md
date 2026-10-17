# Kaggle Notebook Runner Template

## Settings (Before Running)
1. In the right-hand Settings pane, set **Accelerator** to **GPU T4 x2** and toggle **Internet** to **On**.
2. Under **Add-ons -> Secrets**, enable `GITHUB_TOKEN` and `HF_TOKEN`.
3. Paste the following two cells and click **Save & Run All (Commit)**.

## Cell 1: Bootstrap Repo, Secrets & Pinned Packages

```python
# === KAGGLE CELL 1: Bootstrap Repo, Secrets & Pinned Packages ===
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
GITHUB_USER = "YOUR_GITHUB_USERNAME"
GITHUB_REPO = "unanswerability-to-hallucination"
HF_CACHE_REPO = f"{GITHUB_USER}/unanswerability-to-hallucination-caches"

import subprocess
import os

os.chdir("/kaggle/working")
if not os.path.exists(GITHUB_REPO):
    subprocess.run(["git", "clone", f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"], check=True)
os.chdir(f"/kaggle/working/{GITHUB_REPO}")
subprocess.run(["git", "config", "user.email", "kaggle-runner@users.noreply.github.com"], check=True)
subprocess.run(["git", "config", "user.name", "Kaggle GPU Runner"], check=True)
subprocess.run(["git", "pull", "origin", "main"], check=True)

# Install pinned nnsight==0.3.7 and transformers==4.46.3 onto Kaggle image
subprocess.run(["pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)

# Pull any previously cached .pt tensors (skips automatically if already present)
subprocess.run(["python", "scripts/kaggle_sync.py", "--action", "pull", "--hf-repo", HF_CACHE_REPO],
               env={**os.environ, "PYTHONPATH": "."}, check=True)
```

## Cell 2: Execute Stage 0 & Auto-Push Results

```python
# === KAGGLE CELL 2: Execute Resumable Stage(s) & Auto-Push Results ===
import subprocess, os
# Each stage automatically skips if its outputs already exist and pass validation!
subprocess.run(["python", "scripts/stage0_env_check.py", "--allow-kaggle-env", "--dtype", "float16"],
               env={**os.environ, "PYTHONPATH": "."}, check=True)
subprocess.run(["python", "scripts/kaggle_sync.py", "--action", "push", "--stage", "stage0",
                "--hf-repo", HF_CACHE_REPO],
               env={**os.environ, "PYTHONPATH": "."}, check=True)
```
