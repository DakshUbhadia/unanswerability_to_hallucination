import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
﻿# scripts/kaggle_sync.py
import argparse
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi
from src.config import CONFIG


def run_cmd(cmd: list[str]) -> None:
    print(f"[Sync] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=CONFIG.root_dir)


def push_git_artifacts(stage_label: str) -> None:
    """Commits and pushes JSON, CSV, Markdown, and Figure artifacts to GitHub."""
    run_cmd(["git", "add", "results/*.json", "results/*.csv", "tables/", "figures/", "DEVIATIONS.md"])
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=CONFIG.root_dir,
    )
    if status.stdout.strip():
        run_cmd(["git", "commit", "-m", f"Auto-save outputs after {stage_label}"])
        run_cmd(["git", "push", "origin", "main"])
    else:
        print("[Sync] No new Git changes to push.")


def sync_pt_tensors_hf(action: str, hf_repo_id: str) -> None:
    """Uploads or downloads .pt tensor files (stage3_caches.pt, stage4_directions.pt) via HF Hub."""
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN environment variable must be set for tensor sync!"
    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)

    pt_files = ["stage3_caches.pt", "stage4_directions.pt"]
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)

    if action == "push":
        for fname in pt_files:
            fpath = CONFIG.results_dir / fname
            if fpath.exists():
                print(f"[Sync] Uploading {fname} to HF Dataset {hf_repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"results/{fname}",
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                )
    elif action == "pull":
        repo_files = set(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
        for fname in pt_files:
            remote_path = f"results/{fname}"
            local_path = CONFIG.results_dir / fname
            if remote_path in repo_files and not local_path.exists():
                print(f"[Sync] Downloading {remote_path} from {hf_repo_id}...")
                api.hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(CONFIG.root_dir),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle outputs with GitHub and HF Hub")
    parser.add_argument("--action", choices=["pull", "push"], required=True)
    parser.add_argument("--stage", type=str, default="stage0")
    parser.add_argument("--hf-repo", type=str, default="")
    args = parser.parse_args()

    if args.action == "pull" and args.hf_repo:
        sync_pt_tensors_hf("pull", args.hf_repo)
    elif args.action == "push":
        if args.hf_repo:
            sync_pt_tensors_hf("push", args.hf_repo)
        push_git_artifacts(args.stage)
