import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# scripts/kaggle_sync.py
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
    paths = []
    for pattern in ("results/*.json", "results/*.csv", "tables/*", "figures/*", "DEVIATIONS.md"):
        paths.extend(str(path.relative_to(CONFIG.root_dir)) for path in CONFIG.root_dir.glob(pattern) if path.exists())
    if stage_label in ("2", "stage2"):
        for relpath in ("results/stage2_pilot.json", "data/popqa_sample.jsonl", "results/stage0_env.json"):
            if (CONFIG.root_dir / relpath).exists() and relpath not in paths:
                paths.append(relpath)
        for artifact in ("results/stage2_pilot.json", "data/popqa_sample.jsonl"):
            source = CONFIG.root_dir / artifact
            if source.exists():
                import shutil
                target = Path("/kaggle/working") / Path(artifact).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    if stage_label in ("3", "stage3", "4", "stage4"):
        stage_num = "3" if stage_label in ("3", "stage3") else "4"
        manifest = CONFIG.results_dir / f"stage{stage_num}_manifest.json"
        if manifest.exists():
            paths.append(str(manifest.relative_to(CONFIG.root_dir)))
            import shutil
            target = Path("/kaggle/working") / manifest.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest, target)
    if paths:
        run_cmd(["git", "add", "-f", "--", *paths])
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
    if action == "push":
        api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)
    elif action != "pull":
        raise ValueError(f"Unsupported tensor sync action: {action}")

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
        for stage_num in (3, 4):
            manifest = CONFIG.results_dir / f"stage{stage_num}_manifest.json"
            if manifest.exists():
                api.upload_file(path_or_fileobj=str(manifest), path_in_repo=f"results/stage{stage_num}_manifest.json",
                                repo_id=hf_repo_id, repo_type="dataset")
    elif action == "pull":
        repo_files = set(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
        for fname in pt_files:
            remote_path = f"results/{fname}"
            if remote_path in repo_files:
                print(f"[Sync] Downloading {remote_path} from {hf_repo_id}...")
                api.hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(CONFIG.root_dir),
                )


def ensure_stage3_cache_present(repo_dir: Path, hf_repo_id: str) -> Path:
    """Download the Stage 3 cache from the configured private HF dataset when absent."""
    target = Path(repo_dir) / "results" / "stage3_caches.pt"
    if target.exists():
        return target
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to restore the Stage 3 cache")
    if not hf_repo_id:
        raise RuntimeError("HF_TENSOR_REPO_ID is required to restore the Stage 3 cache")
    api = HfApi(token=token)
    api.hf_hub_download(repo_id=hf_repo_id, filename="results/stage3_caches.pt",
                        repo_type="dataset", local_dir=str(repo_dir))
    if not target.exists():
        raise FileNotFoundError(f"HF download did not create {target}")
    return target


def ensure_tensor_cached_from_hf(filename: str, hf_repo_id: str = "", repo_dir: Path = CONFIG.root_dir) -> Path:
    """Ensure a results tensor exists locally, restoring it from the HF dataset if needed."""
    if Path(filename).name != filename or not filename.endswith(".pt"):
        raise ValueError("filename must be a .pt basename")
    target = Path(repo_dir) / "results" / filename
    if target.exists():
        return target
    token = os.environ.get("HF_TOKEN")
    repo_id = hf_repo_id or os.environ.get("HF_TENSOR_REPO_ID", "")
    if not token or not repo_id:
        raise RuntimeError("HF_TOKEN and HF_TENSOR_REPO_ID are required to restore tensor artifacts")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=repo_id, filename=f"results/{filename}", repo_type="dataset",
                    token=token, local_dir=str(repo_dir))
    if not target.is_file():
        raise FileNotFoundError(f"HF download did not create {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle outputs with GitHub and HF Hub")
    parser.add_argument("--action", choices=["pull", "push"], required=True)
    parser.add_argument("--stage", type=str, default="stage0")
    parser.add_argument("--hf-repo", type=str, default=os.environ.get("HF_TENSOR_REPO_ID", ""))
    parser.add_argument("--pull-cache", type=str, choices=["3"], default="")
    parser.add_argument("--ensure-tensor", type=str, default="")
    args = parser.parse_args()

    if args.ensure_tensor:
        ensure_tensor_cached_from_hf(args.ensure_tensor, args.hf_repo)
    elif args.pull_cache == "3":
        ensure_stage3_cache_present(CONFIG.root_dir, args.hf_repo)
    elif args.action == "pull" and args.hf_repo:
        sync_pt_tensors_hf("pull", args.hf_repo)
    elif args.action == "push":
        if args.hf_repo:
            sync_pt_tensors_hf("push", args.hf_repo)
        push_git_artifacts(args.stage)
