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


STAGE_GIT_FILES = {
    "judge": ["data/judge_cache.jsonl", "results/judge_pass_summary.json"],
    "5": ["results/stage5_selection.json", "tables/table2_sensitivity_grid.csv", "tables/table2_sensitivity_grid.md"],
    "6": ["results/stage6_test_h1_h2_h3.json", "results/stage6_generation_samples.jsonl",
          "tables/table1_main_results.csv", "tables/table1_main_results.md",
          "tables/table4_generation_audit.csv", "tables/table4_generation_audit.md"],
    "7": ["results/stage7_capability.json", "tables/table1_main_results.csv", "tables/table1_main_results.md"],
    "8": ["results/stage8_popqa_core.json", "tables/table3_popqa_auroc.csv", "tables/table3_popqa_auroc.md",
          "tables/table1_main_results.csv", "tables/table1_main_results.md"],
    "9": ["results/stage9_summary.json", "README.md"] + [f"figures/{n}.{e}" for n in (
        "fig1_intervention_pareto", "fig2a_popqa_violin", "figS1_layerwise_auroc") for e in ("png", "pdf")],
    "10": ["results/stage10_popqa_steering.json", "figures/fig2b_selective_steering.png",
           "figures/fig2b_selective_steering.pdf", "README.md", "results/stage9_summary.json"],
}


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
    stage_num = stage_label.replace("stage", "")
    if stage_num in STAGE_GIT_FILES:
        import shutil
        for artifact in STAGE_GIT_FILES[stage_num]:
            source = CONFIG.root_dir / artifact
            if source.exists():
                if artifact not in paths:
                    paths.append(artifact)
                target = Path("/kaggle/working") / Path(artifact).name
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                except OSError:
                    pass  # /kaggle/working exists only on Kaggle
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
        last_error = None
        for attempt in range(3):
            pull = subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=CONFIG.root_dir)
            if pull.returncode == 0:
                push = subprocess.run(["git", "push", "origin", "main"], cwd=CONFIG.root_dir)
                if push.returncode == 0:
                    break
                last_error = f"git push failed with status {push.returncode}"
            else:
                last_error = f"git pull --rebase failed with status {pull.returncode}"
                subprocess.run(["git", "rebase", "--abort"], cwd=CONFIG.root_dir)
            if attempt == 2:
                raise RuntimeError(f"Could not push after three attempts: {last_error}")
            import time
            time.sleep(2 ** attempt)
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
        stage5_json = CONFIG.results_dir / "stage5_selection.json"
        if stage5_json.exists():
            api.upload_file(path_or_fileobj=str(stage5_json), path_in_repo="results/stage5_selection.json",
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


def push_exploratory_features(hf_repo_id: str, features_dir: Path = CONFIG.root_dir / "features") -> int:
    """Upload memory-mapped exploratory features to the private tensor dataset, never Git."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required to upload exploratory feature caches")
    features_dir = Path(features_dir)
    files = sorted(path for pattern in ("*_resid.npy", "*_meta.parquet") for path in features_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No exploratory feature arrays or metadata found under {features_dir}")
    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)
    for path in files:
        relative = path.relative_to(features_dir).as_posix()
        target = f"exploratory/features/{relative}"
        print(f"[Sync] Uploading {path} to HF Dataset {hf_repo_id}:{target}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=hf_repo_id, repo_type="dataset")
    return len(files)


def push_exploratory_manifest(manifest_name: str = "features_manifest.json") -> None:
    """Commit and push only the exploratory feature manifest; tensors stay on HF."""
    if manifest_name not in {"features_manifest.json", "features_manifest_neutral.json"}:
        raise ValueError("Unsupported exploratory manifest name")
    relative = f"results/exploratory/{manifest_name}"
    manifest = CONFIG.root_dir / relative
    if not manifest.is_file():
        raise FileNotFoundError(f"Exploratory feature manifest not found: {manifest}")
    run_cmd(["git", "add", "-f", "--", relative])
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative],
                             cwd=CONFIG.root_dir).returncode != 0
    if not changed:
        print("[Sync] Exploratory feature manifest is unchanged.")
        return
    run_cmd(["git", "commit", "-m", "Add exploratory feature manifest", "--only", "--", relative])
    run_cmd(["git", "push", "origin", "main"])


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
    parser.add_argument("--exploratory-features", action="store_true",
                        help="Upload features/*_resid.npy and *_meta.parquet to the HF tensor dataset")
    parser.add_argument("--exploratory-manifest", action="store_true",
                        help="Commit and push only results/exploratory/features_manifest.json")
    parser.add_argument("--manifest-name", choices=("features_manifest.json", "features_manifest_neutral.json"),
                        default="features_manifest.json")
    parser.add_argument("--legacy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if (args.exploratory_features or args.exploratory_manifest) and not args.legacy:
        parser.error("--exploratory-* is superseded by e1_extract_features.py and e3/e4 --push-hf; pass --legacy to run")

    if args.exploratory_features:
        if not args.hf_repo:
            parser.error("--exploratory-features requires --hf-repo or HF_TENSOR_REPO_ID")
        count = push_exploratory_features(args.hf_repo)
        print(f"[Sync] Uploaded {count} exploratory feature files")
    elif args.exploratory_manifest:
        if args.action != "push":
            parser.error("--exploratory-manifest requires --action push")
        push_exploratory_manifest(args.manifest_name)
    elif args.ensure_tensor:
        ensure_tensor_cached_from_hf(args.ensure_tensor, args.hf_repo)
    elif args.pull_cache == "3":
        ensure_stage3_cache_present(CONFIG.root_dir, args.hf_repo)
    elif args.action == "pull" and args.hf_repo:
        sync_pt_tensors_hf("pull", args.hf_repo)
    elif args.action == "push":
        if args.hf_repo:
            sync_pt_tensors_hf("push", args.hf_repo)
        push_git_artifacts(args.stage)
