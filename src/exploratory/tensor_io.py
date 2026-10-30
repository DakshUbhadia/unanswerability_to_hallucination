"""Small checksum-checked transfers for exploratory tensor caches."""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_verified(path, *, repo_path, expected_sha256, repo_id=None):
    """Restore one HF dataset file and verify it before returning its local path."""
    path = Path(path)
    if not path.exists():
        repo_id = repo_id or os.environ.get("HF_TENSOR_REPO_ID")
        if not repo_id:
            raise FileNotFoundError(f"Missing tensor {path}; set HF_TENSOR_REPO_ID to restore it")
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"Tensor checksum mismatch for {path}: expected {expected_sha256}, got {actual}")
    return path


def upload_folder(folder, *, path_in_repo, ignore_patterns=()):
    repo_id = os.environ.get("HF_TENSOR_REPO_ID")
    if not repo_id:
        raise RuntimeError("HF_TENSOR_REPO_ID is required to upload exploratory tensors")
    from huggingface_hub import upload_folder as hf_upload_folder
    return hf_upload_folder(repo_id=repo_id, folder_path=str(folder), path_in_repo=path_in_repo,
                            repo_type="dataset", token=os.environ.get("HF_TOKEN"),
                            ignore_patterns=list(ignore_patterns))
