#!/usr/bin/env python3
"""Verify every archived file against the v4.3 SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root=Path(".")):
    root = Path(root)
    archive = root / "results" / "archive_v43"
    manifest = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
    for relative, metadata in manifest["files"].items():
        path = archive / relative
        if not path.is_file():
            raise FileNotFoundError(f"archive file missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["sha256"] or path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"archive mismatch: {relative}")
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(f"verified {verify(args.root)} archived files")
