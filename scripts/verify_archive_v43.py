#!/usr/bin/env python3
"""Verify every archived file against the v4.3 SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def _hash(data):
    return hashlib.sha256(data).hexdigest(), len(data)


def verify(root=Path(".")):
    root = Path(root)
    archive = root / "results" / "archive_v43"
    manifest = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
    normalised = []
    for relative, metadata in manifest["files"].items():
        path = archive / relative
        if not path.is_file():
            raise FileNotFoundError(f"archive file missing: {relative}")
        raw = path.read_bytes()
        if _hash(raw) == (metadata["sha256"], metadata["size_bytes"]):
            continue
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if _hash(crlf) == (metadata["sha256"], metadata["size_bytes"]):
            normalised.append(relative)
            continue
        raise ValueError(f"archive mismatch: {relative}")
    return normalised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    matches = verify(args.root)
    print(f"verified {len(json.loads((Path(args.root) / 'results/archive_v43/MANIFEST.json').read_text(encoding='utf-8'))['files'])} archived files; EOL-normalised matches: {matches}")
