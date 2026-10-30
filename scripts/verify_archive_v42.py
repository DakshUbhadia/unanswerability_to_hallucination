"""Reproducibility check for an archived snapshot manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def _hash(data):
    return hashlib.sha256(data).hexdigest(), len(data)


def verify(root=Path(".")):
    root = Path(root)
    archive = root / "results/archive_v42"
    data = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
    normalised = []
    for rel, meta in data["files"].items():
        path = archive / rel
        if not path.is_file():
            raise FileNotFoundError(f"archive file missing: {rel}")
        raw = path.read_bytes()
        if _hash(raw) == (meta["sha256"], meta["size_bytes"]):
            continue
        # Historical manifests were made from CRLF working trees. Git checks
        # out text as LF on Linux, so accept only this exact reversible form.
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if _hash(crlf) == (meta["sha256"], meta["size_bytes"]):
            normalised.append(rel)
            continue
        raise ValueError(f"archive mismatch: {rel}")
    return normalised


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    normalised = verify(args.root)
    print(f"verified {len(json.loads((Path(args.root) / 'results/archive_v42/MANIFEST.json').read_text(encoding='utf-8'))['files'])} archived files; EOL-normalised matches: {normalised}")
