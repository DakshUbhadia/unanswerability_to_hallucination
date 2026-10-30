#!/usr/bin/env python3
"""Copy a requested file set into a results archive and write a SHA-256 manifest."""
import argparse
import hashlib
import json
import glob
import subprocess
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_bytes(path):
    """Return stable LF bytes for UTF-8 text, raw bytes for binary files."""
    raw = Path(path).read_bytes()
    if b"\0" in raw:
        return raw
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def build(root, name, files):
    root = Path(root).resolve()
    if not name or Path(name).name != name:
        raise ValueError("--name must be a single directory name")
    if name in {"archive_v4", "archive_v42"}:
        raise ValueError("The v4 and v4.2 archives are immutable")
    archive = root / "results" / name
    if archive.exists() and any(archive.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing archive: {archive}")
    resolved = []
    for item in files:
        matches = [Path(path).resolve() for path in glob.glob(str(root / item))] or [(root / item).resolve()]
        for source in matches:
            if not source.exists():
                raise FileNotFoundError(f"Archive input does not exist: {item}")
            if archive in source.parents or source == archive:
                raise ValueError("Archive inputs cannot come from the target archive")
            candidates = sorted(source.rglob("*") if source.is_dir() else [source])
            resolved.extend(p for p in candidates if p.is_file())
    sources = sorted(set(resolved))
    if not sources:
        raise ValueError("No files selected for the archive")
    archive.mkdir(parents=True, exist_ok=True)
    manifest_files = {}
    for source in sources:
        rel = source.relative_to(root)
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        stable = manifest_bytes(source)
        target.write_bytes(stable)
        manifest_files[rel.as_posix()] = {"sha256": hashlib.sha256(stable).hexdigest(),
                                          "size_bytes": len(stable)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    manifest = {"source_git_commit": commit, "files": manifest_files}
    (archive / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return len(manifest_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--name", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()
    count = build(args.root, args.name, args.files)
    print(f"Archived {count} files to {args.root / 'results' / args.name}")


if __name__ == "__main__":
    main()
