#!/usr/bin/env python3
"""Force-add and push selected exploratory outputs, refusing secrets and tensors."""
from __future__ import annotations

import argparse
import fnmatch
import glob
import subprocess
import time
from pathlib import Path

FORBIDDEN = ("*_key.json", "*.npy", "*.npz", "*checkpoint*")


def validate_paths(paths):
    refused = [str(path) for path in paths
               if any(fnmatch.fnmatch(Path(path).name.lower(), pattern.lower())
                      for pattern in FORBIDDEN)]
    if refused:
        raise ValueError("refusing protected output path(s): " + ", ".join(refused))


def _run(args, *, check=True, root=Path.cwd()):
    return subprocess.run(args, check=check, text=True, cwd=root)


def commit_outputs(paths, message, root=Path.cwd(), retries=3):
    root = Path(root)
    validate_paths(paths)
    expanded = []
    for pattern in paths:
        matches = glob.glob(str(root / pattern), recursive=True)
        expanded.extend(Path(path).resolve().relative_to(root.resolve()).as_posix()
                        for path in matches if Path(path).is_file())
    expanded = sorted(set(expanded))
    validate_paths(expanded)
    if not expanded:
        print("No files matched; nothing to commit.")
        return False
    _run(["git", "add", "-f", "--", *expanded], root=root)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *expanded], cwd=root)
    if staged.returncode == 0:
        print("No staged changes; nothing to commit.")
        return False
    if staged.returncode != 1:
        raise RuntimeError("Could not inspect staged changes")
    _run(["git", "commit", "-m", message, "--", *expanded], root=root)
    last_error = None
    for attempt in range(int(retries)):
        pull = subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=root)
        if pull.returncode == 0:
            push = subprocess.run(["git", "push", "origin", "main"], cwd=root)
            if push.returncode == 0:
                return True
            last_error = f"git push failed with status {push.returncode}"
        else:
            last_error = f"git pull --rebase failed with status {pull.returncode}"
            subprocess.run(["git", "rebase", "--abort"], cwd=root)
        if attempt + 1 < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not push after {retries} attempts: {last_error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args(argv)
    commit_outputs(args.paths, args.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
