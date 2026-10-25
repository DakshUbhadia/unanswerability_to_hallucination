import hashlib
import json
from pathlib import Path

def test_archive_v4_manifest_hashes_match():
    root = Path(__file__).resolve().parents[1]
    archive = root / "results" / "archive_v4"
    manifest = json.loads((archive / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source_git_commit"]
    assert manifest["files"]
    for relative, metadata in manifest["files"].items():
        path = archive / relative
        raw = path.read_bytes()
        assert len(raw) == metadata["size_bytes"], relative
        assert hashlib.sha256(raw).hexdigest() == metadata["sha256"], relative
