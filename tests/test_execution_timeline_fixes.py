import hashlib
import json
import subprocess
import sys
import types

import pytest


@pytest.mark.parametrize("archive_name, verifier", [
    ("archive_v42", "scripts.verify_archive_v42"),
    ("archive_v43", "scripts.verify_archive_v43"),
])
def test_archive_verifier_accepts_manifest_crlf_hashes_and_rejects_edits(tmp_path, archive_name, verifier):
    import importlib
    archive = tmp_path / "results" / archive_name
    archive.mkdir(parents=True)
    lf = b"one\ntwo\n"
    crlf = b"one\r\ntwo\r\n"
    (archive / "sample.txt").write_bytes(lf)
    (archive / "MANIFEST.json").write_text(json.dumps({"files": {
        "sample.txt": {"sha256": hashlib.sha256(crlf).hexdigest(), "size_bytes": len(crlf)}}}), encoding="utf-8")
    verify = importlib.import_module(verifier).verify
    assert verify(tmp_path) == ["sample.txt"]
    (archive / "sample.txt").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="archive mismatch"):
        verify(tmp_path)


def test_build_archive_manifest_hashes_text_as_lf(tmp_path):
    from scripts.build_archive_manifest import build
    (tmp_path / "results").mkdir()
    source = tmp_path / "notes.txt"
    source.write_bytes(b"one\r\ntwo\r\n")
    # git rev-parse is consulted for provenance; this test needs no git repo.
    import subprocess
    original = subprocess.run
    def fake_run(*args, **kwargs):
        if args and args[0] == ["git", "rev-parse", "HEAD"]:
            return types.SimpleNamespace(stdout="test-commit\n")
        return original(*args, **kwargs)
    import scripts.build_archive_manifest as module
    module.subprocess.run = fake_run
    try:
        build(tmp_path, "archive_test", ["notes.txt"])
    finally:
        module.subprocess.run = original
    manifest = json.loads((tmp_path / "results/archive_test/MANIFEST.json").read_text())
    row = manifest["files"]["notes.txt"]
    assert row == {"sha256": hashlib.sha256(b"one\ntwo\n").hexdigest(), "size_bytes": 8}
    assert (tmp_path / "results/archive_test/notes.txt").read_bytes() == b"one\ntwo\n"


@pytest.mark.parametrize("name", ["results/exploratory/a_key.json", "x.npy", "x.npz", "e3_checkpoint.pt"])
def test_commit_outputs_refuses_protected_paths(name):
    from scripts.exploratory.commit_outputs import validate_paths
    with pytest.raises(ValueError, match="refusing protected"):
        validate_paths([name])


def test_commit_outputs_refuses_protected_glob_even_when_unmatched(tmp_path):
    from scripts.exploratory.commit_outputs import commit_outputs
    with pytest.raises(ValueError, match="refusing protected"):
        commit_outputs(["missing.npy"], "must refuse", root=tmp_path)


def test_commit_outputs_adds_selected_file_and_uses_rebase_push_retry(tmp_path, monkeypatch):
    import subprocess
    import scripts.exploratory.commit_outputs as commit
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    output = tmp_path / "results/exploratory/result.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n")
    original = commit.subprocess.run
    network_calls = []
    def fake_run(args, **kwargs):
        if args[:3] == ["git", "pull", "--rebase"] or args[:2] == ["git", "push"]:
            network_calls.append(args)
            return subprocess.CompletedProcess(args, 0)
        return original(args, **kwargs)
    monkeypatch.setattr(commit.subprocess, "run", fake_run)
    assert commit.commit_outputs(["results/exploratory/*.json"], "test outputs", root=tmp_path)
    assert len(network_calls) == 2
    assert subprocess.run(["git", "status", "--porcelain"], cwd=tmp_path, capture_output=True, text=True).stdout == ""


def test_tensor_io_download_checks_sha_and_uploads_dataset(monkeypatch, tmp_path):
    import hashlib
    from src.exploratory import tensor_io
    payload = b"tensor bytes"
    cached = tmp_path / "hub-cache.npy"
    cached.write_bytes(payload)
    calls = {}
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kw: (calls.setdefault("download", kw) and str(cached))
    fake_hub.upload_folder = lambda **kw: calls.setdefault("upload", kw) or "uploaded"
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    target = tmp_path / "features.npy"
    expected = hashlib.sha256(payload).hexdigest()
    tensor_io.download_verified(target, repo_path="exploratory/e3_features/x/a.npy", expected_sha256=expected, repo_id="owner/repo")
    assert target.read_bytes() == payload
    assert calls["download"]["repo_type"] == "dataset"
    monkeypatch.setenv("HF_TENSOR_REPO_ID", "owner/repo")
    tensor_io.upload_folder(tmp_path, path_in_repo="exploratory/e3_features/x", ignore_patterns=["*checkpoint*"])
    assert calls["upload"]["repo_type"] == "dataset"
    assert calls["upload"]["path_in_repo"] == "exploratory/e3_features/x"


def test_e2_dom_vector_sha_must_match_results_manifest(tmp_path):
    import hashlib
    from scripts.exploratory.e2_causal_vocab import verify_dom_vector
    vector = tmp_path / "dom.npy"
    vector.write_bytes(b"vector")
    digest = hashlib.sha256(vector.read_bytes()).hexdigest()
    results = tmp_path / "e2_results.json"
    results.write_text(json.dumps({"parts": {"G": {"dom_vector_sha256": digest}}}))
    assert verify_dom_vector(vector, results) == digest
    vector.write_bytes(b"altered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_dom_vector(vector, results)


def test_release_local_judge_clears_module_cache(monkeypatch):
    import src.abstention_classifier as judge
    monkeypatch.setattr(judge, "_LOCAL_MODEL", object())
    monkeypatch.setattr(judge, "_LOCAL_TOKENIZER", object())
    monkeypatch.setattr(judge, "_LOCAL_SIGNATURE", ("model", "revision", "nf4"))
    judge.release_local_judge()
    assert judge._LOCAL_MODEL is judge._LOCAL_TOKENIZER is judge._LOCAL_SIGNATURE is None


def test_e3_validation_export_scores_cache_offline_before_grader_gate(tmp_path, monkeypatch):
    import scripts.exploratory.e3_export_validation as export
    source = tmp_path / "data" / "exploratory"
    source.mkdir(parents=True)
    rows = [{"popqa_id": "p1", "question": "question", "response": "answer",
             "possible_answers": ["answer"], "obj": "answer"}]
    (source / "popqa_neutral_llama32_3b.jsonl").write_text(json.dumps(rows[0]) + "\n")
    monkeypatch.setattr(export, "CFG", types.SimpleNamespace(data_dir=tmp_path / "data"))
    monkeypatch.setattr(export, "classify_texts", lambda items, offline: [
        {"uid": items[0]["uid"], "final_label": "ANSWER"}])
    monkeypatch.setattr(export, "grade_popqa", lambda *args: {"grade": "correct", "reason": "exact_alias"})
    got = export._rows("llama32_3b")
    assert got[0]["baseline_partition"] == "correct"
    assert got[0]["grade_reason"] == "exact_alias"


def test_legacy_phase_f_entrypoint_fails_closed(tmp_path):
    result = subprocess.run([sys.executable, "scripts/exploratory/e3_neutral_prompt.py"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "superseded by scripts/exploratory/e3_generate_and_capture.py; pass --legacy to run" in result.stderr
