import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.capability_eval import (build_wikitext_batches, format_mmlu_prompt, kl_per_passage,
                                 load_stratified_mmlu_questions, load_wikitext2_passages, ratio_bootstrap,
                                 run_stage7_capability_evaluation, save_stage7_artifacts, score_mmlu_logits,
                                 verify_stage7_outputs)
from src.config import CFG
from tests._tiny_model import StubTokenizer, tiny_contract, tiny_directions, tiny_model, tiny_selection


def _mmlu_rows(n_subjects=5, per=6):
    return [{"question": f"Q{s}-{i}?", "subject": f"subj{s}", "choices": ["w", "x", "y", "z"], "answer": i % 4,
             "hf_row_idx": s * per + i} for s in range(n_subjects) for i in range(per)]


def test_mmlu_4way_accuracy_and_false_abstention_math():
    logits = torch.full((3, 10), -10.0)
    option_ids = [[0, 4], [1, 5], [2, 6], [3, 7]]
    logits[0, 4] = 5.0           # predicts A via the leading-space id
    logits[1, 2] = 5.0           # predicts C
    logits[2, 9] = 8.0; logits[2, 1] = 2.0  # predicts B but Unknown (id 9) dominates
    out = score_mmlu_logits(logits, [0, 1, 1], option_ids, [9])
    assert out["correct"].tolist() == [1.0, 0.0, 1.0]
    assert out["false_abstention"].tolist() == [0.0, 0.0, 1.0]


def test_stratified_mmlu_sampling_is_deterministic_and_balanced():
    rows = _mmlu_rows()
    a = load_stratified_mmlu_questions(12, 5, rows)
    b = load_stratified_mmlu_questions(12, 5, rows)
    assert a == b and len(a) == 12
    counts = sorted(sum(r["subject"] == s for r in a) for s in {r["subject"] for r in rows})
    assert counts == [2, 2, 2, 3, 3]
    assert format_mmlu_prompt(a[0]).endswith("Answer with a single letter (A, B, C, or D):")


def test_wikitext_kl_non_negativity_and_zero_identity():
    tok = StubTokenizer()
    rows = ["= Header =", "short line", " ".join(["word"] * 70), " ".join(["other"] * 80)]
    passages = load_wikitext2_passages(tok, 2, 40, 64, rows)
    assert len(passages) == 2 and all(len(p) == 40 for p in passages)
    batch = build_wikitext_batches([passages[0], passages[1][:20]], 0, 8)[0]
    assert batch["kl_mask"].sum().item() == 39 + 19
    logits = torch.randn(2, 39, 50)
    base = [F.log_softmax(logits[:, s:s + 16], -1) for s in range(0, 39, 16)]
    same, n = kl_per_passage(base, logits, batch["kl_mask"], 16)
    assert torch.allclose(same, torch.zeros(2, dtype=torch.float64), atol=1e-6) and n.tolist() == [39.0, 19.0]
    diff, _ = kl_per_passage(base, logits + torch.randn_like(logits), batch["kl_mask"], 16)
    assert (diff > 0).all()
    lo, hi = ratio_bootstrap(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), 200, 1)
    assert 1.0 <= lo <= hi <= 3.0


def test_stage7_runner_end_to_end_and_table1_enrichment(tmp_path):
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    rows = [" ".join([f"tok{i}{j}" for j in range(70)]) for i in range(4)]
    passages = load_wikitext2_passages(tok, 4, 48, 64, rows)
    questions = load_stratified_mmlu_questions(6, CFG.seed_mmlu, _mmlu_rows(3, 4))
    ckpt = tmp_path / "ck.pt"
    result = run_stage7_capability_evaluation(model, tok, contract, tiny_directions(), tiny_selection(), 4, ckpt,
                                              wikitext_passages=passages, mmlu_questions=questions,
                                              wikitext_batch_size=2, source_stage4_sha256="a" * 64, kl_chunk=16)
    assert ckpt.exists()
    out = tmp_path / "s7.json"
    save_stage7_artifacts(result, out, tmp_path / "t1.csv", tmp_path / "t1.md", ckpt)
    assert not ckpt.exists()
    d = verify_stage7_outputs(out)
    abl = d["controls_wikitext_kl"]["ablate_all"]
    assert len(abl["random"]) == 99 and min(abl["random"]) >= 0
    assert d["learned_direction_capability"][0]["wikitext_kl"] == 0.0
    assert all(r["wikitext_kl"] > 0 for r in d["learned_direction_capability"][1:])
    bad = json.loads(out.read_text()); bad["controls_wikitext_kl"]["ablate_all"]["random"] = [0.1]
    out.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        verify_stage7_outputs(out)


@pytest.mark.integration
def test_stage7_real_artifacts_verified():
    if not CFG.stage7_out_path.exists():
        pytest.skip("Stage 7 runtime artifact is not present")
    verify_stage7_outputs(stage5_path=CFG.stage5_out_path)
