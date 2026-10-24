import json

import numpy as np
import pytest
import torch

from src.config import CFG
from src.eval_common import read_json, slice_vectors_at_star
from src.popqa_uncertainty import (analyze_stage8, compute_binary_auroc, compute_paired_bootstrap_auroc_comparison,
                                   extract_all_layer_projections_and_confidence, load_popqa_groups,
                                   run_stage8_popqa_uncertainty, save_stage8_artifacts, verify_stage8_outputs)
from src.test_evaluation import load_and_validate_test_and_b_ctrl_datasets
from tests._tiny_model import StubTokenizer, tiny_contract, tiny_directions, tiny_model, tiny_selection


def test_compute_binary_auroc_exact_and_ties():
    assert compute_binary_auroc([3, 4], [1, 2]) == 1.0
    assert compute_binary_auroc([1, 2], [3, 4]) == 0.0
    assert compute_binary_auroc([1, 1], [1, 1]) == 0.5
    assert compute_binary_auroc([1, 3], [2, 2]) == 0.5
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); a, b = rng.normal(1, 1, 40), rng.normal(0, 1, 30)
    assert compute_binary_auroc(a, b) == pytest.approx(roc_auc_score([1] * 40 + [0] * 30, np.r_[a, b]))


def test_paired_bootstrap_auroc_comparison_h4_and_h5():
    rng = np.random.default_rng(1); n = 150
    pos = {"proj_learned": rng.normal(1, 1, n), "neg_maxp": rng.normal(0, 1, n)}
    neg = {"proj_learned": rng.normal(0, 1, n), "neg_maxp": rng.normal(0, 1, n)}
    out = compute_paired_bootstrap_auroc_comparison(pos, neg, n_boot=300, seed=3)
    assert out == compute_paired_bootstrap_auroc_comparison(pos, neg, n_boot=300, seed=3)
    a = out["auroc"]["proj_learned"]
    assert a["ci95_low"] < a["point"] < a["ci95_high"] and a["point"] > 0.7
    d = out["delta"]["neg_maxp"]
    assert d["point"] > 0.15 and d["p_boot_one_sided"] == pytest.approx(1 / 301)


def test_load_popqa_groups_real_repo_data_matches_stage2():
    g = load_popqa_groups(stage2_record=read_json(CFG.stage2_out_path))
    assert len(g["wrong"]) == len(g["correct"]) == len(g["pair_ids"]) == 147
    assert all(w["matched_pair_id"] == c["matched_pair_id"] for w, c in zip(g["wrong"], g["correct"]))
    assert len(g["abstain"]) == 517


def _stage6_stub(p1=1e-6, p2=1e-6, p3=1e-4, rank1=True, bctrl_ok=True):
    t = lambda p: {"p_wilcoxon": p, "passes_mc_rank_1": rank1, "concept_confound_flag": False}
    return {"primary_tests": {"H1": t(p1), "H2": t(p2), "H3": {"p_wilcoxon": p3, "b_ctrl_abstention_ok": bctrl_ok}},
            "logit_gen_discrepancy": False}


def _feats(n, L, d, shift, rng):
    acts = rng.normal(0, 1, (n, d))
    return {"proj": rng.normal(shift, 1, (n, L)), "maxp": rng.uniform(.2, .9, n), "entropy": rng.uniform(0, 3, n),
            "log_odds": rng.normal(-3, 1, n), "acts_star": acts}


@pytest.mark.parametrize("shift,expected", [(1.2, "Hypothesis_B_Internal_Uncertainty_Detector"),
                                            (0.0, "Hypothesis_A_Contextual_Insufficiency_Detector")])
def test_hypothesis_classification_branch_a_vs_branch_b(shift, expected):
    rng = np.random.default_rng(2); L, d, n = 3, 32, 60
    feats = {"popqa_wrong": _feats(n, L, d, shift, rng), "popqa_correct": _feats(n, L, d, 0.0, rng),
             "popqa_abstain": _feats(20, L, d, 2.0, rng), "test_unans": _feats(16, L, d, 2, rng),
             "test_ans": _feats(16, L, d, 0, rng), "b_ctrl": _feats(8, L, d, 0, rng)}
    groups = {"wrong": [{"popqa_id": f"w{i}", "matched_pair_id": f"m_{i:03d}", "s_pop": 1.0} for i in range(n)],
              "correct": [{"popqa_id": f"c{i}", "matched_pair_id": f"m_{i:03d}", "s_pop": 1.0} for i in range(n)],
              "abstain": [{}] * 20}
    vectors = slice_vectors_at_star(tiny_directions(), 1, 4)
    out = analyze_stage8(groups, feats, vectors, _stage6_stub(), 1, n_boot=200)
    assert out["interpretation"]["overall"] == expected
    holm = out["holm_bonferroni_table"]
    assert sorted(r["test_id"] for r in holm) == ["H1", "H2", "H3", "H4", "H5"]
    assert [r["raw_p_value"] for r in holm] == sorted(r["raw_p_value"] for r in holm)
    assert len(out["layerwise_auroc_exploratory"]) == L and len(out["h4"]["controls"]["random"]["aurocs"]) == 99
    bad = analyze_stage8(groups, feats, vectors, _stage6_stub(bctrl_ok=False), 1, n_boot=50)
    assert bad["interpretation"]["overall"].startswith("Inconclusive")


def test_stage8_runner_and_exports_on_tiny_model(tmp_path):
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    dirs, sel = tiny_directions(), tiny_selection()
    pairs, bctrl = load_and_validate_test_and_b_ctrl_datasets()
    pairs, bctrl = pairs[:8], [r for r in bctrl if r["pair_id"] in {p["pair_id"] for p in pairs}] or bctrl[:4]
    q = lambda i: {"popqa_id": f"popqa_{i:04d}", "question": f"Where is place {i}?", "s_pop": 10.0 + i}
    groups = {"pair_ids": [f"m_{i:03d}" for i in range(6)],
              "wrong": [{**q(i), "matched_pair_id": f"m_{i:03d}", "baseline_partition": "wrong"} for i in range(6)],
              "correct": [{**q(10 + i), "matched_pair_id": f"m_{i:03d}", "baseline_partition": "correct"} for i in range(6)],
              "abstain": [q(30 + i) for i in range(4)]}
    units = dirs["unit_directions"][:, sel["pos_idx"], :]
    proj = {k: extract_all_layer_projections_and_confidence(model, tok, prompts, contract["system_prompt_used"],
                                                            contract["suffix_token_ids"], units, contract["unknown_token_ids"],
                                                            True, 4, sel["i_star"])["proj"][:, sel["l_star"]].tolist()
            for k, prompts in (("test_unans", [p["unans_prompt"] for p in pairs]),
                               ("test_ans", [p["ans_prompt"] for p in pairs]), ("b_ctrl", [r["prompt"] for r in bctrl]))}
    stage6 = {**_stage6_stub(), "projections": proj}
    res = run_stage8_popqa_uncertainty(model, tok, contract, dirs, sel, groups, pairs, bctrl, stage6, batch_size=4)
    out = tmp_path / "s8.json"
    save_stage8_artifacts(res, out, tmp_path / "t3.csv", tmp_path / "t3.md", tmp_path / "t1.csv", tmp_path / "t1.md")
    d = verify_stage8_outputs(out, tmp_path / "t3.csv", tmp_path / "t3.md")
    assert d["n_matched_per_group"] == 6 and d["n_abstain"] == 4
    assert "H4" in (tmp_path / "t1.md").read_text()
    bad = json.loads(out.read_text()); bad["holm_bonferroni_table"] = bad["holm_bonferroni_table"][:4]
    out.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        verify_stage8_outputs(out, tmp_path / "t3.csv", tmp_path / "t3.md")


@pytest.mark.integration
def test_stage8_real_artifacts_verified():
    if not CFG.stage8_out_path.exists():
        pytest.skip("Stage 8 runtime artifact is not present")
    verify_stage8_outputs(stage5_path=CFG.stage5_out_path)
