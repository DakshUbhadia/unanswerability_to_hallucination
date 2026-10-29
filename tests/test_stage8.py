import json

import numpy as np
import pytest
import torch

from src.config import CFG
from src.eval_common import read_json, slice_vectors_at_star
from src.popqa_uncertainty import (VERDICT_A, VERDICT_B, VERDICT_DOMAIN_FAIL, VERDICT_INCONCLUSIVE, VERDICT_MIXED,
                                   analyze_stage8, classify_interpretation, compute_binary_auroc, compute_paired_bootstrap_auroc_comparison,
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



def test_load_popqa_groups_rejects_missing_hash(tmp_path):
    path = tmp_path / "popqa.jsonl"
    path.write_text(json.dumps({"popqa_id":"x","question":"Q?","baseline_partition":"wrong"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lack input_ids_sha256"):
        load_popqa_groups(path)


def test_load_popqa_groups_rejects_unjudged_stage2_labels():
    # Current rows already have canonical tokenization hashes, but Phase C's
    # human judge gate has not supplied the judged Stage 2 label source yet.
    with pytest.raises(ValueError, match="label_source is not canonical_v1_batch1_judged"):
        load_popqa_groups(stage2_record=read_json(CFG.stage2_out_path))


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
    groups = {"wrong": [{"popqa_id": f"w{i}", "matched_pair_id": f"m_{i:03d}", "s_pop": 1.0, "input_ids_sha256": "a"*64, "label_source": "canonical_v1_batch1_judged"} for i in range(n)],
              "correct": [{"popqa_id": f"c{i}", "matched_pair_id": f"m_{i:03d}", "s_pop": 1.0, "input_ids_sha256": "a"*64, "label_source": "canonical_v1_batch1_judged"} for i in range(n)],
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
    from src.tokenization_contract import CONTRACT_VERSION, canonical_input_ids, chat_template_sha256, input_ids_sha256
    template_hash = chat_template_sha256(tok)
    def q(i):
        question = f"Where is place {i}?"
        ids = canonical_input_ids(tok, question, contract["system_prompt_used"], contract["suffix_token_ids"])
        return {"popqa_id": f"popqa_{i:04d}", "question": question, "s_pop": 10.0 + i,
                "input_ids_sha256": input_ids_sha256(ids, contract["model_name"], template_hash),
                "tokenization_contract_version": CONTRACT_VERSION, "label_source": "canonical_v1_batch1_judged"}
    groups = {"pair_ids": [f"m_{i:03d}" for i in range(6)],
              "wrong": [{**q(i), "matched_pair_id": f"m_{i:03d}", "baseline_partition": "wrong", "grade": "wrong"} for i in range(6)],
              "correct": [{**q(10 + i), "matched_pair_id": f"m_{i:03d}", "baseline_partition": "correct", "grade": "correct"} for i in range(6)],
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



def _holm_rows(h1=True,h2=True,h3=True,h4=True,h5=True):
    return [{"test_id":k,"confirmed":v} for k,v in zip(("H1","H2","H3","H4","H5"),(h1,h2,h3,h4,h5))]

@pytest.mark.parametrize("holm,proj,maxp,abstain,layers,want,want_v4,prereq",[
((1,1,1,1,0),.69,.77,.94,[],VERDICT_MIXED,VERDICT_B,True),
((1,1,1,1,1),.85,.70,.94,[],VERDICT_B,VERDICT_B,True),
((1,1,1,0,0),.55,.77,.94,[],VERDICT_A,VERDICT_A,True),
((1,1,1,0,0),.55,.60,.94,[{"layer":5,"auroc":.65}],VERDICT_MIXED,VERDICT_A,True),
((1,1,0,1,1),.85,.70,.94,[],VERDICT_INCONCLUSIVE,VERDICT_INCONCLUSIVE,False),
((1,1,1,0,0),.52,.53,.53,[],VERDICT_DOMAIN_FAIL,VERDICT_A,True)])
def test_classify_interpretation_decision_table(holm,proj,maxp,abstain,layers,want,want_v4,prereq):
    s6={"primary_tests":{"H1":{"concept_confound_flag":False},"H2":{"concept_confound_flag":False}},"logit_gen_discrepancy":False}
    out=classify_interpretation(_holm_rows(*holm),s6,{"auroc_proj":proj},{"auroc_maxp":maxp},layers,11,abstain)
    assert out["overall"]==want and out["overall_as_registered_v4"]==want_v4
    assert out["rule_version"]=="v4.1-posthoc" and out["prerequisites_met"] is prereq
    assert out["verdict_explanation"]
    if prereq: assert "H4" in out["verdict_explanation"] and "H5" in out["verdict_explanation"]

def test_committed_stage8_interpretation_is_mixed():
    d=read_json(CFG.stage8_out_path); s6=read_json(CFG.stage6_out_path)
    out=classify_interpretation(d["holm_bonferroni_table"],s6,d["h4"],d["h5"],d["layerwise_auroc_exploratory"],d["l_star"],d["auroc_abstain_vs_correct"])
    assert out["overall"]==VERDICT_MIXED and out["overall_as_registered_v4"]==VERDICT_B
    assert d["interpretation"]==out


def test_stage8_hash_mismatch_reports_popqa_ids():
    from src.popqa_uncertainty import _verify_popqa_input_identity
    tok = StubTokenizer()
    row = {"popqa_id":"mismatch-7","question":"A canonical question?","input_ids_sha256":"0"*64}
    with pytest.raises(ValueError, match="mismatch-7"):
        _verify_popqa_input_identity(tok,[row],tiny_contract(tok)["system_prompt_used"],"tiny-llama",tiny_contract(tok)["suffix_token_ids"])
