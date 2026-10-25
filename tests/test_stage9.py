import json
from pathlib import Path

import numpy as np
import pytest

from src.capability_eval import load_stratified_mmlu_questions, load_wikitext2_passages, run_stage7_capability_evaluation
from src.config import CFG
from src.eval_common import atomic_write_json, slice_vectors_at_star
from src.plotting_and_report import README_BEGIN, assess_stage10, build_stage9_executive_summary, render_results_markdown, run_stage9, verify_stage9_outputs
from src.popqa_uncertainty import analyze_stage8
from src.test_evaluation import analyze_stage6, load_and_validate_test_and_b_ctrl_datasets
@pytest.fixture(scope="module", autouse=True)
def _mock_stage6_judge():
    import src.test_evaluation as evaluation
    from src.abstention_classifier import regex_label
    def classify(items, offline=False, **kwargs):
        out=[]
        for item in items:
            label=regex_label(item["response"])
            out.append({**item,"regex_label":label,"judge_label":label,"final_label":label,
                        "agree":True,"classifier_final":True,"classifier_backend":"mock",
                        "judge_model_id":"mock","judge_prompt_version":"v1","judge_prompt_sha256":"a"*64})
        return out
    monkeypatch=pytest.MonkeyPatch()
    monkeypatch.setattr(evaluation,"classify_texts",classify)
    yield
    monkeypatch.undo()

from tests._tiny_model import StubTokenizer, tiny_contract, tiny_directions, tiny_model, tiny_selection
from tests.test_stage10 import synthetic_stage10
from tests.test_stage6 import _gens, _synthetic_cond
from tests.test_stage8 import _feats, _stage6_stub


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("s9")
    pairs, bctrl = load_and_validate_test_and_b_ctrl_datasets()
    cond = _synthetic_cond()
    s6 = {"stage": 6, "model_name": "tiny", "l_star": 1, "i_star": -1, "pos_idx": 4, "cos_concept_pre_ortho_at_star": 0.1,
          **analyze_stage6(cond, _gens(cond), pairs, bctrl, n_boot=100)}
    tok = StubTokenizer()
    rows = [" ".join([f"w{i}{j}" for j in range(70)]) for i in range(2)]
    mmlu = [{"question": f"Q{i}?", "subject": f"s{i % 2}", "choices": list("wxyz"), "answer": i % 4, "hf_row_idx": i} for i in range(4)]
    s7 = run_stage7_capability_evaluation(tiny_model(), tok, tiny_contract(tok), tiny_directions(), tiny_selection(), 4, None,
                                          wikitext_passages=load_wikitext2_passages(tok, 2, 32, 64, rows),
                                          mmlu_questions=load_stratified_mmlu_questions(4, 1, mmlu), wikitext_batch_size=2,
                                          kl_chunk=16)
    rng = np.random.default_rng(3); n = 40
    feats = {"popqa_wrong": _feats(n, 3, 32, 1.0, rng), "popqa_correct": _feats(n, 3, 32, 0, rng),
             "popqa_abstain": _feats(10, 3, 32, 2, rng), "test_unans": _feats(96, 3, 32, 2, rng),
             "test_ans": _feats(96, 3, 32, 0, rng), "b_ctrl": _feats(48, 3, 32, 0, rng)}
    groups = {"wrong": [{"popqa_id": f"w{i}", "matched_pair_id": f"m{i}", "s_pop": 1.0, "input_ids_sha256": "a"*64, "label_source": "canonical_v1_batch1_judged"} for i in range(n)],
              "correct": [{"popqa_id": f"c{i}", "matched_pair_id": f"m{i}", "s_pop": 1.0, "input_ids_sha256": "a"*64, "label_source": "canonical_v1_batch1_judged"} for i in range(n)], "abstain": [{}] * 10}
    s8 = {"stage": 8, "l_star": 1, "i_star": -1, **analyze_stage8(groups, feats, slice_vectors_at_star(tiny_directions(), 1, 4),
                                                                  s6, 1, n_boot=100)}
    s5 = {"l_star": 1, "i_star": -1, "filter_relaxed": False, "r_norm_at_star": 1.0}
    paths = {k: root / f"{k}.json" for k in ("s5", "s6", "s7", "s8", "s10")}
    for k, v in (("s5", s5), ("s6", s6), ("s7", s7), ("s8", s8), ("s10", synthetic_stage10())):
        atomic_write_json(paths[k], v)
    tables = root / "tables"; tables.mkdir()
    (tables / "table2_sensitivity_grid.md").write_text("| tau_kl |\n|---|\n| 0.15 |\n")
    (tables / "table2_sensitivity_grid.csv").write_text("tau_kl\n0.15\n")
    readme = root / "README.md"; readme.write_text("# Project\n\nIntro text.\n")
    return root, paths, tables, readme


def test_generate_all_figures_and_readme_from_synthetic_artifacts(artifacts):
    root, paths, tables, readme = artifacts
    kwargs = dict(stage5_path=paths["s5"], stage6_path=paths["s6"], stage7_path=paths["s7"], stage8_path=paths["s8"],
                  stage10_path=paths["s10"], figures_dir=root / "figures", summary_path=root / "s9.json",
                  readme_path=readme, tables_dir=tables)
    summary = run_stage9(**kwargs)
    run_stage9(**kwargs)  # idempotent README block
    d = verify_stage9_outputs(root / "s9.json", root / "figures", readme, tables)
    text = readme.read_text()
    assert text.startswith("# Project") and text.count(README_BEGIN) == 1 and "| test_id |" in text
    assert d["stage10_ran"] and set(summary["confirmed"]) == {"H1", "H2", "H3", "H4", "H5"}
    assert "Figure 2B" in text


def test_stage9_without_stage7_and_stage10(artifacts, tmp_path):
    root, paths, tables, readme = artifacts
    run_stage9(paths["s5"], paths["s6"], tmp_path / "none7.json", paths["s8"], tmp_path / "none10.json",
               tmp_path / "figs", tmp_path / "s9.json", readme, tables)
    d = verify_stage9_outputs(tmp_path / "s9.json", tmp_path / "figs", readme, tables)
    assert d["capability"] is None and not d["stage10_ran"]


@pytest.mark.integration
def test_stage9_real_artifacts_verified():
    if not CFG.stage9_out_path.exists():
        pytest.skip("Stage 9 runtime artifact is not present")
    verify_stage9_outputs()



def test_assess_stage10_committed_result():
    d=json.loads(CFG.stage10_out_path.read_text(encoding="utf-8")); a=assess_stage10(d)
    assert a["dominated_at_all_comparable_points"] is True
    assert [round(r["alpha"],2) for r in a["comparable_points"]]==[0.0,0.25]
    assert a["largest_gap_points"]==pytest.approx(14.0,abs=.2) and a["largest_gap_alpha"]==.25
    assert a["learned_max_gain"]==pytest.approx(.238095,abs=1e-6)
    assert a["concept_max_gain"]==pytest.approx(.367347,abs=1e-6)

def test_render_committed_results_has_integrity_disclosures():
    read=lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
    s=build_stage9_executive_summary(read(CFG.stage5_out_path),read(CFG.stage6_out_path),read(CFG.stage7_out_path),read(CFG.stage8_out_path),read(CFG.stage10_out_path))
    text=render_results_markdown(s,"","","")
    assert "Mixed_Not_Supported_Discriminating_Tests_Disagree" in text and "4/20" in text and "negative result" in text
    lines=[x for x in text.splitlines() if "Hypothesis_B_Internal_Uncertainty_Detector" in x]
    assert len(lines)==1 and "Under the original v4 aggregation rule" in lines[0]
