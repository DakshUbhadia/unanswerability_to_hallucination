import json

import numpy as np
import pytest
import torch

from src.config import CFG
from src.stats_utils import cluster_bootstrap, holm_bonferroni, monte_carlo_p_greater, wilcoxon_greater
from src.test_evaluation import (BASE, analyze_stage6, apply_holm_bonferroni, ckey,
                                 compute_empirical_control_p_values, compute_template_clustered_bootstrap_ci,
                                 generation_conditions, load_and_validate_test_and_b_ctrl_datasets,
                                 run_stage6_test_evaluation, save_stage6_artifacts, verify_stage6_outputs)
from tests._tiny_model import StubTokenizer, tiny_contract, tiny_directions, tiny_model, tiny_selection


def test_load_and_validate_test_and_b_ctrl_real_repo_data():
    pairs, bctrl = load_and_validate_test_and_b_ctrl_datasets()
    assert len(pairs) == 96 and len(bctrl) == 48
    assert len({p["template_id"] for p in pairs}) == 24
    assert {r["template_id"] for r in bctrl} == {p["template_id"] for p in pairs if p["family"] == "B"}


def test_template_clustered_bootstrap_ci_math_and_determinism():
    tids = [f"t{i // 4:02d}" for i in range(96)]
    vals = np.array([i // 4 for i in range(96)], dtype=float)
    a = compute_template_clustered_bootstrap_ci(vals, tids, 500, 7)
    b = compute_template_clustered_bootstrap_ci(vals, tids, 500, 7)
    assert a == b and a["n_clusters"] == 24 and np.isclose(a["mean"], vals.mean())
    assert a["ci95_low"] < a["mean"] < a["ci95_high"]
    const = cluster_bootstrap(np.ones(96), tids, 200, 1)
    assert const["ci95_low"] == const["ci95_high"] == 1.0


def test_empirical_control_p_values_one_sided_bounds():
    rnd = np.linspace(-1, 1, 99); shf = np.linspace(-0.5, 0.5, 19)
    best = compute_empirical_control_p_values(-5.0, rnd, shf, -0.5, "less")
    assert best["p_random"] == pytest.approx(0.01) and best["p_shuffled"] == pytest.approx(0.05)
    assert best["passes_random_gate"] and best["passes_concept_gate"] and best["concept_ratio"] == pytest.approx(0.1)
    worst = compute_empirical_control_p_values(-5.0, rnd, shf, -5.0, "greater")
    assert worst["p_random"] == 1.0 and not worst["passes_concept_gate"]
    assert monte_carlo_p_greater(3.0, [3.0, 1.0]) == pytest.approx(2 / 3)


def test_holm_bonferroni_adjustment_monotonicity():
    rows = holm_bonferroni({"H1": 0.001, "H2": 0.012, "H3": 0.02, "H4": 0.03, "H5": 0.2})
    thr = [r["holm_threshold"] for r in rows]
    assert thr == pytest.approx([0.01, 0.0125, 0.05 / 3, 0.025, 0.05])
    assert [r["passed_holm_bonferroni"] for r in rows] == [True, True, False, False, False]
    assert apply_holm_bonferroni({"a": 0.5, "b": 0.001}) == {"b": 0.025, "a": 0.05}
    assert wilcoxon_greater(np.zeros(5))["p_value"] == 1.0


def _synthetic_cond(n=96, nb=48, effect=3.0, seed=0):
    rng = np.random.default_rng(seed)
    cond = {}
    def put(key, s):
        cond[key] = {"log_odds": s, "p_abs": 1 / (1 + np.exp(-s))}
    base_u, base_a = rng.normal(4, 1, n), rng.normal(-2, 1, n)
    put(BASE["unans"], base_u); put(BASE["ans"], base_a); put(BASE["bctrl"], rng.normal(-7, 1, nb))
    put(ckey("unans", "ablate_all"), base_u - effect); put(ckey("ans", "ablate_all"), base_a)
    for a in CFG.grid_alpha_nec:
        if a: put(ckey("unans", "steer_layer", alpha=a), base_u + effect * a)
    for a in CFG.grid_alpha_suf:
        if a: put(ckey("ans", "steer_layer", alpha=a), base_a + effect * a)
    put(ckey("ans", "ablate_all", "concept"), base_a)
    for tier, m_n in (("random", 99), ("shuffled", 19), ("concept", 1)):
        for m in range(m_n):
            put(ckey("unans", "ablate_all", tier, m), base_u - rng.normal(0, .1, n))
            put(ckey("unans", "steer_layer", tier, m, -1.0), base_u + rng.normal(0, .1, n))
            put(ckey("ans", "steer_layer", tier, m, 1.0), base_a + rng.normal(0, .1, n))
    cond["proj|unans"] = {"proj": rng.normal(2, .5, n)}
    cond["proj|ans"] = {"proj": rng.normal(0, .5, n)}
    cond["proj|bctrl"] = {"proj": rng.normal(0, .5, nb)}
    return cond


def _gens(cond, n=96, nb=48):
    gens = {}
    for _, part, key in generation_conditions():
        s = cond[key]["log_odds"]
        gens[key] = ["Unknown" if v > 0 else "Answer" for v in s]
    return gens


def test_analyze_stage6_detects_strong_effect_and_rank_one():
    pairs, bctrl = load_and_validate_test_and_b_ctrl_datasets()
    cond = _synthetic_cond()
    out = analyze_stage6(cond, _gens(cond), pairs, bctrl, n_boot=200)
    h1, h2, h3 = (out["primary_tests"][k] for k in ("H1", "H2", "H3"))
    assert h1["mean_effect"] == pytest.approx(3.0) and h1["p_mc"] == pytest.approx(0.01)
    assert h2["mean_effect"] == pytest.approx(3.0) and h2["passes_mc_rank_1"]
    assert h1["p_wilcoxon"] < 1e-5 and h3["p_wilcoxon"] < 0.01 and h3["b_ctrl_abstention_ok"]
    assert len(out["necessity_curve"]) == 6 and len(out["sufficiency_curve"]) == 5
    assert out["logit_gen_discrepancy"] is False
    assert len(out["controls_summary"]["ablate_all_unans"]["effects"]["random"]) == 99


@pytest.fixture(scope="module")
def tiny_stage6(tmp_path_factory):
    pairs, bctrl = load_and_validate_test_and_b_ctrl_datasets()
    keep_a = sorted({p["template_id"] for p in pairs if p["family"] == "A"})[:2]
    keep_b = sorted({p["template_id"] for p in pairs if p["family"] == "B"})[:2]
    pairs = [p for p in pairs if p["template_id"] in keep_a + keep_b]
    bctrl = [r for r in bctrl if r["template_id"] in keep_b]
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    ckpt = tmp_path_factory.mktemp("s6") / "ckpt.pt"
    result, gens = run_stage6_test_evaluation(model, tok, contract, tiny_directions(), tiny_selection(), pairs, bctrl,
                                              batch_size=8, checkpoint_path=ckpt, source_stage4_sha256="a" * 64,
                                              source_stage5_sha256="b" * 64, verify_nnsight=False)
    return result, gens, pairs, bctrl, ckpt


def test_stage6_runner_end_to_end_on_tiny_model(tiny_stage6, tmp_path):
    result, gens, pairs, bctrl, ckpt = tiny_stage6
    assert ckpt.exists()
    assert len(gens) == sum(len(bctrl) if part == "bctrl" else len(pairs) for _, part, _ in generation_conditions())
    out = {k: tmp_path / k for k in ("r.json", "g.jsonl", "t1.csv", "t1.md", "t4.csv", "t4.md")}
    save_stage6_artifacts(result, gens, out["r.json"], out["g.jsonl"], out["t1.csv"], out["t1.md"], ckpt,
                          out["t4.csv"], out["t4.md"])
    assert not ckpt.exists()
    d = verify_stage6_outputs(out["r.json"], out["g.jsonl"], out["t1.csv"], out["t1.md"], out["t4.csv"])
    assert d["n_test_pairs"] == 16 and len(d["projections"]["b_ctrl"]) == 8
    # Corruption is rejected.
    bad = json.loads(out["r.json"].read_text()); bad["primary_tests"]["H1"]["p_mc"] = 0.001
    out["r.json"].write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        verify_stage6_outputs(out["r.json"], out["g.jsonl"], out["t1.csv"], out["t1.md"], out["t4.csv"])


def test_native_hooks_zero_vector_identity_and_ablation_on_tiny_model():
    from src.model_utils import forward_with_native_hooks, native_intervention_hooks
    tok = StubTokenizer(); model = tiny_model(); contract = tiny_contract(tok)
    from src.eval_common import build_chat_batches
    batch = build_chat_batches(tok, ["Question one?", "A longer second question?"], "sys",
                               contract["suffix_token_ids"], 2)[0]
    base = forward_with_native_hooks(model, batch, True)
    zero = forward_with_native_hooks(model, batch, True, "steer_layer", torch.zeros(32), 0, 1.0)
    assert torch.equal(base, zero)
    r = torch.randn(32)
    captured = {}
    with native_intervention_hooks(model, True, "ablate_all", r, capture_layers=[0, 2], capture_pos=-1, captured=captured):
        model(**batch, use_cache=False)
    unit = r / r.norm()
    assert set(captured) == {0, 2} and float((captured[2] @ unit).abs().max()) < 1e-4
    assert len(model.model.layers[0]._forward_hooks) == 0


@pytest.mark.integration
def test_stage6_real_artifacts_verified():
    if not CFG.stage6_out_path.exists():
        pytest.skip("Stage 6 runtime artifact is not present")
    verify_stage6_outputs(stage5_path=CFG.stage5_out_path)
