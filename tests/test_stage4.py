import json

import pytest
import torch

from src.activation_cache import save_stage3_artifacts
from src.direction_extraction import (compute_candidate_abstention_directions,
                                      compute_orthogonalized_concept_control,
                                      generate_monte_carlo_random_controls,
                                      generate_shuffled_label_controls,
                                      get_norm_scaled_random_controls,
                                      save_stage4_artifacts, verify_stage4_outputs)
from src.config import CFG


def _cache(n=20, layers=3, k=5, d=16):
    torch.manual_seed(31)
    a = torch.randn(n, layers, k, d) + torch.arange(n).view(n, 1, 1, 1) * .02
    u = a + 1.5 + torch.randn_like(a) * .2
    families = ["A"] * (n // 2) + ["B"] * (n // 2)
    cache = {"train_ans": a.double().mean(0).float(), "train_unans": u.double().mean(0).float(),
             "train_pair_acts_ans": a, "train_pair_acts_unans": u,
             "pair_ids": [f"pair_{i:04d}" for i in range(n)],
             "template_ids": [f"template_{i}" for i in range(n)], "families": families,
             "model_name": "synthetic", "system_prompt_used": "synthetic", "n_layers": layers,
             "k_suffix": k, "d_model": d, "suffix_indices": list(range(-k, 0)),
             "suffix_token_ids": list(range(k))}
    return cache


def _stage3_manifest(cache):
    n, layers, k, d = cache["train_pair_acts_ans"].shape
    diff = torch.linalg.vector_norm(cache["train_unans"] - cache["train_ans"], dim=-1)
    return {"stage": 3, "model_name": "synthetic", "runtime_dtype": "float32", "cache_dtype": "float32",
            "n_train_pairs": n, "n_family_a_pairs": n//2, "n_family_b_pairs": n//2,
            "n_layers": layers, "k_suffix": k, "d_model": d, "suffix_indices": list(range(-k,0)),
            "suffix_token_ids": list(range(k)), "train_ans_shape": [layers,k,d],
            "train_pair_acts_shape": [n,layers,k,d], "batch_vs_single_min_cosine": 1.0,
            "batch_vs_single_max_rel_l2_diff": 0.0, "batch_vs_single_max_abs_diff": 0.0,
            "batch_vs_single_allclose_passed": True, "batch_vs_single_allclose_mismatch_count": 0,
            "min_prompt_std_ans": float(cache["train_pair_acts_ans"].std(0).min()),
            "min_prompt_std_unans": float(cache["train_pair_acts_unans"].std(0).min()),
            "min_diff_in_means_norm": float(diff.min()), "max_diff_in_means_norm": float(diff.max()),
            "elapsed_sec": .1, "passed_acceptance_gate": True}


def _directions(cache):
    from src.direction_extraction import extract_all_directions_and_controls
    return extract_all_directions_and_controls(cache, pairs_path=None)


def test_candidate_abstention_directions_math_and_unit_norms():
    a, u = torch.randn(4,5,32), torch.randn(4,5,32) + 2
    diff, unit, norms = compute_candidate_abstention_directions(a, u)
    assert torch.allclose(diff, u-a, atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(unit, dim=-1), torch.ones(4,5), atol=1e-6)
    assert torch.allclose(unit * norms.unsqueeze(-1), diff, atol=1e-5)


def test_monte_carlo_random_controls_determinism_and_norm_matching():
    a = generate_monte_carlo_random_controls(99,4,5,32)
    b = generate_monte_carlo_random_controls(99,4,5,32)
    assert torch.equal(a,b)
    assert torch.allclose(torch.linalg.vector_norm(a,dim=-1),torch.ones(99,4,5),atol=1e-6)
    norms = torch.rand(4,5)+1
    scaled = get_norm_scaled_random_controls(a,norms)
    assert torch.allclose(torch.linalg.vector_norm(scaled,dim=-1),norms.expand(99,-1,-1),atol=1e-6)


def test_shuffled_label_controls_vectorized_matches_explicit_loop():
    cache = _cache(n=20,layers=3,d=16)
    diff, unit, norms, signs = generate_shuffled_label_controls(cache["train_pair_acts_ans"],cache["train_pair_acts_unans"],19)
    for m in range(19):
        loop = torch.zeros_like(diff[m])
        for i in range(20):
            loop += signs[m,i].float() * (cache["train_pair_acts_unans"][i]-cache["train_pair_acts_ans"][i]) / 20
        assert torch.allclose(diff[m],loop,atol=1e-6)


def test_orthogonalized_concept_control_orthogonality_and_pre_cosine():
    torch.manual_seed(8)
    n,l,k,d=16,2,5,32
    unit=torch.randn(l,k,d); unit=unit/torch.linalg.vector_norm(unit,dim=-1,keepdim=True)
    orth=torch.randn_like(unit); orth=orth-(orth*unit).sum(-1,keepdim=True)*unit
    orth=orth/torch.linalg.vector_norm(orth,dim=-1,keepdim=True)
    raw=.65*unit+(1-.65**2)**.5*orth
    shared_noise=torch.randn(8,l,k,d)*.1
    acts=torch.cat([shared_noise+raw.unsqueeze(0)/2,
                    shared_noise-raw.unsqueeze(0)/2],dim=0)
    norms=torch.ones(l,k)*2
    _, cu, scaled, cosine=compute_orthogonalized_concept_control(acts,["A"]*8+["B"]*8,unit,norms)
    assert torch.allclose(cosine,torch.full((l,k),.65),atol=1e-5)
    assert (cu*unit).sum(-1).abs().max()<1e-6
    assert torch.allclose(torch.linalg.vector_norm(scaled,dim=-1),norms,atol=1e-6)


def test_verify_stage4_outputs_accepts_valid_and_rejects_corrupted(tmp_path):
    cache=_cache()
    directions, manifest=_directions(cache)
    manifest["source_stage3_sha256"] = "0" * 64
    dp,mp=tmp_path/"stage4.pt",tmp_path/"stage4.json"
    save_stage4_artifacts(directions,manifest,dp,mp)
    verify_stage4_outputs(dp,mp)
    for mutation in ("orthogonal", "norm", "nan"):
        changed={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in directions.items()}
        if mutation=="orthogonal": changed["r_concept_unit"] += .001*changed["unit_directions"]
        elif mutation=="norm": changed["r_concept_ortho_scaled"][0,0,0] += .01
        else: changed["random_controls_unit"][0,0,0,0]=float("nan")
        badm=dict(manifest)
        save_stage4_artifacts(changed,badm,dp,mp)
        with pytest.raises(AssertionError): verify_stage4_outputs(dp,mp)


def test_stage4_cli_end_to_end_and_skip_on_synthetic_stage3(tmp_path,capsys):
    cache=_cache(n=20)
    cp,sp=tmp_path/"stage3.pt",tmp_path/"stage3.json"
    save_stage3_artifacts(cache,_stage3_manifest(cache),cp,sp)
    from scripts.stage4_extract_directions import main
    args=["--stage3-cache-path",str(cp),"--stage3-manifest-path",str(sp),"--pairs-path",str(tmp_path/"none.jsonl"),
          "--stage0-path",str(tmp_path/"none0.json"),"--output-path",str(tmp_path/"out.pt"),"--manifest-path",str(tmp_path/"out.json")]
    assert main(args)==0
    assert main(args)==0
    assert "[SKIP]" in capsys.readouterr().out


def test_real_repo_stage4_from_cached_stage3_if_available():
    if not CFG.stage3_manifest_path.exists(): pytest.skip("Stage 3 manifest unavailable")
    data=json.loads(CFG.stage3_manifest_path.read_text())
    assert data["stage"]==3 and data["passed_acceptance_gate"] is True


@pytest.mark.integration
def test_stage4_real_artifacts_verified():
    if not CFG.stage4_manifest_path.exists(): pytest.skip("Stage 4 manifest unavailable")
    manifest=json.loads(CFG.stage4_manifest_path.read_text())
    assert manifest["passed_acceptance_gate"] is True
    assert manifest["max_post_ortho_abs_dot"]<1e-5
    assert manifest["max_concept_norm_diff"]<1e-5 and manifest["max_random_norm_diff"]<1e-5
    if CFG.stage4_out_path.exists(): verify_stage4_outputs()
