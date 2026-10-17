# tests/test_stage0.py
import json
from pathlib import Path

import pytest
import torch

from src.config import CONFIG
from src.model_utils import apply_norm_scaled_steering, apply_orthogonal_projection


def test_master_config_invariants():
    assert CONFIG.n_templates_train + CONFIG.n_templates_val + CONFIG.n_templates_test == CONFIG.n_templates_total
    assert CONFIG.n_templates_train * CONFIG.n_pairs_per_template == CONFIG.n_pairs_train
    assert CONFIG.n_templates_val * CONFIG.n_pairs_per_template == CONFIG.n_pairs_val
    assert CONFIG.n_templates_test * CONFIG.n_pairs_per_template == CONFIG.n_pairs_test
    assert abs(5 * CONFIG.w_popqa_bottom5 + 5 * CONFIG.w_popqa_top5 - 1.0) < 1e-9
    assert len(CONFIG.holm_thresholds) == 5


def test_orthogonal_projection_nullspace_and_idempotence_fp16_and_fp32():
    torch.manual_seed(CONFIG.seed_global)
    for dtype, tol in [(torch.float32, 1e-5), (torch.float16, 1e-2)]:
        x = torch.randn(4, 12, 2048, dtype=dtype)
        r = torch.randn(2048, dtype=torch.float32) * 7.5
        r_hat = r / torch.norm(r, p=2)
        xp = apply_orthogonal_projection(x, r_hat)
        assert xp.dtype == dtype
        dots = (xp.float() * r_hat).sum(-1)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=tol)
        xpp = apply_orthogonal_projection(xp, r_hat)
        assert torch.allclose(xp.float(), xpp.float(), atol=tol)


def test_norm_scaled_steering_units():
    x = torch.zeros(2, 4, 256, dtype=torch.float16)
    r_vec = torch.full((256,), 0.5, dtype=torch.float32)
    out = apply_norm_scaled_steering(x, r_vec, alpha=1.5)
    assert out.dtype == torch.float16
    measured = float(torch.norm(out[0, 0].float(), p=2))
    expected = 1.5 * float(torch.norm(r_vec, p=2))
    assert abs(measured - expected) < 1e-3


def test_zero_vector_steering_is_identity():
    torch.manual_seed(CONFIG.seed_global)
    x = torch.randn(2, 5, 128, dtype=torch.float16)
    out = apply_norm_scaled_steering(x, torch.zeros(128), alpha=1.0)
    assert torch.allclose(x, out, atol=CONFIG.atol_zero, rtol=CONFIG.rtol_zero)


@pytest.mark.integration
def test_gpu_stage0_contract_from_json():
    p = CONFIG.results_dir / "stage0_env.json"
    if not p.exists():
        pytest.skip("results/stage0_env.json not generated yet")
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["passed_all_checks"] is True
    assert rec["all_layer_activations_finite"] is True
    assert len(rec["max_abs_activation_per_layer"]) == rec["n_layers"]
    assert max(rec["max_abs_activation_per_layer"]) < 60000
    assert rec["python_version"] and rec["torch_cuda_version"] and rec["cuda_driver_version"]
    assert rec["k_suffix"] == CONFIG.k_suffix_default
    assert len(rec["suffix_token_ids"]) == rec["k_suffix"]
    assert rec["zero_intervention_max_abs_diff"] <= CONFIG.atol_zero
    assert rec["fp32_logit_cosine_sim"] >= CONFIG.min_fp16_fp32_logit_cos
    assert rec["fp32_logit_mean_kl_nats"] <= CONFIG.max_fp16_fp32_kl
    assert rec["fp32_logit_mean_kl_nats"] >= -1e-9
