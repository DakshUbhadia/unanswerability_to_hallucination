# tests/test_stage0.py
import pytest
import torch
from src.config import CONFIG
from src.model_utils import (
    apply_orthogonal_projection,
    apply_norm_scaled_steering,
    load_model_and_verify_runtime,
)


def test_master_config_invariants():
    """Verifies internal mathematical consistency of the Master Parameter Table (Table 3.1)."""
    assert CONFIG.n_templates_train + CONFIG.n_templates_val + CONFIG.n_templates_test == CONFIG.n_templates_total
    assert CONFIG.n_templates_train * CONFIG.n_pairs_per_template == CONFIG.n_pairs_train
    assert CONFIG.n_templates_val * CONFIG.n_pairs_per_template == CONFIG.n_pairs_val
    assert CONFIG.n_templates_test * CONFIG.n_pairs_per_template == CONFIG.n_pairs_test
    assert abs(5 * CONFIG.w_popqa_bottom5 + 5 * CONFIG.w_popqa_top5 - 1.0) < 1e-9
    assert len(CONFIG.holm_thresholds) == 5


def test_orthogonal_projection_nullspace_and_idempotence_fp16_and_fp32():
    """Verifies Proj(x) . r_hat == 0 and idempotence in both float32 and Kaggle T4 float16."""
    torch.manual_seed(CONFIG.seed_global)
    for dtype, tol in [(torch.float32, 1e-5), (torch.float16, 1e-2)]:
        x = torch.randn(4, 12, 2048, dtype=dtype)
        r = torch.randn(2048, dtype=torch.float32) * 7.5
        r_hat = r / torch.norm(r, p=2)

        x_proj = apply_orthogonal_projection(x, r_hat)
        assert x_proj.dtype == dtype
        assert torch.isfinite(x_proj).all(), f"Non-finite values in projection for {dtype}"

        dots = torch.matmul(x_proj.float(), r_hat)
        assert torch.allclose(dots, torch.zeros_like(dots), atol=tol)

        x_proj_twice = apply_orthogonal_projection(x_proj, r_hat)
        assert torch.allclose(x_proj.float(), x_proj_twice.float(), atol=tol)


def test_projection_scale_invariance():
    """Verifies Fix 9 note: norm-matching does not affect orthogonal projection ablation."""
    torch.manual_seed(CONFIG.seed_global)
    x = torch.randn(2, 8, 2048, dtype=torch.float16)
    r_unit = torch.randn(2048, dtype=torch.float32)
    r_unit = r_unit / torch.norm(r_unit, p=2)
    r_scaled = r_unit * 42.0

    proj_unit = apply_orthogonal_projection(x, r_unit)
    proj_scaled = apply_orthogonal_projection(x, r_scaled)
    assert torch.allclose(proj_unit, proj_scaled, atol=1e-3, rtol=1e-3)


def test_norm_scaled_steering_units_fp16():
    """Verifies that alpha is strictly in units of ||r_vec||_2 and preserves float16 dtype."""
    x = torch.zeros(2, 5, 256, dtype=torch.float16)
    r_vec = torch.full((256,), 3.0, dtype=torch.float32)  # ||r_vec||_2 = 48.0
    alpha = -1.5
    steered = apply_norm_scaled_steering(x, r_vec, alpha=alpha)
    assert steered.dtype == torch.float16
    assert torch.isfinite(steered).all()
    shift_norm = float(torch.norm(steered[0, 0].float(), p=2).item())
    expected_norm = abs(alpha) * float(torch.norm(r_vec, p=2).item())
    assert abs(shift_norm - expected_norm) < 5e-2


@pytest.mark.integration
def test_kaggle_gpu_stage0_full_contract():
    """
    GPU Integration Test (Run on Kaggle T4): Verifies native layer hook tuple detection,
    'Unknown' single-token encoding, K_suffix == 5, zero NaN/Inf across all 36 layers,
    active-hook zero-intervention logit identity, and float16 vs float32 logit fidelity.
    """
    assert torch.cuda.is_available(), "CUDA GPU required for @pytest.mark.integration"
    _, env_record = load_model_and_verify_runtime(
        model_name=CONFIG.primary_model_name,
        device="cuda:0",
        requested_dtype="float16",
    )
    assert env_record["passed_all_checks"] is True
    assert env_record["all_layer_activations_finite"] is True
    assert env_record["n_layers"] == 36
    assert env_record["d_model"] == 2048
    assert env_record["k_suffix"] == 5
    assert env_record["zero_intervention_max_abs_diff"] <= CONFIG.atol_zero
    assert env_record["fp32_logit_cosine_sim"] >= CONFIG.min_fp16_fp32_logit_cos
    assert env_record["fp32_logit_mean_kl_nats"] <= CONFIG.max_fp16_fp32_kl
