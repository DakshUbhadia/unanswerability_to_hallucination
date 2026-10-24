# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class MasterConfig:
    # --- Paths ---
    root_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = root_dir / "data"
    results_dir: Path = root_dir / "results"
    stage0_out_path: Path = results_dir / "stage0_env.json"
    stage2_out_path: Path = results_dir / "stage2_pilot.json"
    stage3_out_path: Path = results_dir / "stage3_caches.pt"
    stage3_manifest_path: Path = results_dir / "stage3_manifest.json"
    stage4_out_path: Path = results_dir / "stage4_directions.pt"
    stage4_manifest_path: Path = results_dir / "stage4_manifest.json"
    stage5_out_path: Path = results_dir / "stage5_selection.json"
    stage5_checkpoint_path: Path = results_dir / "stage5_checkpoint.pt"
    figures_dir: Path = root_dir / "figures"
    tables_dir: Path = root_dir / "tables"
    table2_csv_path: Path = tables_dir / "table2_sensitivity_grid.csv"
    table2_md_path: Path = tables_dir / "table2_sensitivity_grid.md"
    stage6_out_path: Path = results_dir / "stage6_test_h1_h2_h3.json"
    stage6_gen_path: Path = results_dir / "stage6_generation_samples.jsonl"
    stage6_checkpoint_path: Path = results_dir / "stage6_checkpoint.pt"
    table1_csv_path: Path = tables_dir / "table1_main_results.csv"
    table1_md_path: Path = tables_dir / "table1_main_results.md"
    table4_csv_path: Path = tables_dir / "table4_generation_audit.csv"
    table4_md_path: Path = tables_dir / "table4_generation_audit.md"
    stage7_out_path: Path = results_dir / "stage7_capability.json"
    stage7_checkpoint_path: Path = results_dir / "stage7_checkpoint.pt"
    stage8_out_path: Path = results_dir / "stage8_popqa_core.json"
    table3_csv_path: Path = tables_dir / "table3_popqa_auroc.csv"
    table3_md_path: Path = tables_dir / "table3_popqa_auroc.md"
    stage9_out_path: Path = results_dir / "stage9_summary.json"
    stage10_out_path: Path = results_dir / "stage10_popqa_steering.json"
    stage10_checkpoint_path: Path = results_dir / "stage10_checkpoint.pt"
    fig1_png_path: Path = figures_dir / "fig1_intervention_pareto.png"
    fig2a_png_path: Path = figures_dir / "fig2a_popqa_violin.png"
    figS1_png_path: Path = figures_dir / "figS1_layerwise_auroc.png"
    fig2b_png_path: Path = figures_dir / "fig2b_selective_steering.png"
    readme_path: Path = root_dir / "README.md"

    # --- Models & Precision (Section 2.1 + DEV-001 for Kaggle T4) ---
    primary_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    fallback_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    spec_dtype_str: str = "bfloat16"
    kaggle_fallback_dtype_str: str = "float16"

    # --- Stage 0 float16 vs float32 Fidelity Thresholds (DEV-001) ---
    min_fp16_fp32_logit_cos: float = 0.9990
    max_fp16_fp32_kl: float = 0.0050

    # --- Pinned Package Versions (Section 2.2) ---
    pinned_versions: Dict[str, str] = field(
        default_factory=lambda: {
            "torch": "2.4.1",
            "transformers": "4.46.3",
            "nnsight": "0.3.7",
            "accelerate": "1.1.1",
            "datasets": "3.1.0",
            "scipy": "1.14.1",
            "scikit-learn": "1.5.2",
            "numpy": "1.26.4",
            "pandas": "2.2.3",
            "matplotlib": "3.9.2",
            "seaborn": "0.13.2",
            "pytest": "8.3.3",
            "tqdm": "4.66.6",
        }
    )

    # --- Seeds (Table 3.1) ---
    seed_global: int = 20260401
    seed_split: int = 20260402
    seed_shuffled: int = 20260403
    seed_random: int = 20260404
    seed_mmlu: int = 20260405
    seed_popqa: int = 20260406
    seed_bootstrap: int = 20260407

    # --- Batching, Decoding & Numerical Tolerances (Table 3.1) ---
    batch_size_forward: int = 16
    batch_size_gen: int = 16
    max_new_tokens: int = 16
    eps_norm: float = 1e-8
    stage4_ortho_tol: float = 1e-5
    atol_zero: float = 1e-5
    rtol_zero: float = 1e-4

    # --- Stage 3 batch-vs-single parity tolerances (DEV-005) ---
    stage3_min_batch_single_cosine: float = 0.9995
    stage3_max_batch_single_rel_l2: float = 0.005
    stage3_max_batch_single_abs_fp16: float = 0.0625
    stage3_max_batch_single_abs_fp32: float = 0.005
    stage3_allclose_rtol: float = 0.005
    stage3_allclose_atol: float = 0.005

    # --- Dataset & Quality Gate Parameters (Table 3.1) ---
    n_templates_total: int = 96
    n_templates_train: int = 48
    n_templates_val: int = 24
    n_templates_test: int = 24
    n_templates_cut3: int = 64
    n_pairs_per_template: int = 4
    n_pairs_train: int = 192
    n_pairs_val: int = 96
    n_pairs_test: int = 96
    n_b_ctrl: int = 48
    max_token_diff_pair: int = 2
    max_ngram_phi: float = 0.15

    # --- Pilot Gate Thresholds (Table 3.1 & Branch Rule 1) ---
    pilot_unans_min: float = 0.25
    pilot_unans_max: float = 0.95
    pilot_ans_max: float = 0.15
    pilot_gap_min: float = 0.25
    pilot_max_attempts: int = 2

    # --- Suffix & Negative Controls (Table 3.1) ---
    k_suffix_default: int = 5
    n_ctrl_random: int = 99
    n_ctrl_shuffled: int = 19
    n_ctrl_concept: int = 1

    # --- Validation Selection & Grids (Table 3.1) ---
    alpha_val_select: float = 1.0
    grid_tau_kl: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.25)
    default_tau_kl: float = 0.15
    grid_rho_depth: Tuple[float, ...] = (0.75, 0.85, 0.95)
    default_rho_depth: float = 0.85
    tie_tol_j: float = 1e-6

    # --- Test Intervention Grids (Table 3.1) ---
    grid_alpha_nec: Tuple[float, ...] = (0.0, -0.5, -1.0, -1.5, -2.0)
    grid_alpha_suf: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    ctrl_operating_alphas: Tuple[float, ...] = (-1.0, 1.0)
    n_gen_audit: int = 50
    discrepancy_log_odds: float = 2.0
    discrepancy_rate: float = 0.10
    alpha_primary_neg: float = -1.0
    alpha_primary_pos: float = 1.0
    concept_effect_ratio_max: float = 0.25  # Section 6.3: r_concept > 25% of learned shift = confound
    b_ctrl_abstention_max: float = 0.15
    h4_auroc_min: float = 0.60

    # --- Capability & PopQA Parameters (Table 3.1) ---
    n_wikitext_kl: int = 200
    len_wikitext_tokens: int = 256
    min_wikitext_words: int = 64
    batch_size_wikitext: int = 8  # 256-token full-vocabulary KL at every position fits a 16 GB T4
    n_mmlu: int = 200
    n_popqa_init: int = 1000
    w_popqa_bottom5: float = 0.06
    w_popqa_top5: float = 0.14
    n_popqa_matched_min: int = 100
    n_popqa_step: int = 250
    n_popqa_max: int = 2000
    caliper_sd_default: float = 0.25
    caliper_sd_fallback: float = 0.50
    grid_alpha_stretch: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00)

    # --- Statistical Protocol (Table 3.1 & Section 6.1) ---
    n_bootstrap: int = 1000
    ci_level: float = 0.95
    alpha_fwer: float = 0.05
    holm_thresholds: Tuple[float, ...] = (0.0100, 0.0125, 0.0167, 0.0250, 0.0500)

    # --- Constrained System Prompt (Section 4.4) ---
    system_prompt: str = (
        "You are a precise question-answering assistant. If the answer is known "
        "or determinable from the prompt, reply directly with the concise answer "
        "starting immediately on the first word. If the question cannot be answered "
        "from the provided information or you do not know the answer, reply with "
        "the exact single word: Unknown"
    )
    under_abstention_clause: str = (
        " Never guess or fabricate an answer when information is missing; "
        "you must output Unknown."
    )
    over_abstention_clause: str = (
        " Whenever the prompt contains the required fact, you must state the "
        "answer directly and never output Unknown."
    )


CONFIG = MasterConfig()
CFG = CONFIG
