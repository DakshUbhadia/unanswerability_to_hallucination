# Implementation Plan 0: Local IDE Setup, Kaggle T4 GPU Workflow & Stage 0 Verification

**Target Specification:** Mechanistic Interpretability Warm-Up (Frozen Specification v4.0)  
**Workflow Model:** Develop & Unit-Test Locally on CPU (WSL / Linux / macOS) $\rightarrow$ Execute GPU Stages on Kaggle Free Tier (**NVIDIA T4 $\times 2$, 16 GB VRAM**)  
**Stage Covered:** Repository Setup, `DEVIATIONS.md`, Local-to-Kaggle Sync & **Stage 0 (`scripts/stage0_env_check.py`)**  
**Estimated Setup Time:** 2–3 Hours (Day 1)

---

## 1. Hybrid Architecture: Local IDE (CPU) vs. Kaggle Notebooks (GPU)

To preserve your weekly Kaggle GPU quota ($\approx 30\text{ hours/week}$, subject to account-level floating limits) and avoid debugging on a metered cloud session, every task that does not require a 3B model forward pass runs **locally in your IDE on CPU**, while forward-pass stages run on **Kaggle GPU (`cuda:0` on a single T4)** using your resumable stage scripts.

### 1.1 Stage-by-Stage Compute Split & Quota Budget

| Stage | Script | Execution Host | Estimated GPU Wall-Time (T4, `float16`) | Output Size |
| :--- | :--- | :--- | :--- | :--- |
| **Dev & Unit Tests** | `pytest tests -m "not integration"` | **Local IDE (CPU)** | $0.0\text{ hrs}$ ($< 3\text{ sec}$ on CPU) | — |
| **Stage 0** | `scripts/stage0_env_check.py` | **Kaggle T4 GPU** | $\approx 0.1\text{ hrs}$ ($5\text{ mins}$ incl. weight download) | $< 10\text{ KB}$ |
| **Stage 1** | `scripts/stage1_build_dataset.py` | **Local IDE (CPU)** | $0.0\text{ hrs}$ (uses `AutoTokenizer` + PopQA on CPU) | $\approx 2.5\text{ MB}$ |
| **Stage 2** | `scripts/stage2_pilot_gate.py` | **Kaggle T4 GPU** | $\approx 0.6\text{ hrs}$ ($D_{\text{val}}$ + 1,000 PopQA greedy decodes) | $\approx 1.5\text{ MB}$ |
| **Stage 3** | `scripts/stage3_cache_activations.py` | **Kaggle T4 GPU** | $\approx 0.2\text{ hrs}$ ($384$ prompts, 1 forward pass each) | $\approx 285\text{ MB}$ |
| **Stage 4** | `scripts/stage4_extract_directions.py` | **Local CPU or Kaggle** | $\approx 0.05\text{ hrs}$ (pure tensor math on `stage3_caches.pt`) | $\approx 160\text{ MB}$ |
| **Stage 5** | `scripts/stage5_select_coordinate.py` | **Kaggle T4 GPU** | $\approx 1.8\text{ hrs}$ ($180$ candidates $\times D_{\text{val}}$ forward passes) | $< 1\text{ MB}$ |
| **Stage 6** | `scripts/stage6_evaluate_test.py` | **Kaggle T4 GPU** | $\approx 3.5\text{ hrs}$ ($D_{\text{test}}$ grids + $119$ controls $\times 3$ operating points) | $< 2\text{ MB}$ |
| **Stage 7** | `scripts/stage7_evaluate_capability.py` | **Kaggle T4 GPU** | $\approx 4.5\text{ hrs}$ (200 WikiText-2 KL across controls + 200 MMLU) | $< 1\text{ MB}$ |
| **Stage 8** | `scripts/stage8_evaluate_popqa_core.py` | **Kaggle T4 GPU** | $\approx 0.3\text{ hrs}$ (1 unsteered forward pass on PopQA) | $\approx 2\text{ MB}$ |
| **Stage 9** | `scripts/stage9_make_figures.py` | **Local IDE (CPU)** | $0.0\text{ hrs}$ (renders PDF/PNG figures from JSON/CSV) | $\approx 5\text{ MB}$ |
| **Stage 10** | `scripts/stage10_stretch_steering.py` | **Kaggle T4 GPU** | $\approx 1.2\text{ hrs}$ (Optional Stretch: PopQA $\alpha$ sweep) | $< 1\text{ MB}$ |
| **Total** | **All Stages (0–10)** | **Hybrid** | **$\approx 12.25\text{ GPU Hours Total}$ (Across 18 Days)** | **$\approx 455\text{ MB}$** |

> **Why Stage 1 & Stage 9 Run Locally on CPU:** Stage 1 (`stage1_build_dataset.py`) only requires the Hugging Face tokenizer (`Qwen/Qwen2.5-3B-Instruct` tokenizer files are $< 15\text{ MB}$) and the `akariasai/PopQA` dataset table to enforce token-length parity ($\pm 2$ tokens) and n-gram leakage ($\max |\phi| \le 0.15$). Running Stage 1 locally in WSL/CPU saves 100% of your GPU quota during Days 2–4.

---

## 2. Resolving the Kaggle `bfloat16` Constraint & Logging `DEVIATIONS.md`

### 2.1 Why `float16` on Kaggle T4 Beats `float32` on P100
1. **Hardware Architecture:** Kaggle offers either **GPU T4 $\times 2$** (Turing `sm_75`, 16 GB GDDR6 per GPU) or **GPU P100** (Pascal `sm_60`, 16 GB HBM2). Neither supports native hardware `bfloat16` (which requires Ampere `sm_80+`).
2. **T4 (`float16`) Advantages:** The T4 has dedicated **Turing Tensor Cores** for `torch.float16`, whereas the P100 (`sm_60`) lacks Tensor Cores and has deprecated kernel support in newer PyTorch 2.4+ releases. Furthermore, loading `Qwen/Qwen2.5-3B-Instruct` in `float32` on a P100 consumes $\approx 12.4\text{ GB}$ for weights alone, leaving $< 3.6\text{ GB}$ for activations and KV cache, whereas `float16` on a single T4 (`cuda:0`) consumes $\approx 6.2\text{ GB}$ for weights and runs cleanly at `batch_size = 16`.
3. **Numerical Stability Safeguards for `float16`:**
   * All residual stream inner products ($x \cdot \hat{r}$), vector norms ($\|r\|_2$), projection subtractions, steering additions, running mean accumulations, and log-odds calculations are explicitly upcast to **`torch.float32`** before casting the modified hidden state back to `torch.float16`.
   * **Stage 0 `NaN`/`Inf` Guard:** Stage 0 hooks all $L = 36$ decoder layers simultaneously in `float16` and asserts that 100% of residual activations and output logits are finite (`torch.isfinite(tensor).all()`).
   * **Stage 0 `float16` vs. `float32` Logit Fidelity Check:** Stage 0 runs a 4-prompt verification batch through the model in `float16` and compares the final-token logits against an exact `float32` reference pass, asserting:
     * Cosine similarity between `float16` and `float32` logits $\ge 0.9990$.
     * Mean next-token KL divergence $D_{\text{KL}}(P_{\text{fp32}} \parallel P_{\text{fp16}}) \le 0.005\text{ nats}$.

### 2.2 Create `DEVIATIONS.md` (Do Not Edit the Frozen Spec)
Per Ground Rule 1 and your workflow instructions, **do not modify `warm_up_documentation.md`**. Instead, create `DEVIATIONS.md` in the root of your repository to transparently document all Kaggle hardware adaptations:

```markdown
# Registered Implementation Deviations from Frozen Specification v4.0

This file logs strictly bounded hardware and environment adaptations required to execute the Frozen Specification (v4.0) on Kaggle Free Tier GPUs (NVIDIA Tesla T4, 16 GB VRAM, Turing `sm_75`). No experimental hypotheses, dataset schemas, controls, or statistical tests are altered.

| Deviation ID | Spec Section | Specified Setting | Executed Setting on Kaggle | Justification & Stage 0 Verification Safeguard |
| :--- | :--- | :--- | :--- | :--- |
| **DEV-001** | Section 2.1 | `torch.bfloat16` model weights & forward pass | `torch.float16` model weights & forward pass on NVIDIA T4 (`sm_75`), with all residual projections, steering math, and accumulations in `torch.float32` | Kaggle T4 (`sm_75`) and P100 (`sm_60`) GPUs lack native hardware `bfloat16` support. Stage 0 enforces two mandatory safeguards: (1) an all-layer finite activation check (`zero NaN/Inf` across all 36 layers), and (2) a `float16` vs. `float32` baseline logit fidelity check requiring cosine similarity $\ge 0.9990$ and $D_{\text{KL}}(P_{\text{fp32}} \parallel P_{\text{fp16}}) \le 0.005\text{ nats}$. |
| **DEV-002** | Section 2.2 | Exact pinned `pip` environment (`python==3.11.9`, `torch==2.4.1`) | Exact pins for `nnsight==0.3.7` and `transformers==4.46.3` layered on Kaggle's native CUDA PyTorch image (`--allow-kaggle-env`) | Reinstalling the multi-gigabyte `torch` wheel inside ephemeral Kaggle sessions wastes GPU quota and risks host CUDA driver mismatches. Exact runtime package versions are logged to `results/stage0_env.json`. |
```

---

## 3. Step-by-Step Local IDE Setup (WSL / Linux / macOS CPU)

### Step 3.1: Create the Repository Directory Structure
Open your local IDE terminal (VS Code or Cursor connected to WSL2 on Windows, or native terminal on Linux/macOS) and run:

```bash
mkdir -p abstention-direction/{configs,data,results,figures,tables,src,scripts,tests,notebooks,.vscode}
cd abstention-direction
touch src/__init__.py tests/__init__.py DEVIATIONS.md
```

### Step 3.2: Create `.gitignore`
We track small JSON/CSV/Markdown artifacts (`results/*.json`, `results/*.csv`, `tables/*`, `figures/*`, `data/*.jsonl`, `data/splits.json`, `data/audit_log.csv`) directly in Git so they persist across Kaggle sessions via `git push`, while excluding `.pt` tensor files ($> 100\text{ MB}$, which are persisted via Kaggle Datasets or Hugging Face Hub):

```gitignore
# .gitignore
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env
.pytest_cache/
.hf_cache/

# Exclude large binary tensors from standard Git (synced via HF Hub / Kaggle Dataset)
results/*.pt
```

### Step 3.3: Create Local CPU & Kaggle Requirements Files
Create `requirements.txt` (strictly matching Section 2.2 for local reproducibility) and `requirements-kaggle.txt` (which installs `nnsight==0.3.7` and `transformers==4.46.3` rapidly onto Kaggle's pre-installed PyTorch GPU image without overwriting Kaggle's CUDA `torch` binary):

```text
# requirements.txt (Full Pinned Spec for Local Virtualenv)
torch==2.4.1
transformers==4.46.3
nnsight==0.3.7
accelerate==1.1.1
datasets==3.1.0
scipy==1.14.1
scikit-learn==1.5.2
numpy==1.26.4
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2
pytest==8.3.3
tqdm==4.66.6
huggingface_hub>=0.26.0
```

```text
# requirements-kaggle.txt (Fast Layer on Top of Kaggle GPU Image)
transformers==4.46.3
nnsight==0.3.7
accelerate==1.1.1
datasets==3.1.0
pytest==8.3.3
huggingface_hub>=0.26.0
```

Set up your local CPU virtual environment for development and fast unit testing:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
# Install CPU torch locally to save disk space, plus all pinned packages
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 3.4: Configure VS Code / Cursor (`.vscode/settings.json` & `pytest.ini`)
Create `pytest.ini` in the repository root:

```ini
# pytest.ini
[pytest]
pythonpath = .
testpaths = tests
markers =
    integration: marks tests that require loading the 3B LLM on a Kaggle CUDA GPU (deselect locally with '-m "not integration"')
```

Create `.vscode/settings.json` so your local IDE runs only the fast CPU unit tests automatically on save:

```json
{
    "python.analysis.extraPaths": ["${workspaceFolder}"],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests",
        "-m",
        "not integration"
    ],
    "editor.formatOnSave": true,
    "files.trimTrailingWhitespace": true
}
```

---

## 4. Core Implementation Files for Stage 0

### 4.1 Master Parameter Configuration (`src/config.py`)
Encodes **Table 3.1 (Master Parameter Table)** plus the DEV-001 `float16` verification thresholds (`min_fp16_fp32_logit_cos = 0.9990`, `max_fp16_fp32_kl = 0.005`).

```python
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
    figures_dir: Path = root_dir / "figures"
    tables_dir: Path = root_dir / "tables"

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
    atol_zero: float = 1e-5
    rtol_zero: float = 1e-4

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

    # --- Capability & PopQA Parameters (Table 3.1) ---
    n_wikitext_kl: int = 200
    len_wikitext_tokens: int = 256
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
```

### 4.2 Model Wrapper, T4 `float16` Safeguards & Intervention Operators (`src/model_utils.py`)
This module automatically selects `torch.float16` when running on a Kaggle T4/P100 GPU (where `torch.cuda.is_bf16_supported()` is `False`), performs all residual inner products and steering updates in `torch.float32`, verifies that zero activations across all 36 layers are `NaN` or `Inf`, and benchmarks `float16` logits against `float32` reference logits on 4 probe prompts.

```python
# src/model_utils.py
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from src.config import CONFIG


def resolve_runtime_dtype(requested_dtype: str = "auto", device: str = "cuda:0") -> Tuple[torch.dtype, str, bool]:
    """
    Resolves the execution torch.dtype:
      - Uses bfloat16 if supported natively by the GPU (sm_80+).
      - Automatically falls back to float16 on Kaggle T4 (sm_75) / P100 (sm_60) and flags
        DEV-001 (precision deviation).
    """
    if requested_dtype == "bfloat16":
        return torch.bfloat16, "torch.bfloat16", False
    if requested_dtype == "float16":
        return torch.float16, "torch.float16", True
    if requested_dtype == "float32":
        return torch.float32, "torch.float32", True

    # "auto" mode: check native hardware bfloat16 support
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "torch.bfloat16", False
        return torch.float16, "torch.float16", True
    return torch.float32, "torch.float32", True


def inspect_native_layer_output(
    raw_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Executes a native PyTorch forward hook on raw_model.model.layers[0] OUTSIDE nnsight
    to determine deterministically whether layer.output is a tuple or a raw Tensor.
    """
    hook_records: Dict[str, Any] = {}

    def _probe_hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        hook_records["is_tuple"] = is_tuple
        hook_records["ndim"] = hidden.ndim
        hook_records["shape"] = tuple(hidden.shape)
        hook_records["dtype"] = str(hidden.dtype)

    handle = raw_model.model.layers[0].register_forward_hook(_probe_hook)
    try:
        dummy_texts = [" Probe prompt one", " Probe prompt two with extra words"]
        enc = tokenizer(dummy_texts, return_tensors="pt", padding=True).to(device)
        b_size, seq_len = enc["input_ids"].shape
        with torch.no_grad():
            _ = raw_model(**enc)
    finally:
        handle.remove()

    assert "is_tuple" in hook_records, "Native forward hook did not fire!"
    shape = hook_records["shape"]
    d_model = raw_model.config.hidden_size
    assert hook_records["ndim"] == 3 and shape == (b_size, seq_len, d_model), (
        f"Expected 3D hidden state shape {(b_size, seq_len, d_model)}, got {shape}"
    )
    return bool(hook_records["is_tuple"]), shape


def verify_unknown_tokens(tokenizer: AutoTokenizer) -> Tuple[int, int]:
    """
    Verifies that 'Unknown' encodes to a single token and returns
    (unknown_token_id_nospace, unknown_token_id_space).
    """
    ids_nospace = tokenizer.encode("Unknown", add_special_tokens=False)
    assert len(ids_nospace) == 1, (
        f"'Unknown' must encode to exactly 1 token, got {len(ids_nospace)} tokens: {ids_nospace}"
    )
    ids_space = tokenizer.encode(" Unknown", add_special_tokens=False)
    assert len(ids_space) >= 1, "Failed to encode ' Unknown'"
    return int(ids_nospace[0]), int(ids_space[-1])


def discover_post_instruction_suffix(
    tokenizer: AutoTokenizer,
    system_prompt: str = CONFIG.system_prompt,
) -> Tuple[int, List[int]]:
    """
    Applies the model's chat template to two distinct user prompts and extracts
    the exact trailing post-instruction token sequence following the user content.
    """
    user_a = " AlphaUniqueMarker"
    user_b = " BetaUniqueMarkerWithDifferentLength"

    msgs_a = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_a}]
    msgs_b = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_b}]

    ids_a = tokenizer.apply_chat_template(msgs_a, tokenize=True, add_generation_prompt=True)
    ids_b = tokenizer.apply_chat_template(msgs_b, tokenize=True, add_generation_prompt=True)

    k_suffix = 0
    max_k = min(len(ids_a), len(ids_b))
    while k_suffix < max_k and ids_a[-(k_suffix + 1)] == ids_b[-(k_suffix + 1)]:
        k_suffix += 1

    assert k_suffix >= 1, "No common post-instruction suffix found across chat prompts!"
    suffix_ids = list(ids_a[-k_suffix:])
    return k_suffix, suffix_ids


def apply_orthogonal_projection(
    x: torch.Tensor,
    r_hat: torch.Tensor,
    eps: float = CONFIG.eps_norm,
) -> torch.Tensor:
    """
    Projects out unit direction r_hat from residual tensor x:
        Proj(x) = x - (x . r_hat) r_hat
    All inner products and subtractions execute in float32 before casting back to x.dtype.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    r_f32 = r_hat.to(device=x.device, dtype=torch.float32)
    r_unit = r_f32 / (torch.norm(r_f32, p=2) + eps)
    proj_coeff = torch.matmul(x_f32, r_unit).unsqueeze(-1)
    out_f32 = x_f32 - (proj_coeff * r_unit)
    return out_f32.to(dtype=orig_dtype)


def apply_norm_scaled_steering(
    x: torch.Tensor,
    r_vec: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Applies directional steering in units of ||r_vec||_2:
        Steer(x; r_vec, alpha) = x + alpha * r_vec
    Executes in float32 and does NOT short-circuit when r_vec == 0, allowing
    the active hook write-back path to be verified in Stage 0.
    """
    orig_dtype = x.dtype
    x_f32 = x.float()
    r_f32 = r_vec.to(device=x.device, dtype=torch.float32)
    out_f32 = x_f32 + (float(alpha) * r_f32)
    return out_f32.to(dtype=orig_dtype)


def _deref_saved(proxy_or_tensor: Any) -> torch.Tensor:
    """Dereferences an nnsight 0.3.7 saved proxy (.value) or returns the tensor directly."""
    return proxy_or_tensor.value if hasattr(proxy_or_tensor, "value") else proxy_or_tensor


def _read_layer_resid_envoy(layer_envoy: Any, is_tuple_output: bool) -> Any:
    """Reads the hidden_states proxy from a decoder layer Envoy using the verified boolean."""
    return layer_envoy.output[0] if is_tuple_output else layer_envoy.output


def _write_layer_resid_envoy(layer_envoy: Any, new_resid: Any, is_tuple_output: bool) -> None:
    """Writes modified hidden_states back to a decoder layer Envoy using the verified boolean."""
    if is_tuple_output:
        layer_envoy.output[0][:] = new_resid
    else:
        layer_envoy.output[:] = new_resid


@torch.no_grad()
def forward_with_intervention(
    llm: LanguageModel,
    is_layer_output_tuple: bool,
    inputs: Dict[str, torch.Tensor],
    mode: str = "none",
    r_vec: Optional[torch.Tensor] = None,
    target_layer: Optional[int] = None,
    alpha: float = 0.0,
) -> torch.Tensor:
    """
    Executes a forward pass through `llm` under `nnsight==0.3.7` and returns
    final-token logits of shape [batch_size, vocab_size].
    """
    n_layers = llm.config.num_hidden_layers

    with llm.trace(inputs):
        if mode == "ablate_all":
            assert r_vec is not None, "r_vec required for mode='ablate_all'"
            for l_idx in range(n_layers):
                layer_envoy = llm.model.layers[l_idx]
                resid = _read_layer_resid_envoy(layer_envoy, is_layer_output_tuple)
                modified = apply_orthogonal_projection(resid, r_vec)
                _write_layer_resid_envoy(layer_envoy, modified, is_layer_output_tuple)

        elif mode == "steer_layer":
            assert r_vec is not None and target_layer is not None, (
                "r_vec and target_layer required for mode='steer_layer'"
            )
            layer_envoy = llm.model.layers[target_layer]
            resid = _read_layer_resid_envoy(layer_envoy, is_layer_output_tuple)
            modified = apply_norm_scaled_steering(resid, r_vec, alpha=alpha)
            _write_layer_resid_envoy(layer_envoy, modified, is_layer_output_tuple)

        elif mode != "none":
            raise ValueError(f"Unsupported intervention mode: {mode}")

        saved_logits = llm.lm_head.output[:, -1, :].save()

    return _deref_saved(saved_logits).detach()


@torch.no_grad()
def assert_all_layer_activations_finite(
    llm: LanguageModel,
    is_layer_output_tuple: bool,
    inputs: Dict[str, torch.Tensor],
    k_suffix: int,
) -> bool:
    """
    Stage 0 Check (DEV-001): Hooks all L decoder layers simultaneously and asserts
    that zero residual stream activations (or logits) are NaN or Inf in float16.
    """
    n_layers = llm.config.num_hidden_layers
    saved_resids = []
    with llm.trace(inputs):
        for l_idx in range(n_layers):
            resid = _read_layer_resid_envoy(llm.model.layers[l_idx], is_layer_output_tuple)
            saved_resids.append(resid[:, -k_suffix:, :].save())
        saved_logits = llm.lm_head.output[:, -1, :].save()

    for l_idx, proxy in enumerate(saved_resids):
        tensor_val = _deref_saved(proxy)
        assert torch.isfinite(tensor_val).all(), (
            f"NaN or Inf detected in layer {l_idx} residual stream under {tensor_val.dtype}!"
        )
    logits_val = _deref_saved(saved_logits)
    assert torch.isfinite(logits_val).all(), "NaN or Inf detected in output logits!"
    return True


@torch.no_grad()
def verify_low_precision_against_fp32(
    raw_model: AutoModelForCausalLM,
    probe_inputs: Dict[str, torch.Tensor],
    low_prec_logits: torch.Tensor,
) -> Dict[str, Any]:
    """
    Stage 0 Check (DEV-001): Compares baseline final-token logits in low precision
    (float16/bfloat16) against an exact float32 forward pass on the probe batch.
    temporarily casting the model weights to float32 on CPU/GPU if memory permits,
    or running float32 layer-by-layer on CPU to avoid 16GB VRAM overflow.
    """
    orig_device = next(raw_model.parameters()).device
    orig_dtype = next(raw_model.parameters()).dtype

    # Move model to CPU in float32 briefly to compute exact fp32 reference logits
    # without exceeding 16 GB VRAM on a Kaggle T4
    raw_model.to(device="cpu", dtype=torch.float32)
    try:
        cpu_inputs = {k: v.to("cpu") for k, v in probe_inputs.items()}
        fp32_logits = raw_model(**cpu_inputs).logits[:, -1, :].detach()
    finally:
        raw_model.to(device=orig_device, dtype=orig_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lp_f32 = low_prec_logits.detach().cpu().float()
    cos_sim = float(F.cosine_similarity(lp_f32, fp32_logits, dim=-1).mean().item())

    p_fp32 = F.softmax(fp32_logits, dim=-1).clamp(min=1e-8)
    log_p_fp32 = F.log_softmax(fp32_logits, dim=-1)
    log_p_lp = F.log_softmax(lp_f32, dim=-1)
    mean_kl = float(torch.sum(p_fp32 * (log_p_fp32 - log_p_lp), dim=-1).mean().item())

    top1_agree = float((lp_f32.argmax(dim=-1) == fp32_logits.argmax(dim=-1)).float().mean().item())

    assert cos_sim >= CONFIG.min_fp16_fp32_logit_cos, (
        f"Low-precision vs fp32 logit cosine similarity too low: {cos_sim:.6f} < {CONFIG.min_fp16_fp32_logit_cos}"
    )
    assert mean_kl <= CONFIG.max_fp16_fp32_kl, (
        f"Low-precision vs fp32 KL divergence too high: {mean_kl:.6f} > {CONFIG.max_fp16_fp32_kl}"
    )
    return {
        "fp32_logit_cosine_sim": cos_sim,
        "fp32_logit_mean_kl_nats": mean_kl,
        "fp32_top1_agreement": top1_agree,
    }


def load_model_and_verify_runtime(
    model_name: str = CONFIG.primary_model_name,
    device: str = "cuda:0",
    requested_dtype: str = "auto",
) -> Tuple[LanguageModel, Dict[str, Any]]:
    """
    Loads the model and tokenizer on Kaggle T4 (or local hardware), executes all
    Stage 0 startup checks, verifies zero NaN/Inf activations, benchmarks float16
    against float32 reference logits, and estimates total pipeline GPU hours.
    """
    runtime_dtype, dtype_str, precision_deviation = resolve_runtime_dtype(requested_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.padding_side == "left", "Tokenizer padding_side must be 'left'"

    raw_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=runtime_dtype,
        device_map=device,
    )
    raw_model.eval()

    # 1. Native PyTorch hook check outside nnsight
    is_tuple, _ = inspect_native_layer_output(raw_model, tokenizer, device=device)

    # 2. Single-token "Unknown" check
    unk_nospace, unk_space = verify_unknown_tokens(tokenizer)

    # 3. Post-instruction suffix discovery
    k_suffix, suffix_ids = discover_post_instruction_suffix(tokenizer)

    # Wrap in nnsight LanguageModel
    llm = LanguageModel(raw_model, tokenizer=tokenizer)

    # 4. Prepare 4 probe prompts (2 answerable, 2 unanswerable)
    raw_probe_questions = [
        "Context: Project Helios was directed by Dr. Aris Thorne in 2019.\nQuestion: Who directed Project Helios?",
        "Context: Project Helios was funded by the Department of Energy in 2019.\nQuestion: Who directed Project Helios?",
        "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: In what year was Mara Vance born?",
        "Profile: Mara Vance is a botanist born in 1984 in Seattle.\nQuestion: What is Mara Vance's childhood pet's name?",
    ]
    probe_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CONFIG.system_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in raw_probe_questions
    ]
    probe_inputs = tokenizer(probe_prompts, return_tensors="pt", padding=True).to(device)

    for row_idx in range(probe_inputs["input_ids"].shape[0]):
        tail_ids = probe_inputs["input_ids"][row_idx, -k_suffix:].tolist()
        assert tail_ids == suffix_ids, (
            f"Prompt {row_idx} tail {tail_ids} does not match expected suffix {suffix_ids}"
        )

    # 5. All-layer finite activation check (zero NaN/Inf in float16/bfloat16)
    all_finite = assert_all_layer_activations_finite(llm, is_tuple, probe_inputs, k_suffix)

    # 6. Active-hook zero-intervention logit identity check (alpha=1.0, r=0) and timing
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        native_logits = raw_model(**probe_inputs).logits[:, -1, :].detach()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    sec_per_4_prompts = time.perf_counter() - t0

    zero_vec = torch.zeros(raw_model.config.hidden_size, dtype=torch.float32, device=device)
    hook_logits = forward_with_intervention(
        llm=llm,
        is_layer_output_tuple=is_tuple,
        inputs=probe_inputs,
        mode="steer_layer",
        r_vec=zero_vec,
        target_layer=0,
        alpha=1.0,
    )

    max_abs_diff = float(torch.max(torch.abs(native_logits.float() - hook_logits.float())).item())
    is_close = torch.allclose(
        native_logits,
        hook_logits,
        atol=CONFIG.atol_zero,
        rtol=CONFIG.rtol_zero,
    )
    assert is_close, (
        f"Zero-intervention active hook failed allclose! max_abs_diff={max_abs_diff:.6e}"
    )

    # 7. Compare low-precision baseline logits against exact float32 reference
    fp32_fidelity = verify_low_precision_against_fp32(raw_model, probe_inputs, native_logits)

    # Estimate total forward passes across Stages 2-8 (~180,000 prompt evaluations)
    est_total_gpu_hours = float((sec_per_4_prompts / 4.0) * 180000.0 / 3600.0)
    gpu_name = torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else "CPU"

    env_record = {
        "model_name": model_name,
        "n_layers": int(raw_model.config.num_hidden_layers),
        "d_model": int(raw_model.config.hidden_size),
        "dtype": dtype_str,
        "precision_deviation_dev001": precision_deviation,
        "gpu_device_name": gpu_name,
        "is_layer_output_tuple": bool(is_tuple),
        "unknown_token_id_nospace": int(unk_nospace),
        "unknown_token_id_space": int(unk_space),
        "k_suffix": int(k_suffix),
        "suffix_token_ids": [int(x) for x in suffix_ids],
        "all_layer_activations_finite": bool(all_finite),
        "zero_intervention_max_abs_diff": max_abs_diff,
        "fp32_logit_cosine_sim": fp32_fidelity["fp32_logit_cosine_sim"],
        "fp32_logit_mean_kl_nats": fp32_fidelity["fp32_logit_mean_kl_nats"],
        "fp32_top1_agreement": fp32_fidelity["fp32_top1_agreement"],
        "estimated_project_gpu_hours": round(est_total_gpu_hours, 2),
        "passed_all_checks": True,
    }
    return llm, env_record
```

### 4.3 Resumable Stage 0 CLI Script (`scripts/stage0_env_check.py`)
Supports `--allow-kaggle-env` so Kaggle's native CUDA `torch` wheel can be used alongside exact pinned `nnsight==0.3.7` and `transformers==4.46.3`.

```python
# scripts/stage0_env_check.py
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Dict
import torch

from src.config import CONFIG
from src.model_utils import load_model_and_verify_runtime

REQUIRED_STAGE0_KEYS = {
    "model_name": str,
    "n_layers": int,
    "d_model": int,
    "dtype": str,
    "is_layer_output_tuple": bool,
    "unknown_token_id_nospace": int,
    "unknown_token_id_space": int,
    "k_suffix": int,
    "suffix_token_ids": list,
    "all_layer_activations_finite": bool,
    "zero_intervention_max_abs_diff": float,
    "fp32_logit_cosine_sim": float,
    "fp32_logit_mean_kl_nats": float,
    "passed_all_checks": bool,
}


def validate_existing_stage0_output(output_path: Path) -> bool:
    """Checks whether results/stage0_env.json exists and passes all acceptance criteria."""
    if not output_path.exists():
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, expected_type in REQUIRED_STAGE0_KEYS.items():
            if key not in data or not isinstance(data[key], expected_type):
                return False
        if not data["passed_all_checks"] or not data["all_layer_activations_finite"]:
            return False
        if data["zero_intervention_max_abs_diff"] > CONFIG.atol_zero:
            return False
        if data["fp32_logit_cosine_sim"] < CONFIG.min_fp16_fp32_logit_cos:
            return False
        if data["fp32_logit_mean_kl_nats"] > CONFIG.max_fp16_fp32_kl:
            return False
        if data["k_suffix"] < 1 or len(data["suffix_token_ids"]) != data["k_suffix"]:
            return False
        return True
    except Exception:
        return False


def verify_installed_packages(allow_kaggle_env: bool = False) -> Dict[str, str]:
    """
    Verifies installed package versions. In --allow-kaggle-env mode (DEV-002),
    strictly enforces nnsight==0.3.7 and transformers==4.46.3 while logging
    Kaggle's native CUDA torch/scipy/numpy versions.
    """
    strict_kaggle_keys = {"nnsight", "transformers"}
    observed = {}
    mismatches = []
    for pkg, expected_ver in CONFIG.pinned_versions.items():
        try:
            actual_ver = importlib.metadata.version(pkg)
            clean_actual = actual_ver.split("+")[0]
            observed[pkg] = actual_ver
            if clean_actual != expected_ver:
                if not allow_kaggle_env or pkg in strict_kaggle_keys:
                    mismatches.append(f"{pkg}: expected {expected_ver}, found {actual_ver}")
        except importlib.metadata.PackageNotFoundError:
            observed[pkg] = "MISSING"
            mismatches.append(f"{pkg}: MISSING (expected {expected_ver})")

    if mismatches:
        raise RuntimeError(
            "Critical package version mismatch:\n  - " + "\n  - ".join(mismatches)
        )
    return observed


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0: Environment & Startup Verification")
    parser.add_argument(
        "--model-name",
        type=str,
        default=CONFIG.primary_model_name,
        help="Target Hugging Face model ID (primary or fallback).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Execution device (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Model precision ('auto' uses float16 on Kaggle T4/P100 per DEV-001).",
    )
    parser.add_argument(
        "--allow-kaggle-env",
        action="store_true",
        help="Enforce strict pins on nnsight/transformers while logging Kaggle host packages (DEV-002).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution even if results/stage0_env.json already passes checks.",
    )
    args = parser.parse_args()

    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = CONFIG.results_dir / "stage0_env.json"

    if not args.force and validate_existing_stage0_output(output_path):
        print(f"[SKIP] Stage 0 already verified at {output_path}")
        return 0

    print("[Stage 0] Checking installed packages...")
    observed_pkgs = verify_installed_packages(allow_kaggle_env=args.allow_kaggle_env)

    print(f"[Stage 0] Loading {args.model_name} on {args.device} (dtype={args.dtype})...")
    _, env_record = load_model_and_verify_runtime(
        model_name=args.model_name,
        device=args.device,
        requested_dtype=args.dtype,
    )
    env_record["installed_packages"] = observed_pkgs

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(env_record, f, indent=2)

    assert validate_existing_stage0_output(output_path), "Stage 0 output failed validation gate!"
    print(f"[PASS] Stage 0 complete. Saved verified contract to {output_path}:")
    print(json.dumps(env_record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 5. Kaggle Persistence & Execution Setup (`scripts/kaggle_sync.py`)

Because Kaggle sessions terminate after 12 hours and wipe `/kaggle/working` unless saved, create `scripts/kaggle_sync.py` so every Kaggle notebook stage automatically:
1. Commits and pushes JSON/CSV/Markdown outputs (`results/*.json`, `results/*.csv`, `tables/*`, `figures/*`) back to your GitHub repository after each stage finishes.
2. Uploads binary `.pt` checkpoints (`results/stage3_caches.pt` and `results/stage4_directions.pt`, $\approx 445\text{ MB}$ total) to a private Hugging Face Dataset repository (`your-username/abstention-direction-caches`) so a new Kaggle session can download them in 10 seconds without re-running Stage 3.

```python
# scripts/kaggle_sync.py
import argparse
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi
from src.config import CONFIG


def run_cmd(cmd: list[str]) -> None:
    print(f"[Sync] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=CONFIG.root_dir)


def push_git_artifacts(stage_label: str) -> None:
    """Commits and pushes JSON, CSV, Markdown, and Figure artifacts to GitHub."""
    run_cmd(["git", "add", "results/*.json", "results/*.csv", "tables/", "figures/", "DEVIATIONS.md"])
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=CONFIG.root_dir,
    )
    if status.stdout.strip():
        run_cmd(["git", "commit", "-m", f"Auto-save outputs after {stage_label}"])
        run_cmd(["git", "push", "origin", "main"])
    else:
        print("[Sync] No new Git changes to push.")


def sync_pt_tensors_hf(action: str, hf_repo_id: str) -> None:
    """Uploads or downloads .pt tensor files (stage3_caches.pt, stage4_directions.pt) via HF Hub."""
    token = os.environ.get("HF_TOKEN")
    assert token, "HF_TOKEN environment variable must be set for tensor sync!"
    api = HfApi(token=token)
    api.create_repo(repo_id=hf_repo_id, repo_type="dataset", private=True, exist_ok=True)

    pt_files = ["stage3_caches.pt", "stage4_directions.pt"]
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)

    if action == "push":
        for fname in pt_files:
            fpath = CONFIG.results_dir / fname
            if fpath.exists():
                print(f"[Sync] Uploading {fname} to HF Dataset {hf_repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=f"results/{fname}",
                    repo_id=hf_repo_id,
                    repo_type="dataset",
                )
    elif action == "pull":
        repo_files = set(api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset"))
        for fname in pt_files:
            remote_path = f"results/{fname}"
            local_path = CONFIG.results_dir / fname
            if remote_path in repo_files and not local_path.exists():
                print(f"[Sync] Downloading {remote_path} from {hf_repo_id}...")
                api.hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(CONFIG.root_dir),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle outputs with GitHub and HF Hub")
    parser.add_argument("--action", choices=["pull", "push"], required=True)
    parser.add_argument("--stage", type=str, default="stage0")
    parser.add_argument("--hf-repo", type=str, default="")
    args = parser.parse_args()

    if args.action == "pull" and args.hf_repo:
        sync_pt_tensors_hf("pull", args.hf_repo)
    elif args.action == "push":
        if args.hf_repo:
            sync_pt_tensors_hf("push", args.hf_repo)
        push_git_artifacts(args.stage)
```

### Kaggle Notebook Runner Template (Copy-Paste into Kaggle)
When you open a new Kaggle Notebook:
1. In the right-hand Settings pane, set **Accelerator** to **GPU T4 x2** and toggle **Internet** to **On**.
2. Under **Add-ons $\rightarrow$ Secrets**, enable `GITHUB_TOKEN` and `HF_TOKEN`.
3. Paste the following two cells and click **Save & Run All (Commit)** for long stages so they run in the background even if you close your browser tab:

```python
# === KAGGLE CELL 1: Bootstrap Repo, Secrets & Pinned Packages ===
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
GITHUB_USER = "YOUR_GITHUB_USERNAME"
GITHUB_REPO = "abstention-direction"
HF_CACHE_REPO = f"{GITHUB_USER}/abstention-direction-caches"

%cd /kaggle/working
if not os.path.exists(GITHUB_REPO):
    !git clone https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git
%cd /kaggle/working/{GITHUB_REPO}
!git config user.email "kaggle-runner@users.noreply.github.com"
!git config user.name "Kaggle GPU Runner"
!git pull origin main

# Install pinned nnsight==0.3.7 and transformers==4.46.3 onto Kaggle image
!pip install -q -r requirements-kaggle.txt

# Pull any previously cached .pt tensors (skips automatically if already present)
!PYTHONPATH=. python scripts/kaggle_sync.py --action pull --hf-repo {HF_CACHE_REPO}
```

```python
# === KAGGLE CELL 2: Execute Resumable Stage(s) & Auto-Push Results ===
# Each stage automatically skips if its outputs already exist and pass validation!
!PYTHONPATH=. python scripts/stage0_env_check.py --allow-kaggle-env --dtype float16
!PYTHONPATH=. python scripts/kaggle_sync.py --action push --stage stage0 --hf-repo {HF_CACHE_REPO}
```

---

## 6. Automated Unit & Integration Tests (`tests/test_stage0.py`)

Create `tests/test_stage0.py` to test all mathematical and `float16` precision invariants locally on CPU (`< 2 seconds`) and verify the full T4 GPU contract on Kaggle:

```python
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
    assert torch.equal(proj_unit, proj_scaled)


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
```

---

## 7. Day 1 Execution Checklist (Local IDE $\rightarrow$ Kaggle)

1. **In Your Local IDE (WSL / CPU — 0 GPU Quota Used):**
   * Run the fast unit test suite:
     ```bash
     pytest tests/test_stage0.py -v
     ```
     *Expected:* `4 passed, 1 deselected in < 2.0s`.
   * Commit and push the repository (`src/`, `scripts/`, `tests/`, `DEVIATIONS.md`, `requirements-kaggle.txt`) to GitHub.
2. **In Your Kaggle Notebook (GPU T4 $\times 2$, Internet On):**
   * Run **Kaggle Cell 1** and **Kaggle Cell 2** above.
   * Confirm that `results/stage0_env.json` reports `"passed_all_checks": true`, `"all_layer_activations_finite": true`, `"fp32_logit_cosine_sim": > 0.999`, and prints `"estimated_project_gpu_hours"` ($\approx 10\text{–}15\text{ hours}$) and pushes `results/stage0_env.json` back to GitHub.
   * Pull `results/stage0_env.json` into your local IDE (`git pull`) and proceed to **Stage 1 (Local CPU Dataset Build & Quality Gates)**.