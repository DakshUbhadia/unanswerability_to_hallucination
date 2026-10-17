# From Unanswerability to Hallucination

> **Investigating Refusal Suppression, Orthogonal Projections, and Representation Engineering in LLMs**

This repository contains the experimental framework, mathematical operators, and evaluation pipeline for analyzing how Large Language Models transition from refusal (unanswerability) to hallucination under representation steering and refusal ablation.

---

## ?? Repository Architecture

```text
??? src/
?   ??? config.py             # Frozen Experiment Configuration (v4.0 constants, seeds, thresholds)
?   ??? model_utils.py        # Numerical assertions, FP16/FP32 projections, model loaders
?   ??? template_catalog.py   # 96 minimal-pair templates (48 Fam A, 48 Fam B) + token-parity calibrator
?   ??? dataset_builder.py    # Template splits, pair/B-ctrl instantiation, n-gram phi gate, PopQA sampler
?   ??? __init__.py
??? scripts/
?   ??? stage0_env_check.py   # Hardware verification, T4 VRAM check, logit cosine audit
?   ??? stage1_build_dataset.py # Builds & validates data/ on CPU (resumable; --force, --cut3)
?   ??? kaggle_sync.py        # Remote Kaggle secrets and artifact synchronization
?   ??? __init__.py
??? tests/
?   ??? test_stage0.py        # Invariant unit tests (projection idempotence, scaling invariance)
?   ??? test_stage1.py        # Dataset gates: splits, |delta_L| <= 2, max |phi| <= 0.15, B-ctrl, PopQA deciles
?   ??? __init__.py
??? data/                     # Stage 1 outputs: templates, splits, pairs, b_ctrl, popqa_sample, audit_log
??? notebooks/
?   ??? kaggle_runner.md      # Instructions & cell runner for Kaggle 2x Tesla T4 execution
??? DEVIATIONS.md             # Formal registry of hardware adaptations (DEV-001, DEV-002)
??? requirements.txt          # Primary project dependencies
??? pytest.ini                # Pytest configuration and marker definitions
??? README.md
```

---

## ?? Environment & Hardware Specifications

- **Primary Target Architecture**: NVIDIA Tesla T4 (16 GB VRAM) on Kaggle Free Tier.
- **Precision Adaptations**:
  - **DEV-001**: Fallback from `bfloat16` to `float16` on Tesla T4 due to hardware capability limitations (SM 7.5).
  - **DEV-002**: Python 3.10+ / Kaggle pre-installed environment tolerance.
- **Logit Cosine Similarity Threshold**: >= 0.999 between FP16 and FP32 baselines.

---

## ?? Quickstart & Local Verification

### 1. Setup Virtual Environment
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

### 2. Run Invariant Unit Tests
```bash
python -m pytest tests/test_stage0.py -m "not integration" -v
```

### 3. Run Environment Audit
```bash
python scripts/stage0_env_check.py --allow-kaggle-env
```

### 4. Build the Stage 1 Dataset (CPU only)
```bash
python -m pytest tests/test_stage1.py -v
python scripts/stage1_build_dataset.py          # [SKIP] if data/ already passes every gate
python scripts/stage1_build_dataset.py --force  # rebuild from scratch
```
Downloads only the Qwen tokenizer and the `akariasai/PopQA` test split. Any template that fails a gate must be rewritten in `src/template_catalog.py`; every failure and rewrite is logged to `data/audit_log.csv`.

---

## ?? Remote Kaggle Execution
Refer to [`notebooks/kaggle_runner.md`](notebooks/kaggle_runner.md) for step-by-step instructions on running the GPU integration tests and full model steering passes on Kaggle 2x T4 GPUs.
