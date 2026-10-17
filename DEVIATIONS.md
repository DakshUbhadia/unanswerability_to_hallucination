# Implementation Deviations

This log records implementation choices that differ from the frozen protocol in `documentation.md`. The frozen specification is unchanged.

| ID | Spec section | Specified | Implemented | Reason |
|---|---|---|---|---|
| DEV-001 | 2.1 | `torch.bfloat16` model weights and forward pass | `torch.float16` on NVIDIA T4 (`sm_75`) and P100 (`sm_60`); residual projections, steering math, and accumulations use `torch.float32` | These GPUs lack native hardware bfloat16 support. Stage 0 checks every layer for finite activations and compares fp16 logits with an fp32 reference, requiring cosine similarity at least 0.9990 and KL at most 0.005 nats. |
| DEV-002 | 2.2 | Exact pinned Python and PyTorch environment | Pin `nnsight==0.3.7` and `transformers==4.46.3` while using Kaggle's native CUDA PyTorch image with `--allow-kaggle-env` | Reinstalling the large PyTorch wheel in ephemeral Kaggle sessions wastes GPU time and can mismatch host CUDA drivers. Runtime package versions are recorded in `results/stage0_env.json`. |
| DEV-003 | 4.4, Stages 2 and 8 | Match popularity using the original PopQA `s_pop` field | Popularity matching uses `log_s_pop = log10(s_pop + 1)`; caliper widths are fractions of `SD(log_s_pop)` | PopQA `s_pop` is raw monthly pageviews and is highly skewed. Log scaling gives the intended standardized popularity matching scale while preserving original fields and rank based deciles. |
