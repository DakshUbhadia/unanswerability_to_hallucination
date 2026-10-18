# Implementation Deviations

This log records implementation choices that differ from the frozen protocol in `documentation.md`. The frozen specification is unchanged.

| ID | Spec section | Specified | Implemented | Reason |
|---|---|---|---|---|
| DEV-001 | 2.1 | `torch.bfloat16` model weights and forward pass | `torch.float16` on NVIDIA T4 (`sm_75`) and P100 (`sm_60`); residual projections, steering math, and accumulations use `torch.float32` | These GPUs lack native hardware bfloat16 support. Stage 0 checks every layer for finite activations and compares fp16 logits with an fp32 reference, requiring cosine similarity at least 0.9990 and KL at most 0.005 nats. |
| DEV-002 | 2.2 | Exact pinned Python and PyTorch environment | Pin `nnsight==0.3.7` and `transformers==4.46.3` while using Kaggle's native CUDA PyTorch image with `--allow-kaggle-env` | Reinstalling the large PyTorch wheel in ephemeral Kaggle sessions wastes GPU time and can mismatch host CUDA drivers. Runtime package versions are recorded in `results/stage0_env.json`. |
| DEV-003 | 4.4, Stages 2 and 8 | Match popularity using the original PopQA `s_pop` field | Popularity matching uses `log_s_pop = log10(s_pop + 1)`; caliper widths are fractions of `SD(log_s_pop)` | PopQA `s_pop` is raw monthly pageviews and is highly skewed. Log scaling gives the intended standardized popularity matching scale while preserving original fields and rank based deciles. |

### DEV-003 details: Stage 2 model switch and log-popularity matching

- Llama token parity measurement: 0/384 pairs had |delta_L| > 2; max |delta_L|=2; absolute-delta distribution={'0': 99, '1': 184, '2': 101}. 48 B-control prompts tokenized; lengths min/median/max=43/50.0/63; distribution={'43': 1, '44': 2, '45': 3, '46': 3, '47': 5, '48': 3, '49': 5, '50': 3, '51': 6, '52': 3, '53': 3, '54': 3, '56': 1, '57': 3, '58': 2, '60': 1, '63': 1}. Full per-pair details are in `results/stage2_llama_parity.json`.

- **Stage affected:** Stage 2, with `meta-llama/Llama-3.2-3B-Instruct` locked for downstream stages in `results/stage2_pilot.json`.
- **Pre-registered branch execution:** Qwen Attempt 1 and Attempt 2 both had a validation unanswerable abstention rate of `1.0000`, above the registered `0.95` ceiling, with answerable abstention at `0.0000`. Branch Rule 1 therefore triggered Attempt 3 using the registered Llama fallback. The fallback passed Stage 0 checks (`28` layers, `d_model = 3072`, `k_suffix = 5`) and Stage 2's validation pilot (`0.9479` unanswerable, `0.0104` answerable, `0.9375` gap; log-odds means differ by approximately `15.01` nats).
- **Frozen dataset tokenizer note:** `data/pairs.jsonl` was built and token-length parity checked under the primary Qwen tokenizer. It was not re-tokenized under Llama after the pre-registered fallback. Stage 3 should use the locked Stage 2 model and tokenizer; the token-length parity diagnostic does not transfer across tokenizers by assumption.
- **Popularity-scale deviation:** `s_pop` is retained unchanged in each PopQA row and continues to determine decile ranks. Matching instead uses `log10(max(s_pop, 0) + 1)` for both the population standard deviation and pairwise caliper distances. `caliper_width_s_pop` consequently records a width in log10 popularity units. This is a documented departure from the original raw-scale caliper in the frozen protocol.
