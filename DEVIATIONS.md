# Registered Implementation Deviations from Frozen Specification v4.0

This file logs strictly bounded hardware and environment adaptations required to execute the Frozen Specification (v4.0) on Kaggle Free Tier GPUs (NVIDIA Tesla T4, 16 GB VRAM, Turing \sm_75\). No experimental hypotheses, dataset schemas, controls, or statistical tests are altered.

| Deviation ID | Spec Section | Specified Setting | Executed Setting on Kaggle | Justification & Stage 0 Verification Safeguard |
| :--- | :--- | :--- | :--- | :--- |
| **DEV-001** | Section 2.1 | \	orch.bfloat16\ model weights & forward pass | \	orch.float16\ model weights & forward pass on NVIDIA T4 (\sm_75\), with all residual projections, steering math, and accumulations in \	orch.float32\ | Kaggle T4 (\sm_75\) and P100 (\sm_60\) GPUs lack native hardware \float16\ support. Stage 0 enforces two mandatory safeguards: (1) an all-layer finite activation check (\zero NaN/Inf\ across all 36 layers), and (2) a \loat16\ vs. \loat32\ baseline logit fidelity check requiring cosine similarity >= 0.9990 and D_KL(P_fp32 || P_fp16) <= 0.005 nats. |
| **DEV-002** | Section 2.2 | Exact pinned \pip\ environment (\python==3.11.9\, \	orch==2.4.1\) | Exact pins for nsight==0.3.7\ and \	ransformers==4.46.3\ layered on Kaggle'''s native CUDA PyTorch image (\--allow-kaggle-env\) | Reinstalling the multi-gigabyte \	orch\ wheel inside ephemeral Kaggle sessions wastes GPU quota and risks host CUDA driver mismatches. Exact runtime package versions are logged to esults/stage0_env.json\. |
