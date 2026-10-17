# Mechanistic Interpretability Warm-Up (18-Day Specification): Linear Mediation of Epistemic Abstention in `Qwen2.5-3B-Instruct`

**Document Status:** FINAL FROZEN SPECIFICATION (v4.0)  
**Target Timeline:** 18 Working Days (with Mandatory Day 14 Checkpoint)  
**Primary Model:** `Qwen/Qwen2.5-3B-Instruct` (`torch.bfloat16`)  
**Fallback Model:** `meta-llama/Llama-3.2-3B-Instruct` (`torch.bfloat16`)

---

## 1. Executive Summary & Core Research Question

This document is the complete, frozen engineering and scientific specification for the Mechanistic Interpretability Warm-Up. Extending the difference-in-means linear intervention methodology of Arditi et al. (2024) (*"Refusal in Language Models Is Mediated by a Single Direction"*), this project investigates **Epistemic Abstention**—when a language model declines to answer because a question is unanswerable from context or beyond its knowledge—and prepares the reusable activation-patching infrastructure for downstream internal truth-probing (Option A).

### 1.1 Primary Research Question
> **"Does the linear abstention direction in `Qwen2.5-3B-Instruct` encode an input property ('this question is unanswerable') or internal epistemic uncertainty ('I do not know the answer')—and can steering along this direction make abstention more selective?"**

### 1.2 Competing Mechanistic Hypotheses
* **Hypothesis A (Input Property / Contextual Insufficiency Detector):** The extracted difference-in-means direction $\hat{r}$ encodes structural or contextual incompleteness in the prompt. It mediates abstention on contextually unanswerable inputs (Necessity & Sufficiency), does not activate on answerable private-detail controls ($D_{\text{B-ctrl}}$), and does not separate closed-book factual hallucinations ($D_{\text{factual}}^{\text{wrong, matched}}$) from correct answers ($D_{\text{factual}}^{\text{correct, matched}}$) beyond surface output confidence.
* **Hypothesis B (Internal Epistemic Uncertainty Detector):** The extracted direction $\hat{r}$ couples to the model's internal epistemic state across both contextual and parametric regimes. In addition to mediating abstention on $D_{\text{test}}$ and remaining inactive on $D_{\text{B-ctrl}}$, unsteered residual projections onto $\hat{r}$ are significantly higher on closed-book factual questions the model gets wrong ($D_{\text{factual}}^{\text{wrong, matched}}$) than on popularity-matched questions it answers correctly ($D_{\text{factual}}^{\text{correct, matched}}$), outperforming output token probability baselines.

### 1.3 Core vs. Stretch Scope
* **Core Scope (Stages 0–9, Days 1–16):**
  1. **Template-Split Minimal-Pair Dataset:** 96 distinct templates (48 Family A Context-Sufficiency, 48 Family B Epistemic-Boundary), partitioned strictly at the template level into `train` (48 templates), `val` (24 templates), and `test` (24 templates), plus the 48-item Family B Private-Detail Answerable Control set ($D_{\text{B-ctrl}}$).
  2. **Pilot Feasibility & PopQA Power Gate:** Verification on Day 3 that baseline abstention rates meet pre-registered thresholds and that the 1,000-question stratified PopQA sample yields at least 100 popularity-matched correct and wrong answers.
  3. **Causal Necessity, Sufficiency & Three-Tier Specificity Controls:** Evaluation of all-layer orthogonal projection ablation and single-layer norm-scaled steering at the validation-selected coordinate $(l^*, i^*)$ on held-out $D_{\text{test}}$, benchmarked against a **99-direction Monte Carlo null distribution**, a **19-direction within-pair label-permutation null distribution**, and **1 orthogonalized in-manifold concept direction ($r_{\text{concept}}$)**.
  4. **Capability Retention Diagnostics:** Continuous token-level KL divergence on 200 neutral WikiText-2 test paragraphs for all interventions and controls, plus 200-question chat-formatted MMLU accuracy for the baseline and learned direction.
  5. **Core PopQA Epistemic Separation & Confidence Baseline Test:** Comparison of unsteered projections $x_{i^*}^{(l^*)} \cdot \hat{r}$ across 6 groups (Figure 2A), AUROC evaluation for separating wrong vs. correct closed-book answers against maximum first-token probability and first-token entropy baselines, and an exploratory all-layer AUROC sweep (Figure S1).
* **Stretch Scope (Stage 10, Days 15–16, Optional):**
  * Multi-$\alpha$ additive steering sweep on the popularity-matched PopQA set to construct the Selective Abstention Accuracy-vs.-Coverage curve (Figure 2B).

---

## 2. System Architecture & Runtime Safeguards

### 2.1 Model Specifications & Precision Contract
1. **Primary Model (`Qwen/Qwen2.5-3B-Instruct`):** 36 decoder layers ($L = 36$), residual dimension $d_{\text{model}} = 2048$, 16 query heads, 2 key-value heads, vocabulary size $151,936$.
2. **Fallback Model (`meta-llama/Llama-3.2-3B-Instruct`):** 28 decoder layers ($L = 28$), residual dimension $d_{\text{model}} = 3072$, 24 query heads, 8 key-value heads, vocabulary size $128,256$.
3. **Precision Rule:** Weights and activations are loaded in native `torch.bfloat16` on a single 24 GB CUDA GPU. Running mean accumulations, inner products, vector norms, and statistical tests are computed in `torch.float32` / `numpy.float64`. Quantization (4-bit or 8-bit) is prohibited.

### 2.2 Pinned Software Stack
All stages execute in a single virtual environment with exact package versions pinned:
* `python==3.11.9`
* `torch==2.4.1`
* `transformers==4.46.3`
* `nnsight==0.3.7`
* `accelerate==1.1.1`
* `datasets==3.1.0`
* `scipy==1.14.1`
* `scikit-learn==1.5.2`
* `numpy==1.26.4`
* `pandas==2.2.3`
* `matplotlib==3.9.2`
* `seaborn==0.13.2`
* `pytest==8.3.3`
* `tqdm==4.66.6`

Under `nnsight==0.3.7`, every proxy tensor saved inside a `with llm.trace(...)` context must be accessed outside the trace via its `.value` attribute.

### 2.3 Native PyTorch Startup Verification Contract (Stage 0)
Before wrapping the model in `nnsight`, Stage 0 executes three deterministic checks on the raw Hugging Face `AutoModelForCausalLM` and `AutoTokenizer`:
1. **Native Layer Output Structure Check (`is_layer_output_tuple`):** Attach a standard PyTorch `register_forward_hook` directly to `raw_model.model.layers[0]`, execute one forward pass on a dummy input of shape `(2, 8)`, inspect the Python `type(output)` directly inside the hook, set `is_layer_output_tuple = isinstance(output, tuple)`, verify that the extracted hidden state tensor has 3D shape `(2, 8, d_model)`, and remove the hook handle. This boolean governs all subsequent residual reads and writes.
2. **Single-Token Abstention Target Check:** Verify that `tokenizer.encode("Unknown", add_special_tokens=False)` has length exactly $1$ (recording token ID $t_{\text{abs, nospace}}$) and record the final token ID of `tokenizer.encode(" Unknown", add_special_tokens=False)` as $t_{\text{abs, space}}$. Define the target abstention token set as $\mathcal{T}_{\text{abs}} = \{t_{\text{abs, nospace}}, t_{\text{abs, space}}\}$.
3. **Post-Instruction Suffix Coordinate Verification:** Apply `tokenizer.apply_chat_template` with `add_generation_prompt=True` to two test messages of different lengths, identify the longest common trailing token sequence following the user message content, and set $K_{\text{suffix}}$ to its exact token length. For `Qwen/Qwen2.5-3B-Instruct`, $K_{\text{suffix}} = 5$, corresponding to relative indices $\mathcal{I} = \{-5, -4, -3, -2, -1\}$ (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`).
4. **Fallback Model Re-Verification Rule:** If Stage 2 triggers the fallback model (`meta-llama/Llama-3.2-3B-Instruct`), Stage 0 must be re-run in its entirety on the fallback model to re-verify `is_layer_output_tuple`, confirm single-token encoding of `"Unknown"`, and set $\mathcal{I} = \{-K_{\text{suffix}}, \dots, -1\}$ to the fallback model's exact post-instruction suffix length ($K_{\text{suffix}} = 5$: `<|eot_id|>`, `<|start_header_id|>`, `assistant`, `<|end_header_id|>`, `\n\n`).

---

## 3. Master Parameter Table (Single Source of Truth)

Every numerical parameter, seed, threshold, grid, and sample size across the project is governed by Table 3.1. All subsequent sections reference this table.

| Parameter Name | Symbol / Key | Exact Value | Scope / Stage |
| :--- | :--- | :--- | :--- |
| **Global Master Seed** | `seed_global` | `20260401` | Global initialization |
| **Template Split Seed** | `seed_split` | `20260402` | Stage 1 template assignment |
| **Label-Permutation Control Seed** | `seed_shuffled` | `20260403` | Stage 4 ($M_{\text{shuf}} = 19$ permutations) |
| **Monte Carlo Random Direction Seed** | `seed_random` | `20260404` | Stage 4 ($M_{\text{rand}} = 99$ Gaussian draws) |
| **MMLU Stratified Sampling Seed** | `seed_mmlu` | `20260405` | Stage 7 (200-question sample) |
| **PopQA Sampling & Matching Seed** | `seed_popqa` | `20260406` | Stage 1, Stage 2 & Stage 8 |
| **Bootstrap Resampling Seed** | `seed_bootstrap` | `20260407` | Stages 6, 7, 8 & 10 |
| **Forward Pass Batch Size** | `batch_size_forward` | `16` | All caching & logit evaluations |
| **Generation Batch Size** | `batch_size_gen` | `16` | Stage 2, Stage 6 & Stage 8 decoding |
| **Max New Tokens (Abstention / PopQA)** | `max_new_tokens` | `16` | Greedy decoding (`do_sample=False`) |
| **Numerical Stability Epsilon** | `eps_norm` | `1e-8` | Vector normalization & probabilities |
| **Zero-Intervention `allclose` Tolerance** | `atol_zero`, `rtol_zero` | `1e-5`, `1e-4` | Stage 0 & unit tests |
| **Total Templates (Standard Schedule)** | `n_templates_total` | `96` (48 Fam A, 48 Fam B) | Stage 1 dataset construction |
| **Template Split Counts (Train / Val / Test)** | `n_templates_split` | `48 / 24 / 24` (per family: `24 / 12 / 12`) | Stage 1 template-level split |
| **Cut-3 Reduced Templates (If Triggered)** | `n_templates_cut3` | `64` (32 Fam A, 32 Fam B; `16 / 8 / 8` per family) | Day 14 Checkpoint Cut 3 only |
| **Instantiated Pairs per Template** | `n_pairs_per_template` | `4` | Stage 1 (`384` pairs / `768` prompts total) |
| **Train / Val / Test Pair Counts** | `n_pairs_split` | `192 / 96 / 96` (`384 / 192 / 192` prompts) | Stages 3, 4, 5 & 6 |
| **Family B Private-Detail Control Size** | `n_b_ctrl` | `48` (12 Fam B test templates $\times 4$) | Stage 1 & Stage 6 |
| **Max Prompt Token Length Parity** | `max_token_diff_pair` | `2` tokens | Stage 1 quality gate |
| **Max N-Gram Label Correlation ($\phi$)** | `max_ngram_phi` | `0.15` ($n \in \{1, 2, 3\}$) | Stage 1 lexical leakage gate |
| **Pilot Unanswerable Abstention Bounds** | `pilot_unans_min`, `pilot_unans_max` | `0.25`, `0.95` ($25\%$ to $95\%$) | Stage 2 Pilot Gate |
| **Pilot Answerable Abstention Max** | `pilot_ans_max` | `0.15` ($\le 15\%$) | Stage 2 Pilot Gate |
| **Pilot Minimum Rate Separation** | `pilot_gap_min` | `0.25` ($\ge 25$ percentage points) | Stage 2 Pilot Gate |
| **Pilot Max Prompt Framing Attempts** | `pilot_max_attempts` | `2` (before fallback model switch) | Stage 2 branch rule |
| **Post-Instruction Suffix Length** | `k_suffix` | `5` ($\mathcal{I} = \{-5, -4, -3, -2, -1\}$) | Stage 0 & Stage 3 |
| **Monte Carlo Random Control Count** | `n_ctrl_random` | `99` | Stage 4 & Stage 6 ($p_{\text{min}} = 0.01$) |
| **Shuffled-Label Control Count** | `n_ctrl_shuffled` | `19` | Stage 4 & Stage 6 ($p_{\text{min}} = 0.05$) |
| **Orthogonalized Concept Control Count** | `n_ctrl_concept` | `1` ($r_{\text{concept}}$) | Stage 4 & Stage 6 |
| **Validation Steering Strength** | `alpha_val_select` | `+1.0` | Stage 5 sufficiency scoring |
| **Validation KL Threshold Grid** | `grid_tau_kl` | `[0.05, 0.10, 0.15, 0.25]` nats | Stage 5 sensitivity sweep |
| **Validation KL Default Threshold** | `default_tau_kl` | `0.15` nats | Stage 5 default filter |
| **Validation Depth Cutoff Grid** | `grid_rho_depth` | `[0.75, 0.85, 0.95]` | Stage 5 sensitivity sweep |
| **Validation Depth Default Cutoff** | `default_rho_depth` | `0.85` ($l < 30$ for $L=36$) | Stage 5 default filter |
| **Composite Tie-Break Tolerance** | `tie_tol_j` | `1e-6` | Stage 5 coordinate selection |
| **Necessity $\alpha$ Grid (Subtraction on $D_{\text{test}}^{\text{unans}}$)** | `grid_alpha_nec` | `[0.0, -0.5, -1.0, -1.5, -2.0]` | Stage 6 & Stage 7 (Panel A) |
| **Sufficiency $\alpha$ Grid (Addition on $D_{\text{test}}^{\text{ans}}$)** | `grid_alpha_suf` | `[0.0, +0.5, +1.0, +1.5, +2.0]` | Stage 6 & Stage 7 (Panel B) |
| **Control Operating Points ($\alpha$)** | `ctrl_operating_points` | `ablate_all`, `alpha = -1.0`, `alpha = +1.0` | Stage 6 & Stage 7 |
| **Generation Audit Sample Size** | `n_gen_audit` | `50` prompts per condition (first 2 templates $\times 4$ + 42 stratified) | Stage 2 & Stage 6 |
| **Logit-Generation Discrepancy Margin** | `discrepancy_log_odds`, `discrepancy_rate` | `2.0` nats vs. `0.10` rate shift | Stage 6 decision rule |
| **WikiText-2 Neutral KL Sample Size** | `n_wikitext_kl` | `200` paragraphs ($\ge 64$ words each) | Stage 7 capability evaluation |
| **WikiText-2 Paragraph Token Length** | `len_wikitext_tokens` | `256` tokens (truncated, no chat template) | Stage 7 capability evaluation |
| **MMLU Evaluation Sample Size** | `n_mmlu` | `200` questions (stratified across 57 subjects) | Stage 7 capability evaluation |
| **PopQA Initial Sample Size** | `n_popqa_init` | `1000` questions | Stage 1, Stage 2 & Stage 8 |
| **PopQA Decile 1–5 Sampling Weight** | `w_popqa_bottom5` | `0.06` per decile ($60$ items/decile, $300$ total) | Stage 1 stratified sampling |
| **PopQA Decile 6–10 Sampling Weight** | `w_popqa_top5` | `0.14` per decile ($140$ items/decile, $700$ total) | Stage 1 stratified sampling |
| **PopQA Power Check Min Matched Group** | `n_popqa_matched_min` | `100` in $D_{\text{factual}}^{\text{correct, matched}}$ and `100` in $D_{\text{factual}}^{\text{wrong, matched}}$ | Stage 2 power check |
| **PopQA Top-Up Batch & Hard Cap** | `n_popqa_step`, `n_popqa_max` | `250` per step (`50`/decile in Deciles 6–10), cap `2000` | Stage 2 power check branch rule |
| **PopQA Caliper Width (Default / Fallback)** | `caliper_sd_default`, `caliper_sd_fallback` | `0.25`, `0.50` of $\text{SD}(s_{\text{pop}})$ | Stage 2 & Stage 8 matching |
| **Stretch PopQA Steering $\alpha$ Grid** | `grid_alpha_stretch` | `[0.0, +0.25, +0.50, +0.75, +1.00, +1.25, +1.50, +2.00]` | Stage 10 (Optional Stretch) |
| **Bootstrap Resampling Iterations** | `n_bootstrap` | `1000` | All confidence intervals & H5 |
| **Confidence Interval Coverage** | `ci_level` | `0.95` ($2.5\text{th}$ and $97.5\text{th}$ percentiles) | All tables & figures |
| **Family-Wise Error Rate Target** | `alpha_fwer` | `0.05` (Holm–Bonferroni across 5 primary tests) | Section 6 pre-registered tests |

---

## 4. Dataset Engineering & Template Authoring Protocol

### 4.1 Contrastive Minimal-Pair Families
To prevent surface entity familiarity or lexical artifacts from leaking into the difference-in-means vector, the main dataset consists of two minimal-pair families where every pair $(x_{t,k}^{\text{ans}}, x_{t,k}^{\text{unans}})$ instantiated from template $t$ with slot filling $k \in \{1, 2, 3, 4\}$ shares identical synthetic/rare entity names and syntactic structure:

1. **Family A — Context-Sufficiency Minimal Pairs (48 Templates, 192 Pairs):**
   * Each template provides a 2–3 sentence context paragraph describing a synthetic entity, project, regulatory filing, or experimental log, followed by a specific question.
   * **Answerable ($x_{t,k}^{\text{ans}}$):** Contains the single clause providing the answer to the question.
   * **Unanswerable ($x_{t,k}^{\text{unans}}$):** Replaces the answer-bearing clause with a length-matched clause ($\pm 2$ tokens) stating an orthogonal attribute of the same entity, rendering the question unanswerable from the provided context.
2. **Family B — Epistemic-Boundary Minimal Pairs (48 Templates, 192 Pairs):**
   * Each template provides a concise biographical or organizational profile of a synthetic entity, followed by a question.
   * **Answerable ($x_{t,k}^{\text{ans}}$):** Asks for a factual attribute explicitly stated in the profile.
   * **Unanswerable ($x_{t,k}^{\text{unans}}$):** Keeps the profile identical and replaces only the question predicate with a length-matched question ($\pm 2$ tokens) asking for an unrecorded, private, or ephemeral attribute not derivable from the profile.
3. **Family B Private-Detail Answerable Control Set ($D_{\text{B-ctrl}}$, 48 Prompts):**
   * Instantiated across all 12 Family B test templates (4 entity instances each).
   * Uses the **exact unanswerable private-detail question predicate** from $x_{t,k}^{\text{unans}}$, but augments the profile sentence so the private detail is explicitly stated. This control isolates whether $\hat{r}$ detects *missing information* rather than *questions about private/trivial topics*.

### 4.2 Template-Level Splitting Contract
Splitting is performed strictly at the `template_id` level using `seed_split = 20260402` before instantiating entity slots:
* **$D_{\text{train}}$:** 24 Family A templates + 24 Family B templates = 48 templates $\times 4$ instances = **192 pairs (384 prompts)**.
* **$D_{\text{val}}$:** 12 Family A templates + 12 Family B templates = 24 templates $\times 4$ instances = **96 pairs (192 prompts)**.
* **$D_{\text{test}}$:** 12 Family A templates + 12 Family B templates = 24 templates $\times 4$ instances = **96 pairs (192 prompts)**, plus $D_{\text{B-ctrl}}$ (**48 prompts**).

No template, missing-clause replacement structure, or question predicate in $D_{\text{val}}$ or $D_{\text{test}}$ may appear in $D_{\text{train}}$.

### 4.3 Template Authoring, Human Review Checklist & Audit Procedure
1. **LLM-Assisted Drafting Policy:** Using an external LLM to draft candidate templates is explicitly permitted, provided every template undergoes manual human verification and passes all automated quality gates below.
2. **Mandatory 4-Point Human Review Checklist:** Every template in `data/templates.jsonl` must be manually reviewed by the researcher against four binary criteria:
   * `check_answer_unambiguous`: The answerable version has a single, unambiguous short factual answer directly stated in the text.
   * `check_zero_unans_support`: The unanswerable version provides zero lexical, logical, or pragmatic support for guessing the answer.
   * `check_distinct_predicate`: The template's missing-clause structure (Family A) or unanswerable question predicate (Family B) is semantically distinct from all other 95 templates.
   * `check_b_ctrl_valid` (Family B only): When the private detail is inserted into the profile for $D_{\text{B-ctrl}}$, the question becomes unambiguously answerable.
3. **Automated Quality Gate 1 — Token Length Parity:** Tokenize $x_{t,k}^{\text{ans}}$ and $x_{t,k}^{\text{unans}}$ with the target model's tokenizer. Assert $|L(x_{t,k}^{\text{ans}}) - L(x_{t,k}^{\text{unans}})| \le 2$ tokens for every pair $(t, k)$.
4. **Automated Quality Gate 2 — N-Gram Lexical Leakage Check:** Compute the absolute Phi coefficient ($|\phi|$, equivalent to Pearson correlation for binary variables) between the presence of every word unigram, bigram, and trigram ($n \in \{1, 2, 3\}$) and the binary label (`ans` vs. `unans`) across the full dataset. Assert $\max_{g} |\phi(g)| \le 0.15$.
5. **Failure & Rewrite Protocol:** Any template or pair failing a manual check, length parity, or contributing an n-gram with $|\phi| > 0.15$ must be **rewritten in place** and re-verified until all 96 templates pass. Templates are never silently dropped. Every initial fail and rewrite action is appended as a permanent row in `data/audit_log.csv`.

### 4.4 Constrained Output Protocol & Prompt Formatting
All prompts in $D_{\text{train}}$, $D_{\text{val}}$, $D_{\text{test}}$, $D_{\text{B-ctrl}}$, and $D_{\text{factual}}$ are formatted via `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` with the fixed system prompt:

> `"You are a precise question-answering assistant. If the answer is known or determinable from the prompt, reply directly with the concise answer starting immediately on the first word. If the question cannot be answered from the provided information or you do not know the answer, reply with the exact single word: Unknown"`

### 4.5 Closed-Book Factual Dataset ($D_{\text{factual}}$, PopQA Stratified Sample)
1. **Stratified Sampling (`n_popqa_init = 1000`):** Load `akariasai/PopQA` (`test` split). Partition all rows into 10 equal-sized deciles based on `s_pop` (log monthly Wikipedia pageviews). Using `seed_popqa = 20260406`, sample without replacement:
   * **Deciles 1–5:** weight $0.06$ per decile ($60$ questions per decile $= 300$ questions).
   * **Deciles 6–10:** weight $0.14$ per decile ($140$ questions per decile $= 700$ questions).
2. **Closed-Book Answer Grading:** Generate up to `max_new_tokens = 16` tokens (`do_sample=False`). Normalize both the generated string and each alias in PopQA `possible_answers` by converting to lowercase, stripping ASCII punctuation, removing English articles (`"a"`, `"an"`, `"the"`), and collapsing whitespace.
   * **Abstained ($D_{\text{factual}}^{\text{abstain}}$):** The raw response matches the abstention regex parser (Section 5.4).
   * **Correct ($D_{\text{factual}}^{\text{correct}}$):** Not abstained, and at least one normalized ground-truth alias appears as an exact prefix or token-boundary substring of the normalized response.
   * **Wrong / Hallucinated ($D_{\text{factual}}^{\text{wrong}}$):** Neither abstained nor correct.
3. **1-to-1 Popularity Caliper Matching:** Match items in $D_{\text{factual}}^{\text{wrong}}$ to items in $D_{\text{factual}}^{\text{correct}}$ without replacement via greedy 1-to-1 nearest-neighbor matching on `s_pop` within caliper width $\delta_{\text{pop}} = 0.25 \cdot \text{SD}(s_{\text{pop}})$ (ties broken using `seed_popqa`), yielding the popularity-matched subsets $D_{\text{factual}}^{\text{wrong, matched}}$ and $D_{\text{factual}}^{\text{correct, matched}}$ of equal size $N_{\text{matched}}$.

---

## 5. Mathematical Formulations & Intervention Specifications

### 5.1 Candidate Direction Extraction
For every decoder layer $l \in \{0, \dots, L-1\}$ and post-instruction token index $i \in \mathcal{I} = \{-K_{\text{suffix}}, \dots, -1\}$, let $x_i^{(l)}(p) \in \mathbb{R}^{d_{\text{model}}}$ be the output residual stream activation of layer $l$ at position $i$ for prompt $p$. On $D_{\text{train}}$ ($N_{\text{train}} = 192$ pairs):
$$\mu_i^{(l)}(\text{unans}) = \frac{1}{N_{\text{train}}} \sum_{p \in D_{\text{train}}^{\text{unans}}} x_i^{(l)}(p), \qquad \mu_i^{(l)}(\text{ans}) = \frac{1}{N_{\text{train}}} \sum_{p \in D_{\text{train}}^{\text{ans}}} x_i^{(l)}(p)$$
$$r_i^{(l)} = \mu_i^{(l)}(\text{unans}) - \mu_i^{(l)}(\text{ans}), \qquad \hat{r}_i^{(l)} = \frac{r_i^{(l)}}{\|r_i^{(l)}\|_2 + \epsilon}$$

### 5.2 Three-Tier Negative Control Formulations at $(l^*, i^*)$
Once the optimal coordinate $(l^*, i^*)$ is locked on $D_{\text{val}}$, let $r = r_{i^*}^{(l^*)}$ and $\hat{r} = \hat{r}_{i^*}^{(l^*)}$. We construct three tiers of negative controls:

1. **Tier 1 — Monte Carlo Null Distribution ($M_{\text{rand}} = 99$ Isotropic Random Directions):**
   Using `seed_random = 20260404`, draw $z^{(m)} \sim \mathcal{N}(0, I_{d_{\text{model}}})$ for $m \in \{1, \dots, 99\}$. Normalize $\hat{r}_{\text{rand}}^{(m)} = z^{(m)} / (\|z^{(m)}\|_2 + \epsilon)$ and rescale to match the learned vector's Euclidean norm:
   $$r_{\text{rand}}^{(m)} = \|r\|_2 \cdot \hat{r}_{\text{rand}}^{(m)}$$
   *Note on Norm Invariance:* Norm-matching to $\|r\|_2$ affects only additive/subtractive steering ($x \leftarrow x + \alpha r_{\text{rand}}^{(m)}$) and has zero mathematical effect on orthogonal projection ablation ($x \leftarrow x - (x \cdot \hat{r}_{\text{rand}}^{(m)})\hat{r}_{\text{rand}}^{(m)}$), which depends solely on the unit vector $\hat{r}_{\text{rand}}^{(m)}$.
2. **Tier 2 — Within-Pair Label-Permutation Null Distribution ($M_{\text{shuf}} = 19$ Shuffled Directions):**
   Using `seed_shuffled = 20260403`, for each $m \in \{1, \dots, 19\}$, independently swap the `ans` and `unans` labels within each of the 192 training pairs with probability $0.5$, compute the difference-in-means $r_{\text{shuf}}^{(m)}$ at $(l^*, i^*)$ using each permutation's natural difference-in-means norm $\|r_{\text{shuf}}^{(m)}\|_2$ for steering and its unit vector $\hat{r}_{\text{shuf}}^{(m)}$ for projection ablation.
3. **Tier 3 — Orthogonalized In-Manifold Concept Control ($M_{\text{concept}} = 1$, $r_{\text{concept}}$):**
   Compute the raw difference-in-means between Family A answerable training prompts and Family B answerable training prompts at $(l^*, i^*)$:
   $$r_{\text{concept}}^{\text{raw}} = \mu_{i^*}^{(l^*)}(D_{\text{train}}^{\text{ans, A}}) - \mu_{i^*}^{(l^*)}(D_{\text{train}}^{\text{ans, B}})$$
   First, record and report the pre-orthogonalization cosine similarity with the abstention direction:
   $$\cos(\theta_{\text{pre}}) = \frac{r_{\text{concept}}^{\text{raw}} \cdot \hat{r}}{\|r_{\text{concept}}^{\text{raw}}\|_2 + \epsilon}$$
   Second, project out the component parallel to $\hat{r}$ and rescale to $\|r\|_2$:
   $$r_{\text{concept}}^{\perp} = r_{\text{concept}}^{\text{raw}} - (r_{\text{concept}}^{\text{raw}} \cdot \hat{r})\hat{r}, \qquad \hat{r}_{\text{concept}} = \frac{r_{\text{concept}}^{\perp}}{\|r_{\text{concept}}^{\perp}\|_2 + \epsilon}, \qquad r_{\text{concept}} = \|r\|_2 \cdot \hat{r}_{\text{concept}}$$

### 5.3 Inference-Time Intervention Operators
All residual stream interventions apply across all sequence token positions $t \in \{1, \dots, T\}$ (including system prompt, user prompt, and generated tokens):
1. **All-Layer Orthogonal Projection Ablation (`mode = "ablate_all"`):**
   At every decoder layer $l \in \{0, \dots, L-1\}$, replace residual output $x_t^{(l)}$ with:
   $$\text{Proj}_{\perp \hat{v}}\left(x_t^{(l)}\right) = x_t^{(l)} - \left(x_t^{(l)} \cdot \hat{v}\right)\hat{v}$$
2. **Single-Layer Norm-Scaled Steering (`mode = "steer_layer"`):**
   At the selected layer $l^*$ only, for scalar multiplier $\alpha \in \mathbb{R}$ defined in units of $\|v\|_2$, replace $x_t^{(l^*)}$ with:
   $$\text{Steer}\left(x_t^{(l^*)}; v, \alpha\right) = x_t^{(l^*)} + \alpha \|v\|_2 \hat{v} = x_t^{(l^*)} + \alpha v$$

### 5.4 Behavioral Metrics & Validation Coordinate Selection
1. **Binary Partition Abstention Log-Odds ($S(p)$):**
   From the final prompt position's next-token probability distribution $P(\cdot \mid p) = \text{softmax}(\text{logits}_{-1}(p))$, compute total probability on the abstention token set $P_{\text{abs}}(p) = \text{clamp}\left(\sum_{t \in \mathcal{T}_{\text{abs}}} P(t \mid p), 10^{-7}, 1 - 10^{-7}\right)$, and define:
   $$S(p) = \log P_{\text{abs}}(p) - \log\left(1 - P_{\text{abs}}(p)\right)$$
2. **Generation Abstention Rate ($R_{\text{gen}}$):**
   Under greedy decoding (`max_new_tokens = 16`, `do_sample=False`), we record both:
   * **Strict Protocol Rate ($R_{\text{gen, strict}}$):** Fraction of responses where `response.strip().lower().startswith("unknown")`.
   * **Regex-Expanded Rate ($R_{\text{gen, regex}}$):** Fraction of responses matching case-insensitively at the start of the string or after punctuation any of: `^unknown\b`, `i do not know`, `i don't know`, `cannot be determined`, `cannot be answered`, `not enough information`, `insufficient information`, `not provided in the context`, `not mentioned`, or `unspecified`.
3. **Validation Coordinate Selection ($l^*, i^*$):**
   On $D_{\text{val}}$ ($96$ pairs), evaluate every coordinate $(l, i) \in \{0, \dots, L-1\} \times \mathcal{I}$:
   * $S_{\text{abl}}(l, i)$: Mean abstention log-odds on $D_{\text{val}}^{\text{unans}}$ under `ablate_all` using $\hat{r}_i^{(l)}$ (lower is better).
   * $S_{\text{steer}}(l, i)$: Mean abstention log-odds on $D_{\text{val}}^{\text{ans}}$ under `steer_layer` at layer $l$ with $\alpha = +1.0$ using $r_i^{(l)}$ (higher is better).
   * $\text{KL}_{\text{val}}(l, i)$: Mean next-token KL divergence $D_{\text{KL}}\left(P_{\text{base}}(\cdot \mid p) \parallel P_{\text{abl}}(\cdot \mid p)\right)$ on $D_{\text{val}}^{\text{ans}}$ under `ablate_all`.
   Standardize $S_{\text{abl}}$ and $S_{\text{steer}}$ across all $L \times K_{\text{suffix}}$ candidates via $z$-scores and define the composite score:
   $$J(l, i) = z\left(S_{\text{steer}}(l, i)\right) - z\left(S_{\text{abl}}(l, i)\right)$$
   Maximize $J(l, i)$ subject to $l < \lfloor \rho_{\text{depth}} L \rfloor$ ($\rho_{\text{depth}} = 0.85$) and $\text{KL}_{\text{val}}(l, i) \le \tau_{\text{KL}}$ ($\tau_{\text{KL}} = 0.15$).

### 5.5 Capability Retention Metrics (Stage 7)
1. **Neutral-Text Token-Level KL Divergence (All Interventions & Controls):**
   Evaluated exclusively on the **200 WikiText-2 (`wikitext-2-raw-v1`, test split) paragraphs** (`n_wikitext_kl = 200`), taking the first 200 paragraphs with at least 64 whitespace-delimited words, truncated to exactly `len_wikitext_tokens = 256` tokens with `add_special_tokens=False` and **no chat template**. Compute $D_{\text{KL}}\left(P_{\text{base}}(\cdot \mid w_{<t}) \parallel P_{\text{int}}(\cdot \mid w_{<t})\right)$ at every non-padding token position $t \in \{1, \dots, 255\}$ and average across all $200 \times 255 = 51,000$ token positions.
2. **Chat-Formatted Zero-Shot MMLU Accuracy (Baseline & Learned Direction Only):**
   Evaluated on a stratified 200-question sample (`n_mmlu = 200`, `seed_mmlu = 20260405`) from `cais/mmlu` (`all`, `test` split), wrapped in `apply_chat_template`. For each option $k \in \{\text{A}, \text{B}, \text{C}, \text{D}\}$, sum probability mass across the non-space token ID `encode(k)[-1]` and leading-space token ID `encode(" " + k)[-1]`. MMLU is evaluated strictly for the unsteered baseline and the learned direction $\hat{r}_{i^*}^{(l^*)}$; control directions use WikiText-2 KL only.

---

## 6. Pre-Registered Analysis Plan & Hypothesis Matrix

### 6.1 Primary Hypothesis Tests (Holm–Bonferroni Controlled at $\alpha_{\text{FWER}} = 0.05$)
Exactly **five primary hypothesis tests (H1–H5)** constitute the confirmatory statistical family. Raw one-sided $p$-values $p_1, \dots, p_5$ are sorted in ascending order $p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)}$ and compared sequentially against the Holm–Bonferroni thresholds $\alpha_k^* = 0.05 / (6 - k)$, namely:
$$\alpha_1^* = 0.0100, \quad \alpha_2^* = 0.0125, \quad \alpha_3^* = 0.0167, \quad \alpha_4^* = 0.0250, \quad \alpha_5^* = 0.0500$$
Testing stops at the first index $k$ where $p_{(k)} > \alpha_k^*$; all hypotheses $j \ge k$ are declared non-significant.

| Test ID | Hypothesis Name | Exact Test Statistic | Statistical Test & Direction | Confirmatory Criterion |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | **Causal Necessity on $D_{\text{test}}^{\text{unans}}$** | Template-cluster mean log-odds shift $\Delta S_{\text{abl}}^{(c)} = S_{\text{base}}^{(c)} - S_{\text{abl}}^{(c)}$ across $C = 24$ test template clusters, plus Monte Carlo rank among $1 + 99$ directions | One-sided Wilcoxon signed-rank test ($H_1: \text{median}(\Delta S_{\text{abl}}^{(c)}) > 0$) **AND** Monte Carlo rank $p_{\text{MC}} = \frac{1 + \sum_{m=1}^{99} \mathbb{I}(\Delta S_{\text{rand}}^{(m)} \ge \Delta S_{\text{abl}})}{100}$ | Both $p_{\text{Wilcoxon}}$ passes Holm–Bonferroni **AND** $p_{\text{MC}} = 0.0100$ (Rank 1 of 100) |
| **H2** | **Causal Sufficiency on $D_{\text{test}}^{\text{ans}}$** | Template-cluster mean log-odds shift $\Delta S_{\text{steer}}^{(c)} = S_{+\alpha r}^{(c)} - S_{\text{base}}^{(c)}$ at $\alpha = +1.0$ across $C = 24$ test template clusters, plus Monte Carlo rank | One-sided Wilcoxon signed-rank test ($H_1: \text{median}(\Delta S_{\text{steer}}^{(c)}) > 0$) **AND** Monte Carlo rank $p_{\text{MC}} = \frac{1 + \sum_{m=1}^{99} \mathbb{I}(\Delta S_{\text{rand}}^{(m)} \ge \Delta S_{\text{steer}})}{100}$ | Both $p_{\text{Wilcoxon}}$ passes Holm–Bonferroni **AND** $p_{\text{MC}} = 0.0100$ (Rank 1 of 100) |
| **H3** | **Private-Detail Topic De-Confounding ($D_{\text{B-ctrl}}$)** | Paired template-cluster mean projection difference $\Delta \text{Proj}_{\text{B}}^{(c)} = \text{Proj}(D_{\text{test}}^{\text{unans, B}})^{(c)} - \text{Proj}(D_{\text{B-ctrl}})^{(c)}$ across $C_{\text{B}} = 12$ Family B test templates | One-sided Wilcoxon signed-rank test ($H_1: \text{median}(\Delta \text{Proj}_{\text{B}}^{(c)}) > 0$) | $p_{\text{Wilcoxon}}$ passes Holm–Bonferroni **AND** mean abstention rate on $D_{\text{B-ctrl}} \le 0.15$ |
| **H4** | **Closed-Book Epistemic Uncertainty Separation ($D_{\text{factual}}$)** | $\text{AUROC}_{\text{proj}}$ for classifying $D_{\text{factual}}^{\text{wrong, matched}}$ ($y=1$) vs. $D_{\text{factual}}^{\text{correct, matched}}$ ($y=0$) using scalar projection $s_{\text{proj}}(p) = x_{i^*}^{(l^*)}(p) \cdot \hat{r}$ | One-sided Mann-Whitney U test ($H_1: \text{AUROC}_{\text{proj}} > 0.50$) | $p_{\text{MWU}}$ passes Holm–Bonferroni **AND** $\text{AUROC}_{\text{proj}} \ge 0.60$ |
| **H5** | **Superiority Over Output Token Probability Baseline** | Paired AUROC difference $\Delta \text{AUROC} = \text{AUROC}_{\text{proj}} - \text{AUROC}_{\text{maxp}}$ on $D_{\text{factual}}^{\text{wrong, matched}}$ vs. $D_{\text{factual}}^{\text{correct, matched}}$ | One-sided paired bootstrap test across `n_bootstrap = 1000` resamples ($p_{\text{boot}} = \frac{1 + \sum_{b=1}^{1000} \mathbb{I}(\Delta \text{AUROC}^{(b)} \le 0)}{1001}$) | $p_{\text{boot}}$ passes Holm–Bonferroni ($\Delta \text{AUROC} > 0$) |

### 6.2 Secondary & Exploratory Analyses
* **SECONDARY Analyses (Reported with 95% Bootstrap CIs and Control $z$-scores):**
  1. Comparison against the 19 within-pair label-permutation controls ($p_{\text{perm}} = 1/20 = 0.05$, plus control $z$-score $z_{\text{shuf}} = (\theta_{\text{learned}} - \mu_{\text{shuf}}) / \sigma_{\text{shuf}}$).
  2. Comparison against the orthogonalized in-manifold concept control $r_{\text{concept}}$, plus reporting $\cos(\theta_{\text{pre}})$.
  3. First-token predictive entropy baseline $\text{AUROC}_{\text{ent}}$ and paired difference $\text{AUROC}_{\text{proj}} - \text{AUROC}_{\text{ent}}$ on PopQA.
  4. Full Pareto trade-off curves across `grid_alpha_nec` and `grid_alpha_suf` against WikiText-2 KL and MMLU accuracy.
  5. Validation sensitivity grid across `grid_tau_kl` $\times$ `grid_rho_depth` (Table 2).
* **EXPLORATORY Analyses:**
  1. **All-Layer PopQA Projection AUROC Sweep (Figure S1):** Computing wrong-vs-correct $\text{AUROC}_{\text{proj}}^{(l)}$ using $x_{i^*}^{(l)}(p) \cdot \hat{r}_{i^*}^{(l)}$ across all layers $l \in \{0, \dots, L-1\}$ at token offset $i^*$.
  2. **Selective Abstention Accuracy-vs.-Coverage Curve (Stage 10 Stretch Goal, Figure 2B):** Evaluating coverage and selective accuracy on PopQA across `grid_alpha_stretch`.

### 6.3 Pre-Registered Post-Experiment Interpretation Matrix

| Empirical Test Dimension | Outcome Supporting Hypothesis A (Input Property Detector) | Outcome Supporting Hypothesis B (Internal Uncertainty Detector) | Outcome Indicating Confound / Methodological Failure |
| :--- | :--- | :--- | :--- |
| **1. Causal Mediation on $D_{\text{test}}$ (H1 & H2)** | H1 and H2 both pass Holm–Bonferroni and achieve Rank 1/100 ($p_{\text{MC}} = 0.0100$) against Monte Carlo controls. | H1 and H2 both pass Holm–Bonferroni and achieve Rank 1/100 ($p_{\text{MC}} = 0.0100$) against Monte Carlo controls. | H1 or H2 fails Holm–Bonferroni, or $r_{\text{concept}}$ produces $> 25\%$ of the behavioral shift of $\hat{r}$ (non-specific subspace perturbation). |
| **2. Private-Detail Control $D_{\text{B-ctrl}}$ (H3)** | H3 passes Holm–Bonferroni and abstention rate on $D_{\text{B-ctrl}} \le 0.15$. | H3 passes Holm–Bonferroni and abstention rate on $D_{\text{B-ctrl}} \le 0.15$. | H3 fails or $D_{\text{B-ctrl}}$ abstention rate $> 0.15$ ($\hat{r}$ encodes "private-detail topic" confound). |
| **3. Closed-Book PopQA Separation (H4)** | H4 fails Holm–Bonferroni or $\text{AUROC}_{\text{proj}} < 0.60$ (whilst $D_{\text{factual}}^{\text{abstain}}$ still projects high if the direction triggers only when the model already decides to output `"Unknown"`). | H4 passes Holm–Bonferroni **AND** $\text{AUROC}_{\text{proj}} \ge 0.60$ on popularity-matched wrong vs. correct answers. | Both $D_{\text{factual}}^{\text{abstain}}$ and $D_{\text{factual}}^{\text{wrong, matched}}$ show flat projections indistinguishable from $D_{\text{factual}}^{\text{correct, matched}}$ (complete domain-transfer failure from contextual to closed-book prompts). |
| **4. Value Over Output Confidence (H5) & Layer Sweep (Fig S1)** | H5 fails Holm–Bonferroni ($\text{AUROC}_{\text{proj}} \le \text{AUROC}_{\text{maxp}}$); $\hat{r}$ merely reflects late-layer output logit readiness. | H5 passes Holm–Bonferroni ($\text{AUROC}_{\text{proj}} > \text{AUROC}_{\text{maxp}}$), or Figure S1 reveals an earlier layer $l < l^*$ where $\text{AUROC}_{\text{proj}}^{(l)} > \text{AUROC}_{\text{maxp}}$. | Neither residual projection nor output confidence separates wrong from correct answers ($\text{AUROC} \approx 0.50$). |

---

## 7. Deterministic Decision Rules for All Branch Points

An engineer implementing this specification must follow these five deterministic rules without manual intervention:

1. **Branch Rule 1 — Stage 2 Pilot Gate Failure:**
   * **Pass Condition:** On $D_{\text{val}}$, baseline generation abstention rates satisfy $0.25 \le R_{\text{unans}} \le 0.95$, $R_{\text{ans}} \le 0.15$, and $R_{\text{unans}} - R_{\text{ans}} \ge 0.25$.
   * **Attempt 1 Failure Remedy:**
     * *If $R_{\text{unans}} < 0.25$ (Under-Abstention):* Append to the system prompt: `"Never guess or fabricate an answer when information is missing; you must output Unknown."`
     * *If $R_{\text{ans}} > 0.15$ (Over-Abstention):* Append to the system prompt: `"Whenever the prompt contains the required fact, you must state the answer directly and never output Unknown."`
     * *If $0.25 \le R_{\text{unans}} \le 0.95$ and $R_{\text{ans}} \le 0.15$ but $R_{\text{unans}} - R_{\text{ans}} < 0.25$:* Apply both prompt clauses above simultaneously.
   * **Attempt 2 Failure Remedy (Model Switch):** If Attempt 2 fails the pass condition, immediately switch the target model to `meta-llama/Llama-3.2-3B-Instruct`, re-run Stage 0 and Stage 2 once (Attempt 3). If Attempt 3 fails, terminate the pipeline with exit code `2` and log the failure diagnostics to `results/stage2_pilot.json`.
2. **Branch Rule 2 — Empty Validation Filter Set at Default $\tau_{\text{KL}}$ (Stage 5):**
   * If zero coordinates $(l, i)$ satisfy $l < \lfloor 0.85 L \rfloor$ and $\text{KL}_{\text{val}}(l, i) \le 0.15$, step sequentially to the next looser threshold in `grid_tau_kl`, setting $\tau_{\text{KL}} = 0.25$.
   * If zero coordinates pass at $(\tau_{\text{KL}} = 0.25, \rho_{\text{depth}} = 0.85)$, relax the depth cutoff to $\rho_{\text{depth}} = 0.95$ at $\tau_{\text{KL}} = 0.25$.
   * If still empty, select the coordinate $(l, i)$ with $l < \lfloor 0.85 L \rfloor$ that minimizes $\text{KL}_{\text{val}}(l, i)$ among the top $10\%$ (top 18) of composite scores $J(l, i)$, and set `"filter_relaxed": true` in `results/stage5_selection.json`.
3. **Branch Rule 3 — Composite Score Tie-Breaking (Stage 5):**
   * Any two valid candidate coordinates $(l_1, i_1)$ and $(l_2, i_2)$ whose composite scores satisfy $|J(l_1, i_1) - J_{\max}| \le 10^{-6}$ are considered tied. Break ties deterministically by selecting:
     1. **The earlier decoder layer** ($\min l$).
     2. If still tied within the same layer $l$, **the later token position** ($\max i$, i.e., closer to $-1$).
4. **Branch Rule 4 — Logit-Generation Audit Discrepancy (Stage 6):**
   * Compare the mean log-odds shift $\Delta S$ and the 50-sample generation abstention rate shift $\Delta R_{\text{gen, regex}}$ at the primary operating points (`ablate_all` on $D_{\text{test}}^{\text{unans}}$ and `alpha = +1.0` on $D_{\text{test}}^{\text{ans}}$).
   * Flag `"logit_gen_discrepancy": true` in `results/stage6_test_eval.json` if: (a) $\text{sign}(\Delta S) \ne \text{sign}(\Delta R_{\text{gen, regex}})$, or (b) $|\Delta S| \ge 2.0\text{ nats}$ while $|\Delta R_{\text{gen, regex}}| < 0.10$.
   * **Reporting Rule:** Always report both log-odds and generation rates in Table 1 and Table 3. Whenever `"logit_gen_discrepancy": true`, all qualitative claims in the write-up must defer to the multi-token generation rate ($R_{\text{gen, regex}}$) as the primary behavioral outcome.
5. **Branch Rule 5 — PopQA Power Check Failure (Stage 2):**
   * After grading the initial `n_popqa_init = 1000` PopQA items and running 1-to-1 nearest-neighbor caliper matching ($\delta_{\text{pop}} = 0.25 \cdot \text{SD}(s_{\text{pop}})$), check whether $N_{\text{matched}} \ge 100$ (meaning at least 100 matched pairs, i.e., $\ge 100$ items in $D_{\text{factual}}^{\text{correct, matched}}$ and $\ge 100$ items in $D_{\text{factual}}^{\text{wrong, matched}}$).
   * If $N_{\text{matched}} < 100$, sample additional batches of `n_popqa_step = 250` unseen questions from `akariasai/PopQA` uniformly across the top 5 popularity deciles (Deciles 6–10, weight $0.20$ per decile $= 50$ questions per decile per batch), run baseline grading, and re-run caliper matching on the cumulative pool until $N_{\text{matched}} \ge 100$ or the cumulative sample reaches the hard cap `n_popqa_max = 2000`.
   * If $N_{\text{matched}} < 100$ at the 2,000-question cap, widen the caliper once to `caliper_sd_fallback = 0.50` ($\delta_{\text{pop}} = 0.50 \cdot \text{SD}(s_{\text{pop}})$), record `"caliper_widened": true` and the final $N_{\text{matched}}$ in `results/stage2_pilot.json`, and proceed.

---

## 8. Exact Data & Results File Schemas

All data files reside in `data/` and all output files reside in `results/`, `figures/`, and `tables/`. Every record has a unique primary identifier.

### 8.1 Input & Intermediate Dataset Schemas (`data/`)

| File Path | Format | Exact Field Schema (Name : Type — Description) |
| :--- | :--- | :--- |
| `data/templates.jsonl` | JSONL (96 rows) | `template_id`: `str` (`"famA_001"`..`"famA_048"`, `"famB_001"`..`"famB_048"`)<br>`family`: `str` (`"A"` or `"B"`)<br>`domain`: `str`<br>`ans_template`: `str` (with `{entity}`, `{attr1}`, `{attr2}` slots)<br>`unans_template`: `str`<br>`b_ctrl_template`: `str` or `null` (populated for all 48 Family B templates)<br>`slots`: `list[dict]` (length 4; each dict contains slot values and `ground_truth`: `str`)<br>`check_answer_unambiguous`: `bool`<br>`check_zero_unans_support`: `bool`<br>`check_distinct_predicate`: `bool`<br>`check_b_ctrl_valid`: `bool` |
| `data/splits.json` | JSON | `seed_split`: `int` (`20260402`)<br>`train_template_ids`: `list[str]` (length 48)<br>`val_template_ids`: `list[str]` (length 24)<br>`test_template_ids`: `list[str]` (length 24) |
| `data/pairs.jsonl` | JSONL (384 rows) | `pair_id`: `str` (`"pair_0001"`..`"pair_0384"`)<br>`template_id`: `str`<br>`split`: `str` (`"train"`, `"val"`, or `"test"`)<br>`family`: `str` (`"A"` or `"B"`)<br>`instance_idx`: `int` (`0`..`3`)<br>`ans_prompt`: `str`<br>`unans_prompt`: `str`<br>`ground_truth`: `str`<br>`ans_token_len`: `int`<br>`unans_token_len`: `int`<br>`token_len_diff`: `int` (must be $\in \{-2, -1, 0, 1, 2\}$) |
| `data/b_ctrl.jsonl` | JSONL (48 rows) | `b_ctrl_id`: `str` (`"bctrl_001"`..`"bctrl_048"`)<br>`pair_id`: `str` (links to matched Family B test pair in `data/pairs.jsonl`)<br>`template_id`: `str` (one of the 12 Family B test templates)<br>`instance_idx`: `int` (`0`..`3`)<br>`prompt`: `str`<br>`ground_truth`: `str`<br>`token_len`: `int` |
| `data/popqa_sample.jsonl` | JSONL (1,000–2,000 rows) | `popqa_id`: `str` (`"popqa_0001"`..)<br>`hf_row_idx`: `int`<br>`question`: `str`<br>`prop`: `str`<br>`subj`: `str`<br>`obj`: `str`<br>`possible_answers`: `list[str]`<br>`s_pop`: `float`<br>`s_pop_decile`: `int` (`1`..`10`)<br>`baseline_response`: `str`<br>`baseline_partition`: `str` (`"correct"`, `"wrong"`, or `"abstain"`)<br>`is_popularity_matched`: `bool`<br>`matched_pair_id`: `str` or `null` (`"m_001"`..`"m_NNN"`) |
| `data/audit_log.csv` | CSV | `timestamp_utc`: `str`, `template_id`: `str`, `failure_reason`: `str` (`"manual_check"`, `"length_parity"`, `"ngram_leakage"`), `offending_detail`: `str`, `old_text`: `str`, `new_text`: `str`, `post_fix_passed`: `bool` |

### 8.2 Stage Output & Results Schemas (`results/`)

| File Path | Format | Exact Field Schema (Name : Type — Description) |
| :--- | :--- | :--- |
| `results/stage0_env.json` | JSON | `model_name`: `str`, `n_layers`: `int`, `d_model`: `int`, `dtype`: `str`, `is_layer_output_tuple`: `bool`, `unknown_token_id_nospace`: `int`, `unknown_token_id_space`: `int`, `k_suffix`: `int`, `suffix_token_ids`: `list[int]`, `zero_intervention_max_abs_diff`: `float`, `passed_all_checks`: `bool` |
| `results/stage2_pilot.json` | JSON | `attempt_number`: `int`, `model_name`: `str`, `system_prompt_used`: `str`, `val_unans_abstention_rate`: `float`, `val_ans_abstention_rate`: `float`, `val_rate_gap`: `float`, `val_unans_mean_log_odds`: `float`, `val_ans_mean_log_odds`: `float`, `popqa_evaluated_count`: `int`, `popqa_correct_count`: `int`, `popqa_wrong_count`: `int`, `popqa_abstain_count`: `int`, `popqa_matched_pair_count`: `int`, `caliper_widened`: `bool`, `passed_pilot_and_power_gate`: `bool` |
| `results/stage3_caches.pt` | PyTorch Dict | `train_ans`: `FloatTensor[L, K_suffix, d_model]` (mean), `train_unans`: `FloatTensor[L, K_suffix, d_model]` (mean), `train_pair_acts_ans`: `FloatTensor[192, L, K_suffix, d_model]`, `train_pair_acts_unans`: `FloatTensor[192, L, K_suffix, d_model]` |
| `results/stage4_directions.pt` | PyTorch Dict | `diff_in_means`: `FloatTensor[L, K_suffix, d_model]`, `unit_directions`: `FloatTensor[L, K_suffix, d_model]`, `norms`: `FloatTensor[L, K_suffix]`, `random_controls_unit`: `FloatTensor[99, L, K_suffix, d_model]`, `shuffled_controls_diff`: `FloatTensor[19, L, K_suffix, d_model]`, `r_concept_raw`: `FloatTensor[L, K_suffix, d_model]`, `r_concept_ortho_scaled`: `FloatTensor[L, K_suffix, d_model]`, `cos_pre_ortho`: `FloatTensor[L, K_suffix]` |
| `results/stage5_selection.json` | JSON | `l_star`: `int`, `i_star`: `int`, `pos_idx`: `int`, `r_norm_at_star`: `float`, `cos_concept_pre_ortho_at_star`: `float`, `default_tau_kl`: `float`, `default_rho_depth`: `float`, `filter_relaxed`: `bool`, `ablation_scores_grid`: `list[list[float]]`, `steering_scores_grid`: `list[list[float]]`, `kl_val_grid`: `list[list[float]]`, `composite_j_grid`: `list[list[float]]`, `sensitivity_table`: `list[dict]` |
| `results/stage6_test_eval.json` | JSON | `l_star`: `int`, `i_star`: `int`, `primary_tests`: `dict` (`H1`, `H2`, `H3` raw stats, Wilcoxon $p$-values, Monte Carlo ranks, cluster bootstrap 95% CIs), `necessity_curve`: `list[dict]` (over `grid_alpha_nec` + `ablate_all`), `sufficiency_curve`: `list[dict]` (over `grid_alpha_suf`), `controls_summary`: `dict` (`random_99`, `shuffled_19`, `r_concept` means, SDs, $z$-scores, percentiles), `b_ctrl_summary`: `dict`, `logit_gen_discrepancy`: `bool` |
| `results/stage6_generations.csv` | CSV | `condition`: `str`, `dataset_part`: `str`, `pair_id`: `str`, `template_id`: `str`, `log_odds`: `float`, `p_unknown`: `float`, `generated_text`: `str`, `abstained_strict`: `bool`, `abstained_regex`: `bool` |
| `results/stage7_capability.json` | JSON | `wikitext_baseline_nll`: `float`, `mmlu_baseline_acc`: `float`, `mmlu_baseline_ci`: `list[float]`, `learned_direction_capability`: `list[dict]` (`mode`, `alpha`, `wikitext_kl`, `wikitext_kl_ci`, `mmlu_acc`, `mmlu_ci`), `controls_wikitext_kl`: `dict` (`random_99`, `shuffled_19`, `r_concept` at `ablate_all`, `alpha=-1.0`, `alpha=+1.0`) |
| `results/stage8_popqa_core.json` | JSON | `n_matched_per_group`: `int`, `n_abstain`: `int`, `auroc_proj`: `float`, `auroc_proj_ci`: `list[float]`, `auroc_maxp`: `float`, `auroc_maxp_ci`: `list[float]`, `auroc_ent`: `float`, `auroc_ent_ci`: `list[float]`, `delta_auroc_proj_minus_maxp`: `float`, `delta_auroc_maxp_ci`: `list[float]`, `delta_auroc_proj_minus_ent`: `float`, `delta_auroc_ent_ci`: `list[float]`, `h4_mwu_u`: `float`, `h4_mwu_p`: `float`, `h5_boot_p`: `float`, `holm_bonferroni_table`: `list[dict]`, `group_projections`: `dict[str, list[float]]` (6 groups for Figure 2A), `layerwise_auroc_exploratory`: `list[dict]` (`layer`, `auroc`, `ci_low`, `ci_high` for Figure S1) |
| `results/stage10_stretch.json` | JSON (Optional) | `alpha_grid`: `list[float]`, `coverage_curve`: `list[dict]` (`alpha`, `coverage`, `coverage_ci`, `selective_acc`, `selective_acc_ci`, `wrong_to_abstain_rate`, `correct_to_abstain_rate`) |

---

## 9. Resumable 11-Stage Execution Pipeline

Each stage is implemented as an independent CLI script in `scripts/`. Every script accepts `--force` to recompute from scratch; without `--force`, if all declared output files exist and pass the stage's acceptance check, the script logs `[SKIP] Stage N already verified` and exits with code `0`.

| Stage | Script Name | Inputs | Outputs | Deterministic Acceptance Gate |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | `scripts/stage0_env_check.py` | Raw HF model & tokenizer | `results/stage0_env.json` | Asserts exact package versions, records `is_layer_output_tuple`, asserts single-token encoding of `"Unknown"`, verifies $K_{\text{suffix}}$, and asserts zero-intervention active hook (`alpha=1.0`, $r=\mathbf{0}$) matches native logits within `atol=1e-5, rtol=1e-4`. |
| **Stage 1** | `scripts/stage1_build_dataset.py` | `data/templates.jsonl`, HF `akariasai/PopQA` | `data/splits.json`, `data/pairs.jsonl`, `data/b_ctrl.jsonl`, `data/popqa_sample.jsonl`, `data/audit_log.csv` | Asserts 96 templates with all 4 human review booleans `True`, zero template overlap across `train`/`val`/`test`, token length parity $\vert{}\Delta L\vert{} \le 2$ on all 384 pairs, max n-gram $\vert{}\phi\vert{} \le 0.15$, 48 rows in `b_ctrl.jsonl`, and 1,000 stratified PopQA rows. |
| **Stage 2** | `scripts/stage2_pilot_gate.py` | `results/stage0_env.json`, `data/pairs.jsonl`, `data/popqa_sample.jsonl` | `results/stage2_pilot.json`, updated `data/popqa_sample.jsonl` | Executes Branch Rule 1 (Pilot Gate on $D_{\text{val}}$) and Branch Rule 5 (PopQA baseline grading & caliper matching power check requiring $N_{\text{matched}} \ge 100$ per group, topping up to $\le 2,000$ items if needed). |
| **Stage 3** | `scripts/stage3_cache_activations.py` | `results/stage2_pilot.json`, `data/pairs.jsonl` | `results/stage3_caches.pt` | Caches float32 post-instruction residual activations on $D_{\text{train}}$ (384 prompts). Asserts tensor shapes `[192, L, K_suffix, d_model]`, zero `NaN`/`Inf` entries, and strictly positive variance across prompts. |
| **Stage 4** | `scripts/stage4_extract_directions.py` | `results/stage3_caches.pt`, `data/pairs.jsonl` | `results/stage4_directions.pt` | Computes $r_i^{(l)}$, $\hat{r}_i^{(l)}$, 99 Monte Carlo random controls, 19 label-shuffled controls, and orthogonalized $r_{\text{concept}}$. Asserts $\vert{}\hat{r}_i^{(l)} \cdot \hat{r}_{\text{concept}, i}^{(l)}\vert{} < 10^{-5}$ and exact norm equality $\Vert{}r_{\text{concept}, i}^{(l)}\Vert{}_2 = \Vert{}r_{\text{rand}, i}^{(l)}\Vert{}_2 = \Vert{}r_i^{(l)}\Vert{}_2$ within $10^{-5}$. |
| **Stage 5** | `scripts/stage5_select_coordinate.py` | `results/stage4_directions.pt`, `data/pairs.jsonl` | `results/stage5_selection.json`, `tables/table2_sensitivity_grid.csv` (`.md`) | Sweeps all $L \times K_{\text{suffix}}$ candidates on $D_{\text{val}}$, applies Branch Rules 2 and 3 to lock $(l^*, i^*)$, and outputs the $4 \times 3$ threshold sensitivity grid. Asserts valid integer indices $0 \le l^* < L$ and $i^* \in \mathcal{I}$. |
| **Stage 6** | `scripts/stage6_evaluate_test.py` | `results/stage5_selection.json`, `results/stage4_directions.pt`, `data/pairs.jsonl`, `data/b_ctrl.jsonl` | `results/stage6_test_eval.json`, `results/stage6_generations.csv`, `tables/table3_generation_audit.csv` (`.md`) | Evaluates locked $(l^*, i^*)$ on $D_{\text{test}}$ (`grid_alpha_nec`, `grid_alpha_suf`, `ablate_all`), runs all 119 control vectors at `ctrl_operating_points`, evaluates $D_{\text{B-ctrl}}$, runs 50-sample generation audits, applies Branch Rule 4, and computes template-cluster bootstrap CIs for H1, H2, and H3. |
| **Stage 7** | `scripts/stage7_evaluate_capability.py` | `results/stage5_selection.json`, `results/stage4_directions.pt`, HF `wikitext`, HF `cais/mmlu` | `results/stage7_capability.json` | Computes 200-paragraph WikiText-2 KL divergence for learned direction (full $\alpha$ grids + `ablate_all`) and all 119 controls (`ctrl_operating_points`), and evaluates 200-item chat-formatted MMLU accuracy for baseline and learned direction only. |
| **Stage 8** | `scripts/stage8_evaluate_popqa_core.py` | `results/stage5_selection.json`, `results/stage4_directions.pt`, `results/stage6_test_eval.json`, `data/popqa_sample.jsonl` | `results/stage8_popqa_core.json`, `tables/table1_primary_results.csv` (`.md`) | Caches unsteered activations at $i^*$ across all layers $l \in \{0, \dots, L-1\}$ on PopQA, computes 6-group projections at $(l^*, i^*)$, evaluates $\text{AUROC}_{\text{proj}}$, $\text{AUROC}_{\text{maxp}}$, $\text{AUROC}_{\text{ent}}$ and paired bootstrap differences (H4 & H5), computes exploratory all-layer AUROC (Figure S1), and applies Holm–Bonferroni across H1–H5. |
| **Stage 9** | `scripts/stage9_make_figures.py` | `results/stage5_selection.json`, `results/stage6_test_eval.json`, `results/stage7_capability.json`, `results/stage8_popqa_core.json` | `figures/fig1_pareto_tradeoff.pdf` (`.png`), `figures/fig2a_epistemic_projections.pdf` (`.png`), `figures/fig_s1_layerwise_popqa_auroc.pdf` (`.png`) | Renders all core and supplementary publication figures at 300 DPI and verifies non-empty PDF/PNG outputs. |
| **Stage 10** | `scripts/stage10_stretch_steering.py` | `results/stage5_selection.json`, `results/stage4_directions.pt`, `data/popqa_sample.jsonl` | `results/stage10_stretch.json`, `figures/fig2b_selective_abstention_stretch.pdf` (`.png`) | *(Optional Stretch Goal)* Runs greedy decoding on the matched PopQA evaluation set across `grid_alpha_stretch` at layer $l^*$ and plots the Accuracy-vs.-Coverage curve. |

---

## 10. Output Artifacts Specification (Figures & Tables)

Every generated figure and table must match the exact specification in Table 10.1.

| Artifact Filename | Generating Stage | Exact Layout, Axes, Plotted Series & Error-Bar Specification |
| :--- | :--- | :--- |
| `figures/fig1_pareto_tradeoff.pdf` (`.png`) | Stage 9 | **2×2 Panel Figure (Necessity vs. Sufficiency $\times$ KL vs. MMLU):**<br>• **Top-Left (Panel A1 — Necessity vs. KL):** X-axis: Mean Token-Level KL Divergence on 200 WikiText-2 test paragraphs (nats, inverted so right = better retention). Y-axis: Abstention Rate (%) on $D_{\text{test}}^{\text{unans}}$. Series: (1) Baseline (`alpha=0`), (2) Learned $\hat{r}$ `ablate_all`, (3) Learned $r$ subtraction across `grid_alpha_nec`, (4) 99 Monte Carlo random controls mean $\pm 95\%$ empirical interval at `ablate_all` and `alpha=-1.0`, (5) 19 Shuffled-label controls mean $\pm 95\%$ empirical interval at `ablate_all` and `alpha=-1.0`, (6) Orthogonalized $r_{\text{concept}}$ at `ablate_all` and `alpha=-1.0`.<br>• **Bottom-Left (Panel A2 — Necessity vs. MMLU):** X-axis: 200-item Chat MMLU Accuracy (%). Y-axis: Abstention Rate (%) on $D_{\text{test}}^{\text{unans}}$. Series: Baseline, Learned $\hat{r}$ `ablate_all`, and Learned $r$ across `grid_alpha_nec` (no controls).<br>• **Top-Right (Panel B1 — Sufficiency vs. KL):** X-axis: Mean Token-Level KL on WikiText-2 (nats). Y-axis: Abstention Rate (%) on $D_{\text{test}}^{\text{ans}}$. Series: Baseline, Learned $r$ addition across `grid_alpha_suf`, plus 99 Monte Carlo, 19 Shuffled, and $r_{\text{concept}}$ controls at `alpha=+1.0`.<br>• **Bottom-Right (Panel B2 — Sufficiency vs. MMLU):** X-axis: 200-item Chat MMLU Accuracy (%). Y-axis: Abstention Rate (%) on $D_{\text{test}}^{\text{ans}}$. Series: Baseline and Learned $r$ across `grid_alpha_suf` (no controls).<br>• **Error Bars:** Y-axis uses 1,000-iteration **template-cluster bootstrap** 95% CIs ($C=24$ clusters); X-axis uses 1,000-iteration item-level bootstrap 95% CIs ($N=200$ paragraphs or MMLU items). |
| `figures/fig2a_epistemic_projections.pdf` (`.png`) | Stage 9 | **Single-Panel 6-Group Violin + Strip + Median/IQR Plot:**<br>• **X-axis (6 Categorical Groups in Order):**<br>  1. $D_{\text{test}}^{\text{unans}}$ ($N=96$, Contextual/Boundary Unanswerable — Positive Reference)<br>  2. $D_{\text{test}}^{\text{ans}}$ ($N=96$, Contextual/Boundary Answerable — Negative Reference)<br>  3. $D_{\text{B-ctrl}}$ ($N=48$, Family B Private-Detail Answerable Control)<br>  4. $D_{\text{factual}}^{\text{abstain}}$ ($N=N_{\text{abs}}$, PopQA Closed-Book Abstained — Closed-Book Positive Reference)<br>  5. $D_{\text{factual}}^{\text{wrong, matched}}$ ($N=N_{\text{matched}}$, PopQA Closed-Book Wrong / Hallucinated, Popularity-Matched)<br>  6. $D_{\text{factual}}^{\text{correct, matched}}$ ($N=N_{\text{matched}}$, PopQA Closed-Book Correct, Popularity-Matched)<br>• **Y-axis:** Unsteered scalar projection $x_{i^*}^{(l^*)}(p) \cdot \hat{r}_{i^*}^{(l^*)}$ (float32 units).<br>• **Annotations:** Horizontal bracket between Groups 2 and 3 showing H3 Wilcoxon $p$-value; horizontal bracket between Groups 5 and 6 showing H4 $\text{AUROC}_{\text{proj}}$ (with 95% bootstrap CI) and Mann-Whitney U $p$-value; inset text box reporting $\text{AUROC}_{\text{maxp}}$, $\text{AUROC}_{\text{ent}}$, and $\Delta \text{AUROC}$ (H5). |
| `figures/fig_s1_layerwise_popqa_auroc.pdf` (`.png`) | Stage 9 | **Supplementary Exploratory Line Plot (Figure S1):**<br>• **Title:** explicitly prefixed with `[EXPLORATORY]`.<br>• **X-axis:** Decoder Layer Index $l \in \{0, \dots, L-1\}$.<br>• **Y-axis:** Wrong-vs.-Correct Classification AUROC on Popularity-Matched PopQA ($D_{\text{factual}}^{\text{wrong, matched}}$ vs. $D_{\text{factual}}^{\text{correct, matched}}$).<br>• **Plotted Series:** (1) Layer-wise projection $\text{AUROC}_{\text{proj}}^{(l)}$ using $x_{i^*}^{(l)}(p) \cdot \hat{r}_{i^*}^{(l)}$ with shaded 1,000-iteration bootstrap 95% CI ribbon, (2) Vertical dashed line marking selected layer $l^*$, (3) Horizontal dashed lines for $\text{AUROC}_{\text{maxp}}$, $\text{AUROC}_{\text{ent}}$, and chance level ($0.50$). |
| `figures/fig2b_selective_abstention_stretch.pdf` (`.png`) | Stage 10 | **Optional Stretch Goal Line Plot (Figure 2B):**<br>• **X-axis:** Coverage — Fraction of popularity-matched PopQA prompts ($2 N_{\text{matched}}$) where the steered model provides a non-abstained answer.<br>• **Y-axis:** Selective Accuracy (%) among answered prompts.<br>• **Plotted Series:** (1) Learned abstention steering $x \leftarrow x + \alpha r_{i^*}^{(l^*)}$ across `grid_alpha_stretch`, (2) Output confidence thresholding baseline (sweeping threshold on $\max_v P(v \mid p)$ without activation steering), (3) Random-abstention horizontal baseline. Error bars show 1,000-iteration item-level bootstrap 95% CIs. |
| `tables/table1_primary_results.csv` (`.md`) | Stage 8 | **Confirmatory & Control Summary Table:**<br>Columns: `test_id`, `hypothesis_or_control`, `point_estimate`, `ci_95_low`, `ci_95_high`, `raw_p_value`, `holm_threshold`, `passed_holm_bonferroni`, `mc_rank_out_of_100`, `mc_z_score`, `shuffled_z_score`, `r_concept_effect_ratio`. Includes rows for H1–H5 plus secondary rows for $\text{AUROC}_{\text{ent}}$, $r_{\text{concept}}$ (with $\cos(\theta_{\text{pre}})$), and shuffled-label controls. |
| `tables/table2_sensitivity_grid.csv` (`.md`) | Stage 5 | **Validation Selection Sensitivity Table (12 Rows):**<br>Columns: `tau_kl`, `rho_depth`, `selected_l_star`, `selected_i_star`, `val_ablation_log_odds`, `val_steering_log_odds`, `val_kl_nats`, `composite_j_score`, `matches_default_coordinate`. |
| `tables/table3_generation_audit.csv` (`.md`) | Stage 6 | **Log-Odds vs. Multi-Token Generation Audit Table:**<br>Columns: `condition`, `dataset_partition`, `mean_log_odds`, `log_odds_ci_95`, `mean_p_unknown`, `gen_abstention_rate_strict`, `gen_abstention_rate_regex`, `gen_rate_ci_95`, `discrepancy_flagged`. |

---

## 11. 18-Day Execution Schedule & Day 14 Checkpoint Cut Order

### 11.1 18-Day Working Schedule

| Working Days | Pipeline Stages | Milestones & Verification Deliverables |
| :--- | :--- | :--- |
| **Day 1** | **Stage 0** | Set up pinned virtual environment (`nnsight==0.3.7`, `transformers==4.46.3`), run `scripts/stage0_env_check.py` on `Qwen/Qwen2.5-3B-Instruct`, verify native PyTorch hook tuple check, `"Unknown"` single-token assertion, $K_{\text{suffix}}=5$, and `allclose` zero-intervention logit identity. |
| **Days 2–4** | **Stage 1** | Draft and manually review all 96 minimal-pair templates (48 Family A, 48 Family B + $D_{\text{B-ctrl}}$). Run `scripts/stage1_build_dataset.py` to enforce the 4-point human review checklist, $\pm 2$-token length parity, and n-gram leakage check ($\max \vert{}\phi\vert{} \le 0.15$), logging all rewrites to `data/audit_log.csv`. Draw the 1,000-question stratified PopQA sample. |
| **Day 5** | **Stage 2** | Execute `scripts/stage2_pilot_gate.py`. Verify baseline abstention rates on $D_{\text{val}}$ (Branch Rule 1) and run the PopQA baseline grading and caliper-matching power check requiring $\ge 100$ matched items per group (Branch Rule 5). |
| **Days 6–7** | **Stage 3 & Stage 4** | Execute `scripts/stage3_cache_activations.py` to cache float32 post-instruction activations on $D_{\text{train}}$. Execute `scripts/stage4_extract_directions.py` to extract candidate vectors $r_i^{(l)}$, 99 Monte Carlo random controls, 19 label-shuffled controls, and orthogonalized $r_{\text{concept}}$ (recording $\cos(\theta_{\text{pre}})$). Run unit tests on projection idempotence and orthogonality. |
| **Days 8–9** | **Stage 5** | Execute `scripts/stage5_select_coordinate.py` across all $36 \times 5 = 180$ coordinates on $D_{\text{val}}$. Apply Branch Rules 2 and 3, lock $(l^*, i^*)$, and generate `tables/table2_sensitivity_grid.csv`. |
| **Days 10–11** | **Stage 6** | Execute `scripts/stage6_evaluate_test.py` once on locked $D_{\text{test}}$ and $D_{\text{B-ctrl}}$, including the 99 Monte Carlo, 19 shuffled, and $r_{\text{concept}}$ controls at `ctrl_operating_points`, the 50-sample generation audit, Branch Rule 4 check, and template-cluster bootstrap CIs (H1, H2, H3). |
| **Days 12–13** | **Stage 7** | Execute `scripts/stage7_evaluate_capability.py` to compute neutral-text KL divergence on 200 WikiText-2 test paragraphs (learned direction + controls) and 200-question chat MMLU accuracy (baseline + learned direction only). |
| **Day 14** | **CHECKPOINT & Stage 8** | **Mandatory Day 14 Schedule Checkpoint (apply Section 11.2 Cut Order if behind).** Execute `scripts/stage8_evaluate_popqa_core.py` to run the core PopQA projection test, confidence baselines ($\text{AUROC}_{\text{maxp}}$, $\text{AUROC}_{\text{ent}}$), exploratory all-layer sweep (Figure S1), and Holm–Bonferroni table (`tables/table1_primary_results.csv`). |
| **Days 15–16** | **Stage 9 & Stage 10** | Execute `scripts/stage9_make_figures.py` to produce Figure 1, Figure 2A, and Figure S1. If on schedule at the Day 14 Checkpoint, execute optional `scripts/stage10_stretch_steering.py` to produce Figure 2B. |
| **Days 17–18** | **Final Write-Up** | Populate the final research report (`README.md`) following the Section 12 template, archive all JSON/CSV artifacts and extracted direction tensors (`results/stage4_directions.pt`) for Option A, and freeze repository. |

### 11.2 Mandatory Day 14 Checkpoint & Strict Cut Order
At the start of Day 14 (and on Day 4 if template authoring is delayed), evaluate progress against the schedule and apply the following cuts **strictly in numerical order**:
1. **Cut 1 (First Cut if Behind at Day 14):** Drop **Stage 10 (Stretch Goal)**—omit the multi-$\alpha$ PopQA steering sweep and `figures/fig2b_selective_abstention_stretch.pdf`.
2. **Cut 2 (Second Cut if Stage 7 Is Incomplete at Day 14):** Drop the **200-question MMLU companion evaluation** (`Panel A2` and `Panel B2` of Figure 1), retaining the 200-paragraph WikiText-2 neutral-text KL divergence (`Panel A1` and `Panel B1`) as the sole capability retention metric.
3. **Cut 3 (Early Cut if Still in Stage 1 at End of Day 4):** Reduce total template count from 96 to **64 templates** (`n_templates_cut3 = 64`: 32 Family A and 32 Family B, split **16 train / 8 val / 8 test** per family = 32 train / 16 val / 16 test templates total, maintaining 4 pairs per template = 256 pairs / 512 prompts, and $D_{\text{B-ctrl}} = 32$ prompts).
4. **Protected Core Invariants (NEVER CUT Under Any Circumstances):**
   * The causal necessity and sufficiency tests on $D_{\text{test}}$ with all three control tiers (99 Monte Carlo random directions, 19 shuffled-label directions, and 1 orthogonalized $r_{\text{concept}}$).
   * The Family B private-detail answerable control check ($D_{\text{B-ctrl}}$, H3).
   * The core PopQA unsteered projection and confidence-baseline test (Stage 8: H4, H5, Figure 2A, and exploratory Figure S1).

---

## 12. Final Research Write-Up Template (`README.md`)

The final project write-up must use the exact 7-section structure below, with every claim grounded in the locked outputs of Stages 5–10:

1. **Section 1 — Research Question & Mechanistic Framing:**
   * State the primary research question, contrast Hypothesis A (contextual insufficiency detector) against Hypothesis B (internal epistemic uncertainty detector), and summarize the 5 pre-registered Holm–Bonferroni tests (H1–H5).
2. **Section 2 — Experimental Design & De-Confounded Minimal Pairs:**
   * Describe `Qwen/Qwen2.5-3B-Instruct` (or `meta-llama/Llama-3.2-3B-Instruct` if triggered), the single-token `"Unknown"` protocol, the 96-template dataset across Family A and Family B, the template-level split (`48 / 24 / 24`), the n-gram leakage gate ($\max \vert{}\phi\vert{} \le 0.15$), the $D_{\text{B-ctrl}}$ private-detail control, and the validation selection of $(l^*, i^*)$ with reference to `tables/table2_sensitivity_grid.md`.
3. **Section 3 — Causal Necessity, Sufficiency & Capability Pareto Frontier:**
   * Present H1 (necessity on $D_{\text{test}}^{\text{unans}}$) and H2 (sufficiency on $D_{\text{test}}^{\text{ans}}$) alongside `figures/fig1_pareto_tradeoff.png` and `tables/table3_generation_audit.md`. Discuss the trade-off between behavioral shift and WikiText-2 KL / MMLU accuracy, and explicitly note whether Branch Rule 4 flagged a logit-generation discrepancy.
4. **Section 4 — Specificity Against Monte Carlo, Label-Permutation, and In-Manifold Concept Controls:**
   * Report the empirical rank and $z$-score of $\hat{r}_{i^*}^{(l^*)}$ against the 99-direction Monte Carlo null distribution, the 19-direction label-permutation null distribution, and the orthogonalized concept direction $r_{\text{concept}}$ (reporting pre-orthogonalization cosine $\cos(\theta_{\text{pre}})$) from `tables/table1_primary_results.md`.
5. **Section 5 — Input Unanswerability vs. Internal Uncertainty (PopQA & $D_{\text{B-ctrl}}$ Results):**
   * Present H3 ($D_{\text{B-ctrl}}$), H4 ($\text{AUROC}_{\text{proj}}$ on popularity-matched PopQA), and H5 ($\Delta \text{AUROC}$ over $\text{AUROC}_{\text{maxp}}$ and $\text{AUROC}_{\text{ent}}$) referencing `figures/fig2a_epistemic_projections.png` and exploratory `figures/fig_s1_layerwise_popqa_auroc.png` (plus `figures/fig2b_selective_abstention_stretch.png` if Stage 10 was executed). Map the observed outcomes directly to the Section 6.3 Interpretation Matrix.
6. **Section 6 — Known Limitations & Confound Analysis (Pre-Filled Mandatory Discussion):**
   * **Domain & Format Transfer Shift:** Discuss that $\hat{r}$ is extracted from in-context sufficiency contrasts (Families A & B) and evaluated zero-shot on closed-book parametric retrieval (PopQA); a null result on H4 can reflect format non-transferability rather than absence of an internal uncertainty representation.
   * **Residual Entity Popularity Variance:** Note that while 1-to-1 caliper matching on `s_pop` ($\delta_{\text{pop}} = 0.25 \cdot \text{SD}$) balances marginal pageview popularity between correct and wrong PopQA sets, unmeasured entity familiarity within the model's pretraining corpus can still contribute residual variance.
   * **Single-Token Constrained Protocol vs. Free-Form Abstention:** Discuss how constraining abstention to `"Unknown"` sharpens first-token log-odds measurement ($S(p)$) at the cost of narrowing natural multi-sentence epistemic hedging, mitigated by the 50-sample regex generation audit.
   * **Rank-1 Linear Subspace Assumption:** Acknowledge that difference-in-means extraction isolates a single 1D linear direction at $(l^*, i^*)$, leaving multi-dimensional or non-linear uncertainty manifolds to downstream probing.
7. **Section 7 — Transition & Reusable Assets for Option A:**
   * Document how the verified hook infrastructure, activation caching pipeline, popularity-matched PopQA splits, and extracted abstention subspace $\hat{r}_{i^*}^{(l^*)}$ transfer directly into Option A (probing internal truth vs. epistemic uncertainty representations).