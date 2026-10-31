# Agent task: pin the Llama chat-template date, and make the local judge memory-safe and resumable

**Repository:** `DakshUbhadia/unanswerability_to_hallucination` (written against commit `31cc6e6`)
**Context:** on Kaggle, preflight G1-1 detected that Llama-3.2's chat template renders today's date. The diagnostic (`results/exploratory/date_diagnostic.json`) then measured its effect. Separately, the Tier 1 judge pass ran out of GPU memory on a T4.

**Do not:**
- re-run or modify any registered stage output or archive;
- change any pre-registration file;
- change any threshold, seed or prompt text;
- run anything on GPU.

Commit message prefix: `fix-date-judge:`.

---

## Evidence (from `date_diagnostic.json`; already on `main`)

| Fact | Value |
|---|---|
| Date that reproduces all 1,000 stored `input_ids_sha256` in `data/popqa_sample.jsonl` | **27 Oct 2026** (1000/1000 rows) |
| Stage 4 direction re-derived from train pairs at **19 Oct 2026** (the Stage 3 run date) | cosine 0.99999994, relative norm error 0.0, i.e. **exact** |
| Stage 5 val log-odds re-measured at **24 Oct 2026** (the Stage 5–10 run date) | 6.779 / −11.317 vs recorded 6.781 / −11.324 (batch-1 vs batched padding noise) |
| Direction at 27 Oct vs Stage 4 | cosine 0.999913, relative norm error **0.0014** (> E4's 1e-3 tolerance) |
| Val log-odds at 27 Oct vs Stage 5 | mean differences 0.019 and 0.044 (E4 tolerance 0.05, which is borderline) |
| Max per-prompt \|Δ log-odds\| across dates | up to 0.39 nats. Val AUROC = 1.0 for every date |
| Max \|Δ projection\| across dates | ≤ 0.039 (Stage 8 vs Stage 6 tolerance is 0.05 × scale, so this passes) |

**Conclusion:**
- Every registered stage rendered prompts with **its own run day's date** (Stage 3/4: Oct 19; Stages 5–10: Oct 24; PopQA regeneration: Oct 27).
- The effect is substantively negligible (identical AUROCs, cosine ≥ 0.99988), but it is large enough to break exact-reproduction tolerances.

---

## Fix 1: date pinning

### 1.1 Config

Append to `MasterConfig`. Never edit existing fields.

```python
chat_template_date: str = "27 Oct 2026"          # date used by the canonical PopQA regeneration; all new Llama renders
chat_template_date_stage3_4: str = "19 Oct 2026" # reproduce Stage 3/4 artifacts (parity checks only)
chat_template_date_stage5_10: str = "24 Oct 2026"# reproduce Stage 5–10 artifacts (parity checks only)
```

### 1.2 Thread the date through every Llama chat render

1. **`src/tokenization_contract.canonical_input_ids`:** add the keyword `date_string: str | None = CFG.chat_template_date` and pass it to `apply_chat_template(..., date_string=date_string)`.
   - Templates that do not use `date_string` (Qwen) ignore extra keyword arguments, but verify this on Qwen's template in a test with a stub.
   - Keep the single-BOS and suffix assertions.
2. **Every other place that renders a Llama chat must use the same default:**
   - `src/model_utils.py` (`_chat_ids`, `discover_post_instruction_suffix`, any generation helper);
   - `src/pilot_check.py`;
   - `src/activation_cache.format_and_verify_batch_inputs`;
   - `src/eval_common.build_chat_batches`;
   - `src/exploratory/chat.py` (`render_ids`, `discover_suffix`);
   - E3 generation and capture.

   Find all of them with `grep -rn apply_chat_template src scripts`, then route them through one helper, `src/tokenization_contract.render_chat(tokenizer, messages, *, date_string=CFG.chat_template_date, **kw)`.
   **Do not change the judge.** `src/abstention_classifier.py` renders Qwen chats, and its behaviour must stay byte-identical.
3. **Parity checks that reproduce a past artifact** must render with that artifact's date:
   - **E4 `syn` direction parity** against Stage 4: use `chat_template_date_stage3_4`.
   - **E4 `syn` validation log-odds parity** against Stage 5: use `chat_template_date_stage5_10`.
   - Features themselves are rendered with the default (27 Oct). The two parity checks therefore need their own extra forward passes over the train/val pairs at their artifact dates. That's cheap: 384 + 192 prompts.
   - Audit any other check that compares a fresh measurement against a Stage 3–10 artifact with a tight tolerance, and apply the same rule. Stage 8's comparison against Stage 6 projections (0.05 × scale) and Stage 10's α=0 label identity work at 27 Oct and need no change.
4. **E3 preflight:** if `CFG.chat_template_date` is set, re-render the first 20 rows of `data/popqa_sample.jsonl` with it.
   - If all 20 hashes match, write `"template_date": "pinned"` and `"template_date_value": "27 Oct 2026"`, then continue to the Qwen checks.
   - If not, keep exit code 3.
   - Keep `template_date_probe` in the output for the record.

### 1.3 Tests (`tests/test_date_pin.py`, CPU, stub tokenizers)

- A stub Llama-like template containing `{{ date_string }}` renders the pinned date by default, and a different date when one is passed explicitly.
- `canonical_input_ids` with the default reproduces a fixture hash computed for "27 Oct 2026".
- The Qwen-like stub template output is unchanged whether or not `date_string` is passed.
- The E4 parity helper calls the renderer with the Stage 3/4 date for the direction and the Stage 5–10 date for the log-odds (monkeypatch the renderer and capture the arguments).
- The preflight writes `"pinned"` when the stub hashes match, and exits with code 3 otherwise.

### 1.4 DEV entry (next free ID)

> Llama-3.2 chat template injects the render date. Registered stages rendered with their run dates (Stage 2 pilot: Oct 18; Stage 3/4: Oct 19; Stages 5–10: Oct 24; Tier 1 PopQA regeneration: Oct 27).
>
> Measured effect (`results/exploratory/date_diagnostic.json`): val AUROC 1.0 at all dates; max |Δ log-odds| 0.39 nats; max |Δ projection| 0.039; re-derived direction cosine ≥ 0.99988.
>
> No registered result is re-run. All new Llama renders are pinned to 27 Oct 2026 so they reproduce the Tier 1 regenerated inputs (1000/1000 hashes). Reproduction checks against Stage 3/4 and Stage 5–10 artifacts render with those artifacts' dates.

Also add one sentence to the README's open-issues section.

---

## Fix 2: local judge memory and resumability

### 2.1 Out of memory

In `src/abstention_classifier.py::_score_local_batch` (and the single-item scorer above it), the code calls `model(...).logits.float()` and then `log_softmax` over **all positions × the full 152k vocabulary**, for 3 sequences per item.

**Change:**
1. Run the base model to get hidden states: `model.model(input_ids, attention_mask).last_hidden_state`. Check that this matches the logits path for Qwen2.
2. Gather **only** the positions needed for label scoring (`len(prompt)-1 … len(prompt)-1+len(label)-1` for each sequence).
3. Apply `model.lm_head` to the gathered rows, then `log_softmax` in float32.
4. Default `judge_batch_size` stays 8 (config untouched). Memory now scales with the number of label tokens, not with sequence length × vocabulary.

**Equivalence test** (tiny random Qwen2-style or Llama-style model on CPU): the new scorer's `judge_logprobs` equal the old full-logits scorer's to within 1e-5, for both a padded batch and a single item.

### 2.2 Resumability

`classify_texts` scores **all** pending items before appending anything to the cache, so a crash loses the whole run.

**Change:** score in chunks of `batch_size` and append each chunk's records to the cache immediately. Cache keys and record contents must stay identical.

**Test:** inject a failure into the 3rd chunk. The first two chunks are in the cache, and a rerun scores only the remaining items.

### 2.3 Runner

In `scripts/kaggle_run.sh`, pass `JUDGE_BATCH_SIZE` (if set) to `judge_pass.py --batch-size`.

---

## Record, don't fix: SQuAD shortfall

The prepared SQuAD sample has **210 + 210** rows instead of the pre-registered 300 + 300. The title cap and the 350-token length filter limited it. The E4 plan's stop table says to report this, proceed with what is available, and never relax the filters silently.

Add it to the E4 DEV entry and to `e4_summary.md` (via `e4_analyze.py`, which should print the realised class counts).

---

## Acceptance

- [ ] The CPU suite is green on a fresh Linux clone, including the new tests
- [ ] `grep -rn apply_chat_template src scripts` shows only the shared helper and the judge
- [ ] DEV entries added; pushed to `main`
- [ ] Report back the exact files changed. The owner will re-run Kaggle cell G1-1, which should now print `template_date: pinned`, and then continue with G1-5
