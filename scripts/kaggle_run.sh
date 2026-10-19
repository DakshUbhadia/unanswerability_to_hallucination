#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git_remote_reset() { git remote set-url origin https://github.com/DakshUbhadia/unanswerability_to_hallucination.git; }
trap git_remote_reset EXIT

pip install -q -r requirements-kaggle.txt

stages=("$@")
tensor_stage_requested=0
for stage in "${stages[@]}"; do
  if [[ "$stage" == 3 || "$stage" == 4 ]]; then tensor_stage_requested=1; fi
done
if (( tensor_stage_requested )); then
  : "${HF_TENSOR_REPO_ID:?Set HF_TENSOR_REPO_ID to the private HF dataset repository (owner/name).}"
  python scripts/kaggle_sync.py --action pull --hf-repo "$HF_TENSOR_REPO_ID"
fi

for stage in "${stages[@]}"; do
  matches=(scripts/stage"${stage}"_*.py)
  ((${#matches[@]})) || { echo "No script for stage $stage" >&2; exit 2; }
  if [[ "$stage" =~ ^[0-9]+$ ]] && (( stage > 2 )); then
    [[ -f results/stage2_pilot.json ]] || { echo "Missing results/stage2_pilot.json; run Stage 2 first." >&2; exit 2; }
    locked_model="$(python -c 'from src.stage_contract import resolve_downstream_model_name; print(resolve_downstream_model_name())')"
    [[ -n "$locked_model" ]] || { echo "Stage 2 has no locked model_name." >&2; exit 2; }
    export STAGE2_MODEL_NAME="$locked_model"
  fi
  for script in "${matches[@]}"; do
    args=()
    help="$(python "$script" --help 2>&1)"
    [[ "$help" == *--allow-kaggle-env* ]] && args+=(--allow-kaggle-env)
    [[ "${FORCE:-0}" == 1 && "$help" == *--force* ]] && args+=(--force)
    if [[ "$stage" =~ ^[0-9]+$ ]] && (( stage > 2 )) && [[ "$help" == *--model-name* ]]; then
      args+=(--model-name "$STAGE2_MODEL_NAME")
    fi
    python "$script" "${args[@]}"
  done
  if [[ "$stage" == 2 ]]; then
    python scripts/measure_stage2_tokenizer_parity.py
  fi
  if [[ "$stage" == 3 || "$stage" == 4 ]]; then
    python scripts/kaggle_sync.py --action push --stage "$stage" --hf-repo "$HF_TENSOR_REPO_ID"
  fi
done

# Save completed artifacts before tests so a failing test run does not discard Kaggle work.
artifact_paths=(results/*.json results/*.csv results/*.png results/*.pdf data/*.jsonl data/*.json data/*.csv data/review/*.csv tables/* figures/*)
if [[ -f DEVIATIONS.md ]]; then artifact_paths+=(DEVIATIONS.md); fi
if ((${#artifact_paths[@]})); then
  git add -f -- "${artifact_paths[@]}"
fi
if ! git diff --cached --quiet; then
  git config user.name "Kaggle Runner"
  git config user.email "kaggle-runner@users.noreply.github.com"
  git commit -m "Run Kaggle stages"
fi
git push origin main

set +e
pytest -q tests/test_stage0.py tests/test_stage1.py tests/test_stage2.py tests/test_stage3.py -m "not integration and not slow"
pytest_status=$?
set -e
exit "$pytest_status"
