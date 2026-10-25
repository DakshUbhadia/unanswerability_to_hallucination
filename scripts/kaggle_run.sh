#!/usr/bin/env bash
# Usage: bash scripts/kaggle_run.sh <stage> [<stage> ...]
#   stage: 0..10, a range such as 5-9, or "all" (= 5 6 7 8 9 10).
# Every stage >= 5 syncs its outputs to GitHub right after it finishes, so a session timeout keeps finished work.
set -euo pipefail
shopt -s nullglob
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git_remote_reset() { git remote set-url origin https://github.com/DakshUbhadia/unanswerability_to_hallucination.git; }
trap git_remote_reset EXIT

pip install -q -r requirements-kaggle.txt
git config user.name >/dev/null || git config user.name "Kaggle Runner"
git config user.email >/dev/null || git config user.email "kaggle-runner@users.noreply.github.com"

stages=()
for arg in "$@"; do
  if [[ "$arg" == all ]]; then
    stages+=(5 6 7 8 9 10)
  elif [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    for ((s = BASH_REMATCH[1]; s <= BASH_REMATCH[2]; s++)); do stages+=("$s"); done
  else
    stages+=("$arg")
  fi
done
((${#stages[@]})) || { echo "Usage: bash scripts/kaggle_run.sh <stage|a-b|all> ..." >&2; exit 2; }
echo "[kaggle_run] stages: ${stages[*]}"

tensor_stage_requested=0
for stage in "${stages[@]}"; do
  case "$stage" in 3|4|5|6|7|8|10) tensor_stage_requested=1 ;; esac
done
if (( tensor_stage_requested )); then
  : "${HF_TENSOR_REPO_ID:?Set HF_TENSOR_REPO_ID to the private HF dataset repository (owner/name).}"
  python scripts/kaggle_sync.py --action pull --hf-repo "$HF_TENSOR_REPO_ID"
fi

for stage in "${stages[@]}"; do
  if [[ "$stage" == pin-judge ]]; then
    python -u scripts/pin_judge_model.py
    continue
  fi
  if [[ "$stage" == judge ]]; then
    python -u scripts/judge_pass.py
    continue
  fi
  if [[ "$stage" == rescore6 ]]; then
    python -u scripts/stage6_rescore_generations.py --offline
    continue
  fi
  matches=(scripts/stage"${stage}"_*.py)
  ((${#matches[@]})) || { echo "No script for stage $stage" >&2; exit 2; }
  if [[ "$stage" =~ ^[0-9]+$ ]] && (( stage > 2 )); then
    [[ -f results/stage2_pilot.json ]] || { echo "Missing results/stage2_pilot.json; run Stage 2 first." >&2; exit 2; }
    locked_model="$(python -c 'from src.stage_contract import resolve_downstream_model_name; print(resolve_downstream_model_name())')"
    [[ -n "$locked_model" ]] || { echo "Stage 2 has no locked model_name." >&2; exit 2; }
    export STAGE2_MODEL_NAME="$locked_model"
  fi
  echo "================ Stage $stage ================"
  for script in "${matches[@]}"; do
    args=()
    help="$(python "$script" --help 2>&1)"
    [[ "$help" == *--allow-kaggle-env* ]] && args+=(--allow-kaggle-env)
    [[ "${FORCE:-0}" == 1 && "$help" == *--force* ]] && args+=(--force)
    if [[ "$stage" =~ ^[0-9]+$ ]] && (( stage > 2 )) && [[ "$help" == *--model-name* ]]; then
      args+=(--model-name "$STAGE2_MODEL_NAME")
    fi
    if [[ "$stage" == 7 && "${SKIP_MMLU:-0}" == 1 ]]; then args+=(--skip-mmlu); fi
    if [[ "$stage" == 2 && "${STAGE2_POPQA_REGENERATE:-0}" == 1 ]]; then args+=(--regenerate-popqa-canonical); fi
    if [[ "$stage" == 2 && "${STAGE2_POPQA_REMATCH:-0}" == 1 ]]; then args+=(--rematch-only --offline); fi
    if [[ "$stage" == 6 && "${STAGE6_GENERATION_ONLY:-0}" == 1 ]]; then args+=(--generation-only); fi
    if [[ "$stage" == 10 && "${STAGE10_GENERATION_ONLY:-0}" == 1 ]]; then args+=(--generation-only); fi
    if [[ "$stage" == 10 && "${STAGE10_ANALYZE_ONLY:-0}" == 1 ]]; then args+=(--analyze-only --offline); fi
    python -u "$script" "${args[@]}"
  done
  if [[ "$stage" == 2 ]]; then
    python scripts/measure_stage2_tokenizer_parity.py
  fi
  if [[ "$stage" == 3 || "$stage" == 4 ]]; then
    python scripts/kaggle_sync.py --action push --stage "$stage" --hf-repo "$HF_TENSOR_REPO_ID"
  fi
  skip_artifact_tests=0
  [[ "$stage" == 6 && "${STAGE6_GENERATION_ONLY:-0}" == 1 ]] && skip_artifact_tests=1
  [[ "$stage" == 10 && "${STAGE10_GENERATION_ONLY:-0}" == 1 ]] && skip_artifact_tests=1
  if [[ "$stage" =~ ^[0-9]+$ ]] && (( stage >= 5 )) && (( ! skip_artifact_tests )); then
    pytest -q -p no:warnings "tests/test_stage${stage}.py" -m "not integration" || echo "[kaggle_run] WARNING: Stage $stage unit tests failed" >&2
    pytest -q -p no:warnings "tests/test_stage${stage}.py" -m integration || echo "[kaggle_run] WARNING: Stage $stage artifact check failed" >&2
    python scripts/kaggle_sync.py --action push --stage "$stage"
  fi
done

# Save completed artifacts before tests so a failing test run does not discard Kaggle work.
artifact_paths=(results/*.json results/*.jsonl results/*.csv data/*.jsonl data/*.json data/*.csv data/review/*.csv tables/* figures/*)
if [[ -f DEVIATIONS.md ]]; then artifact_paths+=(DEVIATIONS.md); fi
if [[ -f README.md ]]; then artifact_paths+=(README.md); fi
if ((${#artifact_paths[@]})); then
  git add -f -- "${artifact_paths[@]}"
fi
if ! git diff --cached --quiet; then
  git config user.name "Kaggle Runner"
  git config user.email "kaggle-runner@users.noreply.github.com"
  git commit -m "Run Kaggle stages ${stages[*]}"
fi
git push origin main

set +e
pytest -q -p no:warnings tests/ -m "not integration and not slow"
pytest_status=$?
set -e
exit "$pytest_status"
