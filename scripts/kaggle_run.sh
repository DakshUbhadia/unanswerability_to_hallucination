#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
git_remote_reset() { git remote set-url origin https://github.com/DakshUbhadia/unanswerability_to_hallucination.git; }
trap git_remote_reset EXIT
pip install -q -r requirements-kaggle.txt
for stage in "$@"; do
  matches=(scripts/stage"${stage}"_*.py)
  ((${#matches[@]})) || { echo "No script for stage $stage" >&2; exit 2; }
  for script in "${matches[@]}"; do
    args=()
    help="$(python "$script" --help 2>&1 || true)"
    [[ "$help" == *--allow-kaggle-env* ]] && args+=(--allow-kaggle-env)
    [[ "${FORCE:-0}" == 1 && "$help" == *--force* ]] && args+=(--force)
    python "$script" "${args[@]}"
  done
done
pytest -q -m "not slow"
git add -f results/*.json results/*.csv data/ tables/ figures/ 2>/dev/null || true
if ! git diff --cached --quiet; then
  git config user.name "Kaggle Runner"
  git config user.email "kaggle-runner@users.noreply.github.com"
  git commit -m "Run Kaggle stages"
  git push origin main
fi
