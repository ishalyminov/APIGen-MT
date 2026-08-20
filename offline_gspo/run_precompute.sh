#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${QWEN_PYBIN:-/home/jovyan/.mlspace/envs/qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
PREPARED_DATASET="${PREPARED_DATASET:?PREPARED_DATASET is required}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:?PREPARED_MANIFEST is required}"
OLD_LOGPS_DATASET="${OLD_LOGPS_DATASET:?OLD_LOGPS_DATASET is required}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT is required}"

for required in \
  "$PYBIN" \
  "$BASE_MODEL/config.json" \
  "$PREPARED_DATASET/state.json" \
  "$PREPARED_MANIFEST"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$LOG_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYBIN" -m offline_gspo.precompute_old_logps \
  --dataset "$PREPARED_DATASET" \
  --dataset-manifest "$PREPARED_MANIFEST" \
  --model "$BASE_MODEL" \
  --output "$OLD_LOGPS_DATASET" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$LOG_ROOT/precompute_old_logps.log"
