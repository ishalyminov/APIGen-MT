#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${QWEN_PYBIN:-/home/jovyan/.mlspace/envs/qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
PREPARED_DATASET="${PREPARED_DATASET:?PREPARED_DATASET is required}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:?PREPARED_MANIFEST is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
LOG_ROOT="${LOG_ROOT:?LOG_ROOT is required}"
LR_TAG="${LR_TAG:?LR_TAG is required}"
LEARNING_RATE="${LEARNING_RATE:?LEARNING_RATE is required}"
EPSILON_LOW="${EPSILON_LOW:-0.0003}"
EPSILON_HIGH="${EPSILON_HIGH:-0.0004}"
EPISODES_PER_BATCH="${EPISODES_PER_BATCH:-1}"
TRAINED="$OUTPUT_ROOT/${LR_TAG}_text"
SERVING="$OUTPUT_ROOT/${LR_TAG}_vlserving"
SERVING_BUILDING="$SERVING.building"
PROMPT="$REPO_ROOT/prompts/reasoning_next_action_system_v4.txt"
TEMPLATE="$REPO_ROOT/templates/qwen35_toolonly_base.jinja"
BASE_MODEL_CONTRACT="$REPO_ROOT/offline_gspo/qwen35_2b_base_contract.json"

for required in \
  "$PYBIN" \
  "$BASE_MODEL/config.json" \
  "$PREPARED_DATASET/state.json" \
  "$PREPARED_MANIFEST" \
  "$BASE_MODEL_CONTRACT" \
  "$PROMPT" \
  "$TEMPLATE"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
if [ -f "$SERVING/.vlserving_done" ]; then
  echo "already complete: $SERVING"
  exit 0
fi
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$TRAINED/.train_done" ]; then
  if [ -d "$TRAINED" ] && [ -n "$(ls -A "$TRAINED" 2>/dev/null)" ]; then
    echo "partial training directory exists without .train_done: $TRAINED" >&2
    exit 1
  fi
  "$PYBIN" -m offline_gspo.train_offline_gspo \
    --model "$BASE_MODEL" \
    --dataset "$PREPARED_DATASET" \
    --dataset-manifest "$PREPARED_MANIFEST" \
    --base-model-contract "$BASE_MODEL_CONTRACT" \
    --chat-template "$TEMPLATE" \
    --system-prompt "$PROMPT" \
    --output-dir "$TRAINED" \
    --learning-rate "$LEARNING_RATE" \
    --epochs 1 \
    --episodes-per-batch "$EPISODES_PER_BATCH" \
    --epsilon-low "$EPSILON_LOW" \
    --epsilon-high "$EPSILON_HIGH" \
    2>&1 | tee -a "$LOG_ROOT/${LR_TAG}.train.log"
fi

if [ -d "$SERVING" ] && [ -n "$(ls -A "$SERVING" 2>/dev/null)" ]; then
  echo "partial serving directory exists without .vlserving_done: $SERVING" >&2
  exit 1
fi
if [ -d "$SERVING" ]; then
  rmdir "$SERVING"
fi
if [ -d "$SERVING_BUILDING" ]; then
  mv "$SERVING_BUILDING" "$SERVING_BUILDING.abandoned.$(date +%s)"
fi
"$PYBIN" "$REPO_ROOT/training/convert_to_vllm.py" \
  --base "$BASE_MODEL" \
  --trained "$TRAINED" \
  --out "$SERVING_BUILDING" \
  2>&1 | tee "$LOG_ROOT/${LR_TAG}.convert.log"
cmp --silent "$TEMPLATE" "$SERVING_BUILDING/chat_template.jinja"
test -s "$SERVING_BUILDING/toolonly_contract.json"
touch "$SERVING_BUILDING/.vlserving_done"
mv "$SERVING_BUILDING" "$SERVING"
echo "complete: $SERVING"
