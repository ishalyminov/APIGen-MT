#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${QWEN_PYBIN:-/home/jovyan/.mlspace/envs/qwen35/bin/python}"
BASE_MODEL="${BASE_MODEL:?BASE_MODEL is required}"
BASE_MODEL_CONFIG_SHA256="${BASE_MODEL_CONFIG_SHA256:?BASE_MODEL_CONFIG_SHA256 is required}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
TRAIN_GPU="${TRAIN_GPU:-0}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
DATA_SHA256="${DATA_SHA256:?DATA_SHA256 is required}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH is required}"
MANIFEST_SHA256="${MANIFEST_SHA256:?MANIFEST_SHA256 is required}"
PROMPT_SHA256="${PROMPT_SHA256:?PROMPT_SHA256 is required}"
TEMPLATE_SHA256="${TEMPLATE_SHA256:?TEMPLATE_SHA256 is required}"
CORPUS_RUN_ID="${CORPUS_RUN_ID:?CORPUS_RUN_ID is required}"
LR_TAG="${LR_TAG:?LR_TAG is required}"
LEARNING_RATE="${LEARNING_RATE:?LEARNING_RATE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
NO_TOOL_REPEAT_FACTOR="${NO_TOOL_REPEAT_FACTOR:?NO_TOOL_REPEAT_FACTOR is required}"
SINGLE_CALL_REPEAT_FACTOR="${SINGLE_CALL_REPEAT_FACTOR:?SINGLE_CALL_REPEAT_FACTOR is required}"
PARALLEL_REPEAT_FACTOR="${PARALLEL_REPEAT_FACTOR:?PARALLEL_REPEAT_FACTOR is required}"
OTHER_REPEAT_FACTOR="${OTHER_REPEAT_FACTOR:?OTHER_REPEAT_FACTOR is required}"
SUPERVISION_CONTRACT="${SUPERVISION_CONTRACT:?SUPERVISION_CONTRACT is required}"
PREFIX_UNIT="${PREFIX_UNIT:?PREFIX_UNIT is required}"
TRAINED="$OUTPUT_ROOT/${LR_TAG}_text"
SERVING="$OUTPUT_ROOT/${LR_TAG}_vlserving"
LOG_ROOT="${LOG_ROOT:-/mnt/shared_ru.ml.SZ-5_000264/gambashidze/qwen35_toolonly_sft_sweep_artifacts/logs}"
PROMPT="$REPO_ROOT/prompts/tool_only_system.txt"
TEMPLATE="$REPO_ROOT/templates/qwen35_toolonly_base.jinja"
CATALOG="$REPO_ROOT/data/tools_openai_format.json"

for required in "$PYBIN" "$BASE_MODEL/config.json" "$DATA_PATH" "$MANIFEST_PATH" "$PROMPT" "$TEMPLATE" "$CATALOG"; do
  if [ ! -e "$required" ]; then
    echo "missing required path: $required" >&2
    exit 1
  fi
done

if [ "$EPOCHS" != "1" ]; then
  echo "this sweep contract requires exactly one epoch, got: $EPOCHS" >&2
  exit 1
fi
ACTUAL_BASE_MODEL_CONFIG_SHA256="$(sha256sum "$BASE_MODEL/config.json" | cut -d' ' -f1)"
if [ "$ACTUAL_BASE_MODEL_CONFIG_SHA256" != "$BASE_MODEL_CONFIG_SHA256" ]; then
  echo "base model config changed: expected=$BASE_MODEL_CONFIG_SHA256 actual=$ACTUAL_BASE_MODEL_CONFIG_SHA256" >&2
  exit 1
fi
for repeat_factor in \
  "$NO_TOOL_REPEAT_FACTOR" \
  "$SINGLE_CALL_REPEAT_FACTOR" \
  "$PARALLEL_REPEAT_FACTOR" \
  "$OTHER_REPEAT_FACTOR"; do
  if ! [[ "$repeat_factor" =~ ^[1-9][0-9]*$ ]]; then
    echo "repeat factors must be positive integers, got: $repeat_factor" >&2
    exit 1
  fi
done
# train_toolcalling_toolonly.py fixes ordinary/other examples at one and has
# explicit CLI switches for the other three behavior classes.
if [ "$OTHER_REPEAT_FACTOR" != "1" ]; then
  echo "trainer requires OTHER_REPEAT_FACTOR=1, got: $OTHER_REPEAT_FACTOR" >&2
  exit 1
fi
if [ "$(basename "$OUTPUT_ROOT")" != "$CORPUS_RUN_ID" ]; then
  echo "output root does not match immutable corpus run id: $OUTPUT_ROOT vs $CORPUS_RUN_ID" >&2
  exit 1
fi

ACTUAL_DATA_SHA256="$(sha256sum "$DATA_PATH" | cut -d' ' -f1)"
if [ "$ACTUAL_DATA_SHA256" != "$DATA_SHA256" ]; then
  echo "dataset changed after submission: expected=$DATA_SHA256 actual=$ACTUAL_DATA_SHA256" >&2
  exit 1
fi
ACTUAL_MANIFEST_SHA256="$(sha256sum "$MANIFEST_PATH" | cut -d' ' -f1)"
if [ "$ACTUAL_MANIFEST_SHA256" != "$MANIFEST_SHA256" ]; then
  echo "manifest changed after submission: expected=$MANIFEST_SHA256 actual=$ACTUAL_MANIFEST_SHA256" >&2
  exit 1
fi
ACTUAL_PROMPT_SHA256="$("$PYBIN" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest())' "$PROMPT")"
if [ "$ACTUAL_PROMPT_SHA256" != "$PROMPT_SHA256" ]; then
  echo "system prompt changed after submission: expected=$PROMPT_SHA256 actual=$ACTUAL_PROMPT_SHA256" >&2
  exit 1
fi
ACTUAL_TEMPLATE_SHA256="$(sha256sum "$TEMPLATE" | cut -d' ' -f1)"
if [ "$ACTUAL_TEMPLATE_SHA256" != "$TEMPLATE_SHA256" ]; then
  echo "chat template changed after submission: expected=$TEMPLATE_SHA256 actual=$ACTUAL_TEMPLATE_SHA256" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
if [ -f "$SERVING/.vlserving_done" ]; then
  echo "already complete: $SERVING"
  exit 0
fi

echo "training $MODEL_TAG: base=$BASE_MODEL gpu=$TRAIN_GPU lr=$LEARNING_RATE epochs=$EPOCHS max_length=$MAX_LENGTH"
echo "data=$DATA_PATH output=$TRAINED"
sha256sum "$DATA_PATH" "$PROMPT" "$TEMPLATE"

if [ ! -f "$TRAINED/.train_done" ]; then
  if [ -d "$TRAINED" ] && [ -n "$(ls -A "$TRAINED" 2>/dev/null)" ]; then
    echo "partial training directory exists without .train_done: $TRAINED" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYBIN" "$REPO_ROOT/training/train_toolcalling_toolonly.py" \
      --model "$BASE_MODEL" \
      --model_tag "$MODEL_TAG" \
      --data "$DATA_PATH" \
      --tools_catalog "$CATALOG" \
      --system_prompt_file "$PROMPT" \
      --chat_template_file "$TEMPLATE" \
      --output_dir "$TRAINED" \
      --epochs "$EPOCHS" \
      --lr "$LEARNING_RATE" \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --max_length "$MAX_LENGTH" \
      --val_ratio 0.1 \
      --seed 42 \
      --warmup_ratio 0.03 \
      --weight_decay 0.01 \
      --lr_scheduler_type cosine \
      --no_tool_repeat "$NO_TOOL_REPEAT_FACTOR" \
      --single_call_repeat "$SINGLE_CALL_REPEAT_FACTOR" \
      --parallel_repeat "$PARALLEL_REPEAT_FACTOR" \
      --expected_supervision_contract "$SUPERVISION_CONTRACT" \
      --expected_prefix_unit "$PREFIX_UNIT" \
      --require_all_rows \
      --save_strategy no \
      --save_total_limit 1 \
      --num_workers 4 \
      2>&1 | tee "$LOG_ROOT/${LR_TAG}.train.log"
  touch "$TRAINED/.train_done"
fi

if [ -d "$SERVING" ] && [ -n "$(ls -A "$SERVING" 2>/dev/null)" ]; then
  echo "partial serving directory exists without .vlserving_done: $SERVING" >&2
  exit 1
fi
"$PYBIN" "$REPO_ROOT/training/convert_to_vllm.py" \
  --base "$BASE_MODEL" \
  --trained "$TRAINED" \
  --out "$SERVING" \
  2>&1 | tee "$LOG_ROOT/${LR_TAG}.convert.log"
cmp --silent "$TEMPLATE" "$SERVING/chat_template.jinja" || {
  echo "serving checkpoint chat template differs from the training template" >&2
  exit 1
}
test -s "$SERVING/toolonly_contract.json" || {
  echo "serving checkpoint is missing toolonly_contract.json" >&2
  exit 1
}
touch "$SERVING/.vlserving_done"
echo "complete: $SERVING"
