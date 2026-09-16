#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${APIGEN_PYTHON:-python}"
feature="${APIGEN_FEATURE:-parallel}"   # parallel | refusal | mixed
rows="${APIGEN_ROWS:-500}"
mode="${APIGEN_MODE:-multi-turn}"
turns="${APIGEN_TURNS:-3}"
actions="${APIGEN_ACTIONS:-2}"
blueprint_max_actions="${APIGEN_BLUEPRINT_MAX_ACTIONS:-}"
actions_min="${APIGEN_ACTIONS_MIN:-}"
actions_max="${APIGEN_ACTIONS_MAX:-}"
max_parallel_width="${APIGEN_MAX_PARALLEL_WIDTH:-${actions_max:-$actions}}"
feature_difficulty="${APIGEN_FEATURE_DIFFICULTY:-standard}"
naturalize_queries="${APIGEN_NATURALIZE_QUERIES:-0}"
feature_schedule="${APIGEN_FEATURE_SCHEDULE:-terminal}"
refusal_reason="${APIGEN_REFUSAL_REASON:-random}"
min_total_steps="${APIGEN_MIN_TOTAL_STEPS:-}"
max_total_steps="${APIGEN_MAX_TOTAL_STEPS:-}"
model="${APIGEN_MODEL:-x-ai/grok-4.5}"
judge_model="${APIGEN_JUDGE_MODEL:-$model}"
output_dir="${APIGEN_OUTPUT_DIR:-data/generated}"
run_name="${APIGEN_RUN_NAME:-${feature}_${mode}_${rows}}"
tool_pool="${APIGEN_TOOL_POOL:-magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl}"
invocations="${APIGEN_INVOCATIONS:-magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl}"

mkdir -p "$output_dir"
output="$output_dir/${run_name}.jsonl"
report="$output_dir/${run_name}.audit.json"
internal_tasks="$output_dir/${run_name}.internal_tasks.jsonl"
bfcl_tasks="$output_dir/${run_name}.bfcl_native_tasks.jsonl"

feature_args=(--require-feature)
feature_args+=(
  --feature-difficulty "$feature_difficulty"
  --multi-turn-feature-schedule "$feature_schedule"
  --refusal-reason "$refusal_reason"
)
case "${naturalize_queries,,}" in
  1|true|yes|on)
    feature_args+=(--naturalize-queries)
    ;;
  0|false|no|off)
    feature_args+=(--no-naturalize-queries)
    ;;
  *)
    echo "APIGEN_NATURALIZE_QUERIES must be true/false or 1/0" >&2
    exit 2
    ;;
esac
case "$feature" in
  parallel)
    feature_args+=(--allow-parallel --parallel-rate 1.0 --no-allow-refusal)
    ;;
  refusal)
    feature_args+=(--allow-refusal --refusal-rate 1.0 --no-allow-parallel)
    ;;
  mixed)
    feature_args+=(
      --allow-refusal --refusal-rate "${APIGEN_REFUSAL_WEIGHT:-0.5}"
      --allow-parallel --parallel-rate "${APIGEN_PARALLEL_WEIGHT:-0.5}"
    )
    ;;
  *)
    echo "APIGEN_FEATURE must be parallel, refusal, or mixed" >&2
    exit 2
    ;;
esac

mode_args=(--mode "$mode")
if [[ "$mode" == "multi-turn" ]]; then
  mode_args+=(--num-turns "$turns")
fi

action_args=(--num-actions "$actions")
if [[ -n "$blueprint_max_actions" ]]; then
  action_args+=(
    --blueprint-max-actions-per-turn "$blueprint_max_actions"
  )
fi
if [[ -n "$actions_min" || -n "$actions_max" ]]; then
  if [[ "$mode" != "step-by-step" ]]; then
    echo "APIGEN_ACTIONS_MIN/MAX are supported only in step-by-step mode" >&2
    exit 2
  fi
  if [[ -z "$actions_min" || -z "$actions_max" ]]; then
    echo "Set both APIGEN_ACTIONS_MIN and APIGEN_ACTIONS_MAX" >&2
    exit 2
  fi
  action_args+=(--num-actions-range "$actions_min" "$actions_max")
fi

dedupe_args=()
if [[ -n "${APIGEN_DEDUPE_REGISTRY:-}" ]]; then
  dedupe_args+=(--dedupe-registry "$APIGEN_DEDUPE_REGISTRY")
fi

length_args=()
if [[ -n "$min_total_steps" ]]; then
  length_args+=(--min-total-steps "$min_total_steps")
fi

audit_args=(--require-feature --expected-rows "$rows")
audit_args+=(--expected-difficulty "$feature_difficulty")
if [[ "${naturalize_queries,,}" =~ ^(1|true|yes|on)$ ]]; then
  audit_args+=(--require-naturalized)
fi
if [[ "$feature_schedule" == "interactive-refusal" ]]; then
  audit_args+=(--expected-schedule interactive-refusal --require-recovery)
elif [[ "$feature_schedule" == "combined" ]]; then
  audit_args+=(--expected-schedule combined --require-recovery)
fi
if [[ "$feature" != "mixed" ]]; then
  audit_args+=(--expected-feature "$feature")
fi
if [[ -n "$min_total_steps" ]]; then
  audit_args+=(--min-steps "$min_total_steps")
fi
if [[ -n "$max_total_steps" ]]; then
  audit_args+=(--max-steps "$max_total_steps")
fi
if [[ "$mode" == "multi-turn" ]]; then
  audit_args+=(--expected-turns "$turns")
fi
if [[ -n "$max_total_steps" ]]; then
  length_args+=(--max-total-steps "$max_total_steps")
fi
if [[ -n "${APIGEN_DEDUPE_AGAINST:-}" ]]; then
  IFS=':' read -r -a dedupe_paths <<< "$APIGEN_DEDUPE_AGAINST"
  for path in "${dedupe_paths[@]}"; do
    dedupe_args+=(--dedupe-against "$path")
  done
fi

"$python_bin" src/generate_step_by_step.py \
  "${mode_args[@]}" \
  --num-datapoints "$rows" \
  "${action_args[@]}" \
  --max-parallel-width "$max_parallel_width" \
  --model "$model" \
  --judge-model "$judge_model" \
  --tool-pool "$tool_pool" \
  --invocation-examples "$invocations" \
  "${dedupe_args[@]}" \
  "${length_args[@]}" \
  "${feature_args[@]}" \
  --output "$output"

PYTHONPATH=src "$python_bin" scripts/audit_refuse_parallel_dataset.py \
  --input "$output" \
  --report "$report" \
  "${audit_args[@]}"

PYTHONPATH=src "$python_bin" scripts/export_refuse_parallel_tasks.py \
  --input "$output" \
  --output "$internal_tasks" \
  --target-format internal

PYTHONPATH=src "$python_bin" scripts/export_refuse_parallel_tasks.py \
  --input "$output" \
  --output "$bfcl_tasks" \
  --target-format bfcl-native

printf 'Generated: %s\nAudit: %s\nInternal pass@k tasks: %s\nBFCL-native tasks: %s\n' \
  "$output" "$report" "$internal_tasks" "$bfcl_tasks"
