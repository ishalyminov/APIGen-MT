#!/usr/bin/env bash
set -u
set -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${APIGEN_PYTHON:-/home/jovyan/.mlspace/envs/qwen35/bin/python}"
model="${APIGEN_MODEL:-x-ai/grok-4.5}"
category="${APIGEN_CATEGORY:-}"
output_dir="${APIGEN_OUTPUT_DIR:-data/generated}"
run_name="${APIGEN_RUN_NAME:-rl_quality_yield_500_grok45_20260727}"
total_rows="${APIGEN_TOTAL_ROWS:-500}"
shard_count="${APIGEN_SHARD_COUNT:-20}"
min_actions="${APIGEN_MIN_ACTIONS:-2}"
max_actions="${APIGEN_MAX_ACTIONS:-2}"
dedupe_against="${APIGEN_DEDUPE_AGAINST:-}"
dedupe_registry="${APIGEN_DEDUPE_REGISTRY:-}"
base_rows_per_shard=$((total_rows / shard_count))
extra_row_shards=$((total_rows % shard_count))

mkdir -p "$output_dir"

if (( min_actions < 1 || max_actions < min_actions )); then
    echo "Invalid action range: ${min_actions}-${max_actions}" >&2
    exit 2
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    # Managed server credential; the value is never copied into an artifact.
    source /home/jovyan/.sweethome/openrouter.sh
fi
export OPENAI_API_KEY="$OPENROUTER_API_KEY"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://openrouter.ai/api/v1}"
export PYTHONUNBUFFERED=1

tool_pool="magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
invocation_examples="magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
final_output="$output_dir/${run_name}.jsonl"

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        wc -l < "$path"
    else
        echo 0
    fi
}

run_shard() {
    local shard_index="$1"
    local shard_output="$output_dir/${run_name}.part${shard_index}.jsonl"
    local shard_log="$output_dir/${run_name}.part${shard_index}.log"
    local shard_target="$base_rows_per_shard"
    local rows remaining exit_code new_rows

    if (( shard_index < extra_row_shards )); then
        shard_target=$((shard_target + 1))
    fi

    rows="$(line_count "$shard_output")"
    while (( rows < shard_target )); do
        remaining=$((shard_target - rows))
        {
            echo
            echo "Starting shard ${shard_index}: existing=${rows}, remaining=${remaining}, model=${model}"
        } >> "$shard_log"

        dedupe_args=()
        if [[ -n "$dedupe_against" ]]; then
            dedupe_args+=(--dedupe-against "$dedupe_against")
        fi
        if [[ -n "$dedupe_registry" ]]; then
            dedupe_args+=(--dedupe-registry "$dedupe_registry")
        fi
        category_args=()
        if [[ -n "$category" ]]; then
            category_args+=(--category "$category")
        fi
        action_args=(--num-actions "$min_actions")
        if (( min_actions != max_actions )); then
            action_args=(--num-actions-range "$min_actions" "$max_actions")
        fi

        "$python_bin" src/generate_step_by_step.py \
            --mode step-by-step \
            --num-datapoints "$remaining" \
            "${action_args[@]}" \
            --model "$model" \
            --tool-pool "$tool_pool" \
            --invocation-examples "$invocation_examples" \
            "${dedupe_args[@]}" \
            "${category_args[@]}" \
            --output "$shard_output" >> "$shard_log" 2>&1
        exit_code=$?
        new_rows="$(line_count "$shard_output")"

        if (( new_rows <= rows )); then
            echo "Shard ${shard_index} made no progress (exit=${exit_code}); stopping." >> "$shard_log"
            return 1
        fi
        rows="$new_rows"
    done

    [[ "$rows" -eq "$shard_target" ]]
}

pids=()
for shard_index in $(seq 0 $((shard_count - 1))); do
    run_shard "$shard_index" &
    pids+=("$!")
done

worker_failure=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        worker_failure=1
    fi
done

if (( worker_failure != 0 )); then
    echo "At least one shard failed; preserving partial files for resume."
    exit 1
fi

for shard_index in $(seq 0 $((shard_count - 1))); do
    shard_output="$output_dir/${run_name}.part${shard_index}.jsonl"
    shard_target="$base_rows_per_shard"
    if (( shard_index < extra_row_shards )); then
        shard_target=$((shard_target + 1))
    fi
    rows="$(line_count "$shard_output")"
    if [[ "$rows" -ne "$shard_target" ]]; then
        echo "Shard ${shard_index} has ${rows} rows; expected ${shard_target}."
        exit 1
    fi
done

combined_output="${final_output}.tmp"
{
    for shard_index in $(seq 0 $((shard_count - 1))); do
        cat "$output_dir/${run_name}.part${shard_index}.jsonl"
    done
} > "$combined_output"
mv "$combined_output" "$final_output"

"$python_bin" - "$final_output" "$total_rows" "$min_actions" "$max_actions" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
min_actions = int(sys.argv[3])
max_actions = int(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert len(rows) == expected_rows, len(rows)
for index, row in enumerate(rows):
    assert row["generation_metadata"]["rl_quality_gate_passed"] is True, index
    assert row["verification_result"]["overall_verification_passed"] is True, index
    assert row["verification_result"]["rl_quality_gate"]["passed"] is True, index
    num_steps = len(row["trajectory"]["steps"])
    assert min_actions <= num_steps <= max_actions, (index, num_steps)
    assert row["generation_metadata"]["num_actions"] == num_steps, index
    assert all(
        step["quality_verification"]["passed"]
        for step in row["trajectory"]["steps"]
    ), index
print(
    f"Audit passed: {len(rows)} certified trajectories "
    f"with {min_actions}-{max_actions} steps"
)
PY

echo "Final output: $final_output"
