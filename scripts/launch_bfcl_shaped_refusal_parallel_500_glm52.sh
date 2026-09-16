#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$repo_root/data/generated/runs/refusal_parallel_bfcl_shaped_steps7_15_500_20260728"
python_bin="${APIGEN_PYTHON:-/home/jovyan/.mlspace/envs/qwen35/bin/python}"
child_pid=""

stop_child() {
    if [[ -n "$child_pid" ]]; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    exit 143
}
trap stop_child INT TERM

source /home/jovyan/.sweethome/openrouter.sh
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is not set}"
export APIGEN_PYTHON="$python_bin"
export APIGEN_MODEL="z-ai/glm-5.2"
export APIGEN_JUDGE_MODEL="z-ai/glm-5.2"

cd "$repo_root"
mkdir -p "$output_dir"
printf '%s\n' "$$" > "$output_dir/launcher.pid"

while true; do
    echo "[$(date --iso-8601=seconds)] starting/resuming GLM-5.2 generation"
    "$python_bin" scripts/generate_bfcl_shaped_refusal_parallel_500.py \
        --output-dir "$output_dir" \
        --model "$APIGEN_MODEL" \
        --judge-model "$APIGEN_JUDGE_MODEL" \
        --quiet-schedule \
        --max-workers 10 \
        --max-task-restarts 12 \
        --task-timeout-seconds 7200 \
        --max-output-tokens 8192 &
    child_pid="$!"
    wait "$child_pid"
    status="$?"
    child_pid=""
    if [[ "$status" -eq 0 ]]; then
        echo "[$(date --iso-8601=seconds)] generation and audits completed"
        exit 0
    fi
    echo "[$(date --iso-8601=seconds)] generation exited $status; retrying in 60s"
    sleep 60
done
