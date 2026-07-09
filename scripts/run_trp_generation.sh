#!/usr/bin/env bash
# Launch parallel generation: 1000 datapoints, Qwen 3.6-35B, 7-20 actions
# Features: refuse (15%), parallel calls, NO think, judge=GLM 5.2
#
# Usage: ./scripts/run_trp_generation.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/dpugacheva/venvos/occ/bin/python}"
PYTHON="/home/dpugacheva/venvs/occ/bin/python"
MODEL_ID="qwen/qwen3.6-35b-a3b"
JUDGE_MODEL="z-ai/glm-5.2"
LABEL="qwen3.6-35b"
TOOL_POOL="${REPO_ROOT}/magnet_tool_extraction/bfcl_v3_tools_with_outputs.jsonl"
INVOCATION_EXAMPLES="${REPO_ROOT}/magnet_tool_extraction/bfcl_v3_invocation_examples.jsonl"
OUTDIR="${REPO_ROOT}/data/generated/trp_qwen36-35b_refuse_parallel"

mkdir -p "${OUTDIR}"

NUM_PROCS=10
DP_PER_PROC=100

echo "$(date '+%a %d %b %Y %I:%M:%S %p'): Launching TRP generation"
echo "  Model: ${MODEL_ID}"
echo "  Judge: ${JUDGE_MODEL}"
echo "  Actions/dp: 7-20 (randomized)"
echo "  Features: refuse (15%), parallel calls, NO think"
echo "  Total target: $((NUM_PROCS * DP_PER_PROC)) dp (${NUM_PROCS} procs × ${DP_PER_PROC} dp)"
echo "  Output: ${OUTDIR}"

PIDS=()
for i in $(seq 1 ${NUM_PROCS}); do
  OUTPUT="${OUTDIR}/${LABEL}_p${i}.jsonl"
  LOG="${OUTDIR}/${LABEL}_p${i}.log"

  rm -f "${OUTPUT}"

  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" nohup "${PYTHON}" "${REPO_ROOT}/src/generate_step_by_step.py" \
    --mode step-by-step \
    --model "${MODEL_ID}" \
    --num-actions-range 7 20 \
    --num-datapoints "${DP_PER_PROC}" \
    --output "${OUTPUT}" \
    --tool-pool "${TOOL_POOL}" \
    --invocation-examples "${INVOCATION_EXAMPLES}" \
    --judge-model "${JUDGE_MODEL}" \
    --config-pool \
    --no-enable-think \
    --allow-refusal \
    --refusal-rate 0.15 \
    > "${LOG}" 2>&1 &

  PID=$!
  PIDS+=("${PID}")
  echo "$(date '+%I:%M:%S'): launched ${LABEL}_p${i} PID ${PID}"
done

echo "${PIDS[@]}" > "${OUTDIR}/pids.txt"
echo ""
echo "$(date '+%I:%M:%S'): All ${NUM_PROCS} workers launched. PIDs: ${PIDS[*]}"
echo "PIDs saved to ${OUTDIR}/pids.txt"
echo "Monitor with: tail -f ${OUTDIR}/${LABEL}_p*.log"
echo "Count: wc -l ${OUTDIR}/*.jsonl"
