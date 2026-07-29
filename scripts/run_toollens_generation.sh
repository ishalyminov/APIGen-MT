#!/usr/bin/env bash
# Run ToolLens generation for missing categories, one at a time
set -e

API_BASE="https://opencode.ai/zen/v1"
API_KEY="sk-InDWHwl5DsqTe6pCbDTrxLzhF5s4DWUtaKLkp0yrXgujyafKu4txucuET2s0hFIs"
MODEL="deepseek-v4-flash-free"
LOG="/tmp/toollens_batch_run.log"

CATEGORIES=(
    "News_Media"
    "Location"
    "Travel"
    "Weather"
    "Food"
    "Sports"
    "Transportation"
    "eCommerce"
    "Music"
    "Movies"
    "Health_and_Fitness"
    "Gaming"
    "Video_Images"
)

echo "Starting generation at $(date)" | tee "$LOG"

for cat in "${CATEGORIES[@]}"; do
    CLASS_KEY=$(echo "$cat" | tr '[:upper:]' '[:lower:]')
    PY_FILE="tools/toollens/${CLASS_KEY}.py"

    if [ -f "$PY_FILE" ] && head -5 "$PY_FILE" | grep -q "class" 2>/dev/null; then
        echo "[SKIP] $cat - $PY_FILE exists" | tee -a "$LOG"
        continue
    fi

    echo "============================================================" | tee -a "$LOG"
    echo "Generating: $cat at $(date)" | tee -a "$LOG"
    echo "============================================================" | tee -a "$LOG"

    if python3 -u scripts/generate_toollens_implementations.py \
        --categories "$cat" \
        --model "$MODEL" \
        --api-base "$API_BASE" \
        --api-key "$API_KEY" \
        --max-retries 1 \
        --verbose 2>&1 | tee -a "$LOG"; then
        echo "[OK] $cat" | tee -a "$LOG"
    else
        echo "[FAIL] $cat - retrying once..." | tee -a "$LOG"
        sleep 5
        if python3 -u scripts/generate_toollens_implementations.py \
            --categories "$cat" \
            --model "$MODEL" \
            --api-base "$API_BASE" \
            --api-key "$API_KEY" \
            --max-retries 2 \
            --verbose 2>&1 | tee -a "$LOG"; then
            echo "[OK] $cat (on retry)" | tee -a "$LOG"
        else
            echo "[FAIL] $cat after retry - skipping" | tee -a "$LOG"
        fi
    fi
done

echo "============================================================" | tee -a "$LOG"
echo "Finished at $(date)" | tee -a "$LOG"
