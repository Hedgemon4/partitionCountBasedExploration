#!/bin/bash

# Run from the repo root: submit.sh refers to pqn_atari_with_counts.py,
# requirements.txt, wheels/ and ale/ by relative path.

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

# Derive the array bound from configs.txt rather than hardcoding it: submit.sh
# runs 2 config lines per task, so sizing the array as one-task-per-line (or
# leaving a stale bound behind after editing the grid) silently drops or
# duplicates runs.
NUM_CONFIGS=$(wc -l <"$CONFIG_PATH")
NUM_TASKS=$(((NUM_CONFIGS + 1) / 2)) # 2 runs per task, round up
MAX_INDEX=$((NUM_TASKS - 1))

echo "Launching sweep: $FOLDER_NAME"
echo "Runs: $NUM_CONFIGS  Array tasks: $NUM_TASKS (2 runs/task, array 0-$MAX_INDEX)"
echo "Outputs will be in: $OUTPUT_DIR"

sbatch --array=0-"$MAX_INDEX" \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
