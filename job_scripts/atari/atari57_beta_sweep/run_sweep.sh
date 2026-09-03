#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"

# 2280 runs, 2 runs/task -> 1140 tasks. Cluster MaxArraySize=1000 (indices 0-999),
# so split into two jobs, each starting at array index 0, using TASK_OFFSET to map
# into the shared configs.txt.  effective_task = SLURM_ARRAY_TASK_ID + TASK_OFFSET
#   Job 1: array 0-999   offset 0     -> tasks    0-999   -> config lines 1-2000  (2000 runs)
#   Job 2: array 0-139   offset 1000  -> tasks 1000-1139  -> config lines 2001-2280 (280 runs)
sbatch --array=0-999 \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH",TASK_OFFSET=0 \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"

sbatch --array=0-139 \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH",TASK_OFFSET=1000 \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
