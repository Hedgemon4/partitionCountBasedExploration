#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"

# 12 array tasks x 2 runs/task = 24 runs (8 betas x 3 seeds).
# submit.sh maps task k -> config lines {2*k, 2*k+1} (0-indexed), sharing one GPU.
sbatch --array=0-5 \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
