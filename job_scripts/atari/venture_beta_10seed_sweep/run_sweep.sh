#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"

# 60 array tasks x 2 seeds/task = 120 runs (12 configs x 10 seeds).
# submit.sh maps task k -> (config k/5, seeds {2*(k%5), 2*(k%5)+1}).
sbatch --array=0-59 \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
