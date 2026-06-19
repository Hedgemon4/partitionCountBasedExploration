#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

NUM_CONFIGS=$(wc -l <"$CONFIG_PATH")
MAX_INDEX=$((NUM_CONFIGS - 1))

OUTPUT_DIR="run_outputs/$FOLDER_NAME"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"
echo "Number of configurations: $NUM_CONFIGS"

sbatch --array=0-$MAX_INDEX \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
