#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

NUM_CONFIGS=$(wc -l <"$CONFIG_PATH")
# 5 array tasks per config (each task runs a 2-seed pair -> 10 seeds/config).
MAX_INDEX=$((NUM_CONFIGS * 5 - 1))

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"
echo "Number of configurations: $NUM_CONFIGS"
echo "Array tasks: 0-$MAX_INDEX ($NUM_CONFIGS configs x 5 seed-pairs, 2 seeds/task)"

# submit.sh maps task k -> (config k/5, seeds {2*(k%5), 2*(k%5)+1}).
sbatch --array=0-$MAX_INDEX \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
