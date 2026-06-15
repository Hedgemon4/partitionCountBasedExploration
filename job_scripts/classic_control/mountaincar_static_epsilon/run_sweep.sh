#!/bin/bash

# 1. Get the absolute path and the name of the current folder
SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

# 2. Define the config path
CONFIG_PATH="$SWEEP_DIR/configs.txt"

# 3. Determine array size
NUM_CONFIGS=$(wc -l <"$CONFIG_PATH")
MAX_INDEX=$((NUM_CONFIGS - 1))

# 4. Define the output directory
OUTPUT_DIR="run_outputs/$FOLDER_NAME"

echo "Launching sweep: $FOLDER_NAME"
echo "Outputs will be in: $OUTPUT_DIR"
echo "Number of configurations: $NUM_CONFIGS"

# 5. Submit with the -o (output) flag to override the header in submit.sh
sbatch --array=0-$MAX_INDEX \
    --export=ALL,CONFIG_PATH="$CONFIG_PATH" \
    --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
    "$SWEEP_DIR/submit.sh"
