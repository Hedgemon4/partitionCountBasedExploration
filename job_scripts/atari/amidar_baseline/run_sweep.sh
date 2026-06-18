#!/bin/bash

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Launching: $FOLDER_NAME"
echo "Output: $OUTPUT_DIR"

sbatch --output="$OUTPUT_DIR/${FOLDER_NAME}_%j.out" "$SWEEP_DIR/submit.sh"
