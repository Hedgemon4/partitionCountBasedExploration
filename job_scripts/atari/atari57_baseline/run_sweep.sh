#!/bin/bash

# Run from the repo root: submit.sh refers to pqn_atari.py, requirements.txt,
# wheels/ and ale/ by relative path.
#
# Pass --dry-run to print the sbatch calls without submitting.

DRY_RUN=0
[ "$1" = "--dry-run" ] && DRY_RUN=1

SWEEP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FOLDER_NAME=$(basename "$SWEEP_DIR")

CONFIG_PATH="$SWEEP_DIR/configs.txt"

OUTPUT_DIR="run_outputs/$FOLDER_NAME"
[ "$DRY_RUN" -eq 0 ] && mkdir -p "$OUTPUT_DIR"

# Derive everything from configs.txt and chunk to MaxArraySize, rather than
# hardcoding array bounds: submit.sh runs RUNS_PER_TASK lines per task, so a
# stale hardcoded bound silently drops or duplicates runs.
RUNS_PER_TASK=2
MAX_ARRAY_SIZE=1000

NUM_CONFIGS=$(wc -l <"$CONFIG_PATH")
NUM_TASKS=$(((NUM_CONFIGS + RUNS_PER_TASK - 1) / RUNS_PER_TASK))

echo "Launching sweep: $FOLDER_NAME"
echo "Runs: $NUM_CONFIGS  Array tasks: $NUM_TASKS ($RUNS_PER_TASK runs/task)"
echo "Outputs will be in: $OUTPUT_DIR"

offset=0
job=1
while [ "$offset" -lt "$NUM_TASKS" ]; do
    remaining=$((NUM_TASKS - offset))
    chunk=$((remaining < MAX_ARRAY_SIZE ? remaining : MAX_ARRAY_SIZE))
    first_line=$((offset * RUNS_PER_TASK + 1))
    last_line=$(((offset + chunk) * RUNS_PER_TASK))
    [ "$last_line" -gt "$NUM_CONFIGS" ] && last_line=$NUM_CONFIGS
    echo "  job $job: --array=0-$((chunk - 1)) TASK_OFFSET=$offset -> config lines $first_line-$last_line"

    if [ "$DRY_RUN" -eq 0 ]; then
        sbatch --array=0-$((chunk - 1)) \
            --export=ALL,CONFIG_PATH="$CONFIG_PATH",TASK_OFFSET="$offset" \
            --output="$OUTPUT_DIR/${FOLDER_NAME}_%A_%a.out" \
            "$SWEEP_DIR/submit.sh"
    fi

    offset=$((offset + chunk))
    job=$((job + 1))
done

[ "$DRY_RUN" -eq 1 ] && echo "(dry run: nothing submitted)"
