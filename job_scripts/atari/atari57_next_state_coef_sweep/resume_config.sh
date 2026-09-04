#!/bin/bash

# Write configs_resume.txt: the lines of configs.txt whose run has not finished.
#
# A run writes data/<output-folder-name>/metrics.npz only after training completes
# (pqn_atari_counts_with_seperate_value_head.py:711-758), so the file's presence is an exact
# completeness test -- there is no partially-written case to disambiguate. That makes this
# safe to re-run after any partial sweep, and it replaces the hand-built failed-runs
# directory that job_scripts/classic_control/mountaincar_count_first_layer_failed_runs used,
# which does not scale to 5130 lines.
#
# Then, from the repo root:
#   CONFIG_NAME=configs_resume.txt bash job_scripts/atari/atari57_next_state_coef_sweep/run_sweep.sh
# run_sweep.sh re-derives the array chunking from `wc -l`, so no bounds need editing.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd) # job_scripts/atari/<sweep> -> root

input="$SCRIPT_DIR/${1:-configs.txt}"
output="$SCRIPT_DIR/configs_resume.txt"

if [ ! -f "$input" ]; then
    echo "ERROR: $input not found" >&2
    exit 1
fi

done_count=0
todo_count=0
>"$output"
while IFS= read -r line; do
    [ -n "$line" ] || continue
    # Lines end with `--output-folder-name <path> <activation tokens...>`, so strip up to
    # the flag and then take the first field.
    out=${line##*--output-folder-name }
    out=${out%% *}
    if [ -f "$REPO_ROOT/data/$out/metrics.npz" ]; then
        done_count=$((done_count + 1))
    else
        printf '%s\n' "$line" >>"$output"
        todo_count=$((todo_count + 1))
    fi
done <"$input"

echo "Finished: $done_count   Remaining: $todo_count   -> $(basename "$output")"
if [ "$todo_count" -eq 0 ]; then
    echo "Nothing to resubmit."
    rm -f "$output"
fi
