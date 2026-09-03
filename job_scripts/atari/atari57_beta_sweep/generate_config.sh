#!/bin/bash

# Full Atari-57 beta sweep: 57 games x 8 betas x 5 seeds = 2280 runs.
# One line per run (game/beta/seed baked in). Runs are later paired 2-per-GPU across this
# flat list by submit.sh, so a pair may span different games/betas/seeds.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd)   # job_scripts/atari/<sweep> -> root
GAMES_FILE="$REPO_ROOT/atari57_games.txt"

if [ ! -f "$GAMES_FILE" ]; then
    echo "ERROR: game list not found at $GAMES_FILE" >&2
    exit 1
fi

# Sweep Options
# Read one game per line, skipping blank lines (portable; avoids bash-4-only `mapfile`).
games=()
while IFS= read -r game_line; do
    [ -n "$game_line" ] && games+=("$game_line")
done <"$GAMES_FILE"
betas=(1.0 0.5 0.25 0.1 0.05 0.01 0.005 0.0)
seeds=(0 1 2 3 4)
next_state_coef=0.0
sweep=atari57_beta_sweep

output_file="$SCRIPT_DIR/configs.txt"
>"$output_file"
for game in "${games[@]}"; do
    for beta in "${betas[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "one-block --environment $game --force-xla --beta $beta --network.next_state_coef $next_state_coef --seed $seed --output-folder-name ${sweep}/${game}/beta_${beta}/seed_${seed}" >>"$output_file"
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
