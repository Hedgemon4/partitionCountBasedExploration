#!/bin/bash

# Pure PQN baseline (pqn_atari.py, all-relu conv trunk) across Atari-57, both bootstrap rules.
#
# This is the no-FTA reference arm, and it is the one cell with no clean data on disk.
# pqn_atari.py only received the minibatch-shuffle key fix (96d1988, applied to the counts
# scripts on 2026-07-30) in the commit accompanying this sweep, so every earlier pure-PQN run
# reused the same shuffle permutations on every update step.
#
# The FTA-trunk counterpart is NOT run here. beta=0 on the counts script is equivalent -- the
# intrinsic head receives exactly zero gradient and enters neither action selection nor the
# bootstrap -- and that data already exists:
#   Q-learning: data/atari57_count_layer_sweep/<game>/conv2/beta_0.0/next_0.0/   (57 games x 5)
#   SARSA:      data/atari57_seperate_heads_sarsa_sweep_partial/<game>/beta_0.0/*/epsilon_0.001/
#               (partial: 23 games, alien -> frostbite)
#
# 57 games x 2 rules x 5 seeds = 570 runs, 143 array tasks at 4 runs/task (one array job).
#
# Do not submit before the atari4_baseline_probe result: --time is the one value to set from
# the probe's measured elapsed rather than assumed. 02:59 lost 19.7% of atari57_beta_sweep on
# the counts network, concentrated in slow games.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." &>/dev/null && pwd) # job_scripts/atari/<sweep> -> root
GAMES_FILE="$REPO_ROOT/atari57_games.txt"

if [ ! -f "$GAMES_FILE" ]; then
    echo "ERROR: game list not found at $GAMES_FILE" >&2
    exit 1
fi

# Read one game per line, skipping blanks (portable; avoids bash-4-only `mapfile`).
games=()
while IFS= read -r game_line; do
    [ -n "$game_line" ] && games+=("$game_line")
done <"$GAMES_FILE"

seeds=(0 1 2 3 4)

# name|flag. sarsa_returns picks the lambda-return bootstrap: the action actually taken,
# versus the greedy max over the next state.
declare -a bootstraps=(
    "qlearning|--no-sarsa-returns"
    "sarsa|--sarsa-returns"
)

sweep="atari57_baseline"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim. No subcommand token:
# AtariConfig is already tyro's default= and nothing nested is being swept.
>"$output_file"
for game in "${games[@]}"; do
    for bootstrap in "${bootstraps[@]}"; do
        IFS='|' read -r name flag <<<"$bootstrap"
        for seed in "${seeds[@]}"; do
            out="$sweep/$game/$name/seed_${seed}"
            echo "--environment $game --force-xla $flag --seed $seed --output-folder-name $out" >>"$output_file"
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
