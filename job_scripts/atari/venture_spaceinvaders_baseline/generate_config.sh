#!/bin/bash

# Pure PQN baseline (pqn_atari.py, all-relu conv trunk) on venture and
# space_invaders, across both bootstrap rules.
#
# This is the no-FTA reference arm. It is the one cell with no clean data on disk:
# pqn_atari.py only received the shuffle-key fix (96d1988, applied to the counts
# scripts on 2026-07-30) in the commit that accompanies this sweep, so every
# earlier baseline run reused the same minibatch permutations on every update step.
#
# The FTA-trunk counterpart is NOT run here -- beta=0 on the counts script is
# equivalent (the intrinsic head gets exactly zero gradient and never enters
# action selection or the bootstrap), and that data already exists at
# data/atari57_count_layer_sweep/<game>/conv2/beta_0.0/next_0.0/ for all 57 games.
#
# 2 games x 2 bootstrap rules x 5 seeds = 20 runs.
#
# To scale to Atari-57, replace the `games` array with a read of atari57_games.txt.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

games=(venture space_invaders)

# name|flag. sarsa_returns selects the lambda-return bootstrap: the action actually
# taken, versus the greedy max over the next state.
declare -a bootstraps=(
    "qlearning|--no-sarsa-returns"
    "sarsa|--sarsa-returns"
)

seeds=(0 1 2 3 4)

sweep="venture_spaceinvaders_baseline"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim. No
# subcommand token: AtariConfig is already tyro's default and nothing nested is
# being swept.
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
