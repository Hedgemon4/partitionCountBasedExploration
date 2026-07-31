#!/bin/bash

# Where should the FTA layer + counts table live, across the full Atari-57 suite?
#
# 57 games x 3 count positions x 8 betas x 5 seeds = 6840 runs.
# submit.sh packs 4 runs per GPU, so that is 1710 array tasks -- above
# MaxArraySize=1000, hence the TASK_OFFSET split that run_sweep.sh performs.
#
# next_state_coef is fixed at 0.0 and NOT swept. At 0.0 the network skips building
# the auxiliary next-state head entirely and the rollout stops carrying its target,
# which is what makes 4 runs fit on one GPU at MEM_FRACTION=0.22.
#
# count_layer is 1-indexed over [conv1, conv2, conv3, *network.blocks]; conv1 is
# excluded because its FTA map is 32x20x20xbins. The FTA layer sits *only* at the
# count position, so every line spells out all four activations rather than relying
# on the `default` subcommand's defaults.

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

# name|count_layer|activation subcommands (declaration order: 1, 2, 3, blocks.0)
declare -a positions=(
    "conv2|2|network.cnn-activation-1:relu network.cnn-activation-2:fta network.cnn-activation-3:relu network.blocks.0.activation:relu"
    "conv3|3|network.cnn-activation-1:relu network.cnn-activation-2:relu network.cnn-activation-3:fta network.blocks.0.activation:relu"
    "mlp1|4|network.cnn-activation-1:relu network.cnn-activation-2:relu network.cnn-activation-3:relu network.blocks.0.activation:fta"
)

betas=(1.0 0.5 0.25 0.1 0.05 0.01 0.005 0.0)
seeds=(0 1 2 3 4)
next_state_coef=0.0

sweep="atari57_count_layer_sweep"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim.
# tyro wants the `--flag value` args first and the `key:subcommand` tokens last.
# The next_${coef} path level is redundant here (coef never varies) but is kept so
# the layout matches venture_spaceinvaders_count_layer_sweep and
# plot_count_layer_sweep.py can read this sweep with no changes.
>"$output_file"
for game in "${games[@]}"; do
    for position in "${positions[@]}"; do
        IFS='|' read -r name count_layer activations <<<"$position"
        for beta in "${betas[@]}"; do
            for seed in "${seeds[@]}"; do
                echo "default --environment $game --force-xla --beta $beta --network.next_state_coef $next_state_coef --network.count_layer $count_layer --seed $seed --output-folder-name $sweep/$game/$name/beta_${beta}/next_${next_state_coef}/seed_${seed} $activations" >>"$output_file"
            done
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
