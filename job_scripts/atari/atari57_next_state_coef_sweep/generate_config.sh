#!/bin/bash

# Does the auxiliary next-state forward model help, across Atari-57?
#
# 57 games x 6 betas x 3 next_state_coefs x 5 seeds = 5130 runs. submit.sh packs 4 runs
# per GPU, so 1283 array tasks -- above MaxArraySize=1000, hence the TASK_OFFSET split
# that run_sweep.sh performs (2 jobs: 1000 + 283).
#
# This is the sibling of atari57_seperate_heads_sarsa_sweep. That sweep pinned
# next_state_coef at 0.0, which switches the auxiliary task off entirely -- the head is
# never built and the rollout never carries its target (netwoks.py:599-631,
# pqn_atari_counts_with_seperate_value_head.py:144) -- so its 9405 runs say nothing about
# this axis. The only Atari evidence on it is 2 games, from
# venture_spaceinvaders_count_layer_sweep.
#
# Why these axes:
#   next_state_coef  The point of the sweep. The head predicts the *count layer's*
#                    continuous FTA features one step ahead, not raw pixels, so it is a
#                    forward model in exactly the representation the counts are taken in.
#                    0.5 and 1.0 match venture_spaceinvaders_count_layer_sweep; 2.0
#                    extends past it, to find where the auxiliary loss starts crowding
#                    out the TD loss rather than only where it starts helping.
#   beta             Crossed with it, unchanged from the sarsa sweep, because the two
#                    interact: both act on the same count layer, and the auxiliary task
#                    shapes the very features the counts are built from.
#
# 0.0 is NOT in the grid: atari57_seperate_heads_sarsa_sweep already ran it for every
# (game, beta, seed) at the cell pinned below. Re-running it would duplicate 1710 runs.
# The plotting script reads that arm from the sarsa sweep's tree -- which is why the
# pinned levels stay in the output path below, redundant though they are, so this layout
# is a strict superset of the sarsa one. Same reasoning as
# atari57_count_layer_sweep/generate_config.sh, which keeps a redundant next_ level so
# plot_count_layer_sweep.py can read it unchanged.
#
# Why the pinned values, from graphs/atari57_sarsa_figures/summary/*/global/config_selection.csv
# (all 57 games, n_ballots=57):
#   intrinsic_gamma 0.9   Beats 0.99 at all 6 betas under both the final and auc scores,
#                         and wins 76-78% of the 342 per-game/per-beta selections.
#                         It is also the value the sarsa sweep's beta=0 control ran --
#                         that control takes a single gamma_I, since a frozen intrinsic
#                         head makes gamma_I inert -- so pinning 0.9 means the reused
#                         control arm needs no special case at beta=0.
#   epsilon_end     0.001 Wins at every beta except 5.0 (which prefers 0.01, and is a
#                         reference point for the divergence metrics rather than a
#                         performance candidate). The (0.9, 0.001) pair is the modal
#                         per-game winner at 47%, double the runner-up.
#   gamma           0.99  Pinned sweep-wide in the sarsa sweep; per-game tuning is not
#                         possible at 57 games.
#   count_layer     2     conv2, as in the sarsa sweep. Moving it would break
#                         cell-for-cell comparability with the reused control arm.
#
# Note that beta=0 is still crossed with all three coefficients. It zeroes
# intrinsic_loss_coef and so freezes the intrinsic head, but the auxiliary task still
# trains the shared trunk -- that row is the "auxiliary task alone, no count bonus" arm,
# and is the cleanest read on whether the forward model helps at all.

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

# conv2: FTA at the second conv block, relu everywhere else. The FTA layer sits
# *only* at the count position, so every line spells out all four activations.
count_layer=2
activations="network.cnn-activation-1:relu network.cnn-activation-2:fta network.cnn-activation-3:relu network.blocks.0.activation:relu"

betas=(0.0 0.01 0.1 0.5 1.0 5.0)
next_state_coefs=(0.5 1.0 2.0)
seeds=(0 1 2 3 4)
gamma=0.99
intrinsic_gamma=0.9 # pinned; see header
epsilon=0.001       # pinned; see header

sweep="atari57_next_state_coef_sweep"
output_file="$SCRIPT_DIR/configs.txt"
expected=$((${#games[@]} * ${#betas[@]} * ${#next_state_coefs[@]} * ${#seeds[@]}))

# One line per run, seed baked in, so submit.sh runs the line verbatim.
# tyro wants the `--flag value` args first and the `key:subcommand` tokens last.
>"$output_file"
for game in "${games[@]}"; do
    for beta in "${betas[@]}"; do
        # Matches the sarsa sweep's control arm: beta=0 freezes the intrinsic head
        # rather than merely weighting it out of the behaviour policy.
        if [ "$beta" = "0.0" ]; then
            intrinsic_loss_coef=0.0
        else
            intrinsic_loss_coef=1.0
        fi
        for next in "${next_state_coefs[@]}"; do
            for seed in "${seeds[@]}"; do
                out="$sweep/$game/beta_${beta}/intrinsic_gamma_${intrinsic_gamma}/epsilon_${epsilon}/next_${next}/seed_${seed}"
                echo "default --environment $game --force-xla --sarsa-returns --beta $beta --gamma $gamma --intrinsic-gamma $intrinsic_gamma --epsilon-end $epsilon --network.next_state_coef $next --network.intrinsic_loss_coef $intrinsic_loss_coef --network.count_layer $count_layer --seed $seed --output-folder-name $out $activations" >>"$output_file"
            done
        done
    done
done

written=$(wc -l <"$output_file")
echo "Wrote $written lines from ${#games[@]} games"

# A miscount here means submit.sh would silently drop or duplicate runs, so fail loudly
# rather than leaving it to be noticed in coverage.csv weeks later.
if [ "$written" -ne "$expected" ]; then
    echo "ERROR: expected $expected lines, wrote $written" >&2
    exit 1
fi
