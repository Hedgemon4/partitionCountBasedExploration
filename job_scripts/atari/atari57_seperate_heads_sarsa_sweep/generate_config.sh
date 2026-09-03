#!/bin/bash

# Separate intrinsic/extrinsic value heads with SARSA returns, across Atari-57.
#
# 57 games x 11 cells x 3 epsilons x 5 seeds = 9405 runs. submit.sh packs 4 runs per
# GPU, so 2352 array tasks -- above MaxArraySize=1000, hence the TASK_OFFSET split
# that run_sweep.sh performs (3 jobs: 1000 + 1000 + 352).
#
# The 11 cells are 5 nonzero betas x 2 intrinsic gammas, plus one beta=0 control.
# beta=0 sets intrinsic_loss_coef=0.0, which leaves the intrinsic head untrained and
# makes intrinsic_gamma inert -- so it takes a single gamma_I rather than both, the
# same dedup atari4_seperate_heads_gamma_sweep does with its diagonal.
#
# Why these axes:
#   sarsa_returns  The default bootstrap argmaxes over the *fused* Q_e + beta*Q_i,
#                  i.e. it evaluates the greedy policy rather than maximising over
#                  an action the policy would not take. So --sarsa-returns swaps
#                  greedy-policy evaluation for epsilon-greedy-policy evaluation,
#                  and the two coincide wherever epsilon does not fire.
#   epsilon_end    Which is why epsilon is swept: at the 0.001 default the two rules
#                  differ on ~0.1% of late-training steps.
#   beta           0.01-1.0 is where atari4_seperate_heads_gamma_sweep put the
#                  optimum (no game there preferred beta >= 0.5). 5.0 is included as
#                  a reference point for the new divergence metrics, not as a
#                  performance candidate -- without a cell where the intrinsic head
#                  demonstrably acts there is no scale to read the low betas against.
#   gamma_I        Swept; gamma_E is pinned at 0.99 because per-game tuning is not
#                  possible at 57 games and 0.99 was the modal winner.
#
# next_state_coef is fixed at 0.0 and NOT swept. At 0.0 the network skips building
# the auxiliary next-state head entirely and the rollout stops carrying its target,
# which is what makes 4 runs fit on one GPU at MEM_FRACTION=0.22.

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
intrinsic_gammas=(0.9 0.99)
epsilons=(0.001 0.01 0.1)
seeds=(0 1 2 3 4)
gamma=0.99
next_state_coef=0.0

sweep="atari57_seperate_heads_sarsa_sweep"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim.
# tyro wants the `--flag value` args first and the `key:subcommand` tokens last.
>"$output_file"
for game in "${games[@]}"; do
    for beta in "${betas[@]}"; do
        # See the header: the beta=0 control freezes the intrinsic head, which makes
        # gamma_I inert, so it takes one value and zeroes the coefficient.
        if [ "$beta" = "0.0" ]; then
            intrinsic_loss_coef=0.0
            cell_gammas=("${intrinsic_gammas[0]}")
        else
            intrinsic_loss_coef=1.0
            cell_gammas=("${intrinsic_gammas[@]}")
        fi
        for intrinsic_gamma in "${cell_gammas[@]}"; do
            for epsilon in "${epsilons[@]}"; do
                for seed in "${seeds[@]}"; do
                    out="$sweep/$game/beta_${beta}/intrinsic_gamma_${intrinsic_gamma}/epsilon_${epsilon}/seed_${seed}"
                    echo "default --environment $game --force-xla --sarsa-returns --beta $beta --gamma $gamma --intrinsic-gamma $intrinsic_gamma --epsilon-end $epsilon --network.next_state_coef $next_state_coef --network.intrinsic_loss_coef $intrinsic_loss_coef --network.count_layer $count_layer --seed $seed --output-folder-name $out $activations" >>"$output_file"
                done
            done
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
