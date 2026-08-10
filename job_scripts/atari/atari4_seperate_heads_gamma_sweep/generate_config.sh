#!/bin/bash

# Separate intrinsic/extrinsic value heads: how should the two heads be weighted
# against each other (beta), and should exploration be discounted differently from
# exploitation (gamma_E vs gamma_I)?
#
# 4 games x 39 (beta, gamma_E, gamma_I) combos x 5 seeds = 780 runs.
# submit.sh packs 4 runs per GPU -> 195 array tasks, under MaxArraySize=1000, so
# run_sweep.sh submits this as a single array job (TASK_OFFSET stays 0).
#
# Two arms:
#   beta > 0 -- the full 3x3 gamma grid, intrinsic_loss_coef at its 1.0 default.
#               4 betas x 9 pairs = 36 combos.
#   beta = 0 -- the extrinsic-only control, which ALSO sets
#               intrinsic_loss_coef=0.0. gamma_E is still swept over all three
#               values; gamma_I is pinned to gamma_E. 3 combos.
#
# Why gamma_I is pinned only in the beta=0 arm. At the 1.0 default, beta=0 does
# NOT neutralise the intrinsic head: its TD loss still backprops through the
# shared trunk, so gamma_I changes the intrinsic targets -> the trunk -> Q_e ->
# behaviour. Zeroing intrinsic_loss_coef cuts that path: 0.0 * intrinsic_q_loss
# contributes exactly zero gradient, the intrinsic head never leaves its init, and
# Q_e + 0.0*Q_i is exactly Q_e so the argmax and epsilon-greedy draws (and RNG
# consumption) are independent of gamma_I. All that varies is the logged
# loss_intrinsic_q, a diagnostic on a frozen head -- so the three gamma_I values
# there would be the same run three times.
#
# Zeroing both auxiliary coefficients is also what makes beta=0 a genuine
# extrinsic-only control rather than "extrinsic control plus an intrinsic-value
# auxiliary task".
#
# count_layer is fixed at 2 (conv2), which was the best count position in
# atari57_count_layer_sweep. count_layer is 1-indexed over
# [conv1, conv2, conv3, *network.blocks], and the FTA layer sits *only* at the
# count position, so every line spells out all four activations rather than
# relying on the `default` subcommand's defaults.
#
# next_state_coef is fixed at 0.0 and NOT swept. At 0.0 the network skips building
# the auxiliary next-state head entirely and the rollout stops carrying its target,
# which is what makes 4 runs fit on one GPU at MEM_FRACTION=0.22.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

games=(venture space_invaders freeway seaquest)

# conv2: FTA at the second conv block, relu everywhere else.
count_layer=2
activations="network.cnn-activation-1:relu network.cnn-activation-2:fta network.cnn-activation-3:relu network.blocks.0.activation:relu"

betas=(1.0 0.5 0.1 0.01 0.0)
gammas=(0.9 0.99 0.999)
seeds=(0 1 2 3 4)
next_state_coef=0.0

sweep="atari4_seperate_heads_gamma_sweep"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim.
# tyro wants the `--flag value` args first and the `key:subcommand` tokens last.
>"$output_file"
for game in "${games[@]}"; do
    for beta in "${betas[@]}"; do
        # See the header: the beta=0 control freezes the intrinsic head, which makes
        # gamma_I inert, so it takes the diagonal and zeroes the coefficient.
        if [ "$beta" = "0.0" ]; then
            intrinsic_loss_coef=0.0
        else
            intrinsic_loss_coef=1.0
        fi
        for gamma in "${gammas[@]}"; do
            if [ "$beta" = "0.0" ]; then
                intrinsic_gammas=("$gamma")
            else
                intrinsic_gammas=("${gammas[@]}")
            fi
            for intrinsic_gamma in "${intrinsic_gammas[@]}"; do
                for seed in "${seeds[@]}"; do
                    out="$sweep/$game/beta_${beta}/gamma_${gamma}/intrinsic_gamma_${intrinsic_gamma}/seed_${seed}"
                    echo "default --environment $game --force-xla --beta $beta --gamma $gamma --intrinsic-gamma $intrinsic_gamma --network.next_state_coef $next_state_coef --network.intrinsic_loss_coef $intrinsic_loss_coef --network.count_layer $count_layer --seed $seed --output-folder-name $out $activations" >>"$output_file"
                done
            done
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
