#!/bin/bash

# Separate intrinsic/extrinsic value heads: how should the two heads be weighted
# against each other (beta), and should exploration be discounted differently from
# exploitation (gamma_E vs gamma_I)?

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
