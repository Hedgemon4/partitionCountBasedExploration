#!/bin/bash

# Where in the network should the FTA layer + counts table live?
#
# count_layer is a 1-indexed position into [conv1, conv2, conv3, *network.blocks],
# so for the single-MLP-block atari config the candidate positions are:
#   2 -> conv2      spatial discrete representation, (64, 9, 9, bins)
#   3 -> conv3      raveled before the activation, (3136, bins)
#   4 -> MLP block  (512, bins)
# conv1 is excluded on purpose: its FTA map is 32x20x20xbins, which makes conv2
# 10x wider and pushes the rollout buffer past 4 GB per run.
#
# The FTA layer sits *only* at the count position; every other layer is ReLU, so
# each line spells out all four activations rather than relying on the defaults
# of the `default` subcommand (AtariCountsConfig / QNetworkCNNCountsDefaultConfig).
#
# 2 games x 3 positions x 3 next_state_coefs x 5 betas x 5 seeds = 450 lines,
# paired 2-per-GPU by submit.sh -> 225 array tasks.

declare -a games=(venture space_invaders)

# tyro's subcommand chain is positional and ordered:
#   default -> network:<config> -> cnn-activation-1 -> -2 -> -3 -> blocks.0.activation
# The `network:` token cannot be skipped if any activation token follows it, even
# though it re-selects the subcommand's own default network config.
network_subcommand="network:q-network-cnn-counts-default-config"

# name|count_layer|activation subcommands (declaration order: 1, 2, 3, blocks.0)
declare -a positions=(
    "conv2|2|network.cnn-activation-1:relu network.cnn-activation-2:fta network.cnn-activation-3:relu network.blocks.0.activation:relu"
    "conv3|3|network.cnn-activation-1:relu network.cnn-activation-2:relu network.cnn-activation-3:fta network.blocks.0.activation:relu"
    "mlp1|4|network.cnn-activation-1:relu network.cnn-activation-2:relu network.cnn-activation-3:relu network.blocks.0.activation:fta"
)

next_state_coefs=(0.0 0.5 1.0)
betas=(1.0 0.5 0.25 0.1 0.0)
seeds=(0 1 2 3 4)

sweep="venture_spaceinvaders_count_layer_sweep"
output_file="configs.txt"

# One line per run, seed baked in, so submit.sh can run the line verbatim.
# tyro wants the `--flag value` args first and the `key:subcommand` tokens last.
>"$output_file"
for game in "${games[@]}"; do
    for position in "${positions[@]}"; do
        IFS='|' read -r name count_layer activations <<<"$position"
        for next in "${next_state_coefs[@]}"; do
            for beta in "${betas[@]}"; do
                for seed in "${seeds[@]}"; do
                    echo "default --environment $game --force-xla --beta $beta --network.next_state_coef $next --network.count_layer $count_layer --seed $seed --output-folder-name $sweep/$game/$name/beta_${beta}/next_${next}/seed_${seed} $activations" >>"$output_file"
                done
            done
        done
    done
done

echo "Wrote $(wc -l <"$output_file") config lines to $output_file"
