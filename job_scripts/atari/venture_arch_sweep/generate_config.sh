#!/bin/bash

# Compare the four pqn_atari network architectures on Venture, whole-game
# episodes (--no-life-loss-info). One config line per architecture; the
# tyro subcommand (first token) selects the network, the rest are CLI args.
#
#   pong          -> AtariConfig          (normal CNN, [Block(512)])
#   double-relu   -> AtariTwoReluConfig   ([Block(512), Block(512)])
#   fta           -> AtariFTAConfig       (single FTA block, [FTABlock(512)])
#   fta-two-layer -> AtariTwoFTAConfig    (FTA then ReLU, [FTABlock(512), Block(512)])

declare -a archs=(
    "pong:normal"
    "double-relu:double_relu"
    "fta:fta"
    "fta-two-layer:fta_two_layer"
)

output_file="configs.txt"

>"$output_file"
for entry in "${archs[@]}"; do
    subcommand="${entry%%:*}"
    name="${entry##*:}"
    echo "$subcommand --environment venture --force-xla --no-life-loss-info --output-folder-name venture_arch_sweep/${name}/seed_SEED" >>"$output_file"
done
