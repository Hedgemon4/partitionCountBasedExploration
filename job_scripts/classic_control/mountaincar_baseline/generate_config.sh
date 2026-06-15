#!/bin/bash

# Sweep Options
betas=(0.0 0.1)
activations=(fta fta-original)
max_grad_norm=(10.0 100.0)
episode_lengths=(200 1000)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for activation in "${activations[@]}"; do
        for max_norm in "${max_grad_norm[@]}"; do
            for length in "${episode_lengths[@]}"; do
                echo "--beta $beta --max-grad-norm $max_norm --episode_length $length network.activation2:$activation --output-folder-name mountaincar_baseline/beta_"$beta"__max_grad_norm_"$max_norm"__episode_length_"$length"__"$activation"" >>"$output_file"
            done
        done
    done
done
