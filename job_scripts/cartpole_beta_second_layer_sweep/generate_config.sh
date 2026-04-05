#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.01)
activations=(fta fta-original)
max_grad_norm=(10.0 100.0)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for activation in "${activations[@]}"; do
        for max_norm in "${max_grad_norm[@]}"; do
            echo "--beta $beta --max-grad-norm $max_norm --network.count-layer 2 network.activation1:relu network.activation2:$activation --output-folder-name cartpole_beta_second_layer_sweep/beta_"$beta"__max_grad_norm_"$max_norm"__"$activation" " >>"$output_file"
        done
    done
done
