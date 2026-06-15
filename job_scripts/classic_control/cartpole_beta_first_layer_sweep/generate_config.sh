#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.01)
hidden_size=(64 128)
activations=(fta fta-original)
max_grad_norm=(10.0 100.0)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for size in "${hidden_size[@]}"; do
        for activation in "${activations[@]}"; do
            for max_norm in "${max_grad_norm[@]}"; do
                echo "--beta $beta --max-grad-norm $max_norm --network.hidden-size $size network.activation1:$activation --output-folder-name cartpole_beta_first_layer_sweep/beta_"$beta"__hidden_size_"$size"__max_grad_norm_"$max_norm"__"$activation"" >>"$output_file"
            done
        done
    done
done
