#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.05 1.0)
activations=(fta fta-original)
max_grad_norm=(10.0 100.0)
epsilon_end=(0.05 0.01 0.10)
hidden_size=(64 126 256)
learnable_norm=(learnable-norm-params no-learnable-norm-params)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for activation in "${activations[@]}"; do
        for max_norm in "${max_grad_norm[@]}"; do
            for size in "${hidden_size[@]}"; do
                for epsilon in "${epsilon_end[@]}"; do
                    for norm in "${learnable_norm[@]}"; do
                        echo "--beta $beta --epsilon_end $epsilon --max-grad-norm $max_norm --network.count_layer 2 --network.$norm --network.hidden_size $size network.activation2:$activation network.activation1:relu --output-folder-name mountaincar_count_second_layer/beta_"$beta"__max_grad_norm_"$max_norm"__final_epsilon_"$epsilon"__hidden_size_"$size"__"$activation"__"$norm"" >>"$output_file"
                    done
                done
            done
        done
    done
done
