#!/bin/bash

# Sweep Options
epsilon_decay=(0.2 0.1 0.05)
learning_rate=(0.04 0.01 0.004 0.001)
hidden_size=(64 128 256)
max_grad_norm=(10.0 100.0)
epsilon_end=(0.05 0.01 0.10)
counter=0

output_file="configs.txt"

>"$output_file"
for decay in "${epsilon_decay[@]}"; do
    for rate in "${learning_rate[@]}"; do
        for size in "${hidden_size[@]}"; do
            for max_norm in "${max_grad_norm[@]}"; do
                for epsilon in "${epsilon_end[@]}"; do
                    echo "--epsilon_decay $decay --network.hidden_size $size --epsilon_end $epsilon --max-grad-norm $max_norm --initial_learning_rate $rate --final_learning_rate $rate  --output-folder-name mountaincar_pqn_baseline_sweep/run_$counter" >>"$output_file"
                    ((counter++))
                done
            done
        done
    done
done
