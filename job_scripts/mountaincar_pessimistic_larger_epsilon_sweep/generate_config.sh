#!/bin/bash

# Sweep Options
betas=(1.0 0.5 0.1 0.0)
learning_rate=(0.04 0.01 0.004 0.001)
epsilon_decay=(0.5 0.7 1.0)
final_epsilons=(0.2 0.1 0.05 0.01)
max_grad_norm=(100.0 10.0)
counter=0

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for rate in "${learning_rate[@]}"; do
        for final_epsilon in "${final_epsilons[@]}"; do
            for decay in "${epsilon_decay[@]}"; do
                for max_norm in "${max_grad_norm[@]}"; do
                    echo "--beta $beta --pessimistic  --epsilon-end $final_epsilon --epsilon-decay $decay --max-grad-norm $max_norm --initial_learning_rate $rate --final_learning_rate $rate network:q-network-counts --output-folder-name mountaincar_pessimistic_larger_epsilon_sweep/run_$counter" >>"$output_file"
                    ((counter++))
                done
            done
        done
    done
done
