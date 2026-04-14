#!/bin/bash

# Sweep Options
betas=(1.0 0.5 0.1 0.0)
epsilons=(0.2 0.1 0.05 0.01)
learning_rate=(0.04 0.01 0.004 0.001)
counter=0

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        for rate in "${learning_rate[@]}"; do
            echo "--beta $beta --epsilon_start $epsilon --epsilon_decay 0 --initial_learning_rate $rate --final_learning_rate $rate  --output-folder-name mountaincar_static_epsilon/run_$counter" >>"$output_file"
            ((counter++))
        done
    done
done
