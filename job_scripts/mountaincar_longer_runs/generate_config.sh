#!/bin/bash

# Sweep Options
betas=(1.0 0.5 0.1 0.0)
time_steps=(5e5 1e6 2e6)
epsilon_decay=(0.2 0.1 0.05)
counter=0

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for max_step in "${time_steps[@]}"; do
      for decay in "${epsilon_decay[@]}"; do
        echo "--beta $beta --total_time_steps $max_step --epsilon_decay $decay --output-folder-name mountaincar_longer_runs/run_$counter" >>"$output_file"
        ((counter++))
        done
    done
done
