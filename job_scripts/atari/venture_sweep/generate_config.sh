#!/bin/bash

# Sweep Options
seeds=(0 1 2 3 4 5 6 7 8 9)
betas=(0.1 0.05 0.01 0.005 0.0)
next_state=(1.0 0.5 0.25 0.0)

counter=0

output_file="configs.txt"

>"$output_file"
for seed in "${seeds[@]}"; do
    for beta in "${betas[@]}"; do
        for next in "${next_state[@]}"; do
            echo "--environment venture --force-xla --beta $beta --network.next_state_coef $next --seed $seed --output-folder-name venture_sweep/run_$counter" >>"$output_file"
            ((counter++))
        done
    done
done
