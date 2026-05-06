#!/bin/bash

# Sweep Options
betas=(1.0 0.5 0.1 0.0)
fta_bounds=(3)
learnable_norm=(learnable-norm-params no-learnable-norm-params)
counter=0

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for bound in "${fta_bounds[@]}"; do
        for norm in "${learnable_norm[@]}"; do
            echo "--beta $beta --network.blocks.0.activation.bound $bound --network.blocks.0.$norm --output-folder-name mountaincar_bounds_sweep/run_$counter" >>"$output_file"
            ((counter++))
        done
    done
done
