#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.25 0.5 1.0)
next_state=(0.0 0.25 0.5 1.0)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for next in "${next_state[@]}"; do
        echo "--environment amidar --force-xla --beta $beta --network.next_state_coef $next --output-folder-name amidar_counts_sweep/beta_${beta}_next_${next}/seed_SEED" >>"$output_file"
    done
done
