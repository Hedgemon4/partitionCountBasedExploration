#!/bin/bash

# Sweep Options
betas=(1.0 0.5 0.25)
next_state_coef=0.0
seeds=(0 1 2)

output_file="configs.txt"

# One line per run (seed baked in). 8 betas x 3 seeds = 24 runs.
# Runs are later paired 2-per-GPU across this flat list, so pairs may span
# different betas/seeds (3 seeds/beta does not divide evenly into GPU pairs).
>"$output_file"
for beta in "${betas[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "one-block --environment venture --force-xla --beta $beta --network.next_state_coef $next_state_coef --output-folder-name venture_larger_beta_sweep/beta_${beta}/seed_${seed}" >>"$output_file"
    done
done
