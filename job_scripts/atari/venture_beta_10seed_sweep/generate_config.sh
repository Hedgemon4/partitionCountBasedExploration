#!/bin/bash

# Sweep Options
betas=(0.0 0.00022 0.00073 0.00243 0.0081 0.027 0.09 0.1 0.25 0.3 0.5 1.0)
next_state_coef=0.0

output_file="configs.txt"

# One line per beta (seed templated as seed_SEED, filled in by submit.sh).
# 12 betas x 1 next_state_coef = 12 configs; submit.sh expands each to 10 seeds.
>"$output_file"
for beta in "${betas[@]}"; do
    echo "one-block --environment venture --force-xla --beta $beta --network.next_state_coef $next_state_coef --output-folder-name venture_beta_10seed_sweep/beta_${beta}/seed_SEED" >>"$output_file"
done
