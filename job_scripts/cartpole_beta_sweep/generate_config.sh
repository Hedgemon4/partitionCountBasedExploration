#!/bin/bash

# Sweep Options
betas=(0.05 0.01 0.001)
output_file="configs.txt"
>"$output_file"
for beta in "${betas[@]}"; do
    echo "--beta $beta" --metrics_folder_name cartpole_beta_sweep/beta_$beta >>"$output_file"
done
