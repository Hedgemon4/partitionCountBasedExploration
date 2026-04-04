#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.01)
hidden_size=(64 128)
activation=(fta fta-original)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    echo "--exploration.beta $beta" "network.activation1:$activation" --metrics_folder_name cartpole_beta_sweep/beta_$beta >>"$output_file"
done
