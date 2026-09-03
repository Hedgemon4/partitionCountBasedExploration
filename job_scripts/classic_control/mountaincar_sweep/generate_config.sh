#!/bin/bash

# Sweep Options
betas=(0.0 0.1 0.05 0.01 0.005 0.001)
learning_rate=(0.005 0.001 0.0005 0.0001)
max_grad_norm=(10.0 100.0)
epsilon_end=(0.05 0.01 0.10)

# Probably have a good idea if FTA in last layer
hidden_size=(64 126 256)

# Probably two different sweeps here
count_layer

# Hidden 64 by default

# FTA bounds
# [-20, 20], eta 2.0
# [-1, 1], LayerNorm, don't learn params, eta 0.25

output_file="configs.txt"
>"$output_file"
for beta in "${betas[@]}"; do
    echo "--exploration.beta $beta" --metrics_folder_name cartpole_beta_sweep/beta_$beta >>"$output_file"
done
