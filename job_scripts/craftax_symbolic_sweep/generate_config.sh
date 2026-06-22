#!/bin/bash

# Sweep Options
betas=(0.0 0.001 0.005 0.01 0.05 0.1)

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
        echo "--beta $beta --output-folder-name craftax_symbolic/beta_"$beta"__fta_first_layer" >>"$output_file"
done
