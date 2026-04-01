#!/bin/bash

# Sweep Options
betas=(0.05 0.01 0.001)
output_file="configs.txt"
>"$output_file"
for beta in "${betas[@]}"; do
    echo "--beta $beta" >>"$output_file"
done
