#!/bin/bash

# Sweep Options
seeds=(0 1 2 3 4 5 6 7 8 9)
counter=0

output_file="configs.txt"

>"$output_file"
for seed in "${seeds[@]}"; do
    echo "--environment freeway --force-xla --seed $seed --output-folder-name freeway_baseline_sweep/run_$counter" >>"$output_file"
    ((counter++))
done
