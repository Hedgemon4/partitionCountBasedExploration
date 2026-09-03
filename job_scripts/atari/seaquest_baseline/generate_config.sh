#!/bin/bash

# Sweep Options
seeds=(0 1 2 3 4 5 6 7 8 9)
use_sarsa=(sarsa-returns no-sarsa-returns)
counter=0

output_file="configs.txt"

>"$output_file"
for seed in "${seeds[@]}"; do
    for sarsa in "${use_sarsa[@]}"; do
        echo "--environment seaquest --$sarsa --force-xla --seed $seed --output-folder-name seaquest_baseline/run_$counter" >>"$output_file"
        ((counter++))
    done
done
