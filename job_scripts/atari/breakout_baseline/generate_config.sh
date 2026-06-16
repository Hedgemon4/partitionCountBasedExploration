#!/bin/bash

seeds=(0 1 2 3 4 5 6 7 8 9)

output_file="configs.txt"

>"$output_file"
for seed in "${seeds[@]}"; do
    echo "--environment breakout --force-xla --seed $seed --output-folder-name breakout_baseline/seed_$seed" >>"$output_file"
done
