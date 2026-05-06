#!/bin/bash

# Sweep Options
betas=(0.05 0.01 0.005 0.001 0.0)
learning_rate=(0.04 0.01 0.004 0.001 0.0001)
epsilon_decay=(1.0 0.5 0.2 0.1 0.05)
final_epsilons=(0.1 0.05 0.01)
state_prediction_loss_weights=(0.0)
counter=1125

output_file="configs.txt"

>"$output_file"
for beta in "${betas[@]}"; do
    for rate in "${learning_rate[@]}"; do
        for final_epsilon in "${final_epsilons[@]}"; do
            for decay in "${epsilon_decay[@]}"; do
              for weight in "${state_prediction_loss_weights[@]}"; do
                echo "--total-time-steps 1e6 --beta $beta --pessimistic --epsilon-end $final_epsilon --epsilon-decay $decay --initial_learning_rate $rate --final_learning_rate $rate --network.next-state-coef $weight --output-folder-name mountaincar_pessimistic_state_prediction_sweep/run_$counter" >>"$output_file"
                ((counter++))
                done
            done
        done
    done
done
