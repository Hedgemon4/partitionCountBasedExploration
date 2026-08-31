#!/bin/bash

# Pure PQN baseline (pqn_atari.py, all-relu conv trunk), Q-learning only, on 4 games.
#
# This is the walltime probe for the full atari57_baseline sweep, not a result in its own
# right. It exists because pqn_atari.py has no runtime precedent at 4 runs/GPU: the
# 08:59 / 4-per-GPU setting that ran atari57_count_layer_sweep losslessly was measured on
# the counts network, which is ~4x heavier per update (it carries a
# (num_steps, num_envs, 64, 9, 9, num_bins) discrete_state buffer plus a permuted copy per
# epoch, a counts scatter, and a 10x wider conv3).
#
# Games are chosen as walltime stressors first, comparability second:
#   solaris  0/40 finished at 02:59 in atari57_beta_sweep, and the only game to lose runs
#            even at 08:59 (all 4 of atari57_count_layer_sweep's losses). Worst case.
#   alien    1/40 at 02:59. Has both FTA beta=0 cells, so a full trunk x bootstrap 2x2.
#   freeway  1/40 at 02:59. Also has both FTA beta=0 cells.
#   amidar   Has both FTA beta=0 cells, plus pre-fix pure-PQN runs in data/amidar_baseline
#            for a before/after read on the shuffle-key fix.
#
# SARSA is omitted: the two bootstrap rules differ only by an argmax-vs-index in the target
# computation, which is negligible against a rollout and 2 epochs of updates, so the elapsed
# time measured here transfers to the SARSA arm of the full sweep.
#
# 4 games x 1 rule x 3 seeds = 12 runs, 3 array tasks at 4 runs/task.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

games=(solaris alien freeway amidar)
seeds=(0 1 2)

# name|flag, matching the full sweep's format so the two scripts stay diffable.
declare -a bootstraps=(
    "qlearning|--no-sarsa-returns"
)

sweep="atari4_baseline_probe"
output_file="$SCRIPT_DIR/configs.txt"

# One line per run, seed baked in, so submit.sh runs the line verbatim. No subcommand token:
# AtariConfig is already tyro's default= and nothing nested is being swept.
>"$output_file"
for game in "${games[@]}"; do
    for bootstrap in "${bootstraps[@]}"; do
        IFS='|' read -r name flag <<<"$bootstrap"
        for seed in "${seeds[@]}"; do
            out="$sweep/$game/$name/seed_${seed}"
            echo "--environment $game --force-xla $flag --seed $seed --output-folder-name $out" >>"$output_file"
        done
    done
done

echo "Wrote $(wc -l <"$output_file") lines from ${#games[@]} games"
