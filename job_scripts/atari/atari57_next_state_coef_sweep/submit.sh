#!/bin/bash
# aip-mbowling / full L40S. 4 runs per GPU: keep --ntasks, the `for i` loop, the `4 *` in
# LINE_NO, and RUNS_PER_TASK in run_sweep.sh in step.
#
# MEM_FRACTION stays at 0.22, the same as atari57_seperate_heads_sarsa_sweep and
# atari57_count_layer_sweep, even though this sweep runs next_state_coef > 0 and they did
# not. An earlier revision of this file raised it to 0.24 on the theory that the auxiliary
# next-state target needed more room. That was wrong, and it broke the sweep:
#
#   MEM_FRACTION is a *preallocation*, not a demand. XLA reserves that fraction of the GPU
#   at process start regardless of what the program goes on to use, and the auxiliary
#   buffer is allocated inside that pool -- so raising the fraction cannot "make room" for
#   it. What it does do is raise total reservation from 4 x 0.22 = 0.88 to 4 x 0.24 = 0.96,
#   leaving ~1.9 GB of the 48 GB L40S outside the pools for four CUDA contexts (~300-500 MB
#   each) *plus* four cuBLAS workspaces, which cuBLAS allocates outside XLA's pool. At 0.24
#   some runs died with "failed to create cublas handle: the resource allocation failed"
#   during compilation/autotuning, then segfaulted on the null handle -- before the rollout
#   was ever allocated. Marginal, so only some runs failed.
#
#   The buffer was never the problem. At count_layer=2 the target is
#   (num_steps=32, num_envs=128, 64, 9, 9, 10) float32 = 849 MB (num_bins is
#   int(2*1.0/0.25) + 2 = 10, layers.py:18-26; conv2's features are 64x9x9 = 5184), and the
#   head adds ~2.66M parameters. But discrete_state -- the one-hot, same (64, 9, 9, 10)
#   shape, also float32 since jax.nn.one_hot returns float32 -- is *already* in the rollout
#   at coef=0. Total rollout state goes 1.08 GB -> 1.93 GB inside a 10.56 GB pool. The
#   "~2 GB at peak" in netwoks.py:599-603 is that whole footprint, not a delta on top of a
#   full pool.
#
# So: 4 runs at 0.22. If a run ever does exhaust its own pool (unlikely -- persistent
# rollout state is ~2 GB of the 10.56 GB, parameters plus Adam state are ~4.3M floats, and
# a minibatch is 128 transitions), store the target in bfloat16 at
# pqn_atari_counts_with_seperate_value_head.py:282 with a matching upcast in
# netwoks.py:_next_state_loss, which halves it. Do not raise MEM_FRACTION.
#
#   time 08:59:00 -> 11:59:00
#     The auxiliary head costs one more vmapped forward and backward per minibatch per
#     epoch, on top of the 4-way GPU contention the sarsa sweep already had -- and that
#     sweep still lost runs on its slowest games at 08:59. 11:59 is inside the same
#     sub-12h scheduling band, so the headroom is free of any queue-priority penalty.
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # 4 runs per job
#SBATCH --cpus-per-task=4          # 4 cores per run
#SBATCH --mem-per-cpu=7G           # 16 cores x 7G = 112G/job (host RAM, not GPU)
#SBATCH --time=11:59:00
#SBATCH --gpus=1                   # one L40S, shared by the 4 runs
#SBATCH --mail-user=slakins@ualberta.ca
#SBATCH --mail-type=ALL

module load python/3.12.4
module load gcc/12.3
module load cuda/12.9
module load opencv
module load cmake

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip
python -m pip install -U -r requirements.txt --no-index -f wheels/
python -m pip install ale/ --no-index
cp /home/slakins/scratch/projects/partitionCountBasedExploration/roms/*.bin \
    $SLURM_TMPDIR/env/lib/python3.12/site-packages/ale_py/roms/

# 1283 array tasks cover 5130 runs (57 games x 6 betas x 3 next_state_coefs x 5 seeds),
# 4 runs per GPU. 1283 > MaxArraySize=1000, so run_sweep.sh splits this into two array
# jobs with TASK_OFFSET 0 and 1000.
#   effective_task = SLURM_ARRAY_TASK_ID + TASK_OFFSET
#   task k -> config lines {4*k+1 .. 4*k+4} (1-indexed for sed).
#
# Plain `&` backgrounding, not srun: srun --exclusive serializes the runs on the single
# GPU. --num-env-threads 4 x 4 runs = 16 threads on 16 cores; without it ALE's auto mode
# spawns ~64 threads per run and oversubscribes the node.
TASK_OFFSET=${TASK_OFFSET:-0}
EFFECTIVE_TASK=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
started=()
for i in 0 1 2 3; do
    LINE_NO=$((4 * EFFECTIVE_TASK + i + 1))
    LINE=$(sed -n "${LINE_NO}p" "$CONFIG_PATH")
    [ -z "$LINE" ] && continue         # tolerate a short final task
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 \
        python pqn_atari_counts_with_seperate_value_head.py $LINE --num-env-threads 4 &
    started+=("$!:$LINE_NO")
done

# `wait` with no arguments returns 0 whatever the children did, so the previous
# unconditional "All runs finished" reported success even when all four runs segfaulted --
# which is how the 0.24 cuBLAS failure above came to be noticed only by reading stderr by
# hand. Wait on each PID individually and propagate, so a failed task shows up as FAILED in
# sacct and the offending config lines are named in the log.
failed=0
for entry in "${started[@]}"; do
    if ! wait "${entry%%:*}"; then
        echo "FAILED: config line ${entry##*:}" >&2
        failed=$((failed + 1))
    fi
done

echo "$((${#started[@]} - failed))/${#started[@]} runs finished"
[ "$failed" -eq 0 ] || exit 1
