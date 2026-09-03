#!/bin/bash
# aip-mbowling / full L40S. 4 runs per GPU: keep --ntasks, the `for i` loop, the `4 *` in
# LINE_NO, and RUNS_PER_TASK in run_sweep.sh in step.
#
# Two numbers differ from atari57_seperate_heads_sarsa_sweep/submit.sh, both because this
# sweep runs next_state_coef > 0 and that sweep did not:
#
#   MEM_FRACTION 0.22 -> 0.24
#     A nonzero coefficient builds the auxiliary head and makes the rollout carry its
#     per-transition target, which the 0.0 runs skip entirely (netwoks.py:599-631). At
#     count_layer=2 that target is (num_steps=32, num_envs=128, 64, 9, 9, 10) float32 =
#     849 MB, or ~1.7 GB counting the shuffle copy in process_data. num_bins is
#     int(2*1.0/0.25) + 2 = 10 (layers.py:18-26) and conv2's features are 64x9x9 = 5184.
#     The head itself adds only ~2.66M parameters.
#
#     0.24 gives back ~0.96 GB per run, and 4 x 0.24 = 0.96 of the GPU leaves ~1.9 GB for
#     four CUDA contexts. That is tighter than the arithmetic above wants, so this may
#     OOM. Submit the offset-0 array job first, check a few tasks' logs in
#     run_outputs/atari57_next_state_coef_sweep/ for RESOURCE_EXHAUSTED, and only then
#     submit offset-1000. If it does OOM: drop to 3 runs per task at 0.30 (--ntasks,
#     the `for i` loop, the `4 *` and RUNS_PER_TASK all move together), or store the
#     target in bfloat16 at pqn_atari_counts_with_seperate_value_head.py:282 with a
#     matching upcast in netwoks.py:_next_state_loss, which halves the buffer.
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
#SBATCH --mem-per-cpu=7G           # 16 cores x 4G = 64G/job
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
for i in 0 1 2 3; do
    LINE_NO=$((4 * EFFECTIVE_TASK + i + 1))
    LINE=$(sed -n "${LINE_NO}p" "$CONFIG_PATH")
    [ -z "$LINE" ] && continue         # tolerate a short final task
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.24 \
        python pqn_atari_counts_with_seperate_value_head.py $LINE --num-env-threads 4 &
done

wait
echo "All runs finished"
