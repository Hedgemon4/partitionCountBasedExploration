#!/bin/bash
# aip-mbowling / full L40S. 2 runs per GPU: keep --ntasks, the `for i` loop, the `2 *` in
# LINE_NO, and RUNS_PER_TASK in run_sweep.sh in step.
#
# 2 runs x 8 env threads rather than 4 x 4, because that is the only layout with a measured
# walltime: atari4_baseline_probe fits 2:59 at 2 x 8 (b9fd3d1), and 4 x 4 was never timed on
# pqn_atari.py. Not a cost increase -- the bottleneck is Atari env stepping, not the GPU, so
# two runs at 8 threads finish in roughly half the time four at 4 do and the doubled task
# count cancels. 2 x 8 is also the same 16 cores and 64G/job the 4 x 4 layout requested.
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6G
#SBATCH --time=02:59:00
#SBATCH --gpus=1
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

# effective_task = SLURM_ARRAY_TASK_ID + TASK_OFFSET; task k -> config lines
# {2*k+1 .. 2*k+2} (1-indexed for sed). Keep in step with RUNS_PER_TASK in run_sweep.sh.
TASK_OFFSET=${TASK_OFFSET:-0}
EFFECTIVE_TASK=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
for i in 0 1; do
    LINE_NO=$((2 * EFFECTIVE_TASK + i + 1))
    LINE=$(sed -n "${LINE_NO}p" "$CONFIG_PATH")
    [ -z "$LINE" ] && continue
    # 0.45, not the probe's 0.22: that value was sized as 4 x 0.22 ~ 0.88 of the card, and
    # at two runs it reserves 0.44 and leaves over half the GPU idle. 2 x 0.45 = 0.90
    # restores the headroom the 4-run layout had. This is the one setting here the probe's
    # timing does not cover; it cannot OOM at 0.90 total and should be neutral-to-faster.
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        python pqn_atari.py $LINE --num-env-threads 8 &
done

wait
echo "All runs finished"
