#!/bin/bash
# aip-mbowling / full L40S. 2 runs per GPU: keep the `for i` loop, the `2 *` in LINE_NO, and
# RUNS_PER_TASK in run_sweep.sh in step.
#
# --ntasks=4 --cpus-per-task=4 is left as it ran: this directory is the record of the
# measurement behind atari57_baseline's walltime, so the allocation is not rewritten after
# the fact. It reserves 16 cores either way, which is what the two runs use at 8 env threads
# each. Same for XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 below -- sized for the old 4-run layout,
# so this probe used only 0.44 of the card. atari57_baseline raises it to 0.45.
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
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
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 \
        python pqn_atari.py $LINE --num-env-threads 8 &
done

wait
echo "All runs finished"
