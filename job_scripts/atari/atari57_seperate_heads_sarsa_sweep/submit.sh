#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=4                 # 4 runs per job
#SBATCH --cpus-per-task=4          # 4 cores per run
#SBATCH --mem-per-cpu=4G           # 16 cores x 4G = 64G/job
#SBATCH --time=08:59:00
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

# 2352 array tasks cover 9405 runs (57 games x 11 beta/gamma_I cells x 3 epsilons
# x 5 seeds), 4 runs per GPU. 2352 > MaxArraySize=1000, so run_sweep.sh splits this
# into three array jobs with TASK_OFFSET 0, 1000 and 2000.
#   effective_task = SLURM_ARRAY_TASK_ID + TASK_OFFSET
#   task k -> config lines {4*k+1 .. 4*k+4} (1-indexed for sed).
TASK_OFFSET=${TASK_OFFSET:-0}
EFFECTIVE_TASK=$((SLURM_ARRAY_TASK_ID + TASK_OFFSET))
for i in 0 1 2 3; do
    LINE_NO=$((4 * EFFECTIVE_TASK + i + 1))
    LINE=$(sed -n "${LINE_NO}p" "$CONFIG_PATH")
    [ -z "$LINE" ] && continue         # tolerate a short final task
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 \
        python pqn_atari_counts_with_seperate_value_head.py $LINE --num-env-threads 4 &
done

wait
echo "All runs finished"
