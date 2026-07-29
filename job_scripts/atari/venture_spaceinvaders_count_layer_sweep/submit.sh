#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # 2 runs per job
#SBATCH --cpus-per-task=8          # 8 cores per run
#SBATCH --mem-per-cpu=4G           # 16 cores x 4G = 64G/job
#SBATCH --time=08:59:00
#SBATCH --gpus=1                   # one L40S, shared by the 2 runs
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

# 225 array tasks cover 450 runs (2 games x 3 count positions x 3 next_state_coefs
# x 5 betas x 5 seeds), 2 runs per GPU. 225 < MaxArraySize=1000, so this is a
# single array job and no TASK_OFFSET is needed.
#   task k -> config lines {2*k+1, 2*k+2} (1-indexed for sed).
# 5 seeds is odd, so a pair may straddle two configs; that is fine.
for i in 0 1; do
    LINE_NO=$((2 * SLURM_ARRAY_TASK_ID + i + 1))
    LINE=$(sed -n "${LINE_NO}p" "$CONFIG_PATH")
    [ -z "$LINE" ] && continue         # tolerate an odd/short final task
    # The --seed is baked into each config line, so run the line verbatim.
    # Background both runs (no srun step) so they share the one GPU in this allocation.
    # srun --exclusive serializes them on the single GPU; plain backgrounding does not.
    # --num-env-threads 8 caps each run's ALE pool: 2 runs = 16 threads on 16 cores
    # (without it, ALE auto spawns ~64 threads each -> oversubscription).
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        python pqn_atari_with_counts.py $LINE --num-env-threads 8 &
done

wait
echo "All runs finished"
