#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # 2 seeds per job
#SBATCH --cpus-per-task=8          # 8 cores per seed
#SBATCH --mem-per-cpu=4G           # 16 cores x 4G = 64G/job
#SBATCH --time=02:59:00
#SBATCH --gpus=1                   # one L40S, shared by the 2 seeds
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

LINE=$(sed -n "1p" "$CONFIG_PATH")

for i in 0 1; do
    SEED=$((2 * SLURM_ARRAY_TASK_ID + i)) # task 0->{0,1}, 1->{2,3}, ... 4->{8,9}
    PARAMS="${LINE/SEED/$SEED}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
        srun --exclusive --ntasks=1 --cpus-per-task=8 \
        python pqn_atari_with_counts.py $PARAMS --seed $SEED &
done

wait
echo "All seeds finished"
