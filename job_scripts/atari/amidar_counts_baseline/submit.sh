#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=64
#SBATCH --mem=32GB
#SBATCH --time=08:59:00
#SBATCH --gpus=h100_3g.40gb:1
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
cp /home/slakins/links/scratch/projects/partitionCountBasedExploration/roms/*.bin $SLURM_TMPDIR/env/lib/python3.12/site-packages/ale_py/roms/

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_PATH")

for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "Starting seed $seed"
    PARAMS="${LINE/SEED/$seed}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.08 python pqn_atari_with_counts.py $PARAMS --seed $seed &
done

wait
echo "All seeds finished"
