#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:59:00
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

cp /home/slakins/scratch/projects/partitionCountBasedExploration/roms/*.bin $SLURM_TMPDIR/env/lib/python3.12/site-packages/ale_py/roms/

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_PATH")

echo "Starting Run"
python pqn_atari.py $PARAMS
echo "Finished Run"
