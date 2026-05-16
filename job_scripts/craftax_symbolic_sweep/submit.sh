#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:59:00

module load python/3.12.4

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip install -U -r requirements.txt

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_PATH")

python pqn_with_counts.py craftax $PARAMS
 