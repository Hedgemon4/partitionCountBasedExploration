#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=02:59:00
#SBATCH --gpus=h100_2g.20gb:1

module load python/3.12.4

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
python -m pip install --no-index --upgrade pip

python -m pip install -U -r requirements.txt

python -m pip install ale/

python pqn_atari_with_counts.py
