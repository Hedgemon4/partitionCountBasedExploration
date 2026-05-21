#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=2:59:00
#SBATCH --gpus=h100_2g.20gb:1

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

cp /home/slakins/links/scratch/projects/partitionCountBasedExploration/roms/*.bin    $SLURM_TMPDIR/env/lib/python3.12/site-packages/ale_py/roms/

python pqn_atari_with_counts.py --output-folder-name atari/freeway_testing/ --environment freeway
