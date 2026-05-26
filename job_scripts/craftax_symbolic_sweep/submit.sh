#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:59:00

module load python/3.12.4

mkdir $SLURM_TMPDIR/$SLURM_JOB_ID
cp ~/partitionCountBasedExploration/venv_gpu.tar.xz $SLURM_TMPDIR/$SLURM_JOB_ID
tar -xf $SLURM_TMPDIR/$SLURM_JOB_ID/venv_gpu.tar.xz -C $SLURM_TMPDIR/$SLURM_JOB_ID



PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_PATH")

$SLURM_TMPDIR/$SLURM_JOB_ID/.venv/bin/python3.12 ~/partitionCountBasedExploration/pqn_with_counts.py craftax $PARAMS


