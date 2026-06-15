#!/bin/bash
#SBATCH --account=rrg-mbowling-ad
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=02:59:00

module load python/3.12.4

# Create virtual environment on local scratch
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Upgrade pip locally
python -m pip install --no-index --upgrade pip

# --- Retry Loop Starts Here ---
MAX_RETRIES=3
COUNT=0
SUCCESS=false

while [ $COUNT -lt $MAX_RETRIES ]; do
    echo "Attempt $((COUNT + 1)) to install requirements..."

    # We use --no-index where possible to prioritize the local wheelhouse
    if python -m pip install --no-index --find-links wheels -r requirements.txt; then
        echo "Installation successful!"
        SUCCESS=true
        break
    else
        echo "Installation failed. Retrying in 5 seconds..."
        sleep 5
        ((COUNT++))
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "Error: Failed to install requirements after $MAX_RETRIES attempts."
    exit 1
fi
# --- Retry Loop Ends Here ---

PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_PATH")

echo "Starting Run"
python pqn.py mountaincar $PARAMS
echo "Finished Run"
