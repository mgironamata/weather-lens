#!/bin/bash

# Script to run CRPS calculations for wind speed (winds) 
# for WN2 and GenCast at 24h and 72h leads.

CONDA_RUN="/scratch/mg963/miniconda3/bin/conda run -p /scratch2/mg963/conda_envs/envs/regrid --no-capture-output"
PYTHON_SCRIPT="/scratch2/mg963/aifs-analysis/s116-crps-zarr-v2.py"

# Leads to process
LEADS=(24 72)

# Models to process
MODELS=("WN2" "gencast")

for model in "${MODELS[@]}"; do
    for lead in "${LEADS[@]}"; do
        echo "------------------------------------------------------------"
        echo "🚀 Running CRPS for $model | winds | ${lead}h"
        echo "------------------------------------------------------------"
        
        $CONDA_RUN python $PYTHON_SCRIPT --model "$model" --var winds --lead "$lead"
        
        if [ $? -ne 0 ]; then
            echo "❌ Error occurred while running $model at ${lead}h lead."
        fi
    done
done

# Special case for IFS_ENS (only 72h)
echo "------------------------------------------------------------"
echo "🚀 Running CRPS for IFS_ENS | winds | 72h"
echo "------------------------------------------------------------"
$CONDA_RUN python $PYTHON_SCRIPT --model "IFS_ENS" --var winds --lead 72
if [ $? -ne 0 ]; then
    echo "❌ Error occurred while running IFS_ENS at 72h lead."
fi

echo "------------------------------------------------------------"
echo "✅ All wind CRPS runs completed."
echo "------------------------------------------------------------"
