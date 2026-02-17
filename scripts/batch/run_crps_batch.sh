#!/bin/bash
#
# Batch CRPS computation for WN2 and GenCast
# Runs 4 combinations: {WN2, gencast} × {24h, 72h} for winds variable
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/s116-crps-zarr-v2.py"

echo "======================================"
echo "Batch CRPS Computation"
echo "======================================"
echo ""

# Array of models and lead times
MODELS=("WN2" "gencast")
LEADS=(24 72)
VARIABLE="winds"  # Change to "t2m" if needed

# Loop through all combinations
for MODEL in "${MODELS[@]}"; do
    for LEAD in "${LEADS[@]}"; do
        echo ""
        echo "--------------------------------------"
        echo "Running: ${MODEL} | ${VARIABLE} | ${LEAD}h"
        echo "--------------------------------------"
        
        python "${PYTHON_SCRIPT}" \
            --model "${MODEL}" \
            --var "${VARIABLE}" \
            --lead "${LEAD}"
        
        if [ $? -eq 0 ]; then
            echo "✅ SUCCESS: ${MODEL} ${VARIABLE} ${LEAD}h"
        else
            echo "❌ FAILED: ${MODEL} ${VARIABLE} ${LEAD}h"
            exit 1
        fi
    done
done

echo ""
echo "======================================"
echo "✅ All CRPS computations completed!"
echo "======================================"

# how to call: ./run_crps_batch.sh