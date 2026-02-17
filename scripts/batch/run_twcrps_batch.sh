#!/bin/bash

# Run batch for WN2 first, then GenCast
for model in wn2 gencast; do
    echo "=========================================="
    echo " 🚀 Running twCRPS Batch for: $model"
    echo "=========================================="
    
    for var in t2m winds tp; do
        for lead in 24 72; do
            echo " Plotting $var (${lead}h) for $model..."
            python s138-plot-twcrps.py --var "$var" --lead "$lead" --mask land --model "$model"
        done
    done
    
    echo -e "✅ Finished batch and plotting for $model\n"
done