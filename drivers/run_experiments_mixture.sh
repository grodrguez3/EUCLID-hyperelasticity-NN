#!/bin/bash
# Define an array for all material models
material_models=("NeoHookean" "Isihara" "HainesWilson")
# Define an array for noise levels
noise_levels=("high")

# Loop through each noise level and run the experiment with all material models at once
for noise in "${noise_levels[@]}"; do
    echo "Running experiment with models: ${material_models[*]} and noise level: $noise_levels"
    python -W ignore main2.py "${material_models[@]}" "${noise_levels[@]}"
done

#This will create 2 models one for hgih and one for low... 
#chmod +x run_experiments_mixture.sh grant permision to run file 
 