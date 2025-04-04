#!/bin/bash

# Define arrays for material models and noise levels
material_models=("NeoHookean" "Isihara" "HainesWilson") 
noise_levels=("high" "low")

# Loop through each combination and run the experiment
for material in "${material_models[@]}"; do
    for noise in "${noise_levels[@]}"; do
        echo "Running experiment for material: $material, noise level: $noise"
        python main.py "$material" "$noise"
    done
done
