#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of first parameters (number of q_pairs)
q_pairs=(2 3 4 5 10)

# Array of seeds
seeds=(42 123 456 789 1024)

# Add delay between process launches
delay_seconds=2

for q in "${q_pairs[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Starting process with q_pairs=$q and seed=$seed"
        
        # Run process and wait for the specified delay
        (python -m firl.irl_samples configs/samples/agents/hopper.yml $q $seed > outputs/run_q${q}_seed${seed}.log 2>&1) &
        
        # Add delay between launches
        sleep $delay_seconds
    done
done

wait

echo "All experiments launched and completed!"