#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of first parameters (number of q_pairs)
q_pairs=(2 3 5 10)

# Array of seeds
seeds=(42 123 456 789 1024)

# Array of q_std_clip
q_std_clip=(1)

# Add delay between process launches
delay_seconds=2

for q in "${q_pairs[@]}"; do
    for seed in "${seeds[@]}"; do
        for q_std_clip in "${q_std_clip[@]}"; do
            echo "Starting process with q_pairs=$q and seed=$seed and q_std_clip=$q_std_clip"
            
            # Run process and wait for the specified delay
            (python -m firl.irl_samples --config configs/samples/agents/hopper.yml --num_q_pairs $q --seed $seed --q_std_clip $q_std_clip> outputs/run_q${q}_seed${seed}_clip${q_std_clip}.log 2>&1) &
            
            # Add delay between launches
            sleep $delay_seconds
        done
    done
done

wait

echo "All experiments launched and completed!"
