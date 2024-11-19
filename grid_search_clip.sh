#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of first parameters (number of q_pairs)
# q_pairs=(1 2 3 5 10)
q=4

# Array of seeds
seeds=(42 123 456 789 1024)

# Array of q_std_clip values
q_std_clip=(1 2 3 5 10 20 50)

# Add delay between process launches
delay_seconds=2

env="ant"

# Create environment-specific output directory if it doesn't exist
mkdir -p "outputs/${env}"

for clip in "${q_std_clip[@]}"; do
    for seed in "${seeds[@]}"; do

        echo "Starting process with q_pairs=${q} and seed=${seed} and q_std_clip=${clip}"
        
        # Run process and wait for the specified delay
        (python -m firl.irl_samples --config configs/samples/agents/${env}.yml \
            --num_q_pairs "${q}" \
            --seed "${seed}" \
            --q_std_clip "${clip}" \
            > "outputs/${env}/run_q${q}_seed${seed}_clip${clip}.log" 2>&1) &
        
        sleep $delay_seconds

    done
done

wait

echo "All experiments launched and completed!"