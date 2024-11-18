#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p outputs

# Array of first parameters (number of q_pairs)
q_pairs=(1 2 3 5 10)

# Array of seeds
seeds=(42 123 456 789 1024)

# Array of q_std_clip values
q_std_clip=(1 3 5)

# Add delay between process launches
delay_seconds=2

# Validate arrays are not empty
if [ ${#q_pairs[@]} -eq 0 ] || [ ${#seeds[@]} -eq 0 ] || [ ${#q_std_clip[@]} -eq 0 ]; then
    echo "Error: One or more parameter arrays are empty"
    exit 1
fi

for q in "${q_pairs[@]}"; do
    for seed in "${seeds[@]}"; do
        # If q is 1, only use clipping value of 1
        if [ "${q}" -eq 1 ]; then
            clip=1
            echo "Starting process with q_pairs=${q} and seed=${seed} and q_std_clip=${clip}"
            
            # Run process and wait for the specified delay
            # (python -m firl.irl_samples --config configs/samples/agents/ant.yml \
            #     --num_q_pairs "${q}" \
            #     --seed "${seed}" \
            #     --q_std_clip "${clip}" \
            #     > "outputs/run_q${q}_seed${seed}_clip${clip}.log" 2>&1) &
            
            # sleep $delay_seconds
        else
            # For other q values, iterate through all clipping values
            for clip in "${q_std_clip[@]}"; do
                echo "Starting process with q_pairs=${q} and seed=${seed} and q_std_clip=${clip}"
                
                # Run process and wait for the specified delay
                # (python -m firl.irl_samples --config configs/samples/agents/ant.yml \
                #     --num_q_pairs "${q}" \
                #     --seed "${seed}" \
                #     --q_std_clip "${clip}" \
                #     > "outputs/run_q${q}_seed${seed}_clip${clip}.log" 2>&1) &
                
                # sleep $delay_seconds
            done
        fi
    done
done

wait

echo "All experiments launched and completed!"

