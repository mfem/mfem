#!/bin/bash

# Define the arrays for NP and R
NPS=(4 16 64 128)
RS=(0 1 2 3 4)

# Loop through each NP value
for NP in "${NPS[@]}"
do
    # Loop through each R value
    for R in "${RS[@]}"
    do
        echo "Running: mpirun -np $NP ./par_obstacle_prec -r $R"
        
        # Execute the command and save output to the specific filename
        mpirun --oversubscribe -np "$NP" ./par_obstacle_prec -r "$R" | tee "result_${NP}_${R}.txt"
        
        echo "Finished NP=$NP, R=$R"
        echo "--------------------------"
    done
done
