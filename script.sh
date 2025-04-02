#!/bin/bash

DIR="/home/pfarghad/Myschedulingmodel_3/RL/Instances/"
FAILED_JOBS="failed_jobs.txt"

# Clear previous failed jobs log
> $FAILED_JOBS

# Parameters you want to easily modify
MODEL="2Stage"      # Options: "Average" or "2Stage"
SOLVER="MIP"         # Options: "MIP", "ALNS", "PHA", "BBC"
NR_SCENARIO="5" 
PHA_OBJ="Q"          # Options: "Q" (Quadratic) or "L" (Linear)
PHAPenalty="S"       # Options: "S", "D", or "DL"
ALNSRL=0             # 0: no RL, 1: use RL in ALNS
ALNSRL_DEEPQ=0       # 0: Q-Learning, 1: Deep Q-Learning
BBC_SETTING="NS"     # Options: "NE", "JM", "NM", "JS", "NS", "JW", "NW", "JL", "NL", "AE"

# Arrays for different options (if you wish to loop over multiple values)
SCENARIO_GENERATION=("RQMC")        # Options: "MC", "RQMC", "QMC"
CLUSTER_METHODS=("KMeansPP")          # Options: "NoC", "KM", "KMPP", "SOM"


# Now update the instance parameters.
# Your new instances have six parts (e.g. "5_5_3_4_3_1_CRP"). Here we loop over each part.
for ARG1 in 5
do
    for ARG2 in 5
    do
        for ARG3 in 3
        do
            for ARG4 in 4
            do
                for ARG5 in 3
                do
                    for ARG6 in 1
                    do
                        INSTANCE_NAME="${ARG1}_${ARG2}_${ARG3}_${ARG4}_${ARG5}_${ARG6}_CRP"
                        echo "Submitting job for instance ${INSTANCE_NAME}"
                        
                        for SCENARIO_GEN in "${SCENARIO_GENERATION[@]}"
                        do
                            for CLUSTER_METHOD in "${CLUSTER_METHODS[@]}"
                            do
                                # Submit the job with 11 parameters.
                                sbatch ./qsub.sh \
                                    ${INSTANCE_NAME} \
                                    ${MODEL} \
                                    ${SOLVER} \
                                    ${NR_SCENARIO} \
                                    ${PHA_OBJ} \
                                    ${PHAPenalty} \
                                    ${ALNSRL} \
                                    ${ALNSRL_DEEPQ} \
                                    ${SETTING} \
                                    ${SCENARIO_GEN} \
                                    ${CLUSTER_METHOD}

                                if [ $? -ne 0 ]; then
                                    echo ${INSTANCE_NAME} >> $FAILED_JOBS
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

# Retry failed jobs if any
if [ -s $FAILED_JOBS ]; then
    echo "Retrying failed jobs..."
    while read -r INSTANCE_NAME; do
        echo "Resubmitting job for instance ${INSTANCE_NAME}"
        sbatch ./qsub.sh \
          ${INSTANCE_NAME} \
          ${MODEL} \
          ${SOLVER} \
          ${NR_SCENARIO} \
          ${PHA_OBJ} \
          ${PHAPenalty} \
          ${ALNSRL} \
          ${ALNSRL_DEEPQ} \
          ${BBC_SETTING} \
          ${SCENARIO_GENERATION[0]} \
          ${CLUSTER_METHODS[0]}
        if [ $? -ne 0 ]; then
            echo "Failed again: ${INSTANCE_NAME}"
        fi
    done < $FAILED_JOBS
fi