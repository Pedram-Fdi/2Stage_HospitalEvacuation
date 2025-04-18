#!/bin/bash

DIR="/home/pfarghad/Myschedulingmodel_3/RL/Instances/"
FAILED_JOBS="failed_jobs.txt"

# Clear previous failed jobs log
> $FAILED_JOBS

# Fixed parameter
MODEL="2Stage"    # Options: "Average" or "2Stage"
BBC_SETTING="NS"  # Use the appropriate value

# Independent arrays:
NR_SCENARIOS=("250")
SCENARIO_GENERATION=("RQMC")          # Options: "MC", "RQMC", "QMC"
CLUSTER_METHODS=("KM" "SOM")               # Options: "NoC", "KM", "KMPP", "SOM"

# Array for SOLVERS to loop over:
SOLVERS=("ALNS")

for SOLVER in "${SOLVERS[@]}"; do
  # Set parameter arrays based on the current solver:
  if [ "$SOLVER" = "MIP" ]; then
    PHA_OBJS=("Q")
    PHAPenalties=("S")
    ALNSRLs=(0)
    ALNSRL_DEEPQs=(0)
  elif [ "$SOLVER" = "ALNS" ]; then
    PHA_OBJS=("Q")
    PHAPenalties=("S")
    ALNSRLs=(0 1)
    # For ALNS, ALNSRL_DEEPQs will be set conditionally below
  elif [ "$SOLVER" = "PHA" ]; then
    ALNSRLs=(0)
    ALNSRL_DEEPQs=(0)
    PHA_OBJS=("Q")
    PHAPenalties=("S" "DL")
  fi

  for NR_SCENARIO in "${NR_SCENARIOS[@]}"; do
    for PHA_OBJ in "${PHA_OBJS[@]}"; do
      for PHAPenalty in "${PHAPenalties[@]}"; do
        for ALNSRL in "${ALNSRLs[@]}"; do
          # For the ALNS solver, set ALNSRL_DEEPQs based on ALNSRL value:
          if [ "$SOLVER" = "ALNS" ]; then
            if [ "$ALNSRL" -eq 0 ]; then
              ALNSRL_DEEPQs=(0)
            elif [ "$ALNSRL" -eq 1 ]; then
              ALNSRL_DEEPQs=(0 1)
            fi
          fi

          for ALNSRL_DEEPQ in "${ALNSRL_DEEPQs[@]}"; do
            # Loop over instance parameters:
            for ARG1 in 6; 
            do
              for ARG2 in 10 20; 
              do
                for ARG3 in 5; 
                do
                  for ARG4 in 10 20; 
                  do
                    for ARG5 in 3; 
                    do
                      for ARG6 in 2 3; 
                      do
                        INSTANCE_NAME="${ARG1}_${ARG2}_${ARG3}_${ARG4}_${ARG5}_${ARG6}_CRP"
                        echo "Submitting job for instance ${INSTANCE_NAME} with: SOLVER=${SOLVER}, NR_SCENARIO=${NR_SCENARIO}, PHA_OBJ=${PHA_OBJ}, PHAPenalty=${PHAPenalty}, ALNSRL=${ALNSRL}, ALNSRL_DEEPQ=${ALNSRL_DEEPQ}"
                        
                        for SCENARIO_GEN in "${SCENARIO_GENERATION[@]}"; do
                          for CLUSTER_METHOD in "${CLUSTER_METHODS[@]}"; do
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