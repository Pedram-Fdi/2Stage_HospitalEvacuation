#!/bin/bash

DIR="/home/pfarghad/Myschedulingmodel_3/RL/Instances/"
FAILED_JOBS="failed_jobs.txt"
OUT_DIR="/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs"
SUBMITTED_CSV="${OUT_DIR}/submitted_jobs.csv"

mkdir -p "$OUT_DIR"

# Clear previous failed jobs log
> $FAILED_JOBS

# Fixed parameter
MODEL="2Stage"    # Options: "Average" or "2Stage"
BBC_SETTING="NS"  # Use the appropriate value

# Independent arrays:
NR_SCENARIOS=("50")
SCENARIO_GENERATION=("RQMC")          # Options: "MC", "RQMC", "QMC"
CLUSTER_METHODS=("DB")               # Options: "NoC", "KM", "KMPP", "SOM", "DB"

# Track every submitted JobID with the same config fields used for sizing visibility
if [[ ! -f "$SUBMITTED_CSV" ]]; then
  echo "JobID,Instance,Model,Solver,NrScenario,PHAObj,PHAPenalty,ALNSRL,ALNSRL_DeepQ,BBC_Setting,ScenarioGeneration,ClusteringMethod,SubmitTime" > "$SUBMITTED_CSV"
fi

submit_job() {
  local instance="$1"
  local model="$2"
  local solver="$3"
  local nr_scenario="$4"
  local pha_obj="$5"
  local pha_penalty="$6"
  local alnsrl="$7"
  local alnsrl_deepq="$8"
  local bbc="$9"
  local scenario_gen="${10}"
  local cluster_method="${11}"

  local sbatch_out job_id
  sbatch_out=$(sbatch ./qsub.sh \
    "${instance}" \
    "${model}" \
    "${solver}" \
    "${nr_scenario}" \
    "${pha_obj}" \
    "${pha_penalty}" \
    "${alnsrl}" \
    "${alnsrl_deepq}" \
    "${bbc}" \
    "${scenario_gen}" \
    "${cluster_method}")
  local rc=$?

  if [[ $rc -ne 0 ]]; then
    echo "${instance}" >> "$FAILED_JOBS"
    return "$rc"
  fi

  # sbatch output: "Submitted batch job 123456"
  job_id=$(echo "$sbatch_out" | awk '{print $NF}')
  echo "${job_id},${instance},${model},${solver},${nr_scenario},${pha_obj},${pha_penalty},${alnsrl},${alnsrl_deepq},${bbc},${scenario_gen},${cluster_method},$(date -Iseconds)" >> "$SUBMITTED_CSV"
  echo "Submitted JobID=${job_id} for ${instance}"
  return 0
}

# Array for SOLVERS to loop over:
SOLVERS=("ALNS")            # Options: "MIP", "PHA", "ALNS"

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
    ALNSRLs=(1)
    # For ALNS, ALNSRL_DEEPQs will be set conditionally below
  elif [ "$SOLVER" = "PHA" ]; then
    ALNSRLs=(0)
    ALNSRL_DEEPQs=(0)
    PHA_OBJS=("Q")
    PHAPenalties=("S")
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
              ALNSRL_DEEPQs=(1)
            fi
          fi

          for ALNSRL_DEEPQ in "${ALNSRL_DEEPQs[@]}"; do
            # Loop over instance parameters:
            for ARG1 in 4; 
            do
              for ARG2 in 15; 
              do
                for ARG3 in 5; 
                do
                  for ARG4 in 15; 
                  do
                    for ARG5 in 3; 
                    do
                      for ARG6 in {1..7}; 
                      do
                        INSTANCE_NAME="${ARG1}_${ARG2}_${ARG3}_${ARG4}_${ARG5}_${ARG6}_CRP"
                        echo "Submitting job for instance ${INSTANCE_NAME} with: SOLVER=${SOLVER}, NR_SCENARIO=${NR_SCENARIO}, PHA_OBJ=${PHA_OBJ}, PHAPenalty=${PHAPenalty}, ALNSRL=${ALNSRL}, ALNSRL_DEEPQ=${ALNSRL_DEEPQ}"
                        
                        for SCENARIO_GEN in "${SCENARIO_GENERATION[@]}"; do
                          for CLUSTER_METHOD in "${CLUSTER_METHODS[@]}"; do
                            submit_job \
                              "${INSTANCE_NAME}" \
                              "${MODEL}" \
                              "${SOLVER}" \
                              "${NR_SCENARIO}" \
                              "${PHA_OBJ}" \
                              "${PHAPenalty}" \
                              "${ALNSRL}" \
                              "${ALNSRL_DEEPQ}" \
                              "${BBC_SETTING}" \
                              "${SCENARIO_GEN}" \
                              "${CLUSTER_METHOD}"
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
        submit_job \
          "${INSTANCE_NAME}" \
          "${MODEL}" \
          "${SOLVER}" \
          "${NR_SCENARIO}" \
          "${PHA_OBJ}" \
          "${PHAPenalty}" \
          "${ALNSRL}" \
          "${ALNSRL_DEEPQ}" \
          "${BBC_SETTING}" \
          "${SCENARIO_GENERATION[0]}" \
          "${CLUSTER_METHODS[0]}"
        if [ $? -ne 0 ]; then
            echo "Failed again: ${INSTANCE_NAME}"
        fi
    done < $FAILED_JOBS
fi

echo "Submission log: ${SUBMITTED_CSV}"
echo "After jobs finish, refresh SLURM MaxRSS/Elapsed with: ./collect_sacct.sh"