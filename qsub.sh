#!/bin/bash
#SBATCH --account=def-sehasd
#SBATCH --time=14:00:00       # Specify time directly
#SBATCH --mem=20GB            # Specify memory directly
#SBATCH --output=/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs/%x-%j.out
#SBATCH --error=/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs/%x-%j.err

# Load necessary modules
module load StdEnv/2023
module load gurobi/11.0.1

# Activate the virtual environment
source /home/pfarghad/Myschedulingmodel_3/myenv/bin/activate

# Log all parameters passed to the job
LOGFILE="/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs/job_parameters.log"
echo "Running instance: $1" >> $LOGFILE
echo "Model: $2" >> $LOGFILE
echo "Solver: $3" >> $LOGFILE
echo "NrScenario: $4" >> $LOGFILE
echo "PHAObj: $5" >> $LOGFILE
echo "PHAPenalty: $6" >> $LOGFILE
echo "ALNSRL: $7" >> $LOGFILE
echo "ALNSRL_DeepQ: $8" >> $LOGFILE
echo "BBC Setting: $9" >> $LOGFILE
echo "Scenario Generation: ${10}" >> $LOGFILE
echo "Clustering Method: ${11}" >> $LOGFILE
echo "---------------------" >> $LOGFILE

# Define the command using the passed parameters
COMMAND="python main.py \
--Instance $1 \
--Action Solve \
--Model $2 \
--Solver $3 \
--NrScenario $4 \
--PHAObj $5 \
--PHAPenalty $6 \
--ALNSRL $7 \
--ALNSRL_DeepQ $8 \
-c $9 \
--ScenarioGeneration ${10} \
--ClusteringMethod ${11}"

# Print the command to the log for reference
echo $COMMAND

# Execute the command
$COMMAND
