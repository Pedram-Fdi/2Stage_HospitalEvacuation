#!/bin/bash
#SBATCH --account=def-sehasd
#SBATCH --time=10:00:00
#SBATCH --mem=25G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs/%x-%j.out
#SBATCH --error=/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs/%x-%j.err

set -euo pipefail

# Clean environment (StdEnv is sticky, so force purge)
module --force purge
module load StdEnv/2023
module load python/3.11.5
module load gurobi/12.0.0

# Activate the virtual environment (correct path)
source /home/pfarghad/Myschedulingmodel_3/RL/myenv/bin/activate

# (Optional) keep gurobi threads aligned with SLURM allocation
export GRB_THREADS=${SLURM_CPUS_PER_TASK}

INSTANCE="$1"
MODEL="$2"
SOLVER="$3"
NR_SCENARIO="$4"
PHA_OBJ="$5"
PHA_PENALTY="$6"
ALNSRL="$7"
ALNSRL_DEEPQ="$8"
BBC_SETTING="$9"
SCENARIO_GEN="${10}"
CLUSTER_METHOD="${11}"

OUT_DIR="/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs"
mkdir -p "$OUT_DIR"

# Human-readable parameter log (legacy)
LOGFILE="${OUT_DIR}/job_parameters.log"
{
  echo "JobID: ${SLURM_JOB_ID:-NA}"
  echo "Running instance: $INSTANCE"
  echo "Model: $MODEL"
  echo "Solver: $SOLVER"
  echo "NrScenario: $NR_SCENARIO"
  echo "PHAObj: $PHA_OBJ"
  echo "PHAPenalty: $PHA_PENALTY"
  echo "ALNSRL: $ALNSRL"
  echo "ALNSRL_DeepQ: $ALNSRL_DEEPQ"
  echo "BBC Setting: $BBC_SETTING"
  echo "Scenario Generation: $SCENARIO_GEN"
  echo "Clustering Method: $CLUSTER_METHOD"
  echo "---------------------"
} >> "$LOGFILE"

# Structured resource usage CSV (one row per completed job)
RESOURCE_CSV="${OUT_DIR}/resource_usage.csv"
RESOURCE_HEADER="JobID,Instance,Model,Solver,NrScenario,PHAObj,PHAPenalty,ALNSRL,ALNSRL_DeepQ,BBC_Setting,ScenarioGeneration,ClusteringMethod,ExitCode,Elapsed_sec,MaxRSS_KB,MaxRSS_MB,ReqMem,Timelimit,AllocCPUS,SacctState,SacctElapsed,SacctMaxRSS"

append_resource_row() {
  local row="$1"
  (
    flock 9
    if [[ ! -f "$RESOURCE_CSV" ]]; then
      echo "$RESOURCE_HEADER" > "$RESOURCE_CSV"
    fi
    echo "$row" >> "$RESOURCE_CSV"
  ) 9>>"${RESOURCE_CSV}.lock"
}

# Quick diagnostics (helps debug future issues)
which python
python --version
python -c "import sys; print('python exe:', sys.executable)"
gurobi_cl --version

# Measure wall time + peak RSS of the python process.
# Prefer GNU time (available after StdEnv); fall back to a Python wrapper.
TIMEFILE=$(mktemp "${OUT_DIR}/time_${SLURM_JOB_ID:-local}.XXXXXX")
EXIT_CODE=0
START_EPOCH=$(date +%s)
ELAPSED_SEC=""
MAXRSS_KB=""

MAIN_ARGS=(
  main.py
  --Instance "$INSTANCE"
  --Action Solve
  --Model "$MODEL"
  --Solver "$SOLVER"
  --NrScenario "$NR_SCENARIO"
  --PHAObj "$PHA_OBJ"
  --PHAPenalty "$PHA_PENALTY"
  --ALNSRL "$ALNSRL"
  --ALNSRL_DeepQ "$ALNSRL_DEEPQ"
  -c "$BBC_SETTING"
  --ScenarioGeneration "$SCENARIO_GEN"
  --ClusteringMethod "$CLUSTER_METHOD"
)

GNU_TIME=""
for candidate in /usr/bin/time /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin/time; do
  if [[ -x "$candidate" ]] && "$candidate" -f '%e %M' true >/dev/null 2>&1; then
    GNU_TIME="$candidate"
    break
  fi
done

if [[ -n "$GNU_TIME" ]]; then
  # GNU time: %e = elapsed seconds, %M = max RSS in KB
  set +e
  "$GNU_TIME" -f '%e %M' -o "$TIMEFILE" -- python "${MAIN_ARGS[@]}"
  EXIT_CODE=$?
  set -e
  if [[ -s "$TIMEFILE" ]]; then
    read -r ELAPSED_SEC MAXRSS_KB < "$TIMEFILE" || true
  fi
else
  # Fallback: wrap the solve so RUSAGE_CHILDREN reports MaxRSS (KB on Linux)
  set +e
  python - "$TIMEFILE" "${MAIN_ARGS[@]}" <<'PY'
import resource
import subprocess
import sys

timefile = sys.argv[1]
cmd = ["python", *sys.argv[2:]]
proc = subprocess.run(cmd)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
# Linux: ru_maxrss is kilobytes
with open(timefile, "w", encoding="utf-8") as f:
    f.write(f"{ru.ru_maxrss}\n")
sys.exit(proc.returncode)
PY
  EXIT_CODE=$?
  set -e
  if [[ -s "$TIMEFILE" ]]; then
    MAXRSS_KB=$(tr -d '[:space:]' < "$TIMEFILE")
  fi
fi

rm -f "$TIMEFILE"

# If GNU time path did not set elapsed, fall back to wall clock
if [[ -z "${ELAPSED_SEC}" ]]; then
  END_EPOCH=$(date +%s)
  ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
fi

MAXRSS_MB=""
if [[ -n "${MAXRSS_KB}" ]]; then
  MAXRSS_MB=$(python -c "print(round(float('${MAXRSS_KB}') / 1024.0, 2))")
fi

REQ_MEM="${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-NA}}"
TIME_LIMIT="${SBATCH_TIMELIMIT:-NA}"
# Prefer SLURM env when available
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Timelimit / ReqMem from sacct may still be incomplete at job end; try anyway.
  # MaxRSS is usually on the .batch step; State/Elapsed on the parent job.
  SACCT_OUT=$(sacct -j "${SLURM_JOB_ID}" --parsable2 --noheader \
    --format=JobID,State,Elapsed,MaxRSS,Timelimit,ReqMem,AllocCPUS 2>/dev/null || true)
  SACCT_PARENT=$(echo "$SACCT_OUT" | awk -F'|' '$1 !~ /\./ {print; exit}')
  SACCT_BATCH=$(echo "$SACCT_OUT" | awk -F'|' '$1 ~ /\.batch$/ {print; exit}')
  SACCT_STATE=$(echo "${SACCT_PARENT}" | cut -d'|' -f2)
  SACCT_ELAPSED=$(echo "${SACCT_PARENT}" | cut -d'|' -f3)
  SACCT_MAXRSS=$(echo "${SACCT_BATCH:-${SACCT_PARENT}}" | cut -d'|' -f4)
  SACCT_TIMELIMIT=$(echo "${SACCT_PARENT}" | cut -d'|' -f5)
  SACCT_REQMEM=$(echo "${SACCT_PARENT}" | cut -d'|' -f6)
  SACCT_CPUS=$(echo "${SACCT_PARENT}" | cut -d'|' -f7)
  [[ -n "${SACCT_TIMELIMIT}" ]] && TIME_LIMIT="${SACCT_TIMELIMIT}"
  [[ -n "${SACCT_REQMEM}" ]] && REQ_MEM="${SACCT_REQMEM}"
  ALLOC_CPUS="${SACCT_CPUS:-${SLURM_CPUS_PER_TASK:-NA}}"
else
  SACCT_STATE="NA"
  SACCT_ELAPSED="NA"
  SACCT_MAXRSS="NA"
  ALLOC_CPUS="${SLURM_CPUS_PER_TASK:-NA}"
fi

# Sacct MaxRSS is often blank until accounting flushes after exit.
# Elapsed_sec / MaxRSS_* come from in-job measurement (authoritative for sizing).
append_resource_row "$(
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${SLURM_JOB_ID:-NA}" \
    "$INSTANCE" \
    "$MODEL" \
    "$SOLVER" \
    "$NR_SCENARIO" \
    "$PHA_OBJ" \
    "$PHA_PENALTY" \
    "$ALNSRL" \
    "$ALNSRL_DEEPQ" \
    "$BBC_SETTING" \
    "$SCENARIO_GEN" \
    "$CLUSTER_METHOD" \
    "$EXIT_CODE" \
    "$ELAPSED_SEC" \
    "${MAXRSS_KB:-}" \
    "${MAXRSS_MB:-}" \
    "$REQ_MEM" \
    "$TIME_LIMIT" \
    "$ALLOC_CPUS" \
    "${SACCT_STATE:-}" \
    "${SACCT_ELAPSED:-}" \
    "${SACCT_MAXRSS:-}"
)"

echo "Resource usage logged to ${RESOURCE_CSV}"
echo "ExitCode=${EXIT_CODE} Elapsed_sec=${ELAPSED_SEC} MaxRSS_KB=${MAXRSS_KB:-NA} MaxRSS_MB=${MAXRSS_MB:-NA}"

exit "${EXIT_CODE}"
