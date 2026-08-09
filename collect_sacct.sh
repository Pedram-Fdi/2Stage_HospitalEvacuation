#!/bin/bash
# Refresh SacctElapsed / SacctMaxRSS in resource_usage.csv for completed jobs.
# Sacct accounting is often incomplete at job exit; run this after jobs finish:
#   ./collect_sacct.sh
#   ./collect_sacct.sh JobOutputs/resource_usage.csv

set -euo pipefail

OUT_DIR="/home/pfarghad/Myschedulingmodel_3/RL/JobOutputs"
CSV="${1:-${OUT_DIR}/resource_usage.csv}"
SUBMITTED="${OUT_DIR}/submitted_jobs.csv"
TMP=$(mktemp)

if [[ ! -f "$CSV" ]]; then
  echo "No resource CSV found at: $CSV"
  echo "Jobs may still be running, or none have finished yet."
  if [[ -f "$SUBMITTED" ]]; then
    echo "Found submitted_jobs.csv — querying sacct for those JobIDs..."
    # Build a temporary view from submitted jobs + sacct
    echo "JobID,Instance,Model,Solver,NrScenario,PHAObj,PHAPenalty,ALNSRL,ALNSRL_DeepQ,BBC_Setting,ScenarioGeneration,ClusteringMethod,SacctState,SacctElapsed,SacctMaxRSS,ReqMem,Timelimit,AllocCPUS" > "$TMP"
    tail -n +2 "$SUBMITTED" | while IFS=',' read -r jobid instance model solver nrscen phaobj phapen alnsrl deepq bbc scengen cluster rest; do
      [[ -z "$jobid" || "$jobid" == "NA" ]] && continue
      line=$(sacct -j "$jobid" --parsable2 --noheader \
        --format=JobID,State,Elapsed,MaxRSS,ReqMem,Timelimit,AllocCPUS 2>/dev/null \
        | awk -F'|' '$1 !~ /\./ {print; exit}')
      state=$(echo "$line" | cut -d'|' -f2)
      elapsed=$(echo "$line" | cut -d'|' -f3)
      maxrss=$(echo "$line" | cut -d'|' -f4)
      reqmem=$(echo "$line" | cut -d'|' -f5)
      tlimit=$(echo "$line" | cut -d'|' -f6)
      cpus=$(echo "$line" | cut -d'|' -f7)
      echo "${jobid},${instance},${model},${solver},${nrscen},${phaobj},${phapen},${alnsrl},${deepq},${bbc},${scengen},${cluster},${state},${elapsed},${maxrss},${reqmem},${tlimit},${cpus}"
    done >> "$TMP"
    OUT_REFRESH="${OUT_DIR}/sacct_usage.csv"
    mv "$TMP" "$OUT_REFRESH"
    echo "Wrote ${OUT_REFRESH}"
    exit 0
  fi
  exit 1
fi

# Update Sacct* columns in existing resource_usage.csv
python - "$CSV" <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
rows = []
with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    for row in reader:
        jobid = (row.get("JobID") or "").strip()
        if jobid and jobid != "NA":
            try:
                out = subprocess.check_output(
                    [
                        "sacct", "-j", jobid, "--parsable2", "--noheader",
                        "--format=JobID,State,Elapsed,MaxRSS,ReqMem,Timelimit,AllocCPUS",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                out = ""
            # Prefer the parent job for State/Elapsed; .batch for MaxRSS
            parent = None
            batch = None
            for line in out.splitlines():
                parts = line.split("|")
                if not parts:
                    continue
                jid = parts[0]
                if "." not in jid:
                    parent = parts
                elif jid.endswith(".batch"):
                    batch = parts
            picked = parent or batch
            if picked and len(picked) >= 4:
                row["SacctState"] = picked[1]
                row["SacctElapsed"] = picked[2]
                mem_src = batch or parent
                if mem_src and len(mem_src) >= 4:
                    row["SacctMaxRSS"] = mem_src[3]
                if len(picked) > 4 and picked[4]:
                    row["ReqMem"] = picked[4]
                if len(picked) > 5 and picked[5]:
                    row["Timelimit"] = picked[5]
                if len(picked) > 6 and picked[6]:
                    row["AllocCPUS"] = picked[6]
        rows.append(row)

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated sacct columns for {len(rows)} rows in {csv_path}")
PY

rm -f "$TMP"
