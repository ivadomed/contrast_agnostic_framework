#!/bin/bash
# Queue one run_setting.sh controller (CPU job) per key. Usage: bash scripts/cluster/srcsm_srcmatch/launch.sh <key>...
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/job_runner/run_job.sh
mkdir -p "${SCRATCH}/srcsm_match/_status"
for k in "$@"; do
  run_job --name "srcmatch_${k}" --gpus 0 --cpus 8 --mem 64G --time 12:00:00 \
    --log "${SCRATCH}/srcsm_match/_status/${k}.job.log" -- bash scripts/cluster/srcsm_srcmatch/run_setting.sh "${k}"
  sleep 2
done
