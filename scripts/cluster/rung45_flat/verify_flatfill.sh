#!/usr/bin/env bash
# Rung 4.5 (flat fill) pre-launch verification -- see verify_flatfill.py. CPU job through run_job.
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
OUT=${OUT:-$SCRATCH/rung45_flat/verify}
mkdir -p "$OUT"
source "$REPO/scripts/job_runner/run_job.sh"
run_job --name r45_verify --gpus 0 --cpus 8 --mem 32G --time 01:00:00 --log "$OUT/verify.log" ${WAIT:+--wait} -- \
  bash -lc "module load python/3.11 >/dev/null 2>&1; cd $REPO && .venv/bin/python scripts/cluster/rung45_flat/verify_flatfill.py $OUT"
