#!/usr/bin/env bash
# Raw audit of the TotalSegmentator MRI download (CPU job). Usage: bash 02_00_raw_audit.sh   (set DL=<dir with extracted/>)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
DL="${DL:-${SCRATCH}/totalseg_mri_download}"
run_job --name tsmri_raw_audit --gpus 0 --cpus 4 --mem 16G --time 01:00:00 --log "${DL}/raw_audit.log" --wait -- \
  "${PROJECT_ROOT}/.venv/bin/python" -I "$(dirname "$0")/02_00_raw_audit.py" "${DL}/extracted" "${DL}/raw_audit.tsv"
