#!/usr/bin/env bash
# BIDSify CirrMRI600+ (0_raw already downloaded+extracted by 00_00_download.sh,
# which must run on the LOGIN NODE, not via run_job -- compute nodes' outbound
# proxy returns 403 for osf.io, unlike the S3-hosted MSD-Spleen download).
#   bash 00_01_bidsify.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name cirrmri_liver_bidsify --gpus 0 --slot 0 --mem 8G --time 01:00:00 \
    --log "${PROJECT_ROOT}/datasets/cirrmri-liver/0_raw_cirrmri-liver/_bidsify.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py"
