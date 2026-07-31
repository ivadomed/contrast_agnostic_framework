#!/usr/bin/env bash
# Download CirrMRI600+ (T1_3D + T2_3D) from OSF and extract.
# CIRRMRI-LIVER is evaluation-only (see datasets/cirrmri-liver/README.md).
#   bash 00_00_download.sh                  # download + extract
#   bash 00_00_download.sh --skip-download  # 0_raw already populated
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name cirrmri_liver_download --gpus 0 --slot 0 --mem 8G --time 01:00:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download.py" "$@"
