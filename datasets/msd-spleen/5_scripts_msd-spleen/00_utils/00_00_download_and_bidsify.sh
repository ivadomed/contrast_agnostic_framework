#!/usr/bin/env bash
# Download MSD Task09_Spleen (labeled half) from the msd-for-monai S3 mirror and
# BIDSify the NIfTI volumes. MSD-SPLEEN is evaluation-only (see datasets/msd-spleen/README.md).
#   bash 00_00_download_and_bidsify.sh                  # download + BIDSify
#   bash 00_00_download_and_bidsify.sh --skip-download  # 0_raw already populated
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name msd_spleen_download_bidsify --gpus 0 --slot 0 --mem 8G --time 00:30:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download_and_bidsify.py" "$@"
