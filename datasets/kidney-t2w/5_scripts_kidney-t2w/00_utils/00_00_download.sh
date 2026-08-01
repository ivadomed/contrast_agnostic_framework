#!/usr/bin/env bash
# Download T2-weighted Kidney MRI Segmentation (Zenodo 5153568) and extract.
#   bash 00_00_download.sh                  # download + extract
#   bash 00_00_download.sh --skip-download  # 0_raw already populated
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name kidney_t2w_download --gpus 0 --slot 0 --mem 8G --time 00:30:00 \
    --log "${PROJECT_ROOT}/datasets/kidney-t2w/0_raw_kidney-t2w/_download.log" --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download.py" "$@"
