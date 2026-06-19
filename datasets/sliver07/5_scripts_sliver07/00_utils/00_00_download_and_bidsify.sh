#!/usr/bin/env bash
# Download SLIVER07 (labeled half) from Zenodo and BIDSify the .mhd/.raw volumes.
# SLIVER07 is evaluation-only (see datasets/sliver07/README.md).
#   bash 00_00_download_and_bidsify.sh                  # download + BIDSify
#   bash 00_00_download_and_bidsify.sh --skip-download  # 0_raw already populated
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name sliver07_download_bidsify --gpus 0 --slot 0 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download_and_bidsify.py" "$@"
