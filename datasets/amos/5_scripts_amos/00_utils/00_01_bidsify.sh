#!/usr/bin/env bash
# BIDSify AMOS22: reorganise 0_raw_amos/amos22/ into 1_BIDS_amos/amos-abdominal/.
# Run AFTER 00_00_download_and_extract.py (raw data must already be present).
#
# Usage:
#   bash 00_01_bidsify.sh           # BIDSify all labeled cases
#   bash 00_01_bidsify.sh --skip-bids
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name amos_bidsify --gpus 0 --slot 0 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_01_bidsify.py" "$@"
