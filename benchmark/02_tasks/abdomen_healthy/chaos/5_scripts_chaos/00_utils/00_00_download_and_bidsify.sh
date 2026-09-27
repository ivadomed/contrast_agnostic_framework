#!/usr/bin/env bash
# Download CHAOS, copy raw (train only), and DICOM→NIfTI BIDSify.
# Usage: bash 00_00_download_and_bidsify.sh [--skip-copy] [--skip-bids]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"

run_job --name chaos_download_bidsify --gpus 0 --slot 0 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download_and_bidsify.py" "$@"
