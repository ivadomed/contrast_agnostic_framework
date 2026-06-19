#!/usr/bin/env bash
# Download AMOS22 labeled data (500CT+100MRI, 24 GB) from Zenodo and extract.
# Unlabeled splits are NOT downloaded. See 00_00_download_and_extract.py for details.
#   bash 00_00_download_and_extract.sh
#   bash 00_00_download_and_extract.sh --skip-download  # 0_raw already populated
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${PROJECT_ROOT}"
run_job --name amos_download_extract --gpus 0 --slot 0 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/00_00_download_and_extract.py" "$@"
