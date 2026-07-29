#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name ms3seg_reorient_lps --gpus 0 --slot 0 --time 00:20:00 --mem 8G --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_reorient_to_lps.py" "$@"
