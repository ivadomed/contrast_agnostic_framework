#!/usr/bin/env bash
# Idempotent LPS orientation check for CIRRMRI-LIVER (expected to pass cleanly --
# confirmed already-LPS at BIDSify time). Kept as a standing pipeline step per
# the standardization checklist.
#   bash 03_00_check_orientation.sh          # dry-run report (default)
#   bash 03_00_check_orientation.sh --fix    # only if something regresses
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name cirrmri_liver_orient_check --gpus 0 --slot 0 --mem 8G --time 00:20:00 --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_check_orientation.py" "$@"
