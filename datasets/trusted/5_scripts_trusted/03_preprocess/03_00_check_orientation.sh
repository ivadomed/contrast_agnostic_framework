#!/usr/bin/env bash
# Verify TRUSTED images + masks are LPS (match CHAOS / SLIVER07 voxel convention).
# TRUSTED already ships LPS, so this is an idempotent check (all files report 'ok');
# if any ever differ it reorients losslessly. Leaves 0_raw_trusted pristine. See the .py.
#   bash 03_00_check_orientation.sh             # check BIDS + nnUNet trees
#   bash 03_00_check_orientation.sh --dry-run   # report only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name trusted_check_orientation --gpus 0 --cpus 4 --mem 16G --time 00:30:00 \
    --log /tmp/trusted_check_orientation.log --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_check_orientation.py" "$@"
