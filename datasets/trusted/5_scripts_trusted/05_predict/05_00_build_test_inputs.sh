#!/usr/bin/env bash
# Materialise the nnUNet test inputs for TRUSTED (CT native hardlink; US resampled to
# the model's ~1.5 mm resolution — see 05_00_build_test_inputs.py for the rationale).
# US resampling reads/writes ~570 M-voxel volumes → run on a compute node.
#   bash 05_00_build_test_inputs.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
# US resample is parallelised across the allotted CPUs (~4 min wall for 59 volumes).
# Kept tiny + SHORT so the backfill scheduler can slip it into a gap ahead of the
# higher-priority reservations rather than waiting for the next big window.
run_job --name trusted_build_test_inputs --gpus 0 --cpus 4 --mem 32G --time 00:20:00 \
    --log /tmp/trusted_build_test_inputs.log --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/05_00_build_test_inputs.py"
