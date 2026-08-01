#!/usr/bin/env bash
# Materialise the nnUNet test inputs for KIDNEY-T2W (single t2 item).
#   bash 05_00_build_test_inputs.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "${SCRIPT_DIR}/05_00_build_test_inputs.py"
