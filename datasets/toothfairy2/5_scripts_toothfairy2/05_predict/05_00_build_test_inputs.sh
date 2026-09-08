#!/usr/bin/env bash
# Build the held-out CBCT test-input dirs nnU-Net predicts on. Cheap (hard links).
#   bash 05_00_build_test_inputs.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "${HERE}/05_00_build_test_inputs.py"
