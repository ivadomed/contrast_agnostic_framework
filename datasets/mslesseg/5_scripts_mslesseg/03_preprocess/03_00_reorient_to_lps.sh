#!/usr/bin/env bash
# Reorient MSLesSeg images + masks to LPS (match chaos/sliver07/open-ms voxel convention).
# Lossless axis flip/permute — leaves 0_raw_mslesseg pristine. See the .py for details.
#   bash 03_00_reorient_to_lps.sh             # fix BIDS + nnUNet trees
#   bash 03_00_reorient_to_lps.sh --dry-run   # report only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name mslesseg_reorient_lps --gpus 0 --slot 0 --time 00:20:00 --mem 8G --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/03_00_reorient_to_lps.py" "$@"
