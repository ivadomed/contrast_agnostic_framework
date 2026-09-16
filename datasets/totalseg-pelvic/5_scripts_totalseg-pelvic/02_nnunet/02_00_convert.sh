#!/usr/bin/env bash
# Convert both TotalSegmentator archives (CT + MRI) into this project's nnU-Net raw
# layout, restricted to the pelvic/hip label subset. Run once after
# 01_create_splits/01_01_create_splits.py. Not heavy enough to need run_job (extraction +
# label-merge over ~500 cases total) but keep this off a login node if the full
# un-subsampled CT pool is ever used — submit via run_job/sbatch in that case instead.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python datasets/totalseg-pelvic/5_scripts_totalseg-pelvic/02_nnunet/02_00_convert.py
