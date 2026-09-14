#!/usr/bin/env bash
# Split the AutoPET archive into the two per-modality nnU-Net raw datasets (CT, PET).
# CPU/IO-bound (zip extraction), not GPU work — fine on the login node per CLAUDE.md's
# "compilation/checks under ~10 CPU-min" login-node exception IF it finishes quickly;
# if the extraction of ~1038 cases x 2 files runs long, wrap it in run_job instead of
# assuming — measure first.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python "$(dirname "$0")/02_00_convert.py"
