#!/usr/bin/env bash
# Build CT⊕US chimera test inputs (context-injection experiment). Pastes each US
# kidney into the patient's full-torso CT at the CT-kidney centroid. See the .py for
# the full method. Heavy (loads CT + resamples ~570 M-voxel US per side) → compute node.
#   bash 05_20_build_chimera_inputs.sh            # all patients with ≥1 US side
#   bash 05_20_build_chimera_inputs.sh 263 220    # only these patients (quick QA)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
run_job --name trusted_build_chimera --gpus 0 --cpus 8 --mem 64G --time 00:40:00 \
    --log /tmp/trusted_build_chimera.log --wait -- \
    .venv/bin/python "${SCRIPT_DIR}/05_20_build_chimera_inputs.py" "$@"
