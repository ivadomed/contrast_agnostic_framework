#!/usr/bin/env bash
# Build the picai-prostate patient-level partition + 4-fold CV splits (folds 0/1/2 are the
# ones ever trained — see CLAUDE.md FOLD POLICY). Tiny CPU job (reads cases.json only), so
# it runs inline on the login node rather than through run_job.
#
# Run AFTER 00_utils/00_01_bidsify.sh (cases.json must exist).
# Usage: bash 01_01_create_splits.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

exec .venv/bin/python "${SCRIPT_DIR}/01_01_create_splits.py" "$@"
