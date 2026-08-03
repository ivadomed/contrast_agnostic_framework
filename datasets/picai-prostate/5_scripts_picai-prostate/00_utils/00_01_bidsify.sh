#!/usr/bin/env bash
# BIDSify PI-CAI: 0_raw (MHA, three native grids) → 1_BIDS (NIfTI, one common
# prostate-centred grid per study). See 00_01_bidsify.py for the geometry and for why only
# the csPCa-positive studies are kept.
#
# CPU-only (SimpleITK resampling of ~1500 studies x 4 volumes), dispatched through run_job.
#
# Usage: bash 00_01_bidsify.sh [--limit N] [--workers N]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

LOGDIR="${RESULTS_DIR:-${PROJECT_ROOT}/datasets/picai-prostate/8_results_picai-prostate}/_logs"
mkdir -p "${LOGDIR}"

run_job --name picai_bidsify --gpus 0 --cpus 32 --mem 64G --time 03:00:00 \
    --log "${LOGDIR}/picai_bidsify.log" --wait -- bash -c "
    export RAW_ROOT='${RAW_ROOT}'
    export BIDS_ROOT='${BIDS_ROOT}'
    export BIDSIFY_WORKERS='${BIDSIFY_WORKERS:-32}'
    cd '${PROJECT_ROOT}'
    .venv/bin/python '${SCRIPT_DIR}/00_01_bidsify.py' $*
"
