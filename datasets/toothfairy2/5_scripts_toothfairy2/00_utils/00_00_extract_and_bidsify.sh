#!/usr/bin/env bash
# Convert the ToothFairy2 release zip to BIDS NIfTI (see the .py for the label
# reduction, the 0.3->0.6 mm resample rationale, the per-case orientation audit, and
# why the archive is NEVER fully unzipped). CPU-only, dispatched through run_job.
#
#   bash 00_00_extract_and_bidsify.sh                # all 480 cases
#   TF2_LIMIT=8 bash 00_00_extract_and_bidsify.sh    # smoke run on 8 cases
#
# Everything reads the zip in place and stages single cases through $SLURM_TMPDIR,
# so this needs no 109 GB scratch staging area. TF2_ZIP defaults to the TamIA
# download location (CLAUDE.md: "Slurm jobs go to TamIA, not Vulcan").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
cd "${PROJECT_ROOT}"

TF2_STAGE="${TF2_STAGE:-${SCRATCH:-/scratch/p/paulh}/toothfairy2}"
export TF2_ZIP="${TF2_ZIP:-${TF2_STAGE}/0_raw/ToothFairy2.zip}"
[ -f "${TF2_ZIP}" ] || { echo "release zip not found: ${TF2_ZIP}" >&2; exit 1; }
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}" "${BIDS_ROOT}"

# 16 workers x ~1 case in flight each; --mem sized for 0.3mm volumes held in RAM
# during resampling (a 0.3mm CBCT is ~800^3 float32 ~= 2 GB, and B-spline resampling
# holds source + destination), so ~6 GB/worker with headroom.
run_job --name "tf2_bidsify" --gpus 0 --cpus 16 --mem 110G --time 08:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/bidsify_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export TF2_ZIP='${TF2_ZIP}' BIDS_ROOT='${BIDS_ROOT}'
        export TF2_WORKERS='${TF2_WORKERS:-16}' TF2_TARGET_SPACING='${TF2_TARGET_SPACING:-0.6}'
        export TF2_LIMIT='${TF2_LIMIT:-0}'
        cd '${PROJECT_ROOT}'
        .venv/bin/python '${HERE}/00_00_extract_and_bidsify.py'
    "
