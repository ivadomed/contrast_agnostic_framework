#!/usr/bin/env bash
# Register T1w SynthSeg masks into the native space of all other modalities.
#
# Modalities: T2w, DWI (dir-AP), EPI (dir-AP), bold (task-rest, all acq variants),
#             GRE (echo-1 magnitude).
#
# Launches 5 modalities × 4 CPU-only workers = 20 parallel Slurm jobs.
# Each worker handles its share of sessions sequentially (~40 sessions × ~1 min/reg).
# Expected wall time per job: ~1-2 h (bold worst-case with mb4+mb6 per session).
#
# Outputs land in BIDS-compliant paths under:
#   ${BIDS_ROOT}/derivatives/synthseg_masks/<sub>/<ses>/<subdir>/<name>_synthseg.nii.gz
#
# Logs: ${RESULTS_DIR}/logs/register_synthseg/<job>.log   (persistent on $PROJECT)
#
# Usage:
#   bash 02_04_register_synthseg_masks.sh
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER="${SCRIPT_DIR}/02_04_register_synthseg_worker.py"
LOG_DIR="${RESULTS_DIR}/logs/register_synthseg"
N_WORKERS=4

mkdir -p "${LOG_DIR}"

MODALITIES=(T2w dwi epi bold GRE)

echo "[$(date '+%H:%M:%S')] Launching ${#MODALITIES[@]} modalities × ${N_WORKERS} workers ($(( ${#MODALITIES[@]} * N_WORKERS )) Slurm jobs)"
echo "  BIDS root : ${BIDS_ROOT}"
echo "  Logs      : ${LOG_DIR}"
echo ""

PIDS=()
for MOD in "${MODALITIES[@]}"; do
    for R in $(seq 0 $(( N_WORKERS - 1 ))); do
        JOB_NAME="reg_synthseg_${MOD}_r${R}"
        run_job \
            --name "${JOB_NAME}" \
            --gpus 0 \
            --slot "${R}" \
            --time "03:00:00" \
            --log  "${LOG_DIR}/${JOB_NAME}.log" \
            --wait -- bash -c "
module load ants/2.6.5
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
cd '${PROJECT_ROOT}'
.venv/bin/python '${WORKER}' \
    --modality '${MOD}' \
    --rank '${R}' \
    --world-size '${N_WORKERS}' \
    --bids-root '${BIDS_ROOT}'
" &
        PIDS+=($!)
    done
done

echo "[$(date '+%H:%M:%S')] All ${#PIDS[@]} jobs submitted. Waiting for completion..."

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || FAILED=$(( FAILED + 1 ))
done

echo ""
echo "[$(date '+%H:%M:%S')] Finished. ${#PIDS[@]} jobs total, ${FAILED} failed."
echo "  Logs: ${LOG_DIR}"

[ "${FAILED}" -eq 0 ] || {
    echo "  Check failed job logs above for details."
    exit 1
}
