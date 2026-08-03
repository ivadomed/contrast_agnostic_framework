#!/usr/bin/env bash
# TAMIA SIZING PROBE — measure real per-fold H100 VRAM + epoch time BEFORE committing to a
# pack layout. CLAUDE.md requires this ("size actual per-fold GPU-memory and utilization on
# one real H100 job first ... rather than assuming Vulcan's numbers transfer").
#
# Runs a handful of epochs of the two extreme methods side by side on one node:
#   GPU0 = OURS / DualVal (the heaviest fast method: AugLab GPU synthesis + a SECOND
#          synth-only validation pass every epoch)
#   GPU1 = srcsm          (the slow one, ~3x per epoch on brats/on-harmony)
# From the two numbers you get everything the pack layout needs: how many folds fit per GPU
# (VRAM) and how long a chain must be (epoch time). It also smoke-tests the whole training
# path end to end — trainer discovery via the venv shim, AugLab config loading, DualVal's
# second validation pass — before 36 real folds are committed to the queue.
#
# Writes everything under a THROWAWAY results base on $SCRATCH, so it cannot pollute the real
# 01_predictions tree with 3-epoch runs.
#
# Usage (run ON tamia):
#   bash 04_22_tamia_size_probe.sh              # 3 epochs, 40 min walltime
#   PROBE_EPOCHS=5 PROBE_TIME=01:00:00 bash 04_22_tamia_size_probe.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
PROBE_EPOCHS="${PROBE_EPOCHS:-3}"
PROBE_TIME="${PROBE_TIME:-00:40:00}"
PROBE_DIR="${PROBE_DIR:-${SCRATCH}/picai-prostate/_probe/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PROBE_DIR}"
echo "[probe] PROBE_DIR=${PROBE_DIR}  epochs=${PROBE_EPOCHS}"

# Throwaway results base — keeps probe checkpoints out of 8_results/01_predictions.
export NNUNET_RESULTS_BASE="${PROBE_DIR}/results"
mkdir -p "${NNUNET_RESULTS_BASE}"

# Record ONE fold of each of the two extremes (fold 0; TRAIN_FOLDS restricts the fan-out).
# The OURS RUN_ID must contain "_val000_" — DualVal's on_train_end mirrors off that marker.
export NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}"
export TRAIN_FOLDS="0"
RUN_JOB_PACK_DIR="${PROBE_DIR}" bash "${HERE}/04_06_train_t2w_auglabAug_v26_6_2_train050_val000.sh" \
    "probe_ours_val000_$(date +%H%M%S)"
RUN_JOB_PACK_DIR="${PROBE_DIR}" bash "${HERE}/04_05_train_t2w_srcsm.sh" \
    "probe_srcsm_$(date +%H%M%S)"
echo "[probe] recorded:"; cut -f3 "${PROBE_DIR}/index.tsv" | sed 's/^/  /'

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_CPUS=48 PACK_MEM=0 \
PACK_TIME="${PROBE_TIME}" PACK_CHAIN=1 PACK_GPU_MAP="0 1" \
PACK_JOB_NAME="picai_sizeprobe" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PROBE_DIR}"

echo
echo "[probe] when it finishes, read:"
echo "  epoch times : grep -h 'Epoch time' ${PROBE_DIR}/*.log"
echo "  peak VRAM   : grep -i 'mem' ${PROBE_DIR}/*.log | head"
echo "  pack stdout : ${PROBE_DIR}/pack_*.out"
