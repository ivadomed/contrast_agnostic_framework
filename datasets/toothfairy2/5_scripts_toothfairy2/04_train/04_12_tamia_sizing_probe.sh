#!/usr/bin/env bash
# TAMIA SIZING PROBE for toothfairy2 — REQUIRED before the real pack launches.
#
# CLAUDE.md mandates measuring rather than assuming, and a project memory records
# the specific way this goes wrong: a probe that runs N folds on N IDLE GPUs is
# 2.6-8.5x too optimistic versus the real launch, because the real launch puts
# several folds on the SAME GPU. So this probe reproduces the intended contention
# exactly — PROBE_FOLDS folds packed onto the node at the same folds-per-GPU ratio
# the real launch will use — and runs them for a handful of epochs.
#
# It also answers the srcsm question specifically. srcsm's per-epoch cost ratio does
# NOT transfer between datasets (~3x on brats/on-harmony, ~1.2x on atlas-liver-hcc,
# ~1.5x on ambl, none at all on one abandoned set). If srcsm is the straggler here,
# the real launch must give it a GPU to itself via PACK_GPU_MAP.
#
# Results go to a THROWAWAY results base so they can never pollute real checkpoints.
#
# Usage (run ON tamia, after sourcing env.sh + tamia_env_toothfairy2.sh):
#   bash 04_12_tamia_sizing_probe.sh
#   PROBE_EPOCHS=5 PROBE_TIME=01:00:00 bash 04_12_tamia_sizing_probe.sh
#
# Read the result with 04_13_probe_report.sh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
source "${HERE}/../00_utils/env.sh"

PROBE_EPOCHS="${PROBE_EPOCHS:-6}"
PROBE_TIME="${PROBE_TIME:-01:30:00}"
PROBE_BASE="${PROBE_BASE:-${SCRATCH:-/scratch/p/paulh}/toothfairy2/_probe_$(date +%Y%m%d_%H%M%S)}"
PACK_DIR="${PROBE_BASE}/pack"
mkdir -p "${PACK_DIR}"

# THROWAWAY results root — nothing here is ever read by predict/evaluate.
export PREDICTIONS_ROOT="${PROBE_BASE}/01_predictions"
export nnUNet_results="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
export RESULTS_DIR="${PROBE_BASE}"
mkdir -p "${nnUNet_results}"
echo "[probe] throwaway results base: ${PROBE_BASE}"

TS="$(date +%Y%m%d_%H%M%S)"
# 4 methods x 2 folds = 8 fold-jobs over 4 GPUs = 2 folds/GPU, the layout the real
# suite pack will use. TRAIN_FOLDS is deliberately restricted here (the shared driver
# defaults to "0 1 2"); the probe measures cost, it does not produce results.
export TRAIN_FOLDS="0 1"
# Deliberately spans the CHEAPEST (baseline), the SUSPECTED-EXPENSIVE (srcsm) and
# OURS (DualVal — it runs a SECOND validation pass every epoch, so it is the other
# candidate straggler and must be measured, not assumed cheap).
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_01_train_baseline.sh"                  "probe_baseline_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_02_train_auglab_default.sh"            "probe_auglab_default_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_04_train_synthseg_EM.sh"               "probe_synthseg_EM_${TS}"
NNUNET_NUM_EPOCHS="${PROBE_EPOCHS}" RUN_JOB_PACK_DIR="${PACK_DIR}" \
    bash "${HERE}/04_05_train_srcsm.sh"                     "probe_srcsm_${TS}"

echo "[probe] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
echo "${PROBE_BASE}" > "${HERE}/.last_probe_base"

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PROBE_TIME}" PACK_CHAIN=1 \
PACK_USE_MPS=0 PACK_JOB_NAME="tf2_probe" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
