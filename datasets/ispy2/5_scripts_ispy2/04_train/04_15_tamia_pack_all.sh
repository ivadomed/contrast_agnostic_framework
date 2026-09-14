#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the full 6-method ispy2 suite, BOTH training
# modalities (t1wce + t2w) -- 12 methods x 3 folds (0 1 2) = 36 fold-jobs. Mirrors
# datasets/ambl/5_scripts_ambl/04_train/04_15_tamia_pack_all.sh, doubled for ispy2's
# second training modality.
#
# ⚠️ Kept here as the generic/documented template per this project's convention,
# but the ACTUAL launch used a hand-written script splitting these 36 fold-jobs
# (plus the 24 ladder fold-jobs) across 5 whole-node packs at ~12 fold-jobs/node
# (3 concurrent folds/GPU) -- the layout ambl's real TamIA sizing probe validated.
# Running this script AS-IS would put all 36 on one node at 9 folds/GPU, which was
# never measured for this project and left too little VRAM margin on ambl. See
# datasets/ambl/5_scripts_ambl/04_train/04_15's own header for the full story
# (including the PACK_DIR-collision bug this file's per-modality-safe RUN_ID
# naming already avoids, since RUN_IDs — not bare METHOD — key the pack recording).
#
# REQUIRES: `source datasets/ispy2/5_scripts_ispy2/00_utils/env.sh` and
# `source scripts/cluster/tamia_env_ispy2.sh` already sourced in THIS shell.
#
# Usage (run ON tamia, after sourcing the two env files above):
#   bash 04_15_tamia_pack_all.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_15_...            # extend chain / resume
#   PACK_CHAIN=5 bash 04_15_...                   # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/ispy2/_packruns/all12_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    T1WCE_BASELINE_RUN_ID="ispy2_t1wce_baseline_${TS}"
    T1WCE_AUGLAB_DEFAULT_RUN_ID="ispy2_t1wce_auglab_default_${TS}"
    T1WCE_SYNTHSEG_NOEM_RUN_ID="ispy2_t1wce_synthseg_noEM_${TS}"
    T1WCE_SYNTHSEG_EM_RUN_ID="ispy2_t1wce_synthseg_EM_${TS}"
    T1WCE_SRCSM_RUN_ID="ispy2_t1wce_srcsm_${TS}"
    T1WCE_OURS_RUN_ID="ispy2_t1wce_auglabAug_v26_6_2_train050_val000_${TS}"
    T2W_BASELINE_RUN_ID="ispy2_t2w_baseline_${TS}"
    T2W_AUGLAB_DEFAULT_RUN_ID="ispy2_t2w_auglab_default_${TS}"
    T2W_SYNTHSEG_NOEM_RUN_ID="ispy2_t2w_synthseg_noEM_${TS}"
    T2W_SYNTHSEG_EM_RUN_ID="ispy2_t2w_synthseg_EM_${TS}"
    T2W_SRCSM_RUN_ID="ispy2_t2w_srcsm_${TS}"
    T2W_OURS_RUN_ID="ispy2_t2w_auglabAug_v26_6_2_train050_val000_${TS}"
    printf 'T1WCE_BASELINE_RUN_ID=%s\nT1WCE_AUGLAB_DEFAULT_RUN_ID=%s\nT1WCE_SYNTHSEG_NOEM_RUN_ID=%s\nT1WCE_SYNTHSEG_EM_RUN_ID=%s\nT1WCE_SRCSM_RUN_ID=%s\nT1WCE_OURS_RUN_ID=%s\nT2W_BASELINE_RUN_ID=%s\nT2W_AUGLAB_DEFAULT_RUN_ID=%s\nT2W_SYNTHSEG_NOEM_RUN_ID=%s\nT2W_SYNTHSEG_EM_RUN_ID=%s\nT2W_SRCSM_RUN_ID=%s\nT2W_OURS_RUN_ID=%s\n' \
        "${T1WCE_BASELINE_RUN_ID}" "${T1WCE_AUGLAB_DEFAULT_RUN_ID}" "${T1WCE_SYNTHSEG_NOEM_RUN_ID}" \
        "${T1WCE_SYNTHSEG_EM_RUN_ID}" "${T1WCE_SRCSM_RUN_ID}" "${T1WCE_OURS_RUN_ID}" \
        "${T2W_BASELINE_RUN_ID}" "${T2W_AUGLAB_DEFAULT_RUN_ID}" "${T2W_SYNTHSEG_NOEM_RUN_ID}" \
        "${T2W_SYNTHSEG_EM_RUN_ID}" "${T2W_SRCSM_RUN_ID}" "${T2W_OURS_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_01_train_t1wce_baseline.sh"                 "${T1WCE_BASELINE_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_02_train_t1wce_auglab_default.sh"           "${T1WCE_AUGLAB_DEFAULT_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_03_train_t1wce_synthseg_noEM.sh"            "${T1WCE_SYNTHSEG_NOEM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_04_train_t1wce_synthseg_EM.sh"              "${T1WCE_SYNTHSEG_EM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_05_train_t1wce_srcsm.sh"                    "${T1WCE_SRCSM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh" "${T1WCE_OURS_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_08_train_t2w_baseline.sh"                   "${T2W_BASELINE_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_09_train_t2w_auglab_default.sh"             "${T2W_AUGLAB_DEFAULT_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_10_train_t2w_synthseg_noEM.sh"              "${T2W_SYNTHSEG_NOEM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_11_train_t2w_synthseg_EM.sh"                "${T2W_SYNTHSEG_EM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_12_train_t2w_srcsm.sh"                      "${T2W_SRCSM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_13_train_t2w_auglabAug_v26_6_2_dualval.sh"   "${T2W_OURS_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# 36 fold-jobs / 4 GPUs =~ 9 folds/GPU sequential if run as ONE pack -- see the
# header warning above. Prefer splitting into multiple ~12-fold-job packs instead.
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-4}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-ispy2_pack_all12}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
