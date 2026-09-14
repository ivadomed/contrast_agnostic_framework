#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the autopet PET 6-method suite — mirror of
# 04_24_tamia_pack_suite_ct.sh, see its header for the full rationale.
#
# ⚠️ RUN THE SIZING PROBE FIRST, against PET too (not just CT) — PET's SUV intensity/
# noise profile differs from CT's HU scale, so a CT-measured per-fold cost must NOT be
# assumed to transfer.
#
# Usage:
#   bash 04_26_tamia_pack_suite_pet.sh
#   PACK_DIR=<existing> bash 04_26_...
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_METHODS="${PACK_METHODS:-baseline auglab_default synthseg_noEM synthseg_EM srcsm ours}"
PACK_TAG="${PACK_TAG:-suite_pet}"
PACK_DIR="${PACK_DIR:-/scratch/p/paulh/autopet/_packruns/${PACK_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"
echo "[tamia-pack] methods: ${PACK_METHODS}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"; echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    BASELINE_RUN_ID="autopet_pet_baseline_${TS}"
    AUGLAB_DEFAULT_RUN_ID="autopet_pet_auglab_default_${TS}"
    SYNTHSEG_NOEM_RUN_ID="autopet_pet_synthseg_noEM_${TS}"
    SYNTHSEG_EM_RUN_ID="autopet_pet_synthseg_EM_${TS}"
    SRCSM_RUN_ID="autopet_pet_srcsm_${TS}"
    OURS_RUN_ID="autopet_pet_auglabAug_v26_6_2_train050_val000_${TS}"
    printf 'BASELINE_RUN_ID=%s\nAUGLAB_DEFAULT_RUN_ID=%s\nSYNTHSEG_NOEM_RUN_ID=%s\nSYNTHSEG_EM_RUN_ID=%s\nSRCSM_RUN_ID=%s\nOURS_RUN_ID=%s\n' \
        "${BASELINE_RUN_ID}" "${AUGLAB_DEFAULT_RUN_ID}" "${SYNTHSEG_NOEM_RUN_ID}" \
        "${SYNTHSEG_EM_RUN_ID}" "${SRCSM_RUN_ID}" "${OURS_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi

has() { case " ${PACK_METHODS} " in *" $1 "*) return 0;; *) return 1;; esac; }

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    has baseline        && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_08_train_pet_baseline.sh"                   "${BASELINE_RUN_ID}"
    has auglab_default  && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_09_train_pet_auglab_default.sh"             "${AUGLAB_DEFAULT_RUN_ID}"
    has synthseg_noEM   && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_10_train_pet_synthseg_noEM.sh"              "${SYNTHSEG_NOEM_RUN_ID}"
    has synthseg_EM     && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_11_train_pet_synthseg_EM.sh"                "${SYNTHSEG_EM_RUN_ID}"
    has srcsm           && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_12_train_pet_srcsm.sh"                      "${SRCSM_RUN_ID}"
    has ours            && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_13_train_pet_auglabAug_v26_6_2_dualval.sh"  "${OURS_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-23:59:00}" \
PACK_CHAIN="${PACK_CHAIN:-4}" PACK_USE_MPS="${PACK_USE_MPS:-0}" \
PACK_GPU_MAP="${PACK_GPU_MAP:-}" PACK_JOB_NAME="${PACK_JOB_NAME:-autopet_${PACK_TAG}}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
