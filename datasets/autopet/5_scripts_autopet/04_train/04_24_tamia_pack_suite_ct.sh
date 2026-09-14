#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the autopet CT 6-method suite — 6 methods x
# 3 folds (0 1 2) = 18 fold-jobs.
#
# TamIA allocates GPUs by WHOLE NODE (4x H100), so the unit of work is a node-pack:
# several folds pinned across the node's GPUs, chained across jobs because the 24h
# walltime cap is shorter than a full training. See scripts/job_runner/
# run_job_pack_submit.sh for the mechanism.
#
# ⚠️ RUN THE SIZING PROBE FIRST (04_15_tamia_sizing_probe.sh). Folds-per-GPU is a
# VRAM-and-throughput decision that must be MEASURED on this dataset — whole-body
# PET/CT volumes are far larger than any other dataset trained here so far, and a
# probe on idle GPUs is 2.6-8.5x too optimistic vs. the real contended launch.
#
# PACK_METHODS selects a SUBSET, so 18 fold-jobs can be spread over several node
# packs instead of stacking 4-5 folds on every GPU:
#   PACK_METHODS="baseline auglab_default synthseg_noEM" bash 04_24_...   # 9 folds
#   PACK_METHODS="synthseg_EM srcsm ours"                bash 04_24_...   # 9 folds
# Default: all six.
#
# REQUIRES (run ON tamia): env.sh + scripts/cluster/tamia_env_autopet.sh already
# sourced in THIS shell.
#
# Usage:
#   bash 04_24_tamia_pack_suite_ct.sh              # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_24_...             # extend chain / resume
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_METHODS="${PACK_METHODS:-baseline auglab_default synthseg_noEM synthseg_EM srcsm ours}"
PACK_TAG="${PACK_TAG:-suite_ct}"
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
    BASELINE_RUN_ID="autopet_ct_baseline_${TS}"
    AUGLAB_DEFAULT_RUN_ID="autopet_ct_auglab_default_${TS}"
    SYNTHSEG_NOEM_RUN_ID="autopet_ct_synthseg_noEM_${TS}"
    SYNTHSEG_EM_RUN_ID="autopet_ct_synthseg_EM_${TS}"
    SRCSM_RUN_ID="autopet_ct_srcsm_${TS}"
    # MUST keep "_val000_" exactly once — the DualVal trainer derives the val100
    # mirror directory name from it.
    OURS_RUN_ID="autopet_ct_auglabAug_v26_6_2_train050_val000_${TS}"
    printf 'BASELINE_RUN_ID=%s\nAUGLAB_DEFAULT_RUN_ID=%s\nSYNTHSEG_NOEM_RUN_ID=%s\nSYNTHSEG_EM_RUN_ID=%s\nSRCSM_RUN_ID=%s\nOURS_RUN_ID=%s\n' \
        "${BASELINE_RUN_ID}" "${AUGLAB_DEFAULT_RUN_ID}" "${SYNTHSEG_NOEM_RUN_ID}" \
        "${SYNTHSEG_EM_RUN_ID}" "${SRCSM_RUN_ID}" "${OURS_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi

has() { case " ${PACK_METHODS} " in *" $1 "*) return 0;; *) return 1;; esac; }

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    has baseline        && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_01_train_ct_baseline.sh"                   "${BASELINE_RUN_ID}"
    has auglab_default  && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_02_train_ct_auglab_default.sh"             "${AUGLAB_DEFAULT_RUN_ID}"
    has synthseg_noEM   && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_03_train_ct_synthseg_noEM.sh"              "${SYNTHSEG_NOEM_RUN_ID}"
    has synthseg_EM     && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_04_train_ct_synthseg_EM.sh"                "${SYNTHSEG_EM_RUN_ID}"
    has srcsm           && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_05_train_ct_srcsm.sh"                      "${SRCSM_RUN_ID}"
    has ours            && RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_06_train_ct_auglabAug_v26_6_2_dualval.sh"  "${OURS_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-23:59:00}" \
PACK_CHAIN="${PACK_CHAIN:-4}" PACK_USE_MPS="${PACK_USE_MPS:-0}" \
PACK_GPU_MAP="${PACK_GPU_MAP:-}" PACK_JOB_NAME="${PACK_JOB_NAME:-autopet_${PACK_TAG}}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
