#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the full 6-method atlas-liver-hcc T1w suite (7
# checkpoints via DualVal OURS) -- 6 methods x 3 folds (0 1 2) = 18 fold-jobs, packed
# across ONE 4xH100 node with the run_job_pack_submit.sh dependency-chain mechanism
# (fixed RUN_IDs persisted once, index.tsv recorded once, chain resumes via
# checkpoint_latest). Mirrors 04_33_tamia_pack_t2w_dualval_t1n_val100.sh's pattern.
#
# Sizing probe (2026-08-04, baseline vs srcsm, fold0, 6 epochs, on a real H100):
# srcsm is only ~1.21x baseline's per-epoch cost on THIS dataset (11.6s vs 14.0s/epoch)
# -- nowhere near the ~3x seen on brats/on-harmony (CLAUDE.md: don't assume that figure
# transfers). No PACK_GPU_MAP override needed -- default round-robin (i%4) placement is
# fine. ~12-14s/epoch x 2000 epochs =~ 6.5-8h/fold on 1 GPU exclusive.
#
# REQUIRES: `source datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/00_utils/env.sh`
# and `source scripts/cluster/tamia_env_atlas-liver-hcc.sh` already sourced in THIS
# shell (exported vars propagate into the `bash 04_XX...sh` child processes below --
# do NOT source them inside this script, matching 04_33/04_43's convention).
#
# Usage (run ON tamia, after sourcing the two env files above):
#   bash 04_08_tamia_pack_all.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_08_...            # extend chain / resume
#   PACK_CHAIN=5 bash 04_08_...                   # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/atlas-liver-hcc/_packruns/all6_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    BASELINE_RUN_ID="atlas-liver-hcc_t1w_baseline_${TS}"
    AUGLAB_DEFAULT_RUN_ID="atlas-liver-hcc_t1w_auglab_default_${TS}"
    SYNTHSEG_NOEM_RUN_ID="atlas-liver-hcc_t1w_synthseg_noEM_${TS}"
    SYNTHSEG_EM_RUN_ID="atlas-liver-hcc_t1w_synthseg_EM_${TS}"
    SRCSM_RUN_ID="atlas-liver-hcc_t1w_srcsm_${TS}"
    OURS_RUN_ID="atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_${TS}"
    printf 'BASELINE_RUN_ID=%s\nAUGLAB_DEFAULT_RUN_ID=%s\nSYNTHSEG_NOEM_RUN_ID=%s\nSYNTHSEG_EM_RUN_ID=%s\nSRCSM_RUN_ID=%s\nOURS_RUN_ID=%s\n' \
        "${BASELINE_RUN_ID}" "${AUGLAB_DEFAULT_RUN_ID}" "${SYNTHSEG_NOEM_RUN_ID}" \
        "${SYNTHSEG_EM_RUN_ID}" "${SRCSM_RUN_ID}" "${OURS_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  baseline:       ${BASELINE_RUN_ID}"
echo "  auglab_default: ${AUGLAB_DEFAULT_RUN_ID}"
echo "  synthseg_noEM:  ${SYNTHSEG_NOEM_RUN_ID}"
echo "  synthseg_EM:    ${SYNTHSEG_EM_RUN_ID}"
echo "  srcsm:          ${SRCSM_RUN_ID}"
echo "  OURS (dualval): ${OURS_RUN_ID}"

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_01_train_baseline.sh"                 "${BASELINE_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_02_train_auglab_default.sh"           "${AUGLAB_DEFAULT_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_03_train_synthseg_noEM.sh"            "${SYNTHSEG_NOEM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_04_train_synthseg_EM.sh"              "${SYNTHSEG_EM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_05_train_srcsm.sh"                    "${SRCSM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_06_train_auglabAug_v26_6_2_dualval.sh" "${OURS_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# 18 fold-jobs / 4 GPUs =~ 4-5 folds/GPU sequential x ~6.5-8h/fold =~ 26-40h/GPU total;
# chain of 3 x 24h waves gives ~72h headroom, comfortably enough (resumes via
# checkpoint_latest if a wave doesn't finish -- rerun this script with the same
# PACK_DIR to extend).
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-atlashcc_pack_all6}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
