#!/usr/bin/env bash
# TamIA whole-node pack launcher: full 6-method isles2022 suite, BOTH training contrasts (dwi + flair) =
# 12 methods x 3 folds (0 1 2) = 36 fold-jobs, chain-resumed (PACK_CHAIN x 23:59:00).
# Generated from ispy2's 04_15 (same shape). ⚠️ DO NOT launch as-is before a SIZING PROBE: per-fold VRAM and
# epoch time for the heaviest (OURS DualVal) and slowest (srcsm?) methods, N folds sharing one GPU, on a real
# H100 node with a throwaway results base -- this dataset's volumes are small (~112x112x73), so more folds per
# GPU than ispy2 is likely, and PACK_GPU_MAP may be needed if srcsm is slower here (measure, don't assume 3x).
# Also decide the epoch count (04_00_common.sh: 2000, set by Paul 2026-10-04; re-time via the probe).
# REQUIRES (on TamIA): source .../00_utils/env.sh and scripts/cluster/tamia_env_isles2022.sh first, in THIS shell,
# and RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G (GiB, not MB).
#   bash 04_23_tamia_pack_all.sh      PACK_DIR=<existing> bash 04_23_...   # resume/extend
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/isles2022/_packruns/all12_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    DWI_BASELINE_RUN_ID="isles2022_dwi_baseline_${TS}"
    DWI_AUGLAB_DEFAULT_RUN_ID="isles2022_dwi_auglab_default_${TS}"
    DWI_SYNTHSEG_NOEM_RUN_ID="isles2022_dwi_synthseg_noEM_${TS}"
    DWI_SYNTHSEG_EM_RUN_ID="isles2022_dwi_synthseg_EM_${TS}"
    DWI_SRCSM_RUN_ID="isles2022_dwi_srcsm_${TS}"
    DWI_OURS_RUN_ID="isles2022_dwi_auglabAug_v26_6_2_train050_val000_${TS}"
    FLAIR_BASELINE_RUN_ID="isles2022_flair_baseline_${TS}"
    FLAIR_AUGLAB_DEFAULT_RUN_ID="isles2022_flair_auglab_default_${TS}"
    FLAIR_SYNTHSEG_NOEM_RUN_ID="isles2022_flair_synthseg_noEM_${TS}"
    FLAIR_SYNTHSEG_EM_RUN_ID="isles2022_flair_synthseg_EM_${TS}"
    FLAIR_SRCSM_RUN_ID="isles2022_flair_srcsm_${TS}"
    FLAIR_OURS_RUN_ID="isles2022_flair_auglabAug_v26_6_2_train050_val000_${TS}"
    printf 'DWI_BASELINE_RUN_ID=%s\nDWI_AUGLAB_DEFAULT_RUN_ID=%s\nDWI_SYNTHSEG_NOEM_RUN_ID=%s\nDWI_SYNTHSEG_EM_RUN_ID=%s\nDWI_SRCSM_RUN_ID=%s\nDWI_OURS_RUN_ID=%s\nFLAIR_BASELINE_RUN_ID=%s\nFLAIR_AUGLAB_DEFAULT_RUN_ID=%s\nFLAIR_SYNTHSEG_NOEM_RUN_ID=%s\nFLAIR_SYNTHSEG_EM_RUN_ID=%s\nFLAIR_SRCSM_RUN_ID=%s\nFLAIR_OURS_RUN_ID=%s\n' \
        "${DWI_BASELINE_RUN_ID}" "${DWI_AUGLAB_DEFAULT_RUN_ID}" "${DWI_SYNTHSEG_NOEM_RUN_ID}" \
        "${DWI_SYNTHSEG_EM_RUN_ID}" "${DWI_SRCSM_RUN_ID}" "${DWI_OURS_RUN_ID}" \
        "${FLAIR_BASELINE_RUN_ID}" "${FLAIR_AUGLAB_DEFAULT_RUN_ID}" "${FLAIR_SYNTHSEG_NOEM_RUN_ID}" \
        "${FLAIR_SYNTHSEG_EM_RUN_ID}" "${FLAIR_SRCSM_RUN_ID}" "${FLAIR_OURS_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_01_train_dwi_baseline.sh"                 "${DWI_BASELINE_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_02_train_dwi_auglab_default.sh"           "${DWI_AUGLAB_DEFAULT_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_03_train_dwi_synthseg_noEM.sh"            "${DWI_SYNTHSEG_NOEM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_04_train_dwi_synthseg_EM.sh"              "${DWI_SYNTHSEG_EM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_05_train_dwi_srcsm.sh"                    "${DWI_SRCSM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_06_train_dwi_auglabAug_v26_6_2_dualval.sh" "${DWI_OURS_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_08_train_flair_baseline.sh"                   "${FLAIR_BASELINE_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_09_train_flair_auglab_default.sh"             "${FLAIR_AUGLAB_DEFAULT_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_10_train_flair_synthseg_noEM.sh"              "${FLAIR_SYNTHSEG_NOEM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_11_train_flair_synthseg_EM.sh"                "${FLAIR_SYNTHSEG_EM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_12_train_flair_srcsm.sh"                      "${FLAIR_SRCSM_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_13_train_flair_auglabAug_v26_6_2_dualval.sh"   "${FLAIR_OURS_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# 36 fold-jobs / 4 GPUs =~ 9 folds/GPU sequential if run as ONE pack -- see the
# header warning above. Prefer splitting into multiple ~12-fold-job packs instead.
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-4}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-isles2022_pack_all12}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
