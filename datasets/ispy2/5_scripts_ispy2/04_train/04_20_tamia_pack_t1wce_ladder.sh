#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the ispy2 T1WCE causal-ablation ladder -- 4
# untrained rungs x 3 folds (0 1 2) = 12 fold-jobs. Mirrors
# datasets/ambl/5_scripts_ambl/04_train/04_20_tamia_pack_t1wce_ladder.sh exactly;
# see 04_25_tamia_pack_t2w_ladder.sh for the second training modality.
#
# Rungs (baseline + rungs 6/7 = OURS DualVal already exist -- NOT retrained here):
#   rung2: baseline_kmeans                      (04_16)
#   rung3: baseline_kmeans_label_remap           (04_17)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_18, noise fill)
#   rung5: v26_6_2 alone                         (04_19, real fill -- the causal pair vs rung4)
#
# REQUIRES: `source datasets/ispy2/5_scripts_ispy2/00_utils/env.sh` and
# `source scripts/cluster/tamia_env_ispy2.sh` already sourced in THIS shell.
#
# Usage (run ON tamia, after sourcing the two env files above):
#   bash 04_20_tamia_pack_t1wce_ladder.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_20_...                    # extend chain / resume
#   PACK_CHAIN=3 bash 04_20_...                           # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/ispy2/_packruns/ladder_t1wce_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="ispy2_t1wce_baseline_kmeans_${TS}"
    R3_RUN_ID="ispy2_t1wce_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="ispy2_t1wce_baseline_kmeans_label_remap_voronoi_${TS}"
    R5_RUN_ID="ispy2_t1wce_v26_6_2_train050_val100_${TS}"
    printf 'R2_RUN_ID=%s\nR3_RUN_ID=%s\nR4_RUN_ID=%s\nR5_RUN_ID=%s\n' \
        "${R2_RUN_ID}" "${R3_RUN_ID}" "${R4_RUN_ID}" "${R5_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  rung2 (kmeans):              ${R2_RUN_ID}"
echo "  rung3 (kmeans+label_remap):  ${R3_RUN_ID}"
echo "  rung4 (+voronoi, noise):     ${R4_RUN_ID}"
echo "  rung5 (v26_6_2, real fill):  ${R5_RUN_ID}"

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_16_train_t1wce_baseline_kmeans.sh"                    "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_17_train_t1wce_baseline_kmeans_label_remap.sh"         "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_18_train_t1wce_baseline_kmeans_label_remap_voronoi.sh" "${R4_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_19_train_t1wce_v26_6_2_train050_val100.sh"              "${R5_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-ispy2_pack_ladder_t1wce}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
