#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the BraTS T1n causal-ablation ladder --
# 4 untrained rungs x 3 folds (0 1 2, project fold policy) = 12 folds, packed
# across ONE 4xH100 node (2500 epochs each, ~42h/fold on 1 GPU exclusive --
# packed 3-per-GPU will take substantially longer; this WILL span multiple
# chain links, likely multiple days). Mirrors 04_33_tamia_pack_t2w_dualval_t1n_val100.sh's
# pattern exactly (fixed RUN_IDs persisted once, index.tsv recorded once, chain
# resumes via checkpoint_latest).
#
# Rungs (baseline + rungs 6/7 = OURS deployed already exist -- NOT retrained here):
#   rung2: baseline_kmeans                      (04_40)
#   rung3: baseline_kmeans_label_remap           (04_41)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_42, noise fill)
#   rung5: v26_6_2 alone                         (04_20, real fill -- the causal pair vs rung4)
#
# REQUIRES: `source scripts/cluster/tamia_env.sh` already sourced in THIS shell
# (exported vars propagate into the `bash 04_XX...sh` child processes below --
# do NOT source it inside this script, matching 04_33's convention of expecting
# the caller's shell to have it).
#
# Usage (run ON tamia, after sourcing datasets/brats2024-glioma/.../env.sh +
# scripts/cluster/tamia_env.sh):
#   bash 04_43_tamia_pack_t1n_ladder.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_43_...                  # extend chain / resume
#   PACK_CHAIN=8 bash 04_43_...                         # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/brats2024-glioma/_packruns/t1n_ladder_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="brats2024-glioma_t1n_baseline_kmeans_${TS}"
    R3_RUN_ID="brats2024-glioma_t1n_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_${TS}"
    R5_RUN_ID="brats2024-glioma_t1n_v26_6_2_train050_val100_${TS}"
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
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_40_train_t1n_baseline_kmeans.sh"                       "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_41_train_t1n_baseline_kmeans_label_remap.sh"            "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_42_train_t1n_baseline_kmeans_label_remap_voronoi.sh"    "${R4_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_20_train_t1n_v26_6_2_train050_val100.sh"                "${R5_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-8}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-brats_t1n_ladder}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
