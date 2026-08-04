#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the on-harmony T1w causal-ablation ladder --
# 3 untrained rungs x 3 folds (0 1 2, project fold policy) = 9 folds, packed
# across ONE 4xH100 node (2000 epochs each). Mirrors 04_27_tamia_pack_t1w_dualval_
# t2w_val100_srcsm.sh's pattern exactly (fixed RUN_IDs persisted once, index.tsv
# recorded once, chain resumes via checkpoint_latest). MPS OFF (PACK_USE_MPS=0,
# the run_job_pack_submit.sh default) -- MPS hung on-harmony packs outright in an
# earlier smoke test here (see 04_27's header), plain time-slicing is what works.
#
# Rungs (baseline + rung5/6/7 = v26_6_2-alone + OURS already exist -- NOT retrained here):
#   rung2: baseline_kmeans                      (04_28)
#   rung3: baseline_kmeans_label_remap           (04_29)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_30, noise fill)
#   (rung5 = v26_6_2 alone, real fill -- pre-existing: on-harmony_T1w_v26_6_2_train050_val100_20260623_192811)
#
# REQUIRES: `source scripts/cluster/tamia_env_onharmony.sh` already sourced in THIS
# shell (exported vars propagate into the `bash 04_XX...sh` child processes below --
# do NOT source it inside this script, matching 04_27's convention of expecting
# the caller's shell to have it).
#
# Usage (run ON tamia, after sourcing datasets/on-harmony/.../env.sh +
# scripts/cluster/tamia_env_onharmony.sh):
#   bash 04_31_tamia_pack_t1w_ladder.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_31_...                  # extend chain / resume
#   PACK_CHAIN=3 bash 04_31_...                         # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/on-harmony/_packruns/t1w_ladder_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="on-harmony_T1w_baseline_kmeans_${TS}"
    R3_RUN_ID="on-harmony_T1w_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="on-harmony_T1w_baseline_kmeans_label_remap_voronoi_${TS}"
    printf 'R2_RUN_ID=%s\nR3_RUN_ID=%s\nR4_RUN_ID=%s\n' \
        "${R2_RUN_ID}" "${R3_RUN_ID}" "${R4_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  rung2 (kmeans):              ${R2_RUN_ID}"
echo "  rung3 (kmeans+label_remap):  ${R3_RUN_ID}"
echo "  rung4 (+voronoi, noise):     ${R4_RUN_ID}"

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_28_train_t1w_baseline_kmeans.sh"                    "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_29_train_t1w_baseline_kmeans_label_remap.sh"         "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_30_train_t1w_baseline_kmeans_label_remap_voronoi.sh" "${R4_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-onharmony_t1w_ladder}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
