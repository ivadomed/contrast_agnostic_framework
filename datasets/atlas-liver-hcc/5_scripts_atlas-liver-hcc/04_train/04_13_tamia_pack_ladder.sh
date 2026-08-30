#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the atlas-liver-hcc T1w causal-ablation ladder --
# 4 untrained rungs x 3 folds (0 1 2, project fold policy) = 12 fold-jobs, packed
# across ONE 4xH100 node with the run_job_pack_submit.sh dependency-chain mechanism
# (fixed RUN_IDs persisted once, index.tsv recorded once, chain resumes via
# checkpoint_latest). Mirrors 04_08_tamia_pack_all.sh's pattern exactly. Moved here
# from Vulcan 2026-08-29 (queue congestion there; only 4/12 folds had started after
# ~15min, epoch 16/2000 -- negligible progress, restarted fresh here instead of
# trying to resume).
#
# Sizing (04_08's probe, same dataset/node class): ~12-14s/epoch x 2000 epochs =~
# 6.5-8h/fold on 1 GPU exclusive; packed 3-per-GPU on this node, budget accordingly.
# No PACK_GPU_MAP override needed -- default round-robin (i%4) placement is fine
# (04_08's probe found srcsm only ~1.2x baseline cost here, nowhere near brats/
# on-harmony's ~3x -- this dataset doesn't need the imbalanced placement those do).
#
# Rungs (baseline + rungs 6/7 = OURS DualVal already exist -- NOT retrained here):
#   rung2: baseline_kmeans                      (04_09)
#   rung3: baseline_kmeans_label_remap           (04_10)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_11, noise fill)
#   rung5: v26_6_2 alone                         (04_12, real fill -- the causal pair vs rung4;
#                                                  did NOT exist at all for this dataset before)
#
# REQUIRES: `source datasets/atlas-liver-hcc/5_scripts_atlas-liver-hcc/00_utils/env.sh`
# and `source scripts/cluster/tamia_env_atlas-liver-hcc.sh` already sourced in THIS
# shell (exported vars propagate into the `bash 04_XX...sh` child processes below --
# do NOT source them inside this script, matching 04_08's convention).
#
# Usage (run ON tamia, after sourcing the two env files above):
#   bash 04_13_tamia_pack_ladder.sh                 # fresh launch (new pack dir)
#   PACK_DIR=<existing> bash 04_13_...              # extend chain / resume
#   PACK_CHAIN=3 bash 04_13_...                     # tune chain length
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/atlas-liver-hcc/_packruns/ladder_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"
    echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="atlas-liver-hcc_t1w_baseline_kmeans_${TS}"
    R3_RUN_ID="atlas-liver-hcc_t1w_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_${TS}"
    R5_RUN_ID="atlas-liver-hcc_t1w_v26_6_2_train050_val100_${TS}"
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
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_09_train_baseline_kmeans.sh"                    "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_10_train_baseline_kmeans_label_remap.sh"         "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_11_train_baseline_kmeans_label_remap_voronoi.sh" "${R4_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_12_train_v26_6_2_train050_val100.sh"              "${R5_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-3}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-atlashcc_pack_ladder}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
