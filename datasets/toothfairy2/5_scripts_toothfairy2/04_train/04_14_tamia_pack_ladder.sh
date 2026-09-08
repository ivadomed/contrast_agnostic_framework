#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the toothfairy2 CBCT causal-ablation ladder —
# 4 untrained rungs x 3 folds (0 1 2) = 12 fold-jobs.
#
# Rungs (rung 1 = baseline and rungs 6/7 = OURS DualVal already exist in the suite
# pack — NOT retrained here):
#   rung2: baseline_kmeans                       (04_08)
#   rung3: baseline_kmeans_label_remap           (04_09)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_10, NOISE fill)
#   rung5: v26_6_2 alone                         (04_11, REAL fill — the causal pair vs rung4)
#
# The 4->5 step is the whole point: identical partition, only the fill differs. On a
# boundary-defined target like this one the paper predicts it is near-zero (as on
# chaos organs), unlike the large gain on texture-defined targets.
#
# ⚠️ Run 04_12_tamia_sizing_probe.sh first — see 04_13's header on folds-per-GPU.
#
# REQUIRES (run ON tamia): env.sh + scripts/cluster/tamia_env_toothfairy2.sh sourced.
#
# Usage:
#   bash 04_14_tamia_pack_ladder.sh               # fresh launch
#   PACK_DIR=<existing> bash 04_14_...            # extend chain / resume
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/toothfairy2/_packruns/ladder_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"; echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="toothfairy2_cbct_baseline_kmeans_${TS}"
    R3_RUN_ID="toothfairy2_cbct_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="toothfairy2_cbct_baseline_kmeans_label_remap_voronoi_${TS}"
    # ⚠️ NO "auglabAug_" prefix — see 04_11's header. An auglabAug_-prefixed RUN_ID
    # here would collide with the rung-6 DualVal run's val100 mirror.
    R5_RUN_ID="toothfairy2_cbct_v26_6_2_train050_val100_${TS}"
    printf 'R2_RUN_ID=%s\nR3_RUN_ID=%s\nR4_RUN_ID=%s\nR5_RUN_ID=%s\n' \
        "${R2_RUN_ID}" "${R3_RUN_ID}" "${R4_RUN_ID}" "${R5_RUN_ID}" > "${RUNIDS}"
    echo "[tamia-pack] generated RUN_IDs -> ${RUNIDS}"
fi
echo "  rung2 (kmeans):             ${R2_RUN_ID}"
echo "  rung3 (kmeans+label_remap): ${R3_RUN_ID}"
echo "  rung4 (+voronoi, noise):    ${R4_RUN_ID}"
echo "  rung5 (v26_6_2, real fill): ${R5_RUN_ID}"

if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[tamia-pack] index.tsv exists — reusing existing recording (resume/extend)."
else
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_08_train_baseline_kmeans.sh"                     "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_09_train_baseline_kmeans_label_remap.sh"          "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_10_train_baseline_kmeans_label_remap_voronoi.sh"  "${R4_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_11_train_v26_6_2_train050_val100.sh"              "${R5_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-23:59:00}" \
PACK_CHAIN="${PACK_CHAIN:-4}" PACK_USE_MPS="${PACK_USE_MPS:-0}" \
PACK_GPU_MAP="${PACK_GPU_MAP:-}" PACK_JOB_NAME="${PACK_JOB_NAME:-tf2_ladder}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
