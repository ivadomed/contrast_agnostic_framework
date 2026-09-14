#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the autopet CT causal-ablation ladder —
# 4 untrained rungs x 3 folds (0 1 2) = 12 fold-jobs.
#
# Rungs (rung 1 = baseline and rungs 6/7 = OURS DualVal already exist in the suite
# pack — NOT retrained here):
#   rung2: baseline_kmeans                       (04_16)
#   rung3: baseline_kmeans_label_remap           (04_17)
#   rung4: baseline_kmeans_label_remap_voronoi   (04_18, NOISE fill)
#   rung5: v26_6_2 alone                         (04_19, REAL fill — the causal pair vs rung4)
#
# The 4->5 step is the whole point: identical partition, only the fill differs.
# AutoPET's tumor-lesion target is appearance-defined (PET uptake / CT enhancement
# pattern), same category as open-ms lesions / brats tumour sub-regions — the paper's
# prediction is a LARGE gain here, not near-zero like chaos organs. Measure it, don't
# assume it.
#
# ⚠️ Run 04_15_tamia_sizing_probe.sh first — see its header on folds-per-GPU.
#
# REQUIRES (run ON tamia): env.sh + scripts/cluster/tamia_env_autopet.sh sourced.
#
# Usage:
#   bash 04_25_tamia_pack_ladder_ct.sh             # fresh launch
#   PACK_DIR=<existing> bash 04_25_...             # extend chain / resume
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

PACK_DIR="${PACK_DIR:-/scratch/p/paulh/autopet/_packruns/ladder_ct_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[tamia-pack] PACK_DIR=${PACK_DIR}"

RUNIDS="${PACK_DIR}/RUN_IDS.env"
if [ -f "${RUNIDS}" ]; then
    # shellcheck disable=SC1090
    source "${RUNIDS}"; echo "[tamia-pack] reusing RUN_IDs from ${RUNIDS}"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    R2_RUN_ID="autopet_ct_baseline_kmeans_${TS}"
    R3_RUN_ID="autopet_ct_baseline_kmeans_label_remap_${TS}"
    R4_RUN_ID="autopet_ct_baseline_kmeans_label_remap_voronoi_${TS}"
    # ⚠️ NO "auglabAug_" prefix — see 04_19's header.
    R5_RUN_ID="autopet_ct_v26_6_2_train050_val100_${TS}"
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
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_16_train_ct_baseline_kmeans.sh"                     "${R2_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_17_train_ct_baseline_kmeans_label_remap.sh"          "${R3_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_18_train_ct_baseline_kmeans_label_remap_voronoi.sh"  "${R4_RUN_ID}"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/04_19_train_ct_v26_6_2_train050_val100.sh"              "${R5_RUN_ID}"
    echo "[tamia-pack] recorded folds:"; cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

PACK_GPU_TYPE=h100 PACK_NODE_GPUS="${PACK_NODE_GPUS:-4}" PACK_TIME="${PACK_TIME:-23:59:00}" \
PACK_CHAIN="${PACK_CHAIN:-4}" PACK_USE_MPS="${PACK_USE_MPS:-0}" \
PACK_GPU_MAP="${PACK_GPU_MAP:-}" PACK_JOB_NAME="${PACK_JOB_NAME:-autopet_ladder_ct}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
