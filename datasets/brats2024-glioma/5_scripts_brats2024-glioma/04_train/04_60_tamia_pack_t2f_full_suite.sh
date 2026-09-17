#!/usr/bin/env bash
# TAMIA whole-node pack launcher for the FULL BraTS T2f/FLAIR training-modality
# suite: the 6-method comparison (baseline, auglab_default, synthseg_noEM,
# synthseg_EM, srcsm, OURS DualVal) + the 4 new causal-ablation ladder rungs
# (kmeans, kmeans+label_remap, kmeans+label_remap+voronoi, v26_6_2_train050_val100
# = rung5) = 10 wrappers x 3 folds (0 1 2) = 30 folds total.
#
# IMPORTANT: run_job_pack_submit.sh launches EVERY recorded fold in a PACK_DIR
# CONCURRENTLY (backgrounded, no MPS by default) for the full lifetime of the
# node allocation — it does NOT sequence folds within one pack. Packing all 30
# folds into one PACK_DIR would over-subscribe each GPU (~7-8 concurrent
# processes/GPU with no MPS) well past what a single H100 can hold for a
# batch_size=2 3d_fullres model. Following the established brats packing
# precedent (04_33: 6 folds/4 GPUs = max 2/GPU; 04_43: 12 folds/4 GPUs =
# 3/GPU), this script splits the 30 folds into THREE separate whole-node
# packs (each its own PACK_DIR, its own chain, and — since TamIA has more
# than one node — able to run concurrently on separate nodes rather than
# queueing behind each other):
#
#   Pack A (12 folds, round-robin 3/GPU): baseline, auglab_default,
#           synthseg_EM, synthseg_noEM
#   Pack B (12 folds, round-robin 3/GPU): OURS DualVal, v26_6_2_train050_val100
#           (ladder rung5, real fill), kmeans (rung2), kmeans_label_remap (rung3)
#   Pack C (6 folds, explicit PACK_GPU_MAP): kmeans_label_remap_voronoi (rung4,
#           noise fill — 3 folds share GPU 3) + srcsm (3 folds, ONE dedicated
#           GPU each: 0, 1, 2 — srcsm measured ~3x slower/epoch on this
#           dataset's other training contrasts, so it must not share a GPU
#           with anything else or it becomes the straggler holding the whole
#           node open)
#
# RUN_IDs are FIXED (persisted per pack dir) so a resumed/extended chain always
# targets the same run directories. Recording happens ONCE per pack dir.
#
# Usage (run ON tamia, after sourcing
#   datasets/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env_t2f.sh
#   scripts/cluster/tamia_env.sh):
#   bash 04_60_tamia_pack_t2f_full_suite.sh                  # fresh launch (new pack dirs)
#   PACK_DIR_A=<existing> PACK_DIR_B=<existing> PACK_DIR_C=<existing> bash 04_60_...  # resume/extend
#   PACK_CHAIN=8 bash 04_60_...                              # tune chain length (all 3 packs)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
PACK_DIR_A="${PACK_DIR_A:-/scratch/p/paulh/brats2024-glioma/_packruns/t2f_suiteA_${TS}}"
PACK_DIR_B="${PACK_DIR_B:-/scratch/p/paulh/brats2024-glioma/_packruns/t2f_suiteB_${TS}}"
PACK_DIR_C="${PACK_DIR_C:-/scratch/p/paulh/brats2024-glioma/_packruns/t2f_suiteC_${TS}}"
mkdir -p "${PACK_DIR_A}" "${PACK_DIR_B}" "${PACK_DIR_C}"
echo "[tamia-pack] PACK_DIR_A=${PACK_DIR_A}"
echo "[tamia-pack] PACK_DIR_B=${PACK_DIR_B}"
echo "[tamia-pack] PACK_DIR_C=${PACK_DIR_C}"

run_id_for() {  # $1 = pack dir, $2 = method suffix for RUN_ID
    local runids="${1}/RUN_IDS.env"
    local var="$2"
    if [ -f "${runids}" ] && grep -q "^${var}=" "${runids}"; then
        grep "^${var}=" "${runids}" | tail -1 | cut -d= -f2-
        return
    fi
    echo "brats2024-glioma_t2f_${3}_${TS}"
}

# ---- Pack A: baseline, auglab_default, synthseg_EM, synthseg_noEM ----
A_BASELINE=$(run_id_for "${PACK_DIR_A}" A_BASELINE baseline)
A_AUGLAB=$(run_id_for "${PACK_DIR_A}" A_AUGLAB auglab_default)
A_SSEM=$(run_id_for "${PACK_DIR_A}" A_SSEM synthseg_EM)
A_SSNOEM=$(run_id_for "${PACK_DIR_A}" A_SSNOEM synthseg_noEM)
printf 'A_BASELINE=%s\nA_AUGLAB=%s\nA_SSEM=%s\nA_SSNOEM=%s\n' \
    "${A_BASELINE}" "${A_AUGLAB}" "${A_SSEM}" "${A_SSNOEM}" > "${PACK_DIR_A}/RUN_IDS.env"
echo "  [A] baseline=${A_BASELINE} auglab_default=${A_AUGLAB} synthseg_EM=${A_SSEM} synthseg_noEM=${A_SSNOEM}"
if [ -f "${PACK_DIR_A}/index.tsv" ]; then
    echo "[tamia-pack] Pack A index.tsv exists — reusing recording."
else
    RUN_JOB_PACK_DIR="${PACK_DIR_A}" bash "${HERE}/04_50_train_t2f_baseline.sh"          "${A_BASELINE}"
    RUN_JOB_PACK_DIR="${PACK_DIR_A}" bash "${HERE}/04_51_train_t2f_auglab_default.sh"    "${A_AUGLAB}"
    RUN_JOB_PACK_DIR="${PACK_DIR_A}" bash "${HERE}/04_52_train_t2f_synthseg_EM.sh"       "${A_SSEM}"
    RUN_JOB_PACK_DIR="${PACK_DIR_A}" bash "${HERE}/04_53_train_t2f_synthseg_noEM.sh"     "${A_SSNOEM}"
fi

# ---- Pack B: OURS DualVal, v26_6_2_train050_val100 (rung5), kmeans (rung2), kmeans_label_remap (rung3) ----
B_DUALVAL=$(run_id_for "${PACK_DIR_B}" B_DUALVAL auglabAug_v26_6_2_train050_val000)
B_RUNG5=$(run_id_for "${PACK_DIR_B}" B_RUNG5 v26_6_2_train050_val100)
B_RUNG2=$(run_id_for "${PACK_DIR_B}" B_RUNG2 baseline_kmeans)
B_RUNG3=$(run_id_for "${PACK_DIR_B}" B_RUNG3 baseline_kmeans_label_remap)
printf 'B_DUALVAL=%s\nB_RUNG5=%s\nB_RUNG2=%s\nB_RUNG3=%s\n' \
    "${B_DUALVAL}" "${B_RUNG5}" "${B_RUNG2}" "${B_RUNG3}" > "${PACK_DIR_B}/RUN_IDS.env"
echo "  [B] OURS-DualVal(val000)=${B_DUALVAL} rung5(v26_6_2_train050_val100)=${B_RUNG5} rung2(kmeans)=${B_RUNG2} rung3(kmeans_label_remap)=${B_RUNG3}"
if [ -f "${PACK_DIR_B}/index.tsv" ]; then
    echo "[tamia-pack] Pack B index.tsv exists — reusing recording."
else
    RUN_JOB_PACK_DIR="${PACK_DIR_B}" bash "${HERE}/04_55_train_t2f_auglabAug_v26_6_2_dualval.sh"        "${B_DUALVAL}"
    RUN_JOB_PACK_DIR="${PACK_DIR_B}" bash "${HERE}/04_56_train_t2f_v26_6_2_train050_val100.sh"           "${B_RUNG5}"
    RUN_JOB_PACK_DIR="${PACK_DIR_B}" bash "${HERE}/04_57_train_t2f_baseline_kmeans.sh"                   "${B_RUNG2}"
    RUN_JOB_PACK_DIR="${PACK_DIR_B}" bash "${HERE}/04_58_train_t2f_baseline_kmeans_label_remap.sh"       "${B_RUNG3}"
fi

# ---- Pack C: kmeans_label_remap_voronoi (rung4, noise fill) + srcsm (own GPU each) ----
C_RUNG4=$(run_id_for "${PACK_DIR_C}" C_RUNG4 baseline_kmeans_label_remap_voronoi)
C_SRCSM=$(run_id_for "${PACK_DIR_C}" C_SRCSM srcsm)
printf 'C_RUNG4=%s\nC_SRCSM=%s\n' "${C_RUNG4}" "${C_SRCSM}" > "${PACK_DIR_C}/RUN_IDS.env"
echo "  [C] rung4(kmeans_label_remap_voronoi, noise)=${C_RUNG4} srcsm=${C_SRCSM}"
if [ -f "${PACK_DIR_C}/index.tsv" ]; then
    echo "[tamia-pack] Pack C index.tsv exists — reusing recording."
else
    RUN_JOB_PACK_DIR="${PACK_DIR_C}" bash "${HERE}/04_59_train_t2f_baseline_kmeans_label_remap_voronoi.sh" "${C_RUNG4}"
    RUN_JOB_PACK_DIR="${PACK_DIR_C}" bash "${HERE}/04_54_train_t2f_srcsm.sh"                                "${C_SRCSM}"
fi

echo "[tamia-pack] recorded folds A:"; cut -f3 "${PACK_DIR_A}/index.tsv" | sed 's/^/  /'
echo "[tamia-pack] recorded folds B:"; cut -f3 "${PACK_DIR_B}/index.tsv" | sed 's/^/  /'
echo "[tamia-pack] recorded folds C:"; cut -f3 "${PACK_DIR_C}/index.tsv" | sed 's/^/  /'

# Submit the three whole-node dependency chains (each can land on a different
# node and run concurrently — they are NOT chained to each other).
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-10}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-brats_t2f_suiteA}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR_A}"

PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-10}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-brats_t2f_suiteB}" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR_B}"

# Pack C: explicit placement — 3 voronoi folds share GPU 3, srcsm's 3 folds get
# GPUs 0/1/2 each to itself (index.tsv row order = voronoi f0,f1,f2, srcsm f0,f1,f2).
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" PACK_CHAIN="${PACK_CHAIN:-10}" \
PACK_USE_MPS="${PACK_USE_MPS:-0}" PACK_JOB_NAME="${PACK_JOB_NAME:-brats_t2f_suiteC}" \
PACK_GPU_MAP="3 3 3 0 1 2" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR_C}"
