#!/bin/bash
# Run the chaos t2spir rung-5 retrain's predict+eval NOW (it finishes long before brats/on-harmony), as
#   (a) chaos' OWN test set (standard 05_22 predict + 06_01 evaluate, via the rung5 predict/eval job bodies), and
#   (b) the cross-dataset AMOS (ct, mri) + SLIVER07 (ct) evaluation, CROP-BEFORE-PREDICT to the CHAOS FOV slab
#       (the standard since 2026-10-03: metrics in <contrast>/fov_crop/ablations/nnUNet_<run>, what the t2spir ladder
#       06_35 reads). Same per-dataset settings and the SAME saved crops (amos _fovcrop/391947, sliver07 _fovcrop/391949)
#       as chaos/06_evaluate/06_36_fov_crop_ladder_rungs.sh, so every rung sees byte-identical inputs.
# Driver = a PINNED SNAPSHOT of the RUNS_FILE-capable fov_crop_predict_evaluate.sh (the repo copy on TamIA predates it).
# Modes (ROOT exported):  gpu = own predict, then cross-dataset crop predict;  cpu = own eval, then cross-dataset eval + merge.
set -uo pipefail
MODE="${1:?gpu|cpu}"; : "${ROOT:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
source "${ROOT}/RUN_IDS.env"
export SKIP="onharmony brats"
FOVDIR="${ROOT}/fovcrop"
export RUNS_FILE="${FOVDIR}/roster_t2spir_rung5.txt" CONTRASTS="t2spir" SKIP_REPORT=1
DRIVER="${FOVDIR}/fov_crop_predict_evaluate.sh"
[ -f "${DRIVER}" ] && [ -f "${RUNS_FILE}" ] || { echo "[chaos-now] missing ${DRIVER} or ${RUNS_FILE}"; exit 1; }
case "${MODE}" in gpu) own=tamia_rung5_predict_job.sh; export PHASES="2";; cpu) own=tamia_rung5_eval_job.sh; export PHASES="3 4";; *) exit 2;; esac
rc=0
echo "[chaos-now] mode=${MODE} run=${CHAOS_RUN} roster=$(grep -v '^#' "${RUNS_FILE}")"
bash "scripts/cluster/${own}" || rc=1
DATASET=amos ITEMS="ct mri" ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos EVAL_PARALLEL=6 \
  CROP_REUSE_DIR="${SCRATCH}/amos/_fovcrop/391947" bash "${DRIVER}" || rc=1
DATASET=sliver07 ITEMS="ct" ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver EVAL_PARALLEL=8 \
  CROP_REUSE_DIR="${SCRATCH}/sliver07/_fovcrop/391949" bash "${DRIVER}" || rc=1
echo "[chaos-now] ${MODE} finished rc=${rc}"; exit ${rc}
