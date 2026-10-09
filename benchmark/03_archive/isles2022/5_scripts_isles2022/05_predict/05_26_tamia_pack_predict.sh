#!/usr/bin/env bash
# TamIA whole-node PREDICT pack for one isles2022 training contrast: records every roster method's 3 fold-predict commands
# (RUN_JOB_PACK_DIR mode) via the shared run_all_predict_common.sh, verifies the recorded commands, then submits ONE node job
# (PACK_GPU_MAP-free round-robin; predict is minutes per fold). Run ON TamIA after env + cluster override are sourced:
#   source benchmark/02_tasks/brain_stroke/isles2022/5_scripts_isles2022/00_utils/env.sh   # or env_flair.sh
#   source scripts/cluster/tamia_env_isles2022.sh
#   RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G bash 05_26_tamia_pack_predict.sh dwi|flair
# PACK_DIR is ALWAYS contrast-qualified: sharing one across contrasts overwrites same-named cmd files silently (hit 3x on
# this project). The grep-verify below fails the launch if any recorded command references the other contrast.
set -euo pipefail
C="${1:?usage: $0 dwi|flair}"
case "$C" in dwi|flair) ;; *) echo "contrast must be dwi|flair" >&2; exit 1;; esac
[ "${TRAINING_CONTRAST:-}" = "$C" ] || { echo "ERROR: TRAINING_CONTRAST='${TRAINING_CONTRAST:-}' != '$C' -- source the matching env (env.sh for dwi, env_flair.sh for flair) THEN tamia_env_isles2022.sh" >&2; exit 1; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_JOB_PACK_DIR="${SCRATCH:?}/isles2022/_packruns/predict_${C}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_JOB_PACK_DIR}"
METHOD_SCRIPTS=( "${HERE}"/05_*_predict_${C}_*.sh )
other="flair"; [ "$C" = "flair" ] && other="dwi"
source_launcher() { source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_common.sh"; }
ROSTER_PACK_SUBMIT=0 source_launcher
n=$(grep -c . "${RUN_JOB_PACK_DIR}/index.tsv"); echo "[pack] recorded ${n} fold commands in ${RUN_JOB_PACK_DIR}"
bad=$(grep -l "/isles2022_model/${other}/" "${RUN_JOB_PACK_DIR}"/*.sh 2>/dev/null || true)
[ -z "${bad}" ] || { echo "ERROR: recorded cmds reference the ${other} models: ${bad}" >&2; exit 1; }
miss=$(grep -L "/isles2022_model/${C}/" "${RUN_JOB_PACK_DIR}"/*.sh 2>/dev/null | grep -v "index" || true)
[ -z "${miss}" ] || echo "[pack] WARN: cmd files without the ${C} model path (check): ${miss}"
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-02:00:00}" PACK_CHAIN=1 PACK_JOB_NAME="isles2022_predict_${C}" \
    bash "${PROJECT_ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${RUN_JOB_PACK_DIR}"
