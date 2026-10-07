#!/usr/bin/env bash
# TamIA whole-node predict pack for totalsegmri-pancreas (all source contrasts in ONE pack: the source RUN_IDs name their contrast, so cmd files cannot collide; grep-verified).
# Run ON TamIA after:  source .../00_utils/env.sh ; source scripts/cluster/tamia_env_totalsegmri-pancreas.sh ; export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_JOB_PACK_DIR="${SCRATCH:?}/totalsegmri-pancreas/_packruns/predict_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${RUN_JOB_PACK_DIR}"
export SOURCE_PREFIX="PANSEG" SOURCE_CONTRASTS="t1wce t2w" PREDICT_SHIM="${HERE}/05_01_predict_pansegdata_common.sh"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/run_all_predict_cross_common.sh"
n=$(grep -c . "${RUN_JOB_PACK_DIR}/index.tsv"); echo "[pack] recorded ${n} fold-predict commands"
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-02:00:00}" PACK_CHAIN=1 PACK_JOB_NAME="totalsegmri-pancreas_predict" \
    bash "${PROJECT_ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${RUN_JOB_PACK_DIR}"
