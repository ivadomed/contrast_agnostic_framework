#!/usr/bin/env bash
# Evaluate ONE isles2022 prediction run: every fold x every test item (dwi/flair), scoring the `lesion` label against
# labelsTs_<item>. Thin shim over the shared driver benchmark/00_commun_scripts/00_03_evaluate/evaluate_run_common.sh.
#   bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD=all]
# Which training contrast the run belongs to comes from the environment: TRAINING_CONTRAST=flair (default dwi), exactly as the
# train/predict wrappers do (the run-all launchers export it via env_flair.sh). Optional env: CKPT_TAG, METRICS_SUBDIR, EVAL_TIME.
# Writes 8_results_isles2022/02_metrics/isles2022_model/<contrast>[/<METRICS_SUBDIR>]/<CATEGORY>_<RUN_ID>/fold{F}/eval_all.csv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

EVAL_ITEMS="dwi flair"
EVAL_LABELS="lesion"
EVAL_JOB_PREFIX="isles2022_eval"
case "${TRAINING_CONTRAST}" in
    dwi)   EVAL_DATASET_ID="${DATASET_ID_DWI}" ;;
    flair) EVAL_DATASET_ID="${DATASET_ID_FLAIR}" ;;
    *) echo "ERROR: TRAINING_CONTRAST must be dwi|flair, got '${TRAINING_CONTRAST}'" >&2; exit 1 ;;
esac
# (GT masks are byte-identical across Dataset140/141 -- same conversion -- so the id only affects paths/provenance.)

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_run_common.sh" "$@"
