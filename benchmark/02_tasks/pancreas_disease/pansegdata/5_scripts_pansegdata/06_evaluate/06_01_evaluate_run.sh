#!/usr/bin/env bash
# Evaluate ONE pansegdata prediction run: every fold x every test item (t1wce/t2w), scoring the `pancreas` label against
# labelsTs_<item>. Thin shim over the shared driver benchmark/00_commun_scripts/00_03_evaluate/evaluate_run_common.sh.
#   bash 06_01_evaluate_run.sh <RUN_ID> <CATEGORY:nnUNet|auglab> [FOLD=all]
# Which training contrast the run belongs to comes from the environment: TRAINING_CONTRAST=t2w (default t1wce), exactly as the
# train/predict wrappers do (the run-all launchers export it via env_t2w.sh). Optional env: CKPT_TAG, METRICS_SUBDIR, EVAL_TIME.
# Writes 8_results_pansegdata/02_metrics/pansegdata_model/<contrast>[/<METRICS_SUBDIR>]/<CATEGORY>_<RUN_ID>/fold{F}/eval_all.csv.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../00_utils/env.sh"

EVAL_ITEMS="t1wce t2w"
EVAL_LABELS="pancreas"
EVAL_JOB_PREFIX="pansegdata_eval"
case "${TRAINING_CONTRAST}" in
    t1wce)   EVAL_DATASET_ID="${DATASET_ID_T1WCE}" ;;
    t2w) EVAL_DATASET_ID="${DATASET_ID_T2W}" ;;
    *) echo "ERROR: TRAINING_CONTRAST must be t1wce|t2w, got '${TRAINING_CONTRAST}'" >&2; exit 1 ;;
esac
# (GT masks are byte-identical across Dataset150/151 -- same conversion -- so the id only affects paths/provenance.)

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_run_common.sh" "$@"
