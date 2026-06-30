#!/usr/bin/env bash
# Aggregate TRUSTED chaos-model evaluation results into comparison tables + heatmaps.
# Run after 06_02_evaluate_all_chaos.sh (or 06_04/06_05) has completed.
#
# TRUSTED needs no dataset-specific aggregator: the shared, label-agnostic core at
# datasets/00_commun_scripts/00_03_evaluate/aggregate_results.py reads each run's
# fold{k}/eval_all.csv (kidney label; ct/us groups) and writes the standard
# 02_00_aggregated_metrics.md / 02_01_summary_by_modality.md / heatmaps under the
# metrics dir. We just point it at chaos_model/<contrast> and pass a title.
#
# Usage:
#   bash 06_03_aggregate_results.sh                 # current CHAOS_TRAINING_CONTRAST (default t1in)
#   CHAOS_TRAINING_CONTRAST=t2spir bash 06_03_aggregate_results.sh
#   bash 06_03_aggregate_results.sh --run_keys nnUNet_chaos_t1in_baseline_20260614_153230 ...
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

_AGG_DIR="${METRICS_ROOT}/${CHAOS_MODEL_TYPE}/${CHAOS_TRAINING_CONTRAST}"
[ -d "${_AGG_DIR}" ] || { echo "ERROR: no metrics at ${_AGG_DIR} — run 06_02/06_04/06_05 first." >&2; exit 1; }

.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/aggregate_results.py" \
    --metrics_dir "${_AGG_DIR}" \
    --title "TRUSTED CT+US — kidney (chaos ${CHAOS_TRAINING_CONTRAST} models)" "$@"
