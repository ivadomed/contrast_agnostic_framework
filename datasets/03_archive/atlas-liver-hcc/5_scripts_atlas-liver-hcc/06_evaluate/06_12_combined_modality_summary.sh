#!/usr/bin/env bash
# Cross-contrast + cross-dataset aggregation for this task's single training
# modality (T1w), with an inline significance column vs the reference method
# — companion to 06_10 (bare aggregate_from_config.py, no meta-heatmap
# wiring). Same YAML-config pattern as chaos/open-ms/brats's own
# 06_*_combined_modality_summary.sh, consumed by the shared
# datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py so
# this task plugs into scripts/evaluate/meta_task_heatmap.py the same way
# every other task does. See that script's module docstring for the full
# config format and statistical model.
#
# Usage:
#   bash 06_12_combined_modality_summary.sh <config.yaml>
#   bash 06_12_combined_modality_summary.sh configs/atlas-liver-hcc_combined_01_results.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <config.yaml>" >&2
    exit 1
fi

CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] combined-modality aggregation from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/combined_modality_summary.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
