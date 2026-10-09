#!/usr/bin/env bash
# ispy1's own combined table: ISPY2 t1wce-trained + t2w-trained models pooled,
# both ispy1 test items as columns (same pattern as duke-breast-mri's 06_04).
#   bash 06_04_combined_modality_summary.sh [config.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CONFIG="${1:-${HERE}/configs/ispy1_combined_01_results.yaml}"
[[ "$CONFIG" != /* ]] && CONFIG="${HERE}/${CONFIG}"
echo "[$(date '+%H:%M:%S')] combined-modality aggregation from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
