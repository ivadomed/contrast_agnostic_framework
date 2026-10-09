#!/usr/bin/env bash
# duke-breast-mri's own combined table: pools ISPY2's two training-modality
# models (t1wce-trained + t2w-trained) as scored on duke's own held-out test
# set into one 6-fold-mean-per-method table. Mirrors every other dataset's
# 06_04_combined_modality_summary.sh pattern (chaos/open-ms/brats2024-glioma/
# ispy2) -- this config already existed (duke_combined_01_results.yaml) but
# had no wrapper invoking it until the 2026-09-27 06_evaluate validator
# extension flagged it as an orphan.
#
# Usage:
#   bash 06_04_combined_modality_summary.sh [configs/duke_combined_01_results.yaml]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

CONFIG="${1:-${HERE}/configs/duke_combined_01_results.yaml}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] combined-modality aggregation from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py" "${CONFIG}"
echo "[$(date '+%H:%M:%S')] done"
