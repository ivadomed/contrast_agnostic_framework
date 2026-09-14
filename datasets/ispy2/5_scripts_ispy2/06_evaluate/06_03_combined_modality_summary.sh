#!/usr/bin/env bash
# Cross-training-modality (t1wce + t2w pooled) aggregation for ispy2's OWN-model
# headline suite, with an inline significance column vs the reference method.
# Mirrors every other dataset's 06_1X_combined_modality_summary.sh pattern
# (chaos/open-ms/brats2024-glioma/ambl).
#
# Usage:
#   bash 06_03_combined_modality_summary.sh <config.yaml>
#   bash 06_03_combined_modality_summary.sh configs/ispy2_combined_01_results.yaml
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
