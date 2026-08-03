#!/usr/bin/env bash
# Cross-training-modality aggregation (flair + t1w pooled) with an inline
# significance column vs the reference method — companion to 06_10 (which
# aggregates ONE training modality at a time). Same YAML-config pattern, but
# consumed by the shared datasets/00_commun_scripts/00_03_evaluate/
# combined_modality_summary.py instead of aggregate_from_config.py. See that
# script's module docstring for the full config format and statistical model.
#
# Usage:
#   bash 06_17_combined_modality_summary.sh <config.yaml>
#   bash 06_17_combined_modality_summary.sh configs/picai-prostate_combined_01_results.yaml
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
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
