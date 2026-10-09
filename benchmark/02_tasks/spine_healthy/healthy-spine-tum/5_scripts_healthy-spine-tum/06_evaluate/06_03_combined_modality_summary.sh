#!/usr/bin/env bash
# Cross-training-modality aggregation (ct + inphase pooled) with an inline
# significance column vs the reference method — companion to 06_01 (which
# aggregates ONE training modality at a time). Consumed by the shared
# benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py
# instead of aggregate_from_config.py.
#   bash 06_03_combined_modality_summary.sh configs/healthy-spine-tum_combined_01_results.yaml
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
CFG="${1:?usage: 06_03_combined_modality_summary.sh <configs/*.yaml>}"
[[ "$CFG" != /* ]] && CFG="${HERE}/${CFG}"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/combined_modality_summary.py "${CFG}"
