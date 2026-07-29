#!/usr/bin/env bash
# Paired significance testing from a YAML config (same config as 06_10).
#   bash 06_11_significance_from_config.sh configs/brats-ssa_t1n_01_results.yaml --ref <exact run id>
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ $# -lt 1 ]; then echo "Usage: $0 <config.yaml> [--ref <exact run id>] [--metric dice]" >&2; exit 1; fi
CONFIG="$1"; shift
if [[ "$CONFIG" != /* ]]; then CONFIG="${HERE}/${CONFIG}"; fi
echo "[$(date '+%H:%M:%S')] significance testing from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py" "${CONFIG}" "$@"
echo "[$(date '+%H:%M:%S')] done"
