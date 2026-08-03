#!/usr/bin/env bash
# Paired significance testing from a YAML config (same config as 06_10).
#
# Usage:
#   bash 06_11_significance_from_config.sh <config.yaml> [--ref <exact run id>] [--metric dice]
#   bash 06_11_significance_from_config.sh configs/picai-prostate_flair_01_results.yaml \
#       --ref picai-prostate_flair_auglabAug_v26_6_2_train025_val100_20260706_061243
#
# --ref: pass the EXACT reference run id explicitly whenever the run list also
# contains ablation variants that extend the same run id (e.g. "..._train050_val000")
# — those share the "auglabAug"+"v26_6_2" substring the auto-picker looks for, so
# auto-detection can silently pick the wrong run. See
# datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py for the
# full statistical model (paired Wilcoxon signed-rank on per-case Dice, folds
# capped to 0-2 — see eval_folds.py).
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
[ -f "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh" ] && [ -d /scratch/p/paulh ] && \
    source "${PROJECT_ROOT}/scripts/cluster/tamia_env_picai.sh"
cd "${PROJECT_ROOT}"

HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [--ref <exact run id>] [--metric dice]" >&2
    exit 1
fi

CONFIG="$1"; shift
if [[ "$CONFIG" != /* ]]; then
    CONFIG="${HERE}/${CONFIG}"
fi

echo "[$(date '+%H:%M:%S')] significance testing from ${CONFIG}"
.venv/bin/python "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py" \
    "${CONFIG}" "$@"
echo "[$(date '+%H:%M:%S')] done"
