#!/usr/bin/env bash
# Paired significance testing from a YAML config (same config as 06_10).
#
# Usage:
#   bash 06_11_significance_from_config.sh <config.yaml> [--ref <exact run id>] [--metric dice]
#   bash 06_11_significance_from_config.sh configs/chaos_t1in_02_ablation_auglab_val100.yaml \
#       --ref chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514
#
# --ref: pass the EXACT reference run id explicitly whenever the run list also
# contains ablation variants that extend the same run id (e.g. "..._train025_val000")
# — those share the "auglabAug"+"v26_6_2" substring the auto-picker looks for, so
# auto-detection can silently pick the wrong run. See
# datasets/00_commun_scripts/00_03_evaluate/significance_from_config.py for the
# full statistical model (paired Wilcoxon signed-rank on per-case Dice, folds
# capped to 0-2 — see eval_folds.py).
set -euo pipefail
# HERE must be resolved BEFORE the cd below and via BASH_SOURCE (not $0): once cwd
# changes to PROJECT_ROOT, `dirname "$0"` for a script invoked with a bare/relative
# name (exactly this file's own documented usage, e.g. `bash 06_11_....sh configs/x`
# from inside 06_evaluate/) collapses to "." relative to the NEW cwd, silently
# resolving HERE to PROJECT_ROOT instead of this script's directory and breaking
# every relative CONFIG path. Same bug class as 06_10_aggregate_from_config.sh
# (see that script's comment) -- hit here directly 2026-08-29 while regenerating
# the chaos cross-dataset configs after removing msd-spleen.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

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
