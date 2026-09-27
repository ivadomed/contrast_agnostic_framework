#!/usr/bin/env bash
# One-shot sync of the dwi_ap suite's metrics (+ checkpoints, optionally) from TamIA
# scratch back to Vulcan, once the eval pack (job 475404, chain launched 2026-09-21)
# has finished. Run this ON VULCAN.
#
# Predictions stay on TamIA scratch (large, purge-on-inactivity, not needed here) —
# only 02_metrics/ (small: eval_all.csv + summaries) is pulled, matching how every
# other TamIA-trained arm on this dataset was synced back.
#
# Usage:
#   bash 06_13_sync_dwi_ap_from_tamia.sh            # metrics only (default, cheap)
#   SYNC_CHECKPOINTS=1 bash 06_13_sync_dwi_ap_from_tamia.sh   # + final checkpoints (large)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

REMOTE_METRICS="/scratch/p/paulh/on-harmony/8_results/02_metrics/on_harmony_model/dwi_ap"
LOCAL_METRICS="${METRICS_ROOT}/on_harmony_model/dwi_ap"
mkdir -p "${LOCAL_METRICS}"

echo "[$(date '+%H:%M:%S')] checking eval completion on TamIA…"
N_EXPECT=33
N_DONE="$(ssh -o BatchMode=yes tamia.alliancecan.ca \
    "find '${REMOTE_METRICS}' -name eval_all.csv 2>/dev/null | wc -l")"
echo "  eval_all.csv found: ${N_DONE} / ${N_EXPECT} expected worker units"
if [ "${N_DONE}" -lt "${N_EXPECT}" ]; then
    echo "  NOT all eval units have produced output yet — syncing what exists, but do" >&2
    echo "  not aggregate/ladder from this until the count reaches ${N_EXPECT}." >&2
fi

echo "[$(date '+%H:%M:%S')] rsync metrics: ${REMOTE_METRICS} -> ${LOCAL_METRICS}"
rsync -avz --progress "tamia.alliancecan.ca:${REMOTE_METRICS}/" "${LOCAL_METRICS}/"

if [ "${SYNC_CHECKPOINTS:-0}" = "1" ]; then
    REMOTE_PRED="/scratch/p/paulh/on-harmony/8_results/01_predictions/on_harmony_model/dwi_ap"
    LOCAL_PRED="${PREDICTIONS_ROOT}/on_harmony_model/dwi_ap"
    mkdir -p "${LOCAL_PRED}"
    echo "[$(date '+%H:%M:%S')] rsync checkpoint_final.pth only (predictions stay on TamIA)…"
    rsync -avz --include='*/' --include='checkpoint_final.pth' --include='*.json' --exclude='*' \
        "tamia.alliancecan.ca:${REMOTE_PRED}/" "${LOCAL_PRED}/"
fi

echo "[$(date '+%H:%M:%S')] done. Next steps once ${N_EXPECT}/${N_EXPECT} confirmed:"
echo "  bash 06_06_aggregate_from_config.sh configs/on-harmony_dwi_ap.yaml"
echo "  bash 06_09_combined_modality_summary.sh configs/on-harmony_combined_01_results.yaml"
echo "  .venv/bin/python 06_12_ladder_summary_dwi_ap.py"
echo "  .venv/bin/python 06_12_ladder_summary_dwi_ap.py --restricted"
