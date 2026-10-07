#!/usr/bin/env bash
# ONE-OFF MIGRATION (2026-10-06), run once per machine (Vulcan done 2026-10-06; TamIA: run there before predicting
# on-harmony again). on-harmony used to predict checkpoint_final into the FLAT fold{k}/<contrast>/ layout (old inline
# script); the shared convention -- and on-harmony's default since 2026-10-06 -- is checkpoint_best flat and every
# other checkpoint under fold{k}/<tag>/<contrast>/. This moves each run's flat (final) contrast dirs to
# fold{k}/final/<contrast>/ so a checkpoint_best prediction can take the flat slot without overwriting them.
# Predictions only: metrics are not touched. Refuses to overwrite an existing final/<contrast>.
#   bash 05_90_migrate_flat_final_predictions.sh [--dry-run]
# On TamIA: CLUSTER_ENV=scripts/cluster/tamia_env_onharmony.sh bash 05_90_migrate_flat_final_predictions.sh  (scratch PREDICTIONS_ROOT)
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
[ -n "${CLUSTER_ENV:-}" ] && source "${PROJECT_ROOT}/${CLUSTER_ENV}"   # cluster path override (sourced after env.sh)
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}"
n=0; skip=0
for fold in "${BASE}"/{T1w,T2w,dwi_ap}/{nnUNet,auglab}/*/fold[0-9]; do
    [ -d "${fold}" ] || continue
    for c in T1w T2w bold dwi_ap epi_ap gre_echo1_mag; do
        [ -d "${fold}/${c}" ] || continue
        if [ -e "${fold}/final/${c}" ]; then
            echo "SKIP (final/${c} exists): ${fold}" >&2; skip=$((skip+1)); continue
        fi
        if [ "${DRY}" = 1 ]; then echo "would mv ${fold}/${c} -> ${fold}/final/${c}"
        else mkdir -p "${fold}/final" && mv "${fold}/${c}" "${fold}/final/${c}"; fi
        n=$((n+1))
    done
done
echo "[migrate] ${n} flat contrast dirs $([ "${DRY}" = 1 ] && echo 'to move' || echo 'moved') to fold*/final/, ${skip} skipped (base ${BASE})"
[ "${skip}" = 0 ]
