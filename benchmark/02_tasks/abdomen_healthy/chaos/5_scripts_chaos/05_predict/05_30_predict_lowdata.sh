#!/usr/bin/env bash
# Low-data benchmark — predict ONE trained low-data run across the CHAOS test
# modalities. Method-agnostic: CATEGORY (nnUNet|auglab) and the *LowData TRAINER are
# discovered from the run's own output dir, so this works for every method.
#
# Usage:
#   bash 05_30_predict_lowdata.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"

RUN_ID="${1:?RUN_ID required}"

_base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}"
CATEGORY=""
for _c in nnUNet auglab; do
    [ -d "${_base}/${_c}/${RUN_ID}" ] && CATEGORY="${_c}"
done
[ -z "${CATEGORY}" ] && { echo "ERROR: run '${RUN_ID}' not found under ${_base}/{nnUNet,auglab}/" >&2; exit 1; }

# Trainer dir name → trainer class (…/<RUN_ID>/<DS>/<TRAINER>__nnUNetPlans__3d_fullres).
_tdir="$(ls -d "${_base}/${CATEGORY}/${RUN_ID}"/*/*__nnUNetPlans__3d_fullres 2>/dev/null | head -1 || true)"
[ -z "${_tdir}" ] && { echo "ERROR: no trainer dir under ${_base}/${CATEGORY}/${RUN_ID}/" >&2; exit 1; }
TRAINER="$(basename "${_tdir}" | sed 's/__nnUNetPlans__3d_fullres$//')"

METHOD="lowdata"   # informational (job naming only); identity is the RUN_ID
echo "[lowdata-predict] RUN_ID=${RUN_ID} CATEGORY=${CATEGORY} TRAINER=${TRAINER}"

source "$(dirname "$0")/05_01_predict_common.sh" "$@"
