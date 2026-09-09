#!/usr/bin/env bash
# Score every run's OWN held-out CBCT test set on the MANDIBLE UNION, so the
# cross-modality table compares the same structure in every column (see the .py header:
# the previous table averaged a 3-class in-domain score with a 1-class OOD score).
#
# Does NOT replace the 3-class in-domain table — this is an additional, label-consistent
# view. CPU-only; dispatched through run_job.
#
#   bash 06_08_eval_mandible_union.sh <SUITE_A_PACK> <SUITE_B_PACK> <LADDER_PACK>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
A="${1:?need suiteA pack}"; B="${2:?need suiteB pack}"; L="${3:?need ladder pack}"
get() { grep -E "^$2=" "$1/RUN_IDS.env" | head -1 | cut -d= -f2-; }
OURS0=$(get "$B" OURS_RUN_ID); OURS1="${OURS0/_val000_/_val100_}"

# run:category:subdir  — suiteA supplies the first three, suiteB the next three
ROWS=(
 "$(get "$A" BASELINE_RUN_ID):nnUNet:"
 "$(get "$A" AUGLAB_DEFAULT_RUN_ID):auglab:"
 "$(get "$A" SYNTHSEG_NOEM_RUN_ID):auglab:"
 "$(get "$B" SYNTHSEG_EM_RUN_ID):auglab:"
 "$(get "$B" SRCSM_RUN_ID):auglab:"
 "${OURS0}:auglab:"
 "${OURS1}:auglab:"
 "$(get "$L" R2_RUN_ID):auglab:ablations"
 "$(get "$L" R3_RUN_ID):auglab:ablations"
 "$(get "$L" R4_RUN_ID):auglab:ablations"
 "$(get "$L" R5_RUN_ID):nnUNet:ablations"
)
LOG_DIR="${HERE}/logs"; mkdir -p "${LOG_DIR}"
CMD=""
for row in "${ROWS[@]}"; do
    IFS=: read -r RID CAT SUB <<< "${row}"
    [ -n "${RID}" ] || continue
    CMD+="echo '--- ${RID} (${CAT}${SUB:+,${SUB}})'; .venv/bin/python '${HERE}/06_08_eval_mandible_union.py' --run '${RID}' --category '${CAT}' ${SUB:+--subdir ${SUB}}; "
done

run_job --name "tf2_union_eval" --gpus 0 --cpus 8 --mem 48G --time 03:00:00 \
    --slot 0 --wait --log "${LOG_DIR}/union_eval_$(date +%Y%m%d_%H%M%S).log" -- bash -c "
        export PREDICTIONS_ROOT='${PREDICTIONS_ROOT}' METRICS_ROOT='${METRICS_ROOT}'
        export nnUNet_raw='${nnUNet_raw}' NNUNET_DATASET_ID='${NNUNET_DATASET_ID}'
        cd '${PROJECT_ROOT}'
        ${CMD}
    "
