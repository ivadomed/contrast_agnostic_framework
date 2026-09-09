#!/usr/bin/env bash
# Assemble a LABEL-CONSISTENT metrics view: every column scores the SAME structure
# (mandible union mandible+lower_teeth).
#
# WHY: the headline table mixes label sets — its `cbct` column is a 3-class macro while
# both hanseg columns are the single mandible-union class, so `all` averages
# incommensurable quantities and the in-domain->OOD drop conflates "harder modality"
# with "different structures scored". This builds a parallel tree
# 02_metrics/toothfairy2_model/cbct_union/ exposing ONLY the union column, so a config
# over {that tree + hanseg} is consistent end to end. The 3-class tree is untouched and
# remains the characterisation of the full task.
#
#   bash 06_09_build_union_view.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
SRC="${METRICS_ROOT}/${MODEL_TYPE}/cbct"
DST="${METRICS_ROOT}/${MODEL_TYPE}/cbct_union"
n=0
while IFS= read -r f; do
    rel="${f#${SRC}/}"                      # [ablations/]<cat>_<run>/fold<k>/cbct_union_metrics.csv
    out="${DST}/$(dirname "${rel}")"
    mkdir -p "${out}"
    cp -f "${f}" "${out}/cbct_union_metrics.csv"
    run_dir="$(basename "$(dirname "$(dirname "${rel}")")")"
    # strip the category prefix to recover the run id
    run_id="${run_dir#nnUNet_}"; run_id="${run_id#auglab_}"
    fold="$(basename "$(dirname "${rel}")")"; fold="${fold#fold}"
    .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py \
        "${out}" "${run_id}" "${fold}" --groups cbct_union --group-col contrast \
        --groups-word Contrasts > /dev/null
    n=$((n+1))
done < <(find "${SRC}" -name cbct_union_metrics.csv)
echo "union view: ${n} fold dirs -> ${DST}"
