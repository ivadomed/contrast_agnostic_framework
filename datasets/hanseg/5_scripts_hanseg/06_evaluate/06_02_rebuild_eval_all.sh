#!/usr/bin/env bash
# Rebuild every hanseg fold's eval_all.csv from ALL per-item *_metrics.csv present.
#
# WHY THIS EXISTS: summarize_fold.py regenerates eval_all.csv from the items named in
# --groups. Evaluating a single item (e.g. adding the mrt1 arm on its own) and passing
# only that item silently DROPS the other item's rows from eval_all.csv — the per-item
# CSVs survive untouched, but every downstream table reads eval_all.csv, so the ladder
# and the aggregates then reflect one modality while appearing complete. Rebuilding from
# whatever *_metrics.csv are actually on disk is idempotent and always correct.
#
#   bash 06_02_rebuild_eval_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
ROOT="${METRICS_ROOT}/${TF2_MODEL_TYPE}/${TF2_TRAINING_CONTRAST}"
n=0
while IFS= read -r fold; do
    items=$(ls "${fold}" | sed -n 's/^\(.*\)_metrics\.csv$/\1/p' | sort | tr '\n' ' ')
    [ -n "${items}" ] || continue
    run_dir="$(basename "$(dirname "${fold}")")"
    run_id="${run_dir#nnUNet_}"; run_id="${run_id#auglab_}"
    f="$(basename "${fold}")"; f="${f#fold}"
    .venv/bin/python datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py \
        "${fold}" "${run_id}" "${f}" --groups ${items} --group-col contrast \
        --groups-word Contrasts > /dev/null
    n=$((n+1))
done < <(find "${ROOT}" -type d -name "fold[0-9]")
echo "rebuilt eval_all.csv in ${n} fold dirs"
