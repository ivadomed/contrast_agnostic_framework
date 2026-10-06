#!/usr/bin/env bash
# Roster library (bash): the single place that knows how a dataset's trained RUN_IDs are found, so no
# per-dataset launcher/config hardcodes a timestamp. SOURCE it (after the dataset's env.sh); do not run it.
#
# Source of truth for "which methods exist" = the dataset's 05_predict/05_XX_predict_<...>.sh wrappers, each of which
# declares literal  METHOD="..."  and  CATEGORY="nnUNet|auglab"  lines (parsed here with grep, never executed).
# A training RUN_ID is  ${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_<YYYYMMDD_HHMMSS>  (train_common.sh), living under
#   ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${CATEGORY}/<RUN_ID>/
# Resolution order for a method: (1) a pin file line  "<METHOD>\t<CATEGORY>\t<RUN_ID>"  in roster_run_ids.tsv
# (written by run_all_predict_common.sh on first resolve, hand-editable to pin an older/other run);
# (2) otherwise the NEWEST matching run dir (timestamp sorts lexicographically).
#
# Methods whose metrics go in the non-headline 'ablations/' subdir (causal-ablation ladder rungs 2-5):
ROSTER_ABLATION_METHODS="baseline_kmeans baseline_kmeans_label_remap baseline_kmeans_label_remap_voronoi v26_6_2_train050_val000"

roster_pin_file() { echo "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/roster_run_ids.tsv"; }

# roster_parse_wrapper <wrapper.sh>  -> sets W_METHOD / W_CATEGORY (fails loudly if either is missing)
roster_parse_wrapper() {
    local f="$1"
    W_METHOD="$(grep -m1 -E '^METHOD="' "$f" | sed -E 's/^METHOD="([^"]*)".*/\1/')"
    W_CATEGORY="$(grep -m1 -E '^CATEGORY="' "$f" | sed -E 's/^CATEGORY="([^"]*)".*/\1/')"
    [ -n "${W_METHOD}" ] && [ -n "${W_CATEGORY}" ] || { echo "roster: no literal METHOD=/CATEGORY= line in $f" >&2; return 1; }
}

# roster_is_ablation <METHOD> -> 0 if its metrics belong under ablations/
roster_is_ablation() { local m; for m in ${ROSTER_ABLATION_METHODS}; do [ "$m" = "$1" ] && return 0; done; return 1; }

# roster_resolve <METHOD> <CATEGORY>  -> prints RUN_ID (pin file first, else newest dir); returns 1 if none
roster_resolve() {
    local method="$1" category="$2" pin rid base
    pin="$(roster_pin_file)"
    if [ -f "$pin" ]; then
        rid="$(awk -F'\t' -v m="$method" -v c="$category" '$1==m && $2==c {print $3}' "$pin" | tail -1)"
        [ -n "$rid" ] && { echo "$rid"; return 0; }
    fi
    base="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/${category}"
    # [0-9]* after the method forces a timestamp next, so METHOD=baseline never matches baseline_kmeans_<TS>
    rid="$(ls -1d "${base}/${DATASET_NAME}_${TRAINING_CONTRAST}_${method}_"[0-9]*_[0-9]* 2>/dev/null | xargs -r -n1 basename | sort | tail -1)"
    [ -n "$rid" ] && { echo "$rid"; return 0; }
    return 1
}

# roster_pin <METHOD> <CATEGORY> <RUN_ID>  -> record in the pin file (replacing any earlier line for that method+category)
roster_pin() {
    local pin; pin="$(roster_pin_file)"; mkdir -p "$(dirname "$pin")"; touch "$pin"
    awk -F'\t' -v m="$1" -v c="$2" '!($1==m && $2==c)' "$pin" > "${pin}.tmp" && mv "${pin}.tmp" "$pin"
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$pin"
}

# ── Cross-dataset (EVAL-ONLY COMPANION) helpers ─────────────────────────────────────────────────────────────────────
# A companion dataset predicts with ANOTHER task's trained models. Its env.sh exports the source block <PREFIX>_{DATASET_ROOT,PREDICTIONS_ROOT,
# NNUNET_RAW,NNUNET_PREPROCESSED,DATASET_ID,MODEL_TYPE,TRAINING_CONTRAST,DATASET_JSON} (see predict_common.sh cross mode). The roster is the SOURCE's
# pin file; the trainer class and dataset id are read from the SOURCE RUN DIR itself, so no per-method wrapper and no trainer-name mapping is needed.
roster_src() { local n="${1}_${2}"; echo "${!n}"; }                  # roster_src ISPY2 PREDICTIONS_ROOT
roster_src_pin_file() { echo "$(roster_src "$1" PREDICTIONS_ROOT)/$(roster_src "$1" MODEL_TYPE)/$2/roster_run_ids.tsv"; }   # <PREFIX> <training contrast>
roster_src_run_dir() { echo "$(roster_src "$1" PREDICTIONS_ROOT)/$(roster_src "$1" MODEL_TYPE)/$2/$3/$4"; }                   # <PREFIX> <contrast> <category> <run id>
# roster_run_trainer_id <run dir> -> sets RUN_TRAINER (e.g. nnUNetTrainerISPY2Baseline) and RUN_DATASET_ID (e.g. 100); returns 1 if no trained model there
roster_run_trainer_id() {
    local td; td="$(ls -d "$1"/Dataset*/*__nnUNetPlans__3d_fullres 2>/dev/null | head -1)"
    [ -n "$td" ] || return 1
    RUN_TRAINER="$(basename "$td" | sed 's/__nnUNetPlans__3d_fullres$//')"
    RUN_DATASET_ID="$(basename "$(dirname "$td")" | sed -E 's/^Dataset0*([0-9]+)_.*/\1/')"
}
