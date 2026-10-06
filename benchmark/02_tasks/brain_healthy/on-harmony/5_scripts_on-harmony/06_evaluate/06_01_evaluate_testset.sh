#!/usr/bin/env bash
# Evaluate a trained on-harmony model on the cross-contrast test set, writing into the
# STANDARD results layout (identical to chaos/brats):
#   predictions → PREDICTIONS_ROOT/<model>/<train_contrast>/<category>/<RUN_ID>/fold{k}/<test_contrast>/
#   metrics     → METRICS_ROOT/<model>/<train_contrast>/<category>_<RUN_ID>/fold{k}/eval_all.csv
# Aggregate across runs with the SHARED benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py via
# 06_02_aggregate_from_config.sh — exactly the same as chaos.
#
# PREREQUISITE (2026-09-27 refactor): predictions must already exist, produced by
# 05_predict/05_00_build_test_inputs.sh (once) + the relevant 05_XX_predict_*.sh wrapper
# for this RUN_ID. This script used to call nnUNetv2_predict inline; that logic moved to
# 05_predict/ so on-harmony follows the same 3-tier predict pattern as every other
# dataset (see 05_predict/05_01_predict_common.sh's header for the one on-harmony-only
# post-step it does — resampling RAS predictions back to native — and the checkpoint
# path-layout note: checkpoint_final predictions now land at fold{k}/final/<contrast>/,
# not the old flat fold{k}/<contrast>/, since that's the shared driver's convention).
#
# TWO MODES:
#   LAUNCHER:  bash 06_01_evaluate_testset.sh <RUN_ID>
#       Fans out ONE evaluate job per fold (the job IS the compute — no idle CPU coordinator).
#   WORKER:    bash 06_01_evaluate_testset.sh <RUN_ID> <FOLD>   (runs inside a job)
#       Scores every contrast for its fold against GT → summarize_fold → fold{k}/eval_all.csv
#       (group=test contrast). No prediction happens here anymore.
#
# Optional env: CHECKPOINT (default "checkpoint_final.pth" — unchanged default, matches
#   the predict-stage default in 05_01_predict_common.sh so eval reads the same run it
#   predicted). Set CHECKPOINT=checkpoint_best.pth to score the best-checkpoint arm
#   instead (must have been predicted with the same CHECKPOINT first).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"
cd "${PROJECT_ROOT}"

RUN_ID="${1:?Usage: $0 <RUN_ID> [FOLD]}"
FOLD="${2:-}"
PY=".venv/bin/python"

TEST_CASES="${PROJECT_ROOT}/benchmark/02_tasks/brain_healthy/on-harmony/4_splits_on-harmony/test_cases.json"

# Standard layout (mirror chaos): the trained model is CO-LOCATED with its predictions under
#   01_predictions/<model>/<train_contrast>/<nnUNet|auglab>/<RUN_ID>/DatasetXXX.../
# Training contrast comes from the RUN_ID; discover which category dir actually holds this run
# (nnUNet vs auglab) by looking for the trainer dir under each — same RUN_ID is unique.
TRAIN_CONTRAST="$(echo "$RUN_ID" | grep -oE 'on-harmony_(T[12]w|dwi_ap)_' | head -1 | sed -E 's/^on-harmony_//; s/_$//')"; TRAIN_CONTRAST="${TRAIN_CONTRAST:-T1w}"
RUN_BASE=""; CATEGORY=""
for CAT in nnUNet auglab; do
    cand="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}/${CAT}/${RUN_ID}"
    if ls -d "${cand}"/*/*__nnUNetPlans__3d_fullres >/dev/null 2>&1; then
        RUN_BASE="$cand"; CATEGORY="$CAT"; break
    fi
done
[ -z "$RUN_BASE" ] && { echo "ERROR: no trained model for ${RUN_ID} under ${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}/{nnUNet,auglab}/"; exit 1; }

[ -f "$TEST_CASES" ] || { echo "ERROR: $TEST_CASES not found"; exit 1; }

# Trainer + dataset auto-discovered from the run dir (used only for DJ below; predict has
# already happened by the time evaluate runs).
_TDIR="$(ls -d "${RUN_BASE}"/*/*__nnUNetPlans__3d_fullres 2>/dev/null | head -1)"
[ -z "$_TDIR" ] && { echo "ERROR: no trainer dir under ${RUN_BASE}"; exit 1; }
DS_NAME="$(basename "$(dirname "${_TDIR}")")"
DJ="${nnUNet_raw}/${DS_NAME}/dataset.json"

TESTSET="${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set"                                   # shared, run-independent (built by 05_00_build_test_inputs.sh)
PRED_BASE="${RUN_BASE}"                                                                  # predictions co-located w/ model (chaos-style)
CONTRAST_LIST="T1w T2w bold dwi_ap epi_ap gre_echo1_mag"

CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
_CKPT_TAG="$(basename "${CHECKPOINT}" .pth)"; _CKPT_TAG="${_CKPT_TAG#checkpoint_}"
# NOTE: matches predict_common.sh's convention (flat only for "best"), NOT the old
# inline script's convention (flat only for "final") — see 05_01_predict_common.sh's
# header for why this changed and what it means for on-disk paths.
_PRED_SUBDIR=""; [ "${_CKPT_TAG}" != "best" ] && _PRED_SUBDIR="${_CKPT_TAG}/"
_OUT_SUFFIX=""; [ "${_CKPT_TAG}" != "final" ] && _OUT_SUFFIX="_${_CKPT_TAG}"
# METRICS_SUBDIR (optional): unset (default) writes to the normal flat METRICS_ROOT/.../
# <contrast>/ layout. Set to e.g. "ablations" to write to METRICS_ROOT/.../<contrast>/
# ablations/ instead — for non-headline result sets (CLAUDE.md's "Within 02_metrics/
# <model>/<contrast>/, a non-headline result set gets its own dedicated subdir"
# convention; mirrors chaos's/brats2024-glioma's 06_01_evaluate_run.sh). Only the
# metrics OUTPUT path moves; predictions stay in the normal (flat) location.
METRICS_SUBDIR="${METRICS_SUBDIR:-}"
METRICS_DIR="${METRICS_ROOT}/${MODEL_TYPE}/${TRAIN_CONTRAST}${METRICS_SUBDIR:+/${METRICS_SUBDIR}}/${CATEGORY}_${RUN_ID}${_OUT_SUFFIX}"  # per-run metrics

# ════════════════════════════════════════════════════════════════════════════
# LAUNCHER — one lightweight (no-GPU) evaluate job per fold
# ════════════════════════════════════════════════════════════════════════════
if [ -z "$FOLD" ]; then
    echo "[$(date '+%H:%M:%S')] LAUNCH eval ${RUN_ID}  -> ${TRAIN_CONTRAST}/${CATEGORY}  ckpt=${CHECKPOINT}"
    [ -d "$TESTSET" ] || { echo "ERROR: ${TESTSET} not found — run 05_predict/05_00_build_test_inputs.sh first"; exit 1; }
    echo "[$(date '+%H:%M:%S')] fanning out per-fold eval jobs (folds: ${EVAL_FOLDS:-0 1 2})"
    # Project fold policy: 3 folds (0 1 2) only; override with EVAL_FOLDS for a legacy 4-fold model.
    for F in ${EVAL_FOLDS:-0 1 2}; do
        run_job --name "onheval_${RUN_ID:0:26}_f${F}" --gpus 0 --cpus 4 --mem 16G --time "${ONHEVAL_TIME:-00:30:00}" \
            --log "${SCRATCH:-/tmp}/onheval_${RUN_ID}_fold${F}.log" -- \
            bash "${HERE}/06_01_evaluate_testset.sh" "${RUN_ID}" "${F}"
    done
    echo "[$(date '+%H:%M:%S')] submitted. metrics → ${METRICS_DIR}/fold*/eval_all.csv"
    exit 0
fi

# ════════════════════════════════════════════════════════════════════════════
# WORKER (FOLD set) — score THIS fold's already-predicted contrasts into standard dirs
# ════════════════════════════════════════════════════════════════════════════
echo "[$(date '+%H:%M:%S')] WORKER ${RUN_ID} fold${FOLD} -> ${TRAIN_CONTRAST}/${CATEGORY}"
FOLD_METRICS="${METRICS_DIR}/fold${FOLD}"; mkdir -p "$FOLD_METRICS"
GEOM_FAIL=0

for CONTRAST in $CONTRAST_LIST; do
    GT_DIR="${TESTSET}/${CONTRAST}/gt_native"
    PRED_DIR="${PRED_BASE}/fold${FOLD}/${_PRED_SUBDIR}${CONTRAST}"
    if [ ! -d "$PRED_DIR" ] || [ -z "$(ls -A "$PRED_DIR" 2>/dev/null)" ]; then
        echo "  ! [fold${FOLD}] skip $CONTRAST (no predictions at $PRED_DIR — run 05_predict/ first)"; continue
    fi
    [ -d "$GT_DIR" ] && NGT=$(ls "$GT_DIR" 2>/dev/null | wc -l) || NGT=0
    # Geometry guard (2026-10-06): a prediction left on the RAS input grid (resample-to-native
    # skipped) would be scored x-mirrored on the natively-LAS contrasts, silently. Refuse it.
    if [ "$NGT" -gt 0 ] && ! $PY "${HERE}/../00_utils/check_pred_geometry.py" --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR"; then
        rm -f "$FOLD_METRICS/${CONTRAST}_metrics.csv"; GEOM_FAIL=1
        echo "  ! [fold${FOLD}] REFUSED $CONTRAST: predictions not on the native GT grid" >&2; continue
    fi
    if [ "$NGT" -gt 0 ]; then
        $PY "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/evaluate.py" \
            --pred_dir "$PRED_DIR" --gt_dir "$GT_DIR" \
            --out_csv "$FOLD_METRICS/${CONTRAST}_metrics.csv" --name "$CONTRAST" --dataset_json "$DJ" \
            || echo "  ! [fold${FOLD}] eval failed for $CONTRAST (continuing)"
    fi
    echo "[$(date '+%H:%M:%S')]   [fold${FOLD}] done $CONTRAST"
done

# no eval_all.csv for a fold with refused predictions (a partial one would look complete downstream)
[ "$GEOM_FAIL" = 0 ] || { rm -f "$FOLD_METRICS/eval_all.csv"; echo "[fold${FOLD}] FAILED: off-grid predictions refused (see above)" >&2; exit 1; }

present=()
for c in $CONTRAST_LIST; do [ -f "$FOLD_METRICS/${c}_metrics.csv" ] && present+=("$c"); done
if [ ${#present[@]} -gt 0 ]; then
    $PY "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
        "$FOLD_METRICS" "$RUN_ID" "$FOLD" --group-col contrast --groups-word Contrasts --groups "${present[@]}"
    echo "[$(date '+%H:%M:%S')] [fold${FOLD}] complete → $FOLD_METRICS/eval_all.csv"
else
    echo "[$(date '+%H:%M:%S')] [fold${FOLD}] no metrics produced"
fi
