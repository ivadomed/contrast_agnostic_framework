#!/usr/bin/env bash
# Predict all CHAOS runs that have incomplete or missing predictions.
# Runs with 0 folds predicted get all 4 folds; runs with partial predictions
# get only the missing folds via FOLD argument.
# nnUNetv2_predict is idempotent: existing output files are skipped automatically.
#
# Runs covered (as of 2026-06-16):
#   Already fully predicted (4/4): baseline, v26_6_2_050_val000, v26_6_2_050_val100,
#     v26_6_2_090_val000, auglab_default, synthseg_EM_val000, synthseg_noEM
#
#   Missing predictions (run after training completes):
#     nnUNet:
#       chaos_t1in_v26_6_2_train090_val100_20260615_213615        folds 1,2,3
#       chaos_t1in_v26_6_2_train025_val000_20260616_010628        folds 2,3
#       chaos_t1in_v26_6_2_train025_val100_<TS>                   folds 0,1,2,3  (NEW)
#     auglab:
#       chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615  folds 2,3
#       chaos_t1in_auglabAug_v26_6_2_train025_val000_20260616_010628      folds 2,3
#       chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615      folds 1,2,3
#       chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420      folds 0,1,2,3
#       chaos_t1in_synthseg_EM_train100_val100_20260616_112420            folds 0,1,2,3
#       chaos_t1in_auglabAug_v26_6_2_train025_val100_<TS>                 folds 0,1,2,3  (NEW)
#       chaos_t1in_auglabAug_v26_6_2_train090_val000_<TS>                 folds 0,1,2,3  (NEW)
#       chaos_t1in_auglabAug_v26_6_2_train090_val100_<TS>                 folds 0,1,2,3  (NEW)
#
# Usage:
#   bash 05_20_predict_all_pending.sh <NEW_RUN_TS>
#   where NEW_RUN_TS is the timestamp suffix for batch5 runs, e.g. 20260616_200514
#
# FOLD=all runs folds 0-3 in parallel (fold→slot→GPU).
# To predict a specific fold range, call the individual script directly.

set -euo pipefail
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
NEW_TS="${1:?Usage: $0 <NEW_RUN_TS>  (e.g. 20260616_200514)}"

echo "[$(date '+%H:%M:%S')] === Predicting all pending CHAOS runs (TS=${NEW_TS}) ==="

# ── nnUNet predictions ──────────────────────────────────────────────────────

echo ""
echo "[$(date '+%H:%M:%S')] v26_6_2_train090_val100 — folds 1,2,3"
for F in 1 2 3; do
    bash "${SCRIPTS_DIR}/05_13_predict_v26_6_2_train090_val100.sh" \
        "chaos_t1in_v26_6_2_train090_val100_20260615_213615" "${F}" &
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] v26_6_2_train025_val000 — folds 2,3"
for F in 2 3; do
    bash "${SCRIPTS_DIR}/05_11_predict_v26_6_2_train025_val000.sh" \
        "chaos_t1in_v26_6_2_train025_val000_20260616_010628" "${F}" &
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] v26_6_2_train025_val100 — all folds (NEW)"
bash "${SCRIPTS_DIR}/05_16_predict_v26_6_2_train025_val100.sh" \
    "chaos_t1in_v26_6_2_train025_val100_${NEW_TS}" all

# ── auglab predictions ──────────────────────────────────────────────────────

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_synthseg_EM_train100_val000 — folds 2,3"
for F in 2 3; do
    bash "${SCRIPTS_DIR}/05_08_predict_auglabAug_synthseg_EM.sh" \
        "chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615" "${F}" &
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train025_val000 — folds 2,3"
for F in 2 3; do
    bash "${SCRIPTS_DIR}/05_12_predict_auglabAug_v26_6_2_train025_val000.sh" \
        "chaos_t1in_auglabAug_v26_6_2_train025_val000_20260616_010628" "${F}" &
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train050_val000 — folds 1,2,3"
for F in 1 2 3; do
    bash "${SCRIPTS_DIR}/05_09_predict_auglabAug_v26_6_2_train050_val000.sh" \
        "chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615" "${F}" &
done
wait

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train050_val100 — all folds"
bash "${SCRIPTS_DIR}/05_14_predict_auglabAug_v26_6_2_train050_val100.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420" all

echo ""
echo "[$(date '+%H:%M:%S')] synthseg_EM_train100_val100 — all folds"
bash "${SCRIPTS_DIR}/05_15_predict_synthseg_EM_train100_val100.sh" \
    "chaos_t1in_synthseg_EM_train100_val100_20260616_112420" all

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train025_val100 — all folds (NEW)"
bash "${SCRIPTS_DIR}/05_17_predict_auglabAug_v26_6_2_train025_val100.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train025_val100_${NEW_TS}" all

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train090_val000 — all folds (NEW)"
bash "${SCRIPTS_DIR}/05_18_predict_auglabAug_v26_6_2_train090_val000.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train090_val000_${NEW_TS}" all

echo ""
echo "[$(date '+%H:%M:%S')] auglabAug_v26_6_2_train090_val100 — all folds (NEW)"
bash "${SCRIPTS_DIR}/05_19_predict_auglabAug_v26_6_2_train090_val100.sh" \
    "chaos_t1in_auglabAug_v26_6_2_train090_val100_${NEW_TS}" all

echo ""
echo "[$(date '+%H:%M:%S')] === All pending predictions done ==="
