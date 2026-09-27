#!/usr/bin/env bash
# Predict all CHAOS runs with missing fold predictions.
# 36 total fold-predictions distributed round-robin across slots 1, 2, 3 (12 each).
# All 12 predictions within each slot subshell run in parallel (&).
# Subshells are fire-and-exit orphan processes that continue after this script returns.
#
# Usage: bash 05_21_launch_predict_pending.sh
#
# Missing predictions as of 2026-06-17:
#   nnUNet: v26_6_2_025_val000 f2,3 | v26_6_2_025_val100 f0-3 | v26_6_2_090_val100 f1,2,3
#   auglab: auglabAug_synthseg_EM f2,3 | auglabAug_025_val000 f2,3 | auglabAug_025_val100 f0-3
#           auglabAug_050_val000 f1,2,3 | auglabAug_050_val100 f0-3
#           auglabAug_090_val000 f0-3 | auglabAug_090_val100 f0-3
#           synthseg_EM_val100 f0-3

cd /home/ge.polymtl.ca/pahoa/mri_synthesis_project
S="$(cd "$(dirname "$0")" && pwd)"

# Run IDs
R090_100="chaos_t1in_v26_6_2_train090_val100_20260615_213615"
R025_000="chaos_t1in_v26_6_2_train025_val000_20260616_010628"
R025_100="chaos_t1in_v26_6_2_train025_val100_20260616_200514"
A_SEM="chaos_t1in_auglabAug_synthseg_EM_train100_val000_20260615_213615"
A_025_000="chaos_t1in_auglabAug_v26_6_2_train025_val000_20260616_010628"
A_025_100="chaos_t1in_auglabAug_v26_6_2_train025_val100_20260616_200514"
A_050_000="chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615"
A_050_100="chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420"
A_090_000="chaos_t1in_auglabAug_v26_6_2_train090_val000_20260616_200514"
A_090_100="chaos_t1in_auglabAug_v26_6_2_train090_val100_20260616_200514"
SEM_100="chaos_t1in_synthseg_EM_train100_val100_20260616_112420"

echo "[$(date '+%H:%M:%S')] Launching 36 pending predictions across slots 1, 2, 3"

# ── Slot 1 (GPU 1) — 12 predictions ──────────────────────────────────────────
(
SLOT=1; GPU=1
export SLOT GPU
echo "[slot1 $(date '+%H:%M:%S')] start (12 predictions)"
SLOT=$SLOT GPU=$GPU bash "$S/05_11_predict_v26_6_2_train025_val000.sh"    "$R025_000" 2 & # 1
SLOT=$SLOT GPU=$GPU bash "$S/05_16_predict_v26_6_2_train025_val100.sh"    "$R025_100" 1 & # 4
SLOT=$SLOT GPU=$GPU bash "$S/05_13_predict_v26_6_2_train090_val100.sh"    "$R090_100" 1 & # 7
SLOT=$SLOT GPU=$GPU bash "$S/05_08_predict_auglabAug_synthseg_EM.sh"      "$A_SEM"    2 & # 10
SLOT=$SLOT GPU=$GPU bash "$S/05_12_predict_auglabAug_v26_6_2_train025_val000.sh" "$A_025_000" 3 & # 13
SLOT=$SLOT GPU=$GPU bash "$S/05_17_predict_auglabAug_v26_6_2_train025_val100.sh" "$A_025_100" 2 & # 16
SLOT=$SLOT GPU=$GPU bash "$S/05_09_predict_auglabAug_v26_6_2_train050_val000.sh" "$A_050_000" 2 & # 19
SLOT=$SLOT GPU=$GPU bash "$S/05_14_predict_auglabAug_v26_6_2_train050_val100.sh" "$A_050_100" 1 & # 22
SLOT=$SLOT GPU=$GPU bash "$S/05_18_predict_auglabAug_v26_6_2_train090_val000.sh" "$A_090_000" 0 & # 25
SLOT=$SLOT GPU=$GPU bash "$S/05_18_predict_auglabAug_v26_6_2_train090_val000.sh" "$A_090_000" 3 & # 28
SLOT=$SLOT GPU=$GPU bash "$S/05_19_predict_auglabAug_v26_6_2_train090_val100.sh" "$A_090_100" 2 & # 31
SLOT=$SLOT GPU=$GPU bash "$S/05_15_predict_synthseg_EM_train100_val100.sh"        "$SEM_100"   1 & # 34
wait
echo "[slot1 $(date '+%H:%M:%S')] all 12 done"
) >> /tmp/chaos_predict_slot1.log 2>&1 &

# ── Slot 2 (GPU 2) — 12 predictions ──────────────────────────────────────────
(
SLOT=2; GPU=2
export SLOT GPU
echo "[slot2 $(date '+%H:%M:%S')] start (12 predictions)"
SLOT=$SLOT GPU=$GPU bash "$S/05_11_predict_v26_6_2_train025_val000.sh"    "$R025_000" 3 & # 2
SLOT=$SLOT GPU=$GPU bash "$S/05_16_predict_v26_6_2_train025_val100.sh"    "$R025_100" 2 & # 5
SLOT=$SLOT GPU=$GPU bash "$S/05_13_predict_v26_6_2_train090_val100.sh"    "$R090_100" 2 & # 8
SLOT=$SLOT GPU=$GPU bash "$S/05_08_predict_auglabAug_synthseg_EM.sh"      "$A_SEM"    3 & # 11
SLOT=$SLOT GPU=$GPU bash "$S/05_17_predict_auglabAug_v26_6_2_train025_val100.sh" "$A_025_100" 0 & # 14
SLOT=$SLOT GPU=$GPU bash "$S/05_17_predict_auglabAug_v26_6_2_train025_val100.sh" "$A_025_100" 3 & # 17
SLOT=$SLOT GPU=$GPU bash "$S/05_09_predict_auglabAug_v26_6_2_train050_val000.sh" "$A_050_000" 3 & # 20
SLOT=$SLOT GPU=$GPU bash "$S/05_14_predict_auglabAug_v26_6_2_train050_val100.sh" "$A_050_100" 2 & # 23
SLOT=$SLOT GPU=$GPU bash "$S/05_18_predict_auglabAug_v26_6_2_train090_val000.sh" "$A_090_000" 1 & # 26
SLOT=$SLOT GPU=$GPU bash "$S/05_19_predict_auglabAug_v26_6_2_train090_val100.sh" "$A_090_100" 0 & # 29
SLOT=$SLOT GPU=$GPU bash "$S/05_19_predict_auglabAug_v26_6_2_train090_val100.sh" "$A_090_100" 3 & # 32
SLOT=$SLOT GPU=$GPU bash "$S/05_15_predict_synthseg_EM_train100_val100.sh"        "$SEM_100"   2 & # 35
wait
echo "[slot2 $(date '+%H:%M:%S')] all 12 done"
) >> /tmp/chaos_predict_slot2.log 2>&1 &

# ── Slot 3 (GPU 3) — 12 predictions ──────────────────────────────────────────
(
SLOT=3; GPU=3
export SLOT GPU
echo "[slot3 $(date '+%H:%M:%S')] start (12 predictions)"
SLOT=$SLOT GPU=$GPU bash "$S/05_16_predict_v26_6_2_train025_val100.sh"    "$R025_100" 0 & # 3
SLOT=$SLOT GPU=$GPU bash "$S/05_16_predict_v26_6_2_train025_val100.sh"    "$R025_100" 3 & # 6
SLOT=$SLOT GPU=$GPU bash "$S/05_13_predict_v26_6_2_train090_val100.sh"    "$R090_100" 3 & # 9
SLOT=$SLOT GPU=$GPU bash "$S/05_12_predict_auglabAug_v26_6_2_train025_val000.sh" "$A_025_000" 2 & # 12
SLOT=$SLOT GPU=$GPU bash "$S/05_17_predict_auglabAug_v26_6_2_train025_val100.sh" "$A_025_100" 1 & # 15
SLOT=$SLOT GPU=$GPU bash "$S/05_09_predict_auglabAug_v26_6_2_train050_val000.sh" "$A_050_000" 1 & # 18
SLOT=$SLOT GPU=$GPU bash "$S/05_14_predict_auglabAug_v26_6_2_train050_val100.sh" "$A_050_100" 0 & # 21
SLOT=$SLOT GPU=$GPU bash "$S/05_14_predict_auglabAug_v26_6_2_train050_val100.sh" "$A_050_100" 3 & # 24
SLOT=$SLOT GPU=$GPU bash "$S/05_18_predict_auglabAug_v26_6_2_train090_val000.sh" "$A_090_000" 2 & # 27
SLOT=$SLOT GPU=$GPU bash "$S/05_19_predict_auglabAug_v26_6_2_train090_val100.sh" "$A_090_100" 1 & # 30
SLOT=$SLOT GPU=$GPU bash "$S/05_15_predict_synthseg_EM_train100_val100.sh"        "$SEM_100"   0 & # 33
SLOT=$SLOT GPU=$GPU bash "$S/05_15_predict_synthseg_EM_train100_val100.sh"        "$SEM_100"   3 & # 36
wait
echo "[slot3 $(date '+%H:%M:%S')] all 12 done"
) >> /tmp/chaos_predict_slot3.log 2>&1 &

echo "[$(date '+%H:%M:%S')] All 3 slot queues launched"
echo "  Monitor: tail -f /tmp/chaos_predict_slot{1,2,3}.log"
