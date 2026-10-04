#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --job-name=chaos_ladder_fovcrop
#
# Cross-dataset (amos + sliver07) CROP-before-predict evaluation of the chaos causal-ablation-ladder
# rungs 2-4 (kmeans / label_remap / voronoi) -- thin wrapper over the shared driver
# 00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh (RUNS_FILE roster mode). TamIA.
#
# Re-uses the EXISTING crops of the headline crop run (amos _fovcrop/391947, sliver07 _fovcrop/391949), so
# every rung -- headline rungs 1/6/7 and these -- sees byte-identical cropped inputs.
#
#   sbatch --export=ALL,DRIVER_ROOT=<root> 06_36_fov_crop_ladder_rungs.sh predict       # GPU node
#   sbatch --export=ALL,DRIVER_ROOT=<root> --gpus-per-node=0 --mem=150G --time=03:00:00 \
#          --dependency=afterok:<predict job> 06_36_fov_crop_ladder_rungs.sh eval       # CPU only
# ONLY=amos|sliver07 restricts a run to one dataset (e.g. to redo tasks an OOM-killed eval left missing).
# DRIVER_ROOT: tree containing benchmark/00_commun_scripts (+ this roster); default the TamIA repo.
set -uo pipefail
MODE="${1:?usage: $0 predict|eval}"
DRIVER_ROOT="${DRIVER_ROOT:-/project/aip-jcohen/paulh/mri_synthesis_project}"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
DRIVER="${DRIVER_ROOT}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
# RUNS_FILE overridable (e.g. chaos_fov_crop_rung6_pv_runs.txt for the rung-6 PV branch); default = the ladder-rung roster
export RUNS_FILE="${RUNS_FILE:-${DRIVER_ROOT}/benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/chaos_fov_crop_ladder_runs.txt}"
export CONTRASTS="t1in t2spir" SKIP_REPORT=1
case "${MODE}" in
  predict) export PHASES="2";;
  eval)    export PHASES="3 4";;
  *) echo "mode must be predict|eval" >&2; exit 2;;
esac
rc=0
echo "[ladder-fov] mode=${MODE} roster=${RUNS_FILE}"
# amos: CT + T2w MRI, kidney-anchored slab; chaos->AMOS organ remap evaluator
if [ -z "${ONLY:-}" ] || [ "${ONLY}" = amos ]; then
  DATASET=amos ITEMS="ct mri" ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos EVAL_PARALLEL=6 \
    CROP_REUSE_DIR="${SCRATCH}/amos/_fovcrop/391947" bash "${DRIVER}" || rc=1
fi
# sliver07: liver-only CT, liver-anchored slab
if [ -z "${ONLY:-}" ] || [ "${ONLY}" = sliver07 ]; then
  DATASET=sliver07 ITEMS="ct" ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver EVAL_PARALLEL=8 \
    CROP_REUSE_DIR="${SCRATCH}/sliver07/_fovcrop/391949" bash "${DRIVER}" || rc=1
fi
echo "[ladder-fov] ${MODE} finished rc=${rc}"; exit ${rc}
