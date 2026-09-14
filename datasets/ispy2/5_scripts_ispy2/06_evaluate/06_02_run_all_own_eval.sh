#!/usr/bin/env bash
# Evaluate ALL currently-complete ispy2 own-model headline + ladder runs
# (CPU-only, one run_job call per run via 06_01_evaluate_own_run.sh). Run this
# on TamIA (needs scratch nnUNet_raw/PREDICTIONS_ROOT) after the corresponding
# predict pack job(s) finish. Covers, as of 2026-09-06:
#   - 8 headline runs: basicA (t1wce x4), basicC (t2w x4)
#   - 3 ladderT1wce rungs 2-4 (baseline_kmeans, +label_remap, +voronoi)
# Deliberately EXCLUDES basicB and ladderT1wce rung 5 (v26_6_2_train050_val100)
# per explicit instruction, even though checkpoint_final.pth now exists for
# them too (flagged in the report, not acted on here).
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source datasets/ispy2/5_scripts_ispy2/00_utils/env.sh
source scripts/cluster/tamia_env_ispy2.sh
HERE=datasets/ispy2/5_scripts_ispy2/06_evaluate
TS=20260905_163655

declare -a JOBS=(
  "ispy2_t1wce_baseline_${TS} nnUNet 100"
  "ispy2_t1wce_auglab_default_${TS} auglab 100"
  "ispy2_t1wce_synthseg_noEM_${TS} auglab 100"
  "ispy2_t1wce_srcsm_${TS} auglab 100"
  "ispy2_t2w_synthseg_noEM_${TS} auglab 101"
  "ispy2_t2w_synthseg_EM_${TS} auglab 101"
  "ispy2_t2w_srcsm_${TS} auglab 101"
  "ispy2_t2w_auglabAug_v26_6_2_train050_val000_${TS} auglab 101"
)
for j in "${JOBS[@]}"; do
  read -r run_id category ds_id <<< "$j"
  echo "=== evaluate own: ${run_id} (${category}, DATASET_ID=${ds_id}) ==="
  bash "${HERE}/06_01_evaluate_own_run.sh" "${run_id}" "${category}" "${ds_id}" all
done

# Ladder rungs -> dedicated ablations/ subdir (CLAUDE.md convention: non-headline
# result sets get their own subdir under 02_metrics/<model>/<contrast>/).
declare -a LADDER_JOBS=(
  "ispy2_t1wce_baseline_kmeans_${TS} auglab 100"
  "ispy2_t1wce_baseline_kmeans_label_remap_${TS} auglab 100"
  "ispy2_t1wce_baseline_kmeans_label_remap_voronoi_${TS} auglab 100"
)
for j in "${LADDER_JOBS[@]}"; do
  read -r run_id category ds_id <<< "$j"
  echo "=== evaluate own (ablations): ${run_id} (${category}, DATASET_ID=${ds_id}) ==="
  METRICS_SUBDIR="ablations" bash "${HERE}/06_01_evaluate_own_run.sh" "${run_id}" "${category}" "${ds_id}" all
done
echo "ALL OWN EVAL SUBMITTED (11 runs)"
