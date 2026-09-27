#!/usr/bin/env bash
# Predict every toothfairy2 CBCT run on the FOV-matched HaN-Seg CT test set (TamIA).
#
# Records all 11 runs x 3 folds into ONE node-pack and submits it, so the 4 H100s on a
# whole-node allocation are all used (CLAUDE.md: a job that exercises 1 of 4 GPUs wastes
# the other 3 for the whole wall clock, and this account has already been warned).
#
# HaN-Seg has TWO items (ct + the MR-in-CT-frame arm); both are recorded per fold task. The union merge step that the
# hanseg path used is deliberately NOT here: scoring is MANDIBLE-ONLY (toothfairy2
# label 1 vs HaN-Seg's Mandible, one-to-one), because HaN-Seg's/HaN-Seg's mandible EXCLUDES
# the teeth — the old union premise was disproven 2026-09-17.
#
#   bash 05_30_tamia_predict.sh          # record + submit
#   DRY=1 bash 05_30_tamia_predict.sh    # record only, print the pack
set -uo pipefail
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
# Suffixed _DIR on purpose — a sourced cluster env file setting a same-named short
# variable silently rewrites these (that is exactly how hanseg once recorded 0 tasks).
HS_DIR="${PROJECT_ROOT}/benchmark/02_tasks/mandible_healthy/hanseg/5_scripts_hanseg"
PACK_DIR="${PACK_DIR:-/scratch/p/paulh/hanseg/_packruns/predict_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"

# token : wrapper : run_id : category
ROWS=(
 "baseline:05_02_predict_baseline.sh:toothfairy2_cbct_baseline_20260908_013025:nnUNet"
 "auglab_default:05_03_predict_auglab_default.sh:toothfairy2_cbct_auglab_default_20260908_013025:auglab"
 "synthseg_noEM:05_04_predict_synthseg_noEM.sh:toothfairy2_cbct_synthseg_noEM_20260908_013025:auglab"
 "synthseg_EM:05_05_predict_synthseg_EM.sh:toothfairy2_cbct_synthseg_EM_20260908_013027:auglab"
 "srcsm:05_06_predict_srcsm.sh:toothfairy2_cbct_srcsm_20260908_013027:auglab"
 "ours_val000:05_07_predict_ours_val000.sh:toothfairy2_cbct_auglabAug_v26_6_2_train050_val000_20260908_013027:auglab"
 "ours_val100:05_08_predict_ours_val100.sh:toothfairy2_cbct_auglabAug_v26_6_2_train050_val100_20260908_013027:auglab"
 "r2_kmeans:05_09_predict_baseline_kmeans.sh:toothfairy2_cbct_baseline_kmeans_20260908_013028:auglab"
 "r3_remap:05_10_predict_baseline_kmeans_label_remap.sh:toothfairy2_cbct_baseline_kmeans_label_remap_20260908_013028:auglab"
 "r4_voronoi:05_11_predict_baseline_kmeans_label_remap_voronoi.sh:toothfairy2_cbct_baseline_kmeans_label_remap_voronoi_20260908_013028:auglab"
 "r5_v26alone:05_12_predict_v26_6_2_train050_val100.sh:toothfairy2_cbct_v26_6_2_train050_val100_20260908_013028:nnUNet"
)

echo "[hs] pack=${PACK_DIR}"
n=0
for row in "${ROWS[@]}"; do
  IFS=: read -r TOK W RID CAT <<< "${row}"
  RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HS_DIR}/05_predict/${W}" "${RID}" all ${ITEMS:-ct mrt1} \
    || { echo "[hs] RECORD FAILED ${TOK}" >&2; exit 1; }
  n=$((n+1))
  echo "[hs] recorded ${TOK} (${CAT})"
done

# The pack records one *.sh per fold-task (plus index.tsv) — count the .sh files.
REC=$(ls "${PACK_DIR}"/*.sh 2>/dev/null | wc -l)
echo "[hs] recorded ${n} wrappers -> ${REC} fold-tasks"
# Guard: 11 runs x 3 folds. A short recording means a wrapper resolved to the wrong
# path or the fold loop never ran — hanseg's silent zero-task failure, again.
[ "${REC}" -eq 33 ] || { echo "[hs] ABORT: expected 33 recorded tasks, got ${REC}" >&2; exit 1; }
# Verify the recorded commands really target hanseg, not hanseg (dataset/contrast
# cross-contamination is a documented repeat failure in this repo).
BAD=$(grep -L "hanseg" "${PACK_DIR}"/*.sh 2>/dev/null | wc -l)
[ "${BAD}" -eq 0 ] || { echo "[hs] ABORT: ${BAD} task files do not reference hanseg" >&2; exit 1; }
echo "[hs] verified: all ${REC} task files reference hanseg"

[ -n "${DRY:-}" ] && { echo "[hs] DRY=1, not submitting"; exit 0; }
bash "${PROJECT_ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
