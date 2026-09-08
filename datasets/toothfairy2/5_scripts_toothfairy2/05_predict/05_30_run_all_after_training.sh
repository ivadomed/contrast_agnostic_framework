#!/usr/bin/env bash
# ONE-COMMAND post-training chain for toothfairy2, driven by the packs' own
# RUN_IDS.env so no RUN_ID is ever typed by hand.
#
#   bash 05_30_run_all_after_training.sh <SUITE_A_PACK> <SUITE_B_PACK> <LADDER_PACK>
#
# Does, in order:
#   1. build the held-out CBCT test inputs (own, in-domain)
#   2. predict + evaluate all 11 runs on the own CBCT test set
#   3. predict + evaluate all 11 runs on the FOV-matched hanseg CT set (cross-modality)
#   4. write the results configs and aggregate (+ the auto-wired significance column)
#   5. build the causal-ablation ladder table
#
# Safe to re-run: every stage skips work that already exists. Ladder-only runs are
# routed to the `ablations/` metrics subdir via METRICS_SUBDIR, per CLAUDE.md's
# convention that non-headline result sets get their own subdir.
#
# REQUIRES (run ON tamia): env.sh + scripts/cluster/tamia_env_toothfairy2.sh sourced.
# hanseg stages re-source the hanseg env themselves in a subshell so the two
# datasets' path exports can never leak into each other.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DS="$(cd "${HERE}/.." && pwd)"
ROOT="$(cd "${DS}/../../.." && pwd)"
A="${1:?need suiteA pack dir}"; B="${2:?need suiteB pack dir}"; L="${3:?need ladder pack dir}"

get() { grep -E "^$2=" "$1/RUN_IDS.env" | head -1 | cut -d= -f2-; }

BASELINE=$(get "$A" BASELINE_RUN_ID)
AUGDEF=$(get "$A" AUGLAB_DEFAULT_RUN_ID)
SSNOEM=$(get "$A" SYNTHSEG_NOEM_RUN_ID)
SSEM=$(get "$B" SYNTHSEG_EM_RUN_ID)
SRCSM=$(get "$B" SRCSM_RUN_ID)
OURS0=$(get "$B" OURS_RUN_ID)
OURS1="${OURS0/_val000_/_val100_}"
R2=$(get "$L" R2_RUN_ID); R3=$(get "$L" R3_RUN_ID)
R4=$(get "$L" R4_RUN_ID); R5=$(get "$L" R5_RUN_ID)

# run_id : predict-wrapper : category : metrics-subdir
ROWS=(
  "${BASELINE}:05_02_predict_baseline.sh:nnUNet:"
  "${AUGDEF}:05_03_predict_auglab_default.sh:auglab:"
  "${SSNOEM}:05_04_predict_synthseg_noEM.sh:auglab:"
  "${SSEM}:05_05_predict_synthseg_EM.sh:auglab:"
  "${SRCSM}:05_06_predict_srcsm.sh:auglab:"
  "${OURS0}:05_07_predict_ours_val000.sh:auglab:"
  "${OURS1}:05_08_predict_ours_val100.sh:auglab:"
  "${R2}:05_09_predict_baseline_kmeans.sh:auglab:ablations"
  "${R3}:05_10_predict_baseline_kmeans_label_remap.sh:auglab:ablations"
  "${R4}:05_11_predict_baseline_kmeans_label_remap_voronoi.sh:auglab:ablations"
  "${R5}:05_12_predict_v26_6_2_train050_val100.sh:nnUNet:ablations"
)

echo "########## 1. own held-out CBCT test inputs"
bash "${HERE}/05_00_build_test_inputs.sh"

echo "########## 2. own predict + evaluate (IN-DOMAIN)"
for row in "${ROWS[@]}"; do
    IFS=: read -r RID WRAP CAT SUB <<< "${row}"
    [ -n "${RID}" ] || { echo "  ! missing RUN_ID for ${WRAP} — skipping"; continue; }
    echo "--- ${RID} (${CAT}${SUB:+, ${SUB}})"
    bash "${HERE}/${WRAP}" "${RID}"
    METRICS_SUBDIR="${SUB}" bash "${DS}/06_evaluate/06_01_evaluate_run.sh" "${RID}" "${CAT}"
done

echo "########## 3. hanseg cross-modality predict + evaluate (OOD)"
(
  set -euo pipefail
  cd "${ROOT}"
  source "${ROOT}/datasets/hanseg/5_scripts_hanseg/00_utils/env.sh"
  source "${ROOT}/scripts/cluster/tamia_env_hanseg.sh"
  HS="${ROOT}/datasets/hanseg/5_scripts_hanseg"
  bash "${HS}/01_prepare/01_01_prepare_ct.sh" || true   # idempotent; skip if already built
  for row in "${ROWS[@]}"; do
      IFS=: read -r RID WRAP CAT SUB <<< "${row}"
      [ -n "${RID}" ] || continue
      echo "--- hanseg ${RID}"
      bash "${HS}/05_predict/${WRAP}" "${RID}"
      METRICS_SUBDIR="${SUB}" bash "${HS}/06_evaluate/06_01_evaluate_run.sh" "${RID}" "${CAT}"
  done
)

echo "########## 4. configs + aggregation"
# Both suite packs are passed: each pack's RUN_IDS.env lists all six method ids
# regardless of which three it actually trained, so composing from A + B is the only
# way to get the ids that really exist. (This replaces an earlier regex patch that
# rewrote lines in the generated config after the fact — fragile, since a bare
# "baseline_" prefix also matches "baseline_kmeans_*".)
bash "${DS}/06_evaluate/06_04_write_configs.sh" "${A}" "${B}" "${L}"

bash "${DS}/06_evaluate/06_02_aggregate_from_config.sh"                       # in-domain only
bash "${DS}/06_evaluate/06_06_cross_dataset_summary.sh"                      # HEADLINE (in-domain + hanseg CT)
bash "${DS}/06_evaluate/06_07_combined_modality_summary.sh" || true          # per-task roll-up for the meta-heatmap
bash "${DS}/06_evaluate/06_03_significance_from_config.sh" \
     "${DS}/06_evaluate/configs/toothfairy2_cross_dataset_01_results.yaml" || true

echo "########## 5. causal-ablation ladder"
bash "${DS}/06_evaluate/06_05_ladder_summary.sh" "${A}" "${B}" "${L}" || true

echo "########## DONE"
