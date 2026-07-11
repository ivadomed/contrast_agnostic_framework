#!/usr/bin/env bash
# One-off rename of the open-ms FLAIR causal-ablation result directories on Vulcan, to match
# the renames already applied on Killarney + in the main repo / AugLab repo (2026-07-11):
#   v26_6_2_noisefill_v2_no_voronoi   -> baseline_kmeans_label_remap        (baseline-anchored)
#   v26_6_2_noisefill_v2              -> baseline_kmeans_label_remap_voronoi
#   auglabAug_v26_6_2_noisefill_v2_no_voronoi -> auglab_kmeans_label_remap  (auglab-anchored)
#   auglabAug_v26_6_2_noisefill_v2    -> auglab_kmeans_label_remap_voronoi
#
# Run on Vulcan's LOGIN NODE (plain mv on 8 dirs, well under the ~10 CPU-min exception in
# CLAUDE.md — no sbatch needed). Dry-run by default; pass --execute to actually rename.
#
# Usage:
#   bash rename_openms_flair_ablation_vulcan.sh            # dry run (lists + checks only)
#   bash rename_openms_flair_ablation_vulcan.sh --execute   # actually mv
set -euo pipefail

REPO="${REPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"
EXECUTE=0
[[ "${1:-}" == "--execute" ]] && EXECUTE=1

PRED="${REPO}/datasets/open-ms/8_results_open-ms/01_predictions/open_ms_model/flair"
MET="${REPO}/datasets/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair"

# old_path|new_path pairs
MAPPINGS=(
  "${PRED}/nnUNet/open-ms_flair_v26_6_2_noisefill_v2_no_voronoi_train050_val100_20260708_075800|${PRED}/nnUNet/open-ms_flair_baseline_kmeans_label_remap_train050_val100_20260708_075800"
  "${PRED}/nnUNet/open-ms_flair_v26_6_2_noisefill_v2_train050_val100_20260708_075823|${PRED}/nnUNet/open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val100_20260708_075823"
  "${PRED}/auglab/open-ms_flair_auglabAug_v26_6_2_noisefill_v2_no_voronoi_train025_val100_20260707_065414|${PRED}/auglab/open-ms_flair_auglab_kmeans_label_remap_train025_val100_20260707_065414"
  "${PRED}/auglab/open-ms_flair_auglabAug_v26_6_2_noisefill_v2_train025_val100_20260707_065409|${PRED}/auglab/open-ms_flair_auglab_kmeans_label_remap_voronoi_train025_val100_20260707_065409"
  "${MET}/nnUNet_open-ms_flair_v26_6_2_noisefill_v2_no_voronoi_train050_val100_20260708_075800|${MET}/nnUNet_open-ms_flair_baseline_kmeans_label_remap_train050_val100_20260708_075800"
  "${MET}/nnUNet_open-ms_flair_v26_6_2_noisefill_v2_train050_val100_20260708_075823|${MET}/nnUNet_open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val100_20260708_075823"
  "${MET}/auglab_open-ms_flair_auglabAug_v26_6_2_noisefill_v2_no_voronoi_train025_val100_20260707_065414|${MET}/auglab_open-ms_flair_auglab_kmeans_label_remap_train025_val100_20260707_065414"
  "${MET}/auglab_open-ms_flair_auglabAug_v26_6_2_noisefill_v2_train025_val100_20260707_065409|${MET}/auglab_open-ms_flair_auglab_kmeans_label_remap_voronoi_train025_val100_20260707_065409"
)

missing=0
for pair in "${MAPPINGS[@]}"; do
  old="${pair%%|*}"; new="${pair##*|}"
  if [[ ! -d "${old}" ]]; then
    echo "MISSING (skip): ${old}"
    missing=1
    continue
  fi
  if [[ -e "${new}" ]]; then
    echo "ABORT: destination already exists: ${new}"
    exit 1
  fi
  if [[ "${EXECUTE}" -eq 1 ]]; then
    mv -v -- "${old}" "${new}"
  else
    echo "would mv: ${old}"
    echo "      -> ${new}"
  fi
done

if [[ "${EXECUTE}" -eq 0 ]]; then
  echo
  echo "Dry run only — nothing renamed. Re-run with --execute to apply."
  [[ "${missing}" -eq 1 ]] && echo "NOTE: some source dirs were not found on this machine — confirm REPO=${REPO} is correct, or that this ablation's results actually landed here."
fi
