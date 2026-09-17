#!/usr/bin/env bash
# Score toothfairy2's OWN held-out CBCT test set on MANDIBLE-ONLY.
#
# The in-domain half of the union-bug correction (2026-09-17). Companion to
# datasets/hanseg/5_scripts_hanseg/06_evaluate/06_03_eval_mandible_only.sh, which does
# the two hanseg OOD arms.
#
# WHY BOTH HALVES ARE REQUIRED
# ----------------------------
# The point of a corrected view is that EVERY column scores the SAME single structure.
# The union view achieved that by building a `cbct_union` in-domain column to sit beside
# the union-scored OOD columns; the mandible-only view needs the exact same treatment, or
# the table has no in-domain anchor and cannot be compared against the union table it
# replaces. Scoring only the OOD arms would leave the in-domain -> OOD drop unmeasurable.
#
# WHY THE UNION WAS WRONG (see the hanseg script's header for the full evidence):
# HaN-Seg's `Bone_Mandible` EXCLUDES the teeth (Brouwer et al. 2015 consensus, verbatim
# "the entire mandible bone, without teeth"; HaN-Seg's own paper follows that guideline;
# and empirically the mask carries no enamel population while 96.6-99.7% of enamel-range
# voxels sit outside it). So `mandible u lower_teeth` over-covers the GT by the crowns.
#
# HOW: in-domain GT is toothfairy2's own 3-class mask with mandible=1, so this is a true
# one-to-one comparison — the existing 3-class call with `--labels mandible` instead of
# `--labels mandible lower_teeth pharynx`. No merge, no remap, evaluation only.
#
# SAFETY: writes to a separate `cbct_mandible_only` metrics tree, exactly as the union
# view wrote to `cbct_union`. The 3-class in-domain table and the union tables are NOT
# touched — all three views stay on disk and comparable.
#
# Run ON TamIA (predictions live on its scratch), submitted through Slurm.
set -uo pipefail

PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
TF2_DIR="${PROJECT_ROOT}/datasets/toothfairy2/5_scripts_toothfairy2"
PY="${PROJECT_ROOT}/.venv/bin/python"
SUMMARIZE="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py"

TF2_PRED=/scratch/p/paulh/toothfairy2/8_results/01_predictions/toothfairy2_model/cbct
TF2_METRICS=/scratch/p/paulh/toothfairy2/8_results/02_metrics/toothfairy2_model/cbct_mandible_only
TF2_RAW=/scratch/p/paulh/toothfairy2/2_nnUNet/raw/Dataset110_ToothFairy2CBCT
DJ="${TF2_RAW}/dataset.json"
GT="${TF2_RAW}/labelsTs_cbct"

# Where each run's 3-class metrics currently live, so ablations/ membership is mirrored.
TF2_METRICS_SRC=/scratch/p/paulh/toothfairy2/8_results/02_metrics/toothfairy2_model/cbct

fail=0; n_ok=0
echo "[mo-cbct] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mo-cbct] GT=${GT} ($(find "${GT}" -name '*.nii.gz' 2>/dev/null | wc -l) files)"

for CAT in nnUNet auglab; do
  [ -d "${TF2_PRED}/${CAT}" ] || continue
  for RUNDIR in "${TF2_PRED}/${CAT}"/*/; do
    RID="$(basename "${RUNDIR}")"
    [ "${RID}" = "_logs" ] && continue

    if   [ -d "${TF2_METRICS_SRC}/${CAT}_${RID}" ];           then SUB=""
    elif [ -d "${TF2_METRICS_SRC}/ablations/${CAT}_${RID}" ]; then SUB="ablations"
    else echo "[mo-cbct] WARN no existing metrics dir for ${CAT}_${RID}, defaulting flat" >&2; SUB=""
    fi

    for F in 0 1 2; do
      D="${RUNDIR}fold${F}/cbct"
      [ -d "$D" ] || { echo "[mo-cbct] MISSING ${D}" >&2; fail=1; continue; }
      n=$(find "$D" -name '*.nii.gz' 2>/dev/null | wc -l)
      [ "$n" = "71" ] || { echo "[mo-cbct] BAD COUNT ${n}/71: $D" >&2; fail=1; continue; }

      OUT="${TF2_METRICS}${SUB:+/${SUB}}/${CAT}_${RID}/fold${F}"
      mkdir -p "$OUT"
      "$PY" "${TF2_DIR}/06_evaluate/06_00_evaluate.py" \
          --pred_dir "$D" --gt_dir "${GT}" \
          --dataset_json "$DJ" --labels mandible --name cbct \
          --out_csv "${OUT}/cbct_metrics.csv" --workers 6 \
          > "${OUT}/cbct_eval.log" 2>&1 \
        || { echo "[mo-cbct] EVAL FAILED ${RID} fold${F}" >&2; fail=1; continue; }
      rows=$(( $(wc -l < "${OUT}/cbct_metrics.csv") - 1 ))
      [ "$rows" = "71" ] || echo "[mo-cbct] WARN rows ${rows}/71 ${RID} f${F}" >&2
      echo "[mo-cbct] ${CAT}_${RID} fold${F}: ${rows} rows"
      n_ok=$((n_ok+1))

      "$PY" "$SUMMARIZE" "$OUT" "$RID" "$F" --groups cbct --group-col contrast \
          --groups-word Contrasts >> "${OUT}/summarize.log" 2>&1 \
        || { echo "[mo-cbct] SUMMARIZE FAILED ${RID} fold${F}" >&2; fail=1; }
    done
  done
done

echo "[mo-cbct] done: ${n_ok} evaluations, fail=${fail}"
exit ${fail}
