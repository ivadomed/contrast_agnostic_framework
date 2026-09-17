#!/usr/bin/env bash
# Score the PDDCA CT arm, MANDIBLE-ONLY.
#
# toothfairy2 label 1 (`mandible`) vs PDDCA's `Mandible.nrrd`, one-to-one via
# --label_map '{"mandible":[1,1]}' on the RAW 3-class predictions (labels 2/3 score as
# background). There is deliberately NO union merge step anywhere in this dataset:
# PDDCA's mandible EXCLUDES the teeth — stated in its own protocol doc ("Only the bone
# is segmented, while the teeth are excluded") and confirmed empirically (p99 = 1726 HU
# inside the mask, 99.5% of enamel-range voxels outside it, per-slice hole-fill ratio
# 1.0000). That is the identical convention to hanseg's Bone_Mandible, so the two CT
# arms are label-consistent with each other and with the corrected hanseg scoring.
#
# Unlike hanseg there is no `mandible_only/` subdir: mandible-only is the ONLY scoring
# convention this dataset has ever had, so results go in the normal layout. Ladder rungs
# are routed to ablations/ by run-id pattern (there are no pre-existing metrics dirs to
# mirror, since this dataset is new).
#
# Run ON TamIA (predictions live on its scratch), submitted through Slurm.
set -uo pipefail

PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PD_DIR="${PROJECT_ROOT}/datasets/pddca/5_scripts_pddca"
PY="${PROJECT_ROOT}/.venv/bin/python"
SUMMARIZE="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py"

PD_PRED=/scratch/p/paulh/pddca/8_results/01_predictions/toothfairy2_model/cbct
PD_METRICS=/scratch/p/paulh/pddca/8_results/02_metrics/toothfairy2_model/cbct
PD_GT=/scratch/p/paulh/pddca/2_nnUNet/raw/labelsTs_ct

ITEMS="${ITEMS_OVERRIDE:-ct}"
EXPECT_N="${EXPECT_N:-40}"
fail=0; n_ok=0

echo "[mo-pddca] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mo-pddca] GT=${PD_GT} ($(find "${PD_GT}" -name '*.nii.gz' 2>/dev/null | wc -l) files)"

for CAT in nnUNet auglab; do
  [ -d "${PD_PRED}/${CAT}" ] || continue
  for RUNDIR in "${PD_PRED}/${CAT}"/*/; do
    RID="$(basename "${RUNDIR}")"
    [ "${RID}" = "_logs" ] && continue

    # Ladder rungs -> ablations/, matching the toothfairy2/hanseg layout. Pattern-based
    # because a brand-new dataset has no existing metrics tree to mirror.
    case "${RID}" in
      *baseline_kmeans*|*_v26_6_2_train050_val100_*) SUB="/ablations" ;;
      *) SUB="" ;;
    esac
    # auglabAug_v26_6_2_* is the HEADLINE method, not a ladder rung — don't let the
    # v26_6_2 pattern above swallow it.
    case "${RID}" in *auglabAug_v26_6_2*) SUB="" ;; esac

    for F in 0 1 2; do
      for IT in ${ITEMS}; do
        D="${RUNDIR}fold${F}/${IT}"
        [ -d "$D" ] || { echo "[mo-pddca] MISSING ${D}" >&2; fail=1; continue; }
        n=$(find "$D" -name '*.nii.gz' 2>/dev/null | wc -l)
        [ "$n" = "${EXPECT_N}" ] || { echo "[mo-pddca] BAD COUNT ${n}/${EXPECT_N}: $D" >&2; fail=1; continue; }

        OUT="${PD_METRICS}${SUB}/${CAT}_${RID}/fold${F}"
        mkdir -p "$OUT"
        "$PY" "${PD_DIR}/06_evaluate/06_00_evaluate.py" \
            --pred_dir "$D" --gt_dir "${PD_GT}" \
            --label_map '{"mandible": [1, 1]}' --name "${IT}" \
            --out_csv "${OUT}/${IT}_metrics.csv" --workers 6 \
            > "${OUT}/${IT}_eval.log" 2>&1 \
          || { echo "[mo-pddca] EVAL FAILED ${RID} fold${F} ${IT}" >&2; fail=1; continue; }
        rows=$(( $(wc -l < "${OUT}/${IT}_metrics.csv") - 1 ))
        [ "$rows" = "$n" ] || echo "[mo-pddca] WARN rows ${rows}/${n} ${RID} f${F}" >&2
        echo "[mo-pddca] ${CAT}_${RID}${SUB} fold${F} ${IT}: ${rows} rows"
        n_ok=$((n_ok+1))
      done
      OUT="${PD_METRICS}${SUB}/${CAT}_${RID}/fold${F}"
      if ls "${OUT}"/*_metrics.csv >/dev/null 2>&1; then
        "$PY" "$SUMMARIZE" "$OUT" "$RID" "$F" --groups ${ITEMS} --group-col contrast \
            --groups-word Contrasts >> "${OUT}/summarize.log" 2>&1 \
          || { echo "[mo-pddca] SUMMARIZE FAILED ${RID} fold${F}" >&2; fail=1; }
      fi
    done
  done
done

echo "[mo-pddca] done: ${n_ok} evaluations, fail=${fail}"
exit ${fail}
