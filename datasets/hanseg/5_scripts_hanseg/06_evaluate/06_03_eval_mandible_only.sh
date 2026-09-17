#!/usr/bin/env bash
# Re-score the hanseg OOD arms on MANDIBLE-ONLY instead of the mandible UNION.
#
# WHY THIS EXISTS (2026-09-17)
# ----------------------------
# The union scoring rests on a claim that is FALSE: HaN-Seg's `Bone_Mandible` does NOT
# include the lower dentition. Verified three ways —
#   1. Brouwer et al. 2015 consensus guidelines, verbatim: "The mandible was defined as
#      the entire mandible bone, without teeth."
#   2. HaN-Seg's own paper (Podobnik, Med Phys 2023 sec 2.2) states it followed those
#      guidelines (its ref 10) and declares its deviations — mandible is not among them.
#   3. Empirically: inside the mask p99 ~ 1500 HU (cortical bone, no enamel population),
#      and 96.6-99.7% of enamel-range (>2000 HU) voxels in the mandible's bounding box
#      lie OUTSIDE the mask, as a shell starting at the mask surface.
#
# Scoring `mandible u lower_teeth` against a teeth-excluded GT charges every method a
# systematic false-positive penalty on BOTH hanseg arms. The penalty scales with how much
# tooth a method predicts, so it is NOT uniform across methods and may be masking real
# separation — which is why the 3-way mandible tie has to be re-checked.
#
# NEITHER LABEL IS AN EXACT MATCH — measured, state this when reporting:
#   HaN-Seg Bone_Mandible        = bone + ROOTS, minus crowns  (solid envelope:
#                                  per-slice hole-fill ratio 1.0000 on every case)
#   ToothFairy2 mandible         = bone, minus roots           (sockets carved out:
#                                  fill ratio 1.073) -> UNDER-covers
#   ToothFairy2 mandible u teeth = bone + roots + crowns       -> OVER-covers
# Roots are 28.3% of lower_teeth volume; crowns 9.6% of the union volume. Ceiling Dice for
# a perfect prediction: union 94.93%, mandible-only 97.77%. Mandible-only more than halves
# the systematic error, so it is the better of the two — but it is not exact, and must not
# be reported as a clean one-to-one match.
#
# HOW: the existing hanseg eval already passes --label_map '{"mandible":[1,1]}'; it just
# points at the merged `_mandible_union` dir. Pointing it at the RAW 3-class prediction dir
# instead IS mandible-only (label 1 = mandible; 2/3 score as background). No re-prediction,
# no new merge pass — evaluation only.
#
# SAFETY: writes to a `mandible_only/` subdir. The existing union metrics are NOT touched,
# so both views stay comparable (same convention as `ablations/`).
#
# Run ON TamIA (predictions live on its scratch), submitted through Slurm.
set -uo pipefail

PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
HS_DIR="${PROJECT_ROOT}/datasets/hanseg/5_scripts_hanseg"
PY="${PROJECT_ROOT}/.venv/bin/python"
SUMMARIZE="${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py"

HS_PRED=/scratch/p/paulh/hanseg/8_results/01_predictions/toothfairy2_model/cbct
HS_METRICS=/scratch/p/paulh/hanseg/8_results/02_metrics/toothfairy2_model/cbct
HS_GT=/scratch/p/paulh/hanseg/2_nnUNet/raw/labelsTs_ct

ITEMS="${ITEMS_OVERRIDE:-ct mrt1}"
fail=0; n_ok=0

echo "[mo] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mo] GT=${HS_GT} ($(find "${HS_GT}" -name '*.nii.gz' 2>/dev/null | wc -l) files)"

for CAT in nnUNet auglab; do
  [ -d "${HS_PRED}/${CAT}" ] || continue
  for RUNDIR in "${HS_PRED}/${CAT}"/*/; do
    RID="$(basename "${RUNDIR}")"
    [ "${RID}" = "_logs" ] && continue

    # Mirror the run's EXISTING metrics location so the mandible_only tree matches the
    # union tree one-for-one (headline runs flat, ladder rungs under ablations/).
    if   [ -d "${HS_METRICS}/${CAT}_${RID}" ];            then SUB=""
    elif [ -d "${HS_METRICS}/ablations/${CAT}_${RID}" ];  then SUB="ablations"
    else echo "[mo] WARN no existing metrics dir for ${CAT}_${RID}, defaulting flat" >&2; SUB=""
    fi

    for F in 0 1 2; do
      for IT in ${ITEMS}; do
        D="${RUNDIR}fold${F}/${IT}"
        [ -d "$D" ] || { echo "[mo] MISSING ${D}" >&2; fail=1; continue; }
        n=$(find "$D" -name '*.nii.gz' 2>/dev/null | wc -l)
        [ "$n" -gt 0 ] || { echo "[mo] EMPTY ${D}" >&2; fail=1; continue; }

        OUT="${HS_METRICS}/mandible_only${SUB:+/${SUB}}/${CAT}_${RID}/fold${F}"
        mkdir -p "$OUT"
        "$PY" "${HS_DIR}/06_evaluate/06_00_evaluate.py" \
            --pred_dir "$D" --gt_dir "${HS_GT}" \
            --label_map '{"mandible": [1, 1]}' --name "${IT}" \
            --out_csv "${OUT}/${IT}_metrics.csv" --workers 6 \
            > "${OUT}/${IT}_eval.log" 2>&1 \
          || { echo "[mo] EVAL FAILED ${RID} fold${F} ${IT}" >&2; fail=1; continue; }
        rows=$(( $(wc -l < "${OUT}/${IT}_metrics.csv") - 1 ))
        [ "$rows" = "$n" ] || echo "[mo] WARN row/case mismatch ${rows}/${n} ${RID} f${F} ${IT}" >&2
        echo "[mo] ${CAT}_${RID} fold${F} ${IT}: ${rows} rows (of ${n} preds)"
        n_ok=$((n_ok+1))
      done
      OUT="${HS_METRICS}/mandible_only${SUB:+/${SUB}}/${CAT}_${RID}/fold${F}"
      if ls "${OUT}"/*_metrics.csv >/dev/null 2>&1; then
        "$PY" "$SUMMARIZE" "$OUT" "$RID" "$F" --groups ${ITEMS} --group-col contrast \
            --groups-word Contrasts >> "${OUT}/summarize.log" 2>&1 \
          || { echo "[mo] SUMMARIZE FAILED ${RID} fold${F}" >&2; fail=1; }
      fi
    done
  done
done

echo "[mo] done: ${n_ok} evaluations, fail=${fail}"
exit ${fail}
