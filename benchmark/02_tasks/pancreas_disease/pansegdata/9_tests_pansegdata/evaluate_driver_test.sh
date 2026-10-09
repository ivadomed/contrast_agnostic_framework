#!/usr/bin/env bash
# Evaluate-driver test on FAKE predictions (= copies of the ground truth) for pansegdata, in a throwaway scratch results base. Run inside a CPU job (run_job):
#   run_job --name pansegdata_evaltest --gpus 0 --cpus 2 --mem 8G --time 00:30:00 --log <log> --wait -- bash evaluate_driver_test.sh
# Must show: (A) a complete fake run evaluates, Dice == 1.0, 1 + 42 cases x 2 items rows per fold;  (B) a missing prediction FAILS;  (C) a wrong CATEGORY FAILS.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS="$(cd "${HERE}/.." && pwd)"
T="${SCRATCH:?}/pansegdata_evaltest_$(date +%Y%m%d_%H%M%S)"
export PREDICTIONS_ROOT="${T}/01_predictions" METRICS_ROOT="${T}/02_metrics" EVAL_INLINE=1
RUN="pansegdata_t1wce_baseline_TESTFAKE"
RAW="${DS}/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE"
for F in 0 1 2; do for I in t1wce t2w; do
    D="${PREDICTIONS_ROOT}/pansegdata_model/t1wce/nnUNet/${RUN}/fold${F}/${I}"; mkdir -p "${D}"; cp "${RAW}/labelsTs_${I}/"*.nii.gz "${D}/"
done; done
n_cases=$(ls "${RAW}/labelsTs_t1wce" | wc -l)
EV="${DS}/5_scripts_pansegdata/06_evaluate/06_01_evaluate_run.sh"
ok=1
echo "== (A) complete fake run"
if bash "${EV}" "${RUN}" nnUNet > "${T}/A.log" 2>&1; then
    for F in 0 1 2; do
        CSV="${METRICS_ROOT}/pansegdata_model/t1wce/nnUNet_${RUN}/fold${F}/eval_all.csv"
        [ -f "${CSV}" ] || { echo "FAIL fold${F}: no eval_all.csv"; ok=0; continue; }
        rows=$(( $(wc -l < "${CSV}") - 1 )); want=$(( n_cases * 2 ))
        dice=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if(tolower($i)=="dice") c=i; next} {s+=$c; n++} END{printf "%.4f", s/n}' "${CSV}")
        echo "fold${F}: rows ${rows} (want ${want}), mean Dice ${dice}"
        [ "${rows}" = "${want}" ] && [ "${dice}" = "1.0000" ] || { echo "FAIL fold${F}"; ok=0; }
    done
else echo "FAIL (A): driver errored"; tail -15 "${T}/A.log"; ok=0; fi
echo "== (B) one missing prediction must FAIL"
rm "$(ls "${PREDICTIONS_ROOT}/pansegdata_model/t1wce/nnUNet/${RUN}/fold1/t2w/"*.nii.gz | head -1)"
if bash "${EV}" "${RUN}" nnUNet > "${T}/B.log" 2>&1; then echo "FAIL (B): evaluated despite a missing prediction"; ok=0; else echo "PASS (B): failed as it should"; grep -i -E "error|expected|missing|!=" "${T}/B.log" | head -3; fi
echo "== (C) wrong CATEGORY must FAIL"
if bash "${EV}" "${RUN}" auglab > "${T}/C.log" 2>&1; then echo "FAIL (C): accepted a wrong category"; ok=0; else echo "PASS (C): failed as it should"; grep -i error "${T}/C.log" | head -2; fi
echo "== VERDICT: $([ ${ok} = 1 ] && echo EVALUATE_DRIVER_TEST_PASSED || echo EVALUATE_DRIVER_TEST_FAILED)"
