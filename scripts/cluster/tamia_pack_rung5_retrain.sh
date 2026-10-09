#!/usr/bin/env bash
# TamIA launcher: RE-TRAIN ladder rung 5 (PALETTE alone, real fill; AugLab GPU synth + standard
# spatial DA only, every other AugLab transform p=0) for the 3 ladders whose rung 5 was trained
# (fully or partly) with the pre-2026-06-25 native src/ synthesis path:
#   chaos t2spir       200 ep  (04_28)  old run: nnUNetTrainerCHAOSV26_6_2_p50, no AugLab transform json
#   on-harmony T1w    2000 ep  (04_08)  old run: no AugLab transform json
#   brats t2w         2500 ep  (04_14)  old run: ~800 ep native src, then resumed on AugLab (mixed)
# 3 runs x folds 0 1 2 = 9 folds, ONE whole-node 4xH100 pack. Fresh RUN_IDs; the old runs are untouched.
#
#   cd /project/aip-jcohen/paulh/mri_synthesis_project        (on TamIA)
#   bash scripts/cluster/tamia_pack_rung5_retrain.sh record       # record + verify, no submit
#   bash scripts/cluster/tamia_pack_rung5_retrain.sh submit       # submit the training chain
#   bash scripts/cluster/tamia_pack_rung5_retrain.sh queue-post   # queue predict (afterany: last train job) + eval (afterany: predict)
# Pack root persisted in STATE so record/submit/queue-post share one recording + one set of RUN_IDs.
# Predict/eval are separate jobs (tamia_rung5_predict_job.sh / tamia_rung5_eval_job.sh); metrics go to ablations/.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
T=benchmark/02_tasks
STATE="${SCRATCH}/_packruns_rung5_retrain_root.txt"
MODE="${1:?usage: record|submit|queue-post}"

if [ -f "${STATE}" ]; then
    ROOT="$(cat "${STATE}")"
else
    [ "${MODE}" = record ] || { echo "nothing recorded yet"; exit 1; }
    ROOT="${SCRATCH}/_packruns/rung5_retrain_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${ROOT}"; echo "${ROOT}" > "${STATE}"
fi
echo "[rung5] ROOT=${ROOT}"
RUNIDS="${ROOT}/RUN_IDS.env"

W_BRATS="$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/04_train/04_14_train_t2w_v26_6_2_train050_val100.sh"
W_ONH="$T/brain_healthy/on-harmony/5_scripts_on-harmony/04_train/04_08_train_t1w_v26_6_2_train050_val100.sh"
W_CHAOS="$T/abdomen_healthy/chaos/5_scripts_chaos/04_train/04_28_train_t2spir_v26_6_2_train050_val100.sh"

if [ "${MODE}" = record ]; then
    if [ -f "${ROOT}/index.tsv" ]; then echo "[rung5] already recorded (${ROOT}/index.tsv) -- not re-recording"; exit 0; fi
    TS="$(date +%Y%m%d_%H%M%S)"
    BRATS_RUN="brats2024-glioma_t2w_v26_6_2_train050_val100_${TS}"
    ONH_RUN="on-harmony_T1w_v26_6_2_train050_val100_${TS}"
    CHAOS_RUN="chaos_t2spir_v26_6_2_train050_val100_${TS}"
    printf 'BRATS_RUN=%s\nONH_RUN=%s\nCHAOS_RUN=%s\n' "${BRATS_RUN}" "${ONH_RUN}" "${CHAOS_RUN}" > "${RUNIDS}"

    # record <wrapper> <tamia env file> <RUN_ID>   (same recipe as tamia_pack_rung6_pv.sh)
    record() {
        local wrapper="$1" tenv="$2" run="$3"
        local envf; envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
        echo "[record] $(basename "${wrapper}") run=${run} (env ${envf}, ${tenv})"
        (
            # shellcheck disable=SC1090
            source "$(dirname "${wrapper}")/../00_utils/${envf}"
            # shellcheck disable=SC1090
            source "scripts/cluster/${tenv}"
            # wrappers re-source env*.sh (re-exports nnUNet_results to /project); train_common honours NNUNET_RESULTS_BASE first.
            # Pin it to the CO-LOCATED layout predict_common/06_01 read from (<PREDICTIONS_ROOT>/<model>/<contrast>/nnUNet/<RUN_ID>):
            # brats' tamia_env.sh alone would put it in the flat 8_results/nnUNet/, which predict cannot find. (chaos/on-harmony already match.)
            export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/nnUNet"
            RUN_JOB_PACK_DIR="${ROOT}" bash "${wrapper}" "${run}"
        )
    }
    # ORDER MATTERS: PACK_GPU_MAP in `submit` is positional (brats f0-2, on-harmony f0-2, chaos f0-2)
    record "${W_BRATS}" tamia_env.sh           "${BRATS_RUN}"
    record "${W_ONH}"   tamia_env_onharmony.sh "${ONH_RUN}"
    record "${W_CHAOS}" tamia_env_chaos.sh     "${CHAOS_RUN}"

    echo; echo "=== verification (every recorded fold) ==="
    bad=0
    n=$(wc -l < "${ROOT}/index.tsv"); echo "-- ${n} folds (expect 9)"; [ "${n}" = 9 ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    while IFS=$'\t' read -r cmd log name done_; do
        clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
        pre=$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_preprocessed=//p' | head -1)
        res=$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)
        cfg=$(printf '%s\n' "$clean" | sed -n 's/^ *export AUGLAB_PARAMS_GPU_JSON=//p' | head -1 | sed 's|.*/||')
        vcfg=$(printf '%s\n' "$clean" | sed -n 's/^ *export AUGLAB_VAL_PARAMS_GPU_JSON=//p' | head -1 | sed 's|.*/||')
        ep=$(printf '%s\n' "$clean" | sed -n 's/^ *export NNUNET_NUM_EPOCHS=//p' | head -1)
        tr=$(printf '%s\n' "$clean" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
        case "$name" in
            *brats*) expect_ep=2500; expect_tr=nnUNetTrainerBraTS2024GliomaT2wV26_6_2_train050_val100;;
            *on-harmony*) expect_ep=2000; expect_tr=nnUNetTrainerOnHarmonyV26_6_2_train050_val100;;
            *chaos*) expect_ep=200; expect_tr=nnUNetTrainerCHAOSV26_6_2_p50;;
            *) expect_ep=?; expect_tr=?;;
        esac
        ok=OK
        case "$pre" in /scratch/*) [ -d "$pre" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
        case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
        grep -q '/01_predictions/.*_model/.*/nnUNet/' <<<"$res" || ok="${ok},RESULTS_NOT_COLOCATED"
        [ "$cfg" = "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json" ] || ok="${ok},WRONG_TRAIN_CFG($cfg)"
        [ "$vcfg" = "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json" ] || ok="${ok},WRONG_VAL_CFG($vcfg)"
        [ "$ep" = "$expect_ep" ] || ok="${ok},EPOCHS($ep!=${expect_ep})"
        [ "$tr" = "$expect_tr" ] || ok="${ok},TRAINER($tr!=${expect_tr})"
        grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
        grep -q '_pv_' <<<"$res$cfg" && ok="${ok},PV_RUNG6_LEAK"
        runid_ok=0; for r in "${BRATS_RUN}" "${ONH_RUN}" "${CHAOS_RUN}"; do grep -q "$r" <<<"$res" && runid_ok=1; done
        [ "$runid_ok" = 1 ] || ok="${ok},RESULTS_NOT_FRESH_RUNID"
        [ -e "$res" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
        [ "$ok" = OK ] || bad=1
        printf '   %-6s %s\n      trainer=%s ep=%s\n      preproc=%s\n      results=%s\n' "$ok" "$name" "$tr" "$ep" "$pre" "$res"
    done < "${ROOT}/index.tsv"
    [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"
    exit 0
fi

if [ "${MODE}" = submit ]; then
    [ -f "${ROOT}/index.tsv" ] || { echo "run record first"; exit 1; }
    # brats f0 f1 f2 -> GPUs 0 1 2 (heaviest, one per GPU); on-harmony f0-2 -> GPU 3; chaos f0-2 (200 ep, short) -> GPUs 0 1 2
    # NOT sized with a probe (see report) -- chain of 5 is safe: tail jobs exit immediately once every checkpoint_final exists.
    PACK_GPU_MAP="0 1 2 3 3 3 0 1 2" PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${PACK_TIME:-23:59:00}" \
    PACK_CHAIN="${PACK_CHAIN:-5}" PACK_USE_MPS=0 PACK_JOB_NAME="rung5_retrain" \
      bash scripts/job_runner/run_job_pack_submit.sh "${ROOT}" | tee "${ROOT}/submit.log"
    grep -oE 'last job id=[0-9]+' "${ROOT}/submit.log" | grep -oE '[0-9]+' > "${ROOT}/TRAIN_LAST_JOB"
    echo "[rung5] last training job: $(cat "${ROOT}/TRAIN_LAST_JOB")"
    exit 0
fi

if [ "${MODE}" = queue-post ]; then
    LAST="$(cat "${ROOT}/TRAIN_LAST_JOB")"
    PJ="$(sbatch --parsable --dependency=afterany:"${LAST}" --job-name=rung5_predict --time=05:00:00 \
          --output="${ROOT}/predict_job_%j.out" --export="ALL,ROOT=${ROOT},SKIP=${SKIP:-}" scripts/cluster/tamia_rung5_predict_job.sh)"
    echo "${PJ}" > "${ROOT}/PREDICT_JOB"; echo "[rung5] predict job ${PJ} (afterany:${LAST})"
    submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
               echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
               sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
    EJ="$(submit --dependency=afterany:"${PJ}" --job-name=rung5_eval --time=05:00:00 --cpus-per-task=16 --mem=96G \
          --account=aip-jcohen --output="${ROOT}/eval_job_%j.out" --export="ALL,ROOT=${ROOT},SKIP=${SKIP:-}" scripts/cluster/tamia_rung5_eval_job.sh)"
    echo "${EJ}" > "${ROOT}/EVAL_JOB"; echo "[rung5] eval job ${EJ} (afterany:${PJ})"
    exit 0
fi
echo "unknown mode ${MODE}"; exit 2
