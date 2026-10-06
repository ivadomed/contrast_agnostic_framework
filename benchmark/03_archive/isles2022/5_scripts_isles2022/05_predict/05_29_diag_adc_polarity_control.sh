#!/usr/bin/env bash
# DIAGNOSTIC (NOT a benchmark item, results never enter the headline/ladder tables): ADC POLARITY POSITIVE CONTROL for isles2022 (2026-10-06).
# Question: ISLES' two training contrasts (DWI, FLAIR) both show the lesion BRIGHT, so the baseline may transfer between them with no real contrast shift for domain
# randomization to fix. ADC shows the lesion DARK (a true polarity shift). If the code is sound: baseline should collapse on ADC, contrast-randomized methods should hold.
# Re-uses the REAL trained runs through symlinks (read-only), writes only under $SCRATCH/isles2022/_diag_adc. Run ON TamIA:
#   bash 05_29_diag_adc_polarity_control.sh submit     # records predict commands (whole-node pack) + queues the eval job behind it
#   bash 05_29_diag_adc_polarity_control.sh eval       # (run by the queued job) evaluates every diag run INLINE on the ADC item
# Needs the ADC test inputs staged at $DIAG/2_nnUNet/raw/Dataset14{0,1}_*/{imagesTs_adc,labelsTs_adc} + dataset.json (rsync from Vulcan's toDelete copy).
set -euo pipefail
MODE="${1:?submit|eval}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/../../../../../.." && pwd)"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}" RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
source "${HERE}/../00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_isles2022.sh"
source "${ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
DIAG="${SCRATCH}/isles2022/_diag_adc"; REAL_PRED="${PREDICTIONS_ROOT}"; REAL_MT="${MODEL_TYPE}"
METHODS="${DIAG_METHODS:-baseline auglab_default srcsm auglabAug_v26_6_2_train050_val000}"
export nnUNet_raw="${DIAG}/2_nnUNet/raw"
declare -A DSDIR=( [dwi]="Dataset${DATASET_ID_DWI}_ISLES2022_DWI" [flair]="Dataset${DATASET_ID_FLAIR}_ISLES2022_FLAIR" )

if [ "${MODE}" = submit ]; then
    PACK="${DIAG}/pack_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${PACK}"; export RUN_JOB_PACK_DIR="${PACK}"
    for c in dwi flair; do
        pins="${REAL_PRED}/${REAL_MT}/${c}/roster_run_ids.tsv"; [ -f "${pins}" ] || { echo "ERROR: no pins ${pins}" >&2; exit 1; }
        for m in ${METHODS}; do
            read -r _m cat rid < <(awk -F'\t' -v m="${m}" '$1==m' "${pins}")
            rd="${REAL_PRED}/${REAL_MT}/${c}/${cat}/${rid}"; roster_run_trainer_id "${rd}" || { echo "ERROR: no model in ${rd}" >&2; exit 1; }
            drd="${DIAG}/01_predictions/${REAL_MT}/${c}/${cat}/${rid}"; mkdir -p "${drd}"
            ln -sfn "$(ls -d "${rd}"/Dataset*)" "${drd}/$(basename "$(ls -d "${rd}"/Dataset*)")"      # read-only view of the real model
            for F in 0 1 2; do
                out="${drd}/fold${F}/adc"; mkdir -p "${out}"
                run_job --name "diag_adc_${c}_${m}_f${F}" --gpus 1 --slot "${F}" --log "${PACK}/diag_${c}_${m}_f${F}.log" --wait -- bash -c "
                    export nnUNet_raw='${nnUNet_raw}' nnUNet_preprocessed='${nnUNet_preprocessed}' nnUNet_results='${drd}'
                    export NNUNET_PROJECT_ROOT='${ROOT}' PYTHONPATH='${PYTHONPATH}' TF_USE_LEGACY_KERAS=1
                    cd '${ROOT}'
                    .venv/bin/nnUNetv2_predict -i '${nnUNet_raw}/${DSDIR[$c]}/imagesTs_adc' -o '${out}' -d ${RUN_DATASET_ID} -c 3d_fullres -tr ${RUN_TRAINER} -f ${F} --disable_tta -chk checkpoint_best.pth -npp 4 -nps 2"
            done
        done
    done
    n=$(grep -c . "${PACK}/index.tsv"); echo "[diag] recorded ${n} predict commands (expect $(( $(echo ${METHODS} | wc -w) * 6 )))"
    pid="$(PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME=01:00:00 PACK_CHAIN=1 PACK_JOB_NAME=isles_diag_adc bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK}" | grep -o 'Submitted batch job [0-9]*' | awk '{print $4}')"
    echo "[diag] predict pack job ${pid}"
    RUN_JOB_DEPENDENCY="afterany:${pid}" run_job --name isles_diag_adc_eval --gpus 0 --cpus 8 --mem 32G --time 02:00:00 --log "${DIAG}/eval_diag.log" -- bash "${HERE}/05_29_diag_adc_polarity_control.sh" eval
    echo "[diag] eval job queued behind ${pid}; log ${DIAG}/eval_diag.log"; exit 0
fi

# ── eval: shared companion-free own-model driver, item adc only, diag base dirs (never the real ones) ──
export EVAL_INLINE=1 PREDICTIONS_ROOT="${DIAG}/01_predictions" METRICS_ROOT="${DIAG}/02_metrics"
for c in dwi flair; do
    pins="${REAL_PRED}/${REAL_MT}/${c}/roster_run_ids.tsv"
    for m in ${METHODS}; do
        read -r _m cat rid < <(awk -F'\t' -v m="${m}" '$1==m' "${pins}")
        ( export TRAINING_CONTRAST="${c}"; source "${HERE}/../00_utils/env.sh"        # re-derive MODEL_TYPE-based paths for this contrast
          export PREDICTIONS_ROOT="${DIAG}/01_predictions" METRICS_ROOT="${DIAG}/02_metrics" nnUNet_raw="${DIAG}/2_nnUNet/raw"
          EVAL_ITEMS="adc"; EVAL_LABELS="lesion"; EVAL_JOB_PREFIX="isles_diag_adc"
          [ "${c}" = dwi ] && EVAL_DATASET_ID="${DATASET_ID_DWI}" || EVAL_DATASET_ID="${DATASET_ID_FLAIR}"
          source "${ROOT}/benchmark/00_commun_scripts/00_03_evaluate/evaluate_run_common.sh" "${rid}" "${cat}" all ) || echo "[diag] EVAL FAILED ${c}/${m}"
    done
done
echo "[diag] done: ${DIAG}/02_metrics"
