#!/bin/bash
#SBATCH --job-name=cirrmri_liver_pack_predict_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/cirrmri-liver/_packruns/slurm_%j.out
# CIRRMRI-LIVER pack PREDICT + EVALUATE, single TamIA whole-node H100 job.
# Timing probe (job 390876): 628 cases (310 t1 + 318 t2) predicted in 5m47s on 1
# GPU -> ~1.5-2h for all 48 fold-tasks at true bounded 4-way concurrency. 8h budget
# leaves generous margin (well under TamIA's 24h cap).
#
# Every lesson from msd-spleen's onboarding applied from the START this time (see
# project_msd_spleen_onboarding.md memory for the full incident history):
#  1. Source the TamIA scratch-path env override BEFORE Phase 1 (predict), not
#     just before evaluate -- each predict wrapper is a fresh subprocess that
#     re-sources env.sh itself; only an already-exported override survives that.
#  2. TRUE bounded 4-way worker pool (one task at a time per GPU slot, waiting for
#     completion before starting the next) -- NOT a stagger-launch-everything
#     loop, which oversubscribes CPU/RAM via nnU-Net's internal preprocessing
#     workers and can silently truncate predictions without a nonzero exit code.
#  3. Evaluate calls the shared evaluate.py DIRECTLY (bypassing 06_01_evaluate_run.sh
#     and run_job entirely) -- run_job's internal sbatch call would itself fail on
#     tamia (whole-node-only scheduler rejects a --gpus 1 nested request; hit this
#     exact failure on the first timing-probe attempt) even for a CPU-only job,
#     and even if it didn't, nesting more sbatch submissions from inside a running
#     job just adds needless queue overhead.
#  4. Evaluate concurrency throttled (4 concurrent x 2 workers = 8 processes) to
#     avoid the OOM the first eval attempt hit at 12x4=48.
#  5. Verify EXACT file/row counts for every output, every time -- never trust
#     existence or a self-reported exit code alone.
set -uo pipefail
HERE_CL="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/cirrmri-liver/5_scripts_cirrmri-liver"
HERE_CHAOS="/project/aip-jcohen/paulh/mri_synthesis_project/datasets/chaos/5_scripts_chaos"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/cirrmri-liver/_packruns/predict_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE_CL}/00_utils/env.sh"
source /project/aip-jcohen/paulh/mri_synthesis_project/scripts/cluster/tamia_env_cirrmri-liver_chaoscross.sh
echo "[pack] CHAOS_DATASET_ROOT=${CHAOS_DATASET_ROOT}"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

echo "[pack] ==== PHASE 1: record all 16 runs x 3 folds (=48), items=t1+t2 ===="
cd "${HERE_CL}/05_predict"
for wrapper in \
    "05_02_predict_chaos_baseline.sh" \
    "05_03_predict_chaos_v26_6_2.sh" \
    "05_04_predict_chaos_synthseg_EM.sh" \
    "05_05_predict_chaos_auglab_default.sh" \
    "05_06_predict_chaos_synthseg_noEM.sh" \
    "05_07_predict_chaos_srcsm.sh" \
    "05_08_predict_chaos_auglabAug_v26_6_2_train050_val000.sh" \
    "05_09_predict_chaos_auglabAug_v26_6_2_train050_val100.sh" \
    "05_11_predict_chaos_t2spir_baseline.sh" \
    "05_12_predict_chaos_t2spir_v26_6_2.sh" \
    "05_13_predict_chaos_t2spir_synthseg_EM.sh" \
    "05_14_predict_chaos_t2spir_auglab_default.sh" \
    "05_15_predict_chaos_t2spir_synthseg_noEM.sh" \
    "05_16_predict_chaos_t2spir_srcsm.sh" \
    "05_17_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val000.sh" \
    "05_18_predict_chaos_t2spir_auglabAug_v26_6_2_train050_val100.sh" ; do
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "" all t1 t2
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: bounded worker pool, 2 concurrent tasks PER GPU (8-way total) ===="
# GPU memory is nowhere near the bottleneck here (timing probe showed ~2.3GB/80GB
# used per task) -- CPU is the real constraint, so PREDICT_EXTRA_FLAGS was halved
# (-npp 12/-nps 6 -> -npp 6/-nps 3, in 05_01_predict_common.sh) to keep total
# preprocessing-worker count reasonable at 2x concurrency (9 workers/task x 8
# concurrent tasks = 72, a modest ~1.5x oversubscription of 48 CPUs -- nowhere
# near the ~15x oversubscription (42 tasks x 18 workers) that caused the original
# silent-truncation bug at unbounded concurrency).
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
SLOTS=8
G=4
worker() {
    local slot="$1" idx="$1"
    local gpu=$(( slot % G ))
    while [ "${idx}" -lt "${N}" ]; do
        IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
        local plog="${cmdfile%.sh}.log"
        echo "[pack] slot${slot}(GPU${gpu}): launch '${name}' -> ${plog}"
        CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
        echo "[pack] slot${slot}(GPU${gpu}): '${name}' exit=$?"
        idx=$(( idx + SLOTS ))
    done
}
pids=()
for slot in 0 1 2 3 4 5 6 7; do worker "${slot}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done"

echo "[pack] ==== PHASE 3: verify EXACT file counts (310 t1 / 318 t2 per fold) ===="
verify_fail=0
for d in "${PREDICTIONS_ROOT}"/chaos_model/*/*/*/fold*/t1 "${PREDICTIONS_ROOT}"/chaos_model/*/*/*/fold*/t2; do
    [ -d "${d}" ] || continue
    item="$(basename "${d}")"
    expect=310; [ "${item}" = "t2" ] && expect=318
    n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne "${expect}" ]; then
        echo "[pack] BAD COUNT (${n}/${expect}): ${d}" >&2
        verify_fail=1
    fi
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"

echo "[pack] ==== PHASE 4: evaluate (CPU-only, direct python calls, throttled) ===="
EVALUATE_PY="${HERE_CHAOS}/06_evaluate/06_00_evaluate.py"
FOV_JSON="${HERE_CHAOS}/06_evaluate/chaos_fov_margins.json"
GT_BASE="${nnUNet_raw}"

run_eval() {   # $1=contrast $2=category $3=run_id $4=fold $5=item(t1|t2)
    local contrast="$1" category="$2" run_id="$3" fold="$4" item="$5"
    local pred_dir="${PREDICTIONS_ROOT}/chaos_model/${contrast}/${category}/${run_id}/fold${fold}/${item}"
    local gt_dir="${GT_BASE}/labelsTs_${item}"
    local eval_dir="${METRICS_ROOT}/chaos_model/${contrast}/${category}_${run_id}/fold${fold}"
    local expect=310; [ "${item}" = "t2" ] && expect=318
    local n
    n=$(find "${pred_dir}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne "${expect}" ]; then
        echo "  ! ${run_id} fold${fold} ${item}: ${n}/${expect} predictions -- refusing to evaluate" >&2
        return 1
    fi
    mkdir -p "${eval_dir}"
    local mm
    mm=$("${PROJECT_ROOT}/.venv/bin/python" -c "import json;d=json.load(open('${FOV_JSON}'))['${contrast}']['liver'];print(d['sup_mm'],d['inf_mm'])")
    "${PROJECT_ROOT}/.venv/bin/python" "${EVALUATE_PY}" \
        --pred_dir "${pred_dir}" --gt_dir "${gt_dir}" --dataset_json "${CHAOS_DATASET_JSON}" \
        --labels liver --name "${item}" --out_csv "${eval_dir}/${item}_metrics.csv" --workers 2 \
        --fov_anchor_gt_ids 1 --fov_sup_mm ${mm% *} --fov_inf_mm ${mm#* } \
        > "${eval_dir}/${item}_eval.log" 2>&1
    local rows
    rows=$(( $(wc -l < "${eval_dir}/${item}_metrics.csv") - 1 ))
    echo "[pack] eval done: ${run_id} fold${fold} ${item} (${rows}/${expect} rows)"
    [ "${rows}" -eq "${expect}" ] || { echo "[pack] WARNING: ${run_id} fold${fold} ${item} wrote only ${rows}/${expect} rows" >&2; return 1; }
}
export -f run_eval
export EVALUATE_PY FOV_JSON GT_BASE PREDICTIONS_ROOT METRICS_ROOT PROJECT_ROOT CHAOS_DATASET_JSON

cat > "${PACK_DIR}/eval_tasks.txt" <<'TASKS'
t1in nnUNet chaos_t1in_baseline_20260614_153230
t1in nnUNet chaos_t1in_v26_6_2_train050_val100_20260615_213615
t1in auglab chaos_t1in_synthseg_EM_train100_val000_20260611_120000
t1in auglab chaos_t1in_auglab_default_20260611_120000
t1in auglab chaos_t1in_synthseg_noEM_train100_val000_20260611_120000
t1in auglab chaos_t1in_srcsm_20260710_011817
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615
t1in auglab chaos_t1in_auglabAug_v26_6_2_train050_val100_20260616_112420
t2spir nnUNet chaos_t2spir_baseline_20260620_111146
t2spir nnUNet chaos_t2spir_v26_6_2_train050_val100_20260620_112122
t2spir auglab chaos_t2spir_synthseg_EM_20260620_112357
t2spir auglab chaos_t2spir_auglab_default_20260620_112240
t2spir auglab chaos_t2spir_synthseg_noEM_20260620_112515
t2spir auglab chaos_t2spir_srcsm_20260709_121945
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202
t2spir auglab chaos_t2spir_auglabAug_v26_6_2_train050_val100_20260723_194053
TASKS

: > "${PACK_DIR}/eval_joblist.txt"
while read -r contrast category run_id; do
    [ -n "${contrast}" ] || continue
    for fold in 0 1 2; do
        for item in t1 t2; do
            echo "${contrast} ${category} ${run_id} ${fold} ${item}" >> "${PACK_DIR}/eval_joblist.txt"
        done
    done
done < "${PACK_DIR}/eval_tasks.txt"

cat "${PACK_DIR}/eval_joblist.txt" | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
rc_eval=$?
echo "[pack] evaluate phase done rc=${rc_eval}"

echo "[pack] ==== PHASE 5: merge each fold's t1+t2 into eval_all.csv (summarize_fold.py) ===="
# NOTE: this step was MISSING in the run that produced this dataset's first real
# results -- 06_04_aggregate_results.py and the cross-dataset aggregate_from_config.py
# both read eval_all.csv (not the raw t1_metrics.csv/t2_metrics.csv directly), so
# skipping this silently left "no evaluation data found" until caught and backfilled
# by hand on Vulcan afterward. Kept here so a future re-run doesn't repeat the gap.
find "${METRICS_ROOT}" -mindepth 2 -maxdepth 2 -type d | while read -r run_dir; do
    run_name="$(basename "${run_dir}")"
    run_id="${run_name#nnUNet_}"; run_id="${run_id#auglab_}"
    for fold_dir in "${run_dir}"/fold*/; do
        [ -d "${fold_dir}" ] || continue
        fold="$(basename "${fold_dir}" | sed 's/fold//')"
        if [ -f "${fold_dir}t1_metrics.csv" ] && [ -f "${fold_dir}t2_metrics.csv" ]; then
            "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/datasets/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
                "${fold_dir}" "${run_id}" "${fold}" --group-col modality --groups-word Modalities \
                --title-suffix " | CIRRMRI-LIVER MRI | liver only" --groups t1 t2 > /dev/null 2>&1
        fi
    done
done
echo "[pack] eval_all.csv merge done"

echo "[pack] ==== FINAL COMPREHENSIVE CHECK: all metrics files, exact row counts ===="
find "${METRICS_ROOT}" -name "t1_metrics.csv" -o -name "t2_metrics.csv" 2>/dev/null | sort | while read -r f; do
    n=$(( $(wc -l < "$f") - 1 ))
    exp=310; [[ "$f" == *t2_metrics.csv ]] && exp=318
    [ "${n}" -eq "${exp}" ] || echo "[pack] BAD ROW COUNT (${n}/${exp}): ${f}"
done
total=$(find "${METRICS_ROOT}" \( -name "t1_metrics.csv" -o -name "t2_metrics.csv" \) 2>/dev/null | wc -l)
echo "[pack] total metrics files: ${total} (expect 96 = 48 folds x 2 items)"
echo "[pack] ==== DONE (verify_fail=${verify_fail}, eval rc=${rc_eval}) ===="
