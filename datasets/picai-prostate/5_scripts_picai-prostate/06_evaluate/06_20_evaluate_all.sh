#!/usr/bin/env bash
# Evaluate ALL 14 picai-prostate runs (7 arms x 2 training modalities) — every fold, every
# test contrast — against the held-out GT. Thin driver over 06_01_evaluate_run.sh (T2W) and
# 06_02_evaluate_adc.sh (ADC), which is where the real work lives.
#
# Each of the 14 invocations submits its OWN CPU-only Slurm job (06_01 uses run_job --wait),
# so they are launched in parallel here and waited on together — 14 sequential --wait calls
# would take hours of wall clock for work that is embarrassingly parallel. The login-node
# cost is 14 sleeping waiters, which is well inside CLAUDE.md's login-node allowance; the
# actual metric computation all happens on compute nodes.
#
# RUN_IDs are read from the training packs' RUN_IDS.env (single source of truth — the same
# resolution 05_20_tamia_pack_predict_all.sh uses), so this cannot drift from what was
# trained and predicted. The val100 arm is derived from its val000 sibling by the
# "_val000_" -> "_val100_" substitution the DualVal trainer used to materialize it.
#
# Run AFTER 05_20_tamia_pack_predict_all.sh has completed.
#
# Usage (run ON tamia):
#   bash 06_20_evaluate_all.sh
#   EVAL_STAGGER=10 bash 06_20_evaluate_all.sh    # seconds between sbatch submissions
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
PACKRUNS="${SCRATCH}/picai-prostate/_packruns"
STAGGER="${EVAL_STAGGER:-5}"     # CLAUDE.md: space out rapid sbatch calls

rid() {
    grep -m1 -- "$2" "${PACKRUNS}/$1/RUN_IDS.env" \
        || { echo "ERROR: no run id matching '$2' in ${PACKRUNS}/$1/RUN_IDS.env" >&2; exit 1; }
}
T2W_OURS000=$(rid t2w_ours_srcsm _val000_)
ADC_OURS000=$(rid adc_ours_srcsm _val000_)

# script | RUN_ID | CATEGORY
JOBS=(
  "06_01_evaluate_run.sh|$(rid t2w_base_auglab _baseline_)|nnUNet"
  "06_01_evaluate_run.sh|$(rid t2w_base_auglab _auglab_default_)|auglab"
  "06_01_evaluate_run.sh|$(rid t2w_synthseg _synthseg_noEM_)|auglab"
  "06_01_evaluate_run.sh|$(rid t2w_synthseg _synthseg_EM_)|auglab"
  "06_01_evaluate_run.sh|$(rid t2w_ours_srcsm _srcsm_)|auglab"
  "06_01_evaluate_run.sh|${T2W_OURS000}|auglab"
  "06_01_evaluate_run.sh|${T2W_OURS000/_val000_/_val100_}|auglab"
  "06_02_evaluate_adc.sh|$(rid adc_base_auglab _baseline_)|nnUNet"
  "06_02_evaluate_adc.sh|$(rid adc_base_auglab _auglab_default_)|auglab"
  "06_02_evaluate_adc.sh|$(rid adc_synthseg _synthseg_noEM_)|auglab"
  "06_02_evaluate_adc.sh|$(rid adc_synthseg _synthseg_EM_)|auglab"
  "06_02_evaluate_adc.sh|$(rid adc_ours_srcsm _srcsm_)|auglab"
  "06_02_evaluate_adc.sh|${ADC_OURS000}|auglab"
  "06_02_evaluate_adc.sh|${ADC_OURS000/_val000_/_val100_}|auglab"
)

echo "[evaluate-all] launching ${#JOBS[@]} evaluation jobs"
declare -a PIDS NAMES
i=0
for spec in "${JOBS[@]}"; do
    IFS='|' read -r script runid category <<< "${spec}"
    echo "  -> ${runid}  (${category})"
    bash "${HERE}/${script}" "${runid}" "${category}" > "${SCRATCH}/picai-prostate/_logs/eval_${runid}.log" 2>&1 &
    PIDS[$i]=$!; NAMES[$i]="${runid}"
    i=$((i + 1))
    sleep "${STAGGER}"
done

echo "[evaluate-all] ${i} launched; waiting..."
rc_all=0
for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then echo "[evaluate-all] OK   ${NAMES[$j]}"
    else rc_all=1; echo "[evaluate-all] FAIL ${NAMES[$j]} — see ${SCRATCH}/picai-prostate/_logs/eval_${NAMES[$j]}.log"; fi
done
echo "[evaluate-all] done rc_all=${rc_all}"
exit ${rc_all}
