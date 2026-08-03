#!/usr/bin/env bash
# TAMIA whole-node PREDICT pack for picai-prostate: all 7 arms x both training modalities
# (T2W, ADC) x 3 folds x 3 test contrasts (t2w/adc/hbv) — i.e. the complete cross-contrast
# test matrix, in ONE whole-node job.
#
# 14 run ids x 3 folds = 42 recorded fold-commands (predict_common.sh emits one run_job per
# FOLD, with that fold's three contrasts running sequentially inside it), packed round-robin
# across the node's 4 H100s. Prediction is minutes per fold-contrast, so unlike training this
# needs no dependency chain — PACK_CHAIN=1.
#
# RUN_IDs are read from the training packs' RUN_IDS.env files, so this cannot drift from what
# was actually trained. The val100 arm is derived from its val000 sibling by the same
# "_val000_" -> "_val100_" substitution the DualVal trainer used to materialize it.
#
# PREDICT NOTE: the val000 AND val100 wrappers both set
# TRAINER=nnUNetTrainerPICAIProstateAugLabDualVal — both mirrors live under that trainer's
# directory name (see 04_train/04_06's header).
#
# Usage (run ON tamia):
#   bash 05_20_tamia_pack_predict_all.sh
#   PACK_DIR=<existing> bash 05_20_tamia_pack_predict_all.sh   # re-submit an existing pack
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"

export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
PACKRUNS="${SCRATCH}/picai-prostate/_packruns"
PACK_DIR="${PACK_DIR:-${PACKRUNS}/predict_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${PACK_DIR}"
echo "[predict-pack] PACK_DIR=${PACK_DIR}"

# ── Resolve the trained RUN_IDs from the training packs (single source of truth) ──
rid() {  # rid <pack> <substring>
    grep -m1 -- "$2" "${PACKRUNS}/$1/RUN_IDS.env" \
        || { echo "ERROR: no run id matching '$2' in ${PACKRUNS}/$1/RUN_IDS.env" >&2; exit 1; }
}
T2W_BASELINE=$(rid t2w_base_auglab _baseline_)
T2W_AUGLAB=$(rid t2w_base_auglab _auglab_default_)
T2W_SS_NOEM=$(rid t2w_synthseg _synthseg_noEM_)
T2W_SS_EM=$(rid t2w_synthseg _synthseg_EM_)
T2W_SRCSM=$(rid t2w_ours_srcsm _srcsm_)
T2W_OURS000=$(rid t2w_ours_srcsm _val000_)
T2W_OURS100="${T2W_OURS000/_val000_/_val100_}"

ADC_BASELINE=$(rid adc_base_auglab _baseline_)
ADC_AUGLAB=$(rid adc_base_auglab _auglab_default_)
ADC_SS_NOEM=$(rid adc_synthseg _synthseg_noEM_)
ADC_SS_EM=$(rid adc_synthseg _synthseg_EM_)
ADC_SRCSM=$(rid adc_ours_srcsm _srcsm_)
ADC_OURS000=$(rid adc_ours_srcsm _val000_)
ADC_OURS100="${ADC_OURS000/_val000_/_val100_}"

# ── Record all 42 fold-commands ONCE (each wrapper in its own subshell → no env bleed) ──
if [ -f "${PACK_DIR}/index.tsv" ]; then
    echo "[predict-pack] index.tsv exists — reusing existing recording."
else
    while read -r wrapper runid; do
        [ -n "${wrapper}" ] || continue
        echo "[predict-pack] recording $(basename "${wrapper}") ${runid}"
        RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${HERE}/${wrapper}" "${runid}" all
    done <<EOF
05_02_predict_t2w_baseline.sh                             ${T2W_BASELINE}
05_03_predict_t2w_auglab_default.sh                       ${T2W_AUGLAB}
05_04_predict_t2w_synthseg_noEM.sh                        ${T2W_SS_NOEM}
05_05_predict_t2w_synthseg_EM.sh                          ${T2W_SS_EM}
05_06_predict_t2w_srcsm.sh                                ${T2W_SRCSM}
05_07_predict_t2w_auglabAug_v26_6_2_train050_val000.sh    ${T2W_OURS000}
05_08_predict_t2w_auglabAug_v26_6_2_train050_val100.sh    ${T2W_OURS100}
05_12_predict_adc_baseline.sh                             ${ADC_BASELINE}
05_13_predict_adc_auglab_default.sh                       ${ADC_AUGLAB}
05_14_predict_adc_synthseg_noEM.sh                        ${ADC_SS_NOEM}
05_15_predict_adc_synthseg_EM.sh                          ${ADC_SS_EM}
05_16_predict_adc_srcsm.sh                                ${ADC_SRCSM}
05_17_predict_adc_auglabAug_v26_6_2_train050_val000.sh    ${ADC_OURS000}
05_18_predict_adc_auglabAug_v26_6_2_train050_val100.sh    ${ADC_OURS100}
EOF
    echo "[predict-pack] recorded $(wc -l < "${PACK_DIR}/index.tsv") fold-commands:"
    cut -f3 "${PACK_DIR}/index.tsv" | sed 's/^/  /'
fi

# No done-marker exists for prediction (RUN_JOB_PACK_DONEFILE is unset here), so the pack
# body's "all folds already finished" short-circuit never fires — hence PACK_CHAIN=1.
PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_CPUS=48 PACK_MEM=0 \
PACK_TIME="${PACK_TIME:-08:00:00}" PACK_CHAIN=1 \
PACK_JOB_NAME="picai_predict_all" \
  bash "${ROOT}/scripts/job_runner/run_job_pack_submit.sh" "${PACK_DIR}"
