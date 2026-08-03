#!/usr/bin/env bash
# TAMIA: launch the ENTIRE picai-prostate training campaign — the 6-method suite on BOTH
# training modalities (T2W + ADC), 3 folds each = 36 fold-trainings, on 6 whole H100 nodes.
#
#   6 methods x 3 folds x 2 modalities = 36 fold-trainings
#   -> 7 headline arms, because the 6th method uses the DualVal trainer: ONE run emits both
#      auglabAug_v26_6_2_train050_val000 AND ..._val100 (see 04_06's header). That is the
#      project standard since 2026-07-25 and is selection-equivalent to training the two
#      separately — do NOT add a 7th training run.
#
# PLACEMENT — why each pack looks the way it does:
#   tamia allocates H100s by WHOLE NODE (4 GPUs), so a pack must keep all 4 busy or the idle
#   ones are visible waste. SRCSM trains ~3x slower per epoch than every other method, so a
#   srcsm fold sharing a GPU would become the straggler holding the whole node (and its whole
#   chain) open. Hence:
#     * the four "fast" packs put 2 methods x 3 folds = 6 folds on 4 GPUs, round-robin
#       (2,2,1,1) — every GPU busy, nothing slower than ~2x;
#     * the two "ours + srcsm" packs use an EXPLICIT map "0 0 0 1 2 3": all 3 DualVal folds
#       share GPU0 while each srcsm fold gets a GPU to itself. Both sides then cost ~3-4x a
#       lone fold, so the four GPUs finish at roughly the same time instead of three of them
#       idling behind srcsm. (nnU-Net plans to a ~8 GB VRAM budget per training, so 3 on one
#       80 GB H100 is comfortable.) This is what PACK_GPU_MAP was added to
#       run_job_pack_submit.sh for.
#   Chain lengths follow from the same asymmetry: ~2x folds need fewer 24h jobs than ~3.5x
#   ones. Over-provisioning is safe — a chain job whose folds are all finished exits at once.
#
# PREREQUISITES (in order): 00_00_download.sh, 00_01_bidsify.sh, 01_01_create_splits.sh,
# 02_00_convert.sh + 02_01_convert_adc.sh, 03_00_preprocess.sh + 03_01_preprocess_adc.sh.
# Recording reads nnUNet_raw to resolve the dataset dir name, so conversion MUST have run.
#
# Usage (run ON tamia):
#   bash 04_21_tamia_pack_launch_all.sh              # launch all 6 packs
#   bash 04_21_tamia_pack_launch_all.sh t2w_ours     # launch/extend just the named pack(s)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# pack_name | gpu_map ("-" = round-robin) | chain_len | wrappers...
PACKS=(
  "t2w_base_auglab|-|5|04_01_train_t2w_baseline.sh 04_02_train_t2w_auglab_default.sh"
  "t2w_synthseg|-|5|04_03_train_t2w_synthseg_noEM.sh 04_04_train_t2w_synthseg_EM.sh"
  "t2w_ours_srcsm|0 0 0 1 2 3|8|04_06_train_t2w_auglabAug_v26_6_2_train050_val000.sh 04_05_train_t2w_srcsm.sh"
  "adc_base_auglab|-|5|04_11_train_adc_baseline.sh 04_12_train_adc_auglab_default.sh"
  "adc_synthseg|-|5|04_13_train_adc_synthseg_noEM.sh 04_14_train_adc_synthseg_EM.sh"
  "adc_ours_srcsm|0 0 0 1 2 3|8|04_16_train_adc_auglabAug_v26_6_2_train050_val000.sh 04_15_train_adc_srcsm.sh"
)

WANTED=("$@")
want() {
    [ "${#WANTED[@]}" -eq 0 ] && return 0
    for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
    return 1
}

launched=0
for spec in "${PACKS[@]}"; do
    IFS='|' read -r name gpumap chain wrappers <<< "${spec}"
    want "${name}" || { echo "[launch-all] skip ${name}"; continue; }
    echo
    echo "================ pack: ${name} (chain=${chain}, gpu_map='${gpumap}') ================"
    # shellcheck disable=SC2086 — $wrappers is an intentional word-split list
    bash "${HERE}/04_20_tamia_pack.sh" "${name}" "${gpumap}" "${chain}" ${wrappers}
    launched=$((launched + 1))
    sleep 5      # space out sbatch bursts (CLAUDE.md: don't hammer the scheduler)
done
echo
echo "[launch-all] ${launched} pack chain(s) submitted."
echo "  queue:  squeue -u \$USER -o '%.10i %.22j %.9P %.2t %.11M %.11l %R'"
echo "  logs:   ls ${SCRATCH:-/scratch/p/paulh}/picai-prostate/_packruns/*/"
