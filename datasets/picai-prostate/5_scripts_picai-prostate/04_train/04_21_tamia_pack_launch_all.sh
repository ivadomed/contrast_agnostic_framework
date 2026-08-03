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
#   ones are visible waste. The layout was designed around srcsm being the ~3x-per-epoch
#   outlier it is on brats2024-glioma / on-harmony — a srcsm fold sharing a GPU would then be
#   the straggler holding the whole node (and its whole chain) open. Hence:
#     * the four "fast" packs put 2 methods x 3 folds = 6 folds on 4 GPUs, round-robin
#       (2,2,1,1) — every GPU busy, nothing carrying more than 2 folds;
#     * the two "ours + srcsm" packs use an EXPLICIT map "0 0 1 1 2 3" so that 2 of the 3
#       srcsm folds get a GPU entirely to themselves and the third shares with only one
#       DualVal fold — instead of round-robin, which would put TWO srcsm folds on one GPU and
#       leave the node hostage to them. This is what PACK_GPU_MAP was added to
#       run_job_pack_submit.sh for.
#   That premise turned out NOT to hold on this dataset (see MEASURED below) — srcsm is
#   roughly as cheap as everything else here — so the explicit map buys little in practice.
#   It is kept because it is still the correct shape if the asymmetry reappears, and because
#   these packs are already recorded and running with it.
#
#   KNOWN INEFFICIENCY: 6 folds over 4 GPUs means two GPUs carry 2 folds and two carry 1, so
#   the single-fold GPUs go idle at roughly the halfway point (~75% node utilisation over the
#   pack's life). Packing 12 folds per node (4 methods x 3 folds, 3 per GPU) would reach
#   ~100% and hold half as many nodes, but needs the per-fold VRAM measurement first.
#   NOTE ON DENSITY: every pack here caps at 2 folds per GPU — the density actually verified
#   on this cluster. A 3-on-one-GPU map would hold fewer nodes, but per-fold VRAM has not
#   been measured, and an OOM across a multi-day campaign is not worth the saving. Re-run
#   04_22_tamia_size_probe.sh (extended to report VRAM) if you want to revisit that.
#
#   MEASURED 2026-08-02 by 04_22_tamia_size_probe.sh, on a dedicated H100, this dataset:
#     OURS / DualVal  ~18 s/epoch      srcsm  ~15 s/epoch
#   So 2000 epochs is ~10 h solo, ~20 h with 2 folds sharing a GPU -> a chain of 3 x 23:59
#   is already generous. IMPORTANT: srcsm was NOT the ~3x outlier here that it is on
#   brats2024-glioma / on-harmony — it is actually slightly CHEAPER than DualVal on
#   picai-prostate. The explicit map is kept anyway because it still guarantees no two srcsm
#   folds share a GPU, but do not assume the 3x figure transfers to a new dataset: measure it.
#   A chain that turns out too SHORT is fixed by re-invoking 04_20 with the same PACK_DIR
#   (it reuses the recording + RUN_IDs and submits a fresh chain that resumes).
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
  "t2w_base_auglab|-|3|04_01_train_t2w_baseline.sh 04_02_train_t2w_auglab_default.sh"
  "t2w_synthseg|-|3|04_03_train_t2w_synthseg_noEM.sh 04_04_train_t2w_synthseg_EM.sh"
  "t2w_ours_srcsm|0 0 1 1 2 3|3|04_06_train_t2w_auglabAug_v26_6_2_train050_val000.sh 04_05_train_t2w_srcsm.sh"
  "adc_base_auglab|-|3|04_11_train_adc_baseline.sh 04_12_train_adc_auglab_default.sh"
  "adc_synthseg|-|3|04_13_train_adc_synthseg_noEM.sh 04_14_train_adc_synthseg_EM.sh"
  "adc_ours_srcsm|0 0 1 1 2 3|3|04_16_train_adc_auglabAug_v26_6_2_train050_val000.sh 04_15_train_adc_srcsm.sh"
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
