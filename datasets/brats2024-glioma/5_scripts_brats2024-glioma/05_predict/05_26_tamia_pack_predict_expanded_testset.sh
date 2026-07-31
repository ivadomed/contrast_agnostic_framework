#!/usr/bin/env bash
# TAMIA whole-node PREDICT job for the EXPANDED brats2024-glioma test set (341 cases,
# +271 from BraTS-GLI 2024 AdditionalTrainingData). Predicts all 7 canonical methods x
# both trained-model families (T1n, T2w) x all 4 contrasts, all 3 folds. Checkpoints
# already staged on tamia scratch (synced from Vulcan); no training dependency needed
# (unlike 05_25) since every checkpoint already exists.
#
# Usage (run ON tamia, from datasets/brats2024-glioma/5_scripts_brats2024-glioma/05_predict/):
#   bash 05_26_tamia_pack_predict_expanded_testset.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

PACK_DIR="/scratch/p/paulh/brats2024-glioma/_packruns/predict_expanded_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"
echo "${PACK_DIR}" > /scratch/p/paulh/brats2024-glioma/_packruns/.last_expanded_predict_pack_dir

BODY="${PACK_DIR}/_predict_job_body.sh"
cat > "${BODY}" <<SBODY
#!/bin/bash
#SBATCH --job-name=brats_expanded_predict
#SBATCH --account=aip-jcohen
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=${PACK_DIR}/predict_job_%j.out
set -uo pipefail
cd '${HERE}'

# Self-contained env setup — do NOT rely on the submitting shell's inherited
# environment (that dependency bit this project once already, per
# 00_commun_scripts/00_00_utils/common_env.sh's comment on the mslesseg->open-ms
# tamia cross-predict incident). Source both explicitly, in the documented order.
source ../00_utils/env.sh
source ../../../../scripts/cluster/tamia_env.sh
echo "[predict-job] nnUNet_raw=\$nnUNet_raw"
echo "[predict-job] PREDICTIONS_ROOT=\$PREDICTIONS_ROOT"
echo "[predict-job] SPLITS_DIR=\$SPLITS_DIR"

echo "[predict-job] recording all 14 runs' fold-commands..."

RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_01_predict_t1n_baseline.sh      brats2024-glioma_t1n_baseline_20260622_044535
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_05_predict_auglab_default.sh    brats2024-glioma_t1n_auglab_default_20260622_044535
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_07_predict_synthseg_noEM.sh     brats2024-glioma_t1n_synthseg_noEM_20260622_044535
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_08_predict_synthseg_EM.sh       brats2024-glioma_t1n_synthseg_EM_20260622_044535
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_18_predict_t1n_srcsm.sh         brats2024-glioma_t1n_srcsm_20260709_122015
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_24_predict_t1n_auglabAug_v26_6_2_train050_val100.sh brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val100_20260725_113540

RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_09_predict_t2w_baseline.sh      brats2024-glioma_t2w_baseline_20260620_125115
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_10_predict_t2w_auglab_default.sh brats2024-glioma_t2w_auglab_default_20260620_125306
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_11_predict_t2w_synthseg_EM.sh   brats2024-glioma_t2w_synthseg_EM_20260620_125354
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_12_predict_t2w_synthseg_noEM.sh brats2024-glioma_t2w_synthseg_noEM_20260620_125442
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_19_predict_t2w_srcsm.sh         brats2024-glioma_t2w_srcsm_20260709_122045
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540
RUN_JOB_PACK_DIR='${PACK_DIR}' bash 05_23_predict_t2w_auglabAug_v26_6_2_train050_val100_dualval.sh brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val100_20260725_113540

echo "[predict-job] recorded \$(wc -l < '${PACK_DIR}/index.tsv') fold-commands:"
cut -f3 '${PACK_DIR}/index.tsv' | sed 's/^/  /'

echo "[predict-job] host=\$(hostname) job=\$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

declare -a PIDS NAMES GPUS
i=0
G=4
while IFS=\$'\t' read -r cmdfile log name donefile; do
    [ -n "\$cmdfile" ] || continue
    gpu=\$(( i % G ))
    plog="\${cmdfile%.sh}.log"
    echo "[predict-job] launch '\$name' on GPU \$gpu -> \$plog"
    CUDA_VISIBLE_DEVICES=\$gpu bash "\$cmdfile" > "\$plog" 2>&1 &
    PIDS[\$i]=\$!; NAMES[\$i]="\$name"; GPUS[\$i]=\$gpu
    i=\$(( i + 1 ))
    sleep 5
done < '${PACK_DIR}/index.tsv'

echo "[predict-job] \$i fold-predict commands launched; waiting..."
rc_all=0
for j in "\${!PIDS[@]}"; do
    if wait "\${PIDS[\$j]}"; then echo "[predict-job] OK   '\${NAMES[\$j]}' (GPU \${GPUS[\$j]})"
    else rc=\$?; rc_all=1; echo "[predict-job] FAIL '\${NAMES[\$j]}' (GPU \${GPUS[\$j]}) exit=\$rc"; fi
done
echo "[predict-job] done rc_all=\$rc_all"
exit \$rc_all
SBODY

echo "[tamia-predict-pack] submitting expanded-testset predict job"
sbatch "${BODY}"
