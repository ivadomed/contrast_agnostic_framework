#!/usr/bin/env bash
# Source at the top of every totalsegmri-pancreas pipeline script.   source "$(dirname "$0")/../00_utils/env.sh"
# >>> SCAFFOLDED by benchmark/create_eval_companion_scripts.py: review the header, BIDS_SUBDIR and the resource lines.
# totalsegmri-pancreas = EVAL-ONLY companion of the pansegdata task: its test items (t1gre t2like) are predicted with the pansegdata-trained models (source contrasts: t1wce t2w) and scored here.
# No training role, no Dataset<id>/ dir: flat 2_nnUNet_totalsegmri-pancreas/raw/{imagesTs,labelsTs}_<item>/ . License / citation: see 1_BIDS_totalsegmri-pancreas/<leaf>/README + LICENSE.
DATASET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DATASET_NAME="totalsegmri-pancreas"
export MODEL_TYPE="pansegdata_model"                       # results are filed under the SOURCE model type (predict cross mode / evaluate driver)
export DATASET_ROLE="eval_only"
export TRAINING_CONTRAST="${TRAINING_CONTRAST:-t1wce}"
export RUN_JOB_TIME_DEFAULT="${RUN_JOB_TIME_DEFAULT:-04:00:00}"
# set BEFORE common_env.sh (it sources run_job and freezes the defaults); adapt to the test volumes' size
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}"
export RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-40G}"
export BIDS_SUBDIR="abdomen-totalsegmri"
CE_SUBDIRS="preprocessed splits"
source "${DATASET_ROOT}/../../../00_commun_scripts/00_00_utils/common_env.sh"
export METRICS_ROOT="${METRICS_ROOT:-${DATASET_ROOT}/8_results_totalsegmri-pancreas/02_metrics}"
export CHECKPOINTS_DIR="${DATASET_ROOT}/6_checkpoints_totalsegmri-pancreas"
export RESULTS_DIR="${DATASET_ROOT}/8_results_totalsegmri-pancreas"

# ── SOURCE block (pansegdata): every var guarded (${VAR:-default}) so cluster override files (tamia_env_totalsegmri-pancreas.sh) win ──
export PANSEG_DATASET_ROOT="${PANSEG_DATASET_ROOT:-${DATASET_ROOT}/../../pancreas_disease/pansegdata}"
export PANSEG_PREDICTIONS_ROOT="${PANSEG_PREDICTIONS_ROOT:-${PANSEG_DATASET_ROOT}/8_results_pansegdata/01_predictions}"
export PANSEG_NNUNET_RAW="${PANSEG_NNUNET_RAW:-${PANSEG_DATASET_ROOT}/2_nnUNet_pansegdata/raw}"
export PANSEG_NNUNET_PREPROCESSED="${PANSEG_NNUNET_PREPROCESSED:-${PANSEG_DATASET_ROOT}/2_nnUNet_pansegdata/preprocessed}"
export PANSEG_MODEL_TYPE="pansegdata_model"
export PANSEG_TRAINING_CONTRAST="${PANSEG_TRAINING_CONTRAST:-t1wce}"   # the roster driver sets these two per source contrast / run
export PANSEG_DATASET_ID="${PANSEG_DATASET_ID:-150}"
# label numbering of the source (background 0, foreground 1 ...): the companion's GT masks MUST use the same numbering (else add --label_map in the evaluate step)
export PANSEG_DATASET_JSON="${PANSEG_DATASET_JSON:-${PANSEG_NNUNET_RAW}/Dataset150_PanSegData_T1WCE/dataset.json}"
export PYTHONPATH="${PANSEG_DATASET_ROOT}/5_scripts_pansegdata:${PYTHONPATH:-}"      # the source's trainer package (nnUNetTrainer<NAME>*), needed by nnUNetv2_predict
