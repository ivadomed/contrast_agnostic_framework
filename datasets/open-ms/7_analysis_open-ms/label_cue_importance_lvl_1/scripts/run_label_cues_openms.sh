#!/usr/bin/env bash
# open-ms label boundary-cue analysis: MS lesion surfaces on FLAIR and T1w.
# One job per training modality; the shared driver does the work.
#   bash run_label_cues_openms.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RAW="${OPENMS_RAW:-${REPO}/datasets/open-ms/2_nnUNet_open-ms/raw}"
DS=open-ms
N_SUBJECTS=${N_SUBJECTS:-40}       # open-ms has 22 train subjects; this takes all of them
JOB_TIME=${JOB_TIME:-01:00:00}
BRANCHES="flair|${RAW}/Dataset070_OpenMS_FLAIR/imagesTr|${RAW}/Dataset070_OpenMS_FLAIR/labelsTr|${RAW}/Dataset070_OpenMS_FLAIR/dataset.json
t1w|${RAW}/Dataset071_OpenMS_T1W/imagesTr|${RAW}/Dataset071_OpenMS_T1W/labelsTr|${RAW}/Dataset071_OpenMS_T1W/dataset.json"
source "${REPO}/datasets/00_commun_scripts/00_04_analysis/label_cue_importance/run_label_cues_common.sh"
