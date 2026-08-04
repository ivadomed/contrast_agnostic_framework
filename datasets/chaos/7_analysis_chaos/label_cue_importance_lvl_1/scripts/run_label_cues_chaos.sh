#!/usr/bin/env bash
# chaos label boundary-cue analysis: abdominal ORGAN surfaces (liver/kidneys/spleen) on T1in and
# T2spir. This is the boundary-dominant end of the comparison -- the rows that make "these regions
# rely on texture far more than those ones" a statement with a right-hand side.
#   bash run_label_cues_chaos.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RAW="${CHAOS_RAW:-${REPO}/datasets/chaos/2_nnUNet_chaos/raw}"
DS=chaos
N_SUBJECTS=${N_SUBJECTS:-40}       # chaos has 16 train subjects; this takes all of them
JOB_TIME=${JOB_TIME:-01:00:00}
BRANCHES="t1in|${RAW}/Dataset060_CHAOS_MR_T1in/imagesTr|${RAW}/Dataset060_CHAOS_MR_T1in/labelsTr|${RAW}/Dataset060_CHAOS_MR_T1in/dataset.json
t2spir|${RAW}/Dataset061_CHAOS_MR_T2spir/imagesTr|${RAW}/Dataset061_CHAOS_MR_T2spir/labelsTr|${RAW}/Dataset061_CHAOS_MR_T2spir/dataset.json"
source "${REPO}/datasets/00_commun_scripts/00_04_analysis/label_cue_importance/run_label_cues_common.sh"
