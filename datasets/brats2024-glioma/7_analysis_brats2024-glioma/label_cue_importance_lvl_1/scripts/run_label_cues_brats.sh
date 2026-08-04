#!/usr/bin/env bash
# brats2024-glioma label boundary-cue analysis: tumour SUB-REGION surfaces (NCR/SNFH/ET/RC) plus
# their union (whole tumour), on all four modalities.
#
# Why the union is scored too: whole tumour on T2w/FLAIR is plausibly the one BraTS target that IS
# fairly boundary-visible, so reporting only sub-regions would invite exactly the objection the
# analysis exists to answer. WT is the --extra-union row.
#
# t1n and t2w come from nnUNet raw. t1c and t2f exist only in the BIDS tree, so this script stages
# them as a flat dir of symlinks in the nnUNet naming the shared driver expects (uncompressed
# .nii, hence the --image-suffix override the driver takes). BraTS modalities are co-registered by
# construction, so the same dseg applies to all four.
#
#   bash run_label_cues_brats.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
RAW="${BRATS_RAW:-${REPO}/datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/raw}"
BIDS="${BRATS_BIDS:-${REPO}/datasets/brats2024-glioma/1_BIDS_brats2024-glioma/glioma-brain-brats2024}"
LVL1="${REPO}/datasets/brats2024-glioma/7_analysis_brats2024-glioma/label_cue_importance_lvl_1"
STAGE="${LVL1}/outputs/stage"
DS=brats2024-glioma
N_SUBJECTS=${N_SUBJECTS:-40}       # of 700; sampled with a fixed seed by the driver
JOB_TIME=${JOB_TIME:-04:00:00}     # 5 regions x 4 modalities is the heaviest branch set
UNION_IDS=${UNION_IDS:-1,2,3,4}
UNION_NAME=${UNION_NAME:-whole_tumour}

# ── stage t1c / t2f from BIDS as flat symlinks ────────────────────────────────────────────────
# BIDS names: sub-<ID>_ce-gadolinium_T1w.nii (t1c), sub-<ID>_FLAIR.nii (t2f),
#             derivatives/manual_masks/sub-<ID>/anat/sub-<ID>_dseg.nii
for pair in "t1c:ce-gadolinium_T1w" "t2f:FLAIR"; do
    mod="${pair%%:*}"; bidsname="${pair##*:}"
    mkdir -p "${STAGE}/${mod}/images" "${STAGE}/${mod}/labels"
    for sub in "${BIDS}"/sub-*; do
        id="$(basename "${sub}")"; case_id="${id#sub-}"
        img="${sub}/anat/${id}_${bidsname}.nii"
        seg="${BIDS}/derivatives/manual_masks/${id}/anat/${id}_dseg.nii"
        [ -f "${img}" ] && [ -f "${seg}" ] || continue
        ln -sf "${img}" "${STAGE}/${mod}/images/${case_id}_0000.nii"
        ln -sf "${seg}" "${STAGE}/${mod}/labels/${case_id}.nii"
    done
    echo "staged ${mod}: $(ls "${STAGE}/${mod}/images" | wc -l) cases"
done

DSJSON="${RAW}/Dataset051_BraTS2024GliomaT1n/dataset.json"
BRANCHES="t1n|${RAW}/Dataset051_BraTS2024GliomaT1n/imagesTr|${RAW}/Dataset051_BraTS2024GliomaT1n/labelsTr|${DSJSON}
t2w|${RAW}/Dataset052_BraTS2024GliomaT2w/imagesTr|${RAW}/Dataset052_BraTS2024GliomaT2w/labelsTr|${DSJSON}"
source "${REPO}/datasets/00_commun_scripts/00_04_analysis/label_cue_importance/run_label_cues_common.sh"

# The two BIDS-staged branches need the .nii suffixes, so they are submitted directly rather than
# through the BRANCHES loop above (which assumes nnUNet naming).
for mod in t1c t2f; do
    run_job --name "cues_${DS}_${mod}" --gpus 1 --time "${JOB_TIME}" \
            --log "${LVL1}/outputs/logs/${mod}.log" -- \
        "${REPO}/.venv/bin/python" \
        "${REPO}/datasets/00_commun_scripts/00_04_analysis/label_cue_importance/compute_label_cues.py" \
            --images "${STAGE}/${mod}/images" --labels "${STAGE}/${mod}/labels" \
            --image-suffix "_0000.nii" --label-suffix ".nii" \
            --labels-json "${DSJSON}" --out-dir "${LVL1}/outputs/data" \
            --dataset "${DS}" --modality "${mod}" --n-subjects "${N_SUBJECTS}" \
            --extra-union "${UNION_IDS}" --union-name "${UNION_NAME}"
    echo "submitted ${DS}/${mod}"
done
