#!/usr/bin/env python3
"""
BIDSify the LLD-MMRI-MedSAM2 HCC subset (0_raw_lld-mmri-hcc/LLD-MMRI-MedSAM2/) ->
1_BIDS_lld-mmri-hcc/lld-mmri-hcc/.

157 hepatocellular-carcinoma patients (category 6 in LLD_MMRI_Annotation_full.json,
filtered from the full 498-patient/7-lesion-type LLD-MMRI dataset -- see
0_raw_lld-mmri-hcc/hcc_patient_ids.txt). Originally T2WI + DWI only (the two phases
genuinely different in contrast mechanism from atlas-liver-hcc's CE-T1w training
contrast). Extended 2026-09-01 to all 8 phases the source repo actually provides
masks for -- the source (wanglab/LLD-MMRI-MedSAM2) always had C-pre/C+A/C+V/
C+Delay (the CE-T1w-family phases -- same phase structure as liverhccseg's
pre/art/pv/del) and InPhase/OutPhase (T1 Dixon-style, not yet used anywhere else
in this project), just never downloaded originally. Same 157 patients, same mask
provenance -- this is "more test items from an already-vetted source", not a new
independent cohort; see project memory for why a genuinely new independent HCC-MRI
cohort could not be found (everything else is either already in use, CT-only, or
lacks tumor masks). Image/label files are HARD-LINKED. Orientation/spacing NOT
touched by this script. The original T2WI/DWI phases were verified 100% LPS-
consistent at onboarding (no reorient script exists for this dataset because none
was needed) -- do NOT assume the same holds for the 6 new phases just because
they're the same source repo; re-verify nib.aff2axcodes on every case of every
new phase before trusting anything downstream (this project has hit the opposite
assumption twice already -- see feedback_verify_orientation_on_onboarding memory).

Subject ids: sequential lldhcc000..lldhcc156 (raw patient ids mix hyphenated/
non-hyphenated formats -- MR-400851 vs MR102385 -- and BIDS entity labels must be
alphanumeric only, no hyphens). Mapping preserved in participants.tsv
(source_patient_id column) and case_id_map.json for traceability.

Segmentation is LESION ONLY (binary {0,1}) -- this dataset has no separate liver-organ
mask, unlike atlas-liver-hcc's 3-class (background/liver/tumour). Cross-dataset
evaluation must remap: atlas model's tumour class (2) -> this dataset's GT lesion
class (1), via evaluate.py's --label_map (see 06_evaluate/06_01_evaluate_run.sh).

Reads:   0_raw_lld-mmri-hcc/LLD-MMRI-MedSAM2/images/<PID>_<inst>_{T2WI,DWI}_0000.nii.gz
         0_raw_lld-mmri-hcc/LLD-MMRI-MedSAM2/labels/<PID>_<inst>_{T2WI,DWI}.nii.gz
         0_raw_lld-mmri-hcc/hcc_patient_ids.txt
Writes:  1_BIDS_lld-mmri-hcc/lld-mmri-hcc/
           dataset_description.json, participants.tsv, case_id_map.json
           sub-lldhccNNN/anat/sub-lldhccNNN_{T2w,dwi}.nii.gz (+ .json sidecars)
           derivatives/manual_masks/sub-lldhccNNN/anat/sub-lldhccNNN_{T2w,dwi}_dseg.nii.gz

Usage:  python 00_01_bidsify.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]                    # datasets/lld-mmri-hcc
RAW = DATASET_ROOT / "0_raw_lld-mmri-hcc" / "LLD-MMRI-MedSAM2"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_lld-mmri-hcc" / "lld-mmri-hcc"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

PHASES = {
    "T2WI": "T2w", "DWI": "dwi",
    "C-pre": "ce-pre_T1w", "C+A": "ce-art_T1w", "C+V": "ce-pv_T1w", "C+Delay": "ce-del_T1w",
    "InPhase": "inphase_T1w", "OutPhase": "outphase_T1w",
}   # raw phase name -> BIDS suffix. ce-* phases match liverhccseg's own
    # pre/art/pv/del naming exactly (same underlying phase structure).


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    pids = sorted(
        l.strip() for l in
        (DATASET_ROOT / "0_raw_lld-mmri-hcc" / "hcc_patient_ids.txt").read_text().splitlines()
        if l.strip()
    )
    if len(pids) != 157:
        raise SystemExit(f"expected 157 HCC patient ids, found {len(pids)}")

    # patient id -> lesion-instance suffix (varies per patient, e.g. MR11115 -> "6")
    # discovered from the actual downloaded filenames.
    inst_re = re.compile(
        r"^([A-Za-z0-9\-]+)_([0-9]+)_(T2WI|DWI|C-pre|C\+A|C\+V|C\+Delay|InPhase|OutPhase)_0000\.nii\.gz$")
    pid_to_inst = {}
    for f in (RAW / "images").glob("*.nii.gz"):
        m = inst_re.match(f.name)
        if m:
            pid_to_inst[m.group(1)] = m.group(2)
    missing = [p for p in pids if p not in pid_to_inst]
    if missing:
        raise SystemExit(f"no image files found for patient ids: {missing}")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "LLD-MMRI-HCC -- hepatocellular carcinoma subset of LLD-MMRI-MedSAM2 "
                "(8 phases: T2WI/DWI/C-pre/C+A/C+V/C+Delay/InPhase/OutPhase), "
                "cross-contrast eval-only set for atlas-liver-hcc",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-NC 4.0 (research use only, no commercial use)",
        "Authors": ["Lou et al. (2025, LLD-MMRI)", "Ma, Yang et al. (2025, MedSAM2 annotation)"],
        "ReferencesAndLinks": [
            "https://github.com/LMMMEng/LLD-MMRI-Dataset",
            "https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2",
            "https://arxiv.org/abs/2504.03600",
        ],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "LLD-MMRI-HCC lesion segmentation masks (MedSAM2, human-in-the-loop)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "MedSAM2 human-in-the-loop annotation (Ma, Yang et al. 2025)"}],
    })

    case_id_map = {}
    rows = ["participant_id\tsource_patient_id\tlesion_instance"]
    for i, pid in enumerate(pids):
        sub = f"sub-lldhcc{i:03d}"
        inst = pid_to_inst[pid]
        case_id_map[sub] = {"source_patient_id": pid, "lesion_instance": inst}
        for raw_phase, suffix in PHASES.items():
            img_src = RAW / "images" / f"{pid}_{inst}_{raw_phase}_0000.nii.gz"
            lab_src = RAW / "labels" / f"{pid}_{inst}_{raw_phase}.nii.gz"
            _link(img_src, BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz")
            _json(BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.json",
                  {"Modality": "MR", "Description": f"{raw_phase}, HCC lesion (category 6)"})
            _link(lab_src, DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii.gz")
            _json(DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.json",
                  {"Manual": False, "GeneratedBy": "MedSAM2 human-in-the-loop",
                   "Labels": {"1": "lesion"}})
        rows.append(f"{sub}\t{pid}\t{inst}")

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    _json(BIDS_ROOT / "case_id_map.json", case_id_map)
    print(f"BIDSified {len(pids)} HCC patients -> {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
