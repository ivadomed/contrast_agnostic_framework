#!/usr/bin/env python3
"""
BIDSify the ATLAS challenge raw archive (0_raw_atlas-liver-hcc/train/) →
1_BIDS_atlas-liver-hcc/atlas-liver-hcc/.

Pure directory-layout consistency with the other datasets — nothing in this project's
pipeline reads from this tree; 02_00_convert.py reads directly from 0_raw (see
00_utils/env.sh). Image/label files are HARD-LINKED (no duplication on the same
filesystem). Orientation/spacing are NOT touched.

Subject ids: raw case index i (im{i}.nii.gz / lb{i}.nii.gz, i=0..59) -> sub-atlas{i:03d}.
Not "atlas_{i:03d}" -- BIDS entity labels must be alphanumeric only, no underscores.
The nnUNet/splits case id "atlas_XXX" (datasets/atlas-liver-hcc/4_splits_atlas-liver-hcc/
partition.json) maps 1:1 to this BIDS subject via the same zero-padded index.

Single T1w CE-MRI channel per patient (contrast phase/scanner vary by patient as
metadata, not as separate BIDS series -- see patient_info_train.json). Segmentation
(0=background, 1=liver, 2=tumour) stored once as a multi-label dseg.

Reads:   0_raw_atlas-liver-hcc/train/{imagesTr,labelsTr}/{im,lb}{i}.nii.gz
         0_raw_atlas-liver-hcc/train/patient_info_train.json
Writes:  1_BIDS_atlas-liver-hcc/atlas-liver-hcc/
           dataset_description.json, participants.tsv
           sub-atlasXXX/anat/sub-atlasXXX_T1w.nii.gz (+ .json sidecar)
           derivatives/manual_masks/dataset_description.json
           derivatives/manual_masks/sub-atlasXXX/anat/sub-atlasXXX_T1w_dseg.nii.gz (+ .json)

Usage:  python 00_01_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]                    # datasets/atlas-liver-hcc
RAW = DATASET_ROOT / "0_raw_atlas-liver-hcc" / "train"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_atlas-liver-hcc" / "atlas-liver-hcc"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

N_CASES = 60


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:                      # cross-device -> copy
        import shutil
        shutil.copyfile(src, dst)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "ATLAS challenge — HCC liver tumour segmentation on CE-MRI",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-NC-SA 4.0",
        "Authors": ["Quinton et al. (2023)"],
        "ReferencesAndLinks": [
            "https://atlas-challenge.u-bourgogne.fr",
            "https://doi.org/10.3390/data8050079",
        ],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "ATLAS liver + HCC tumour segmentation masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Quinton et al. (2023) expert annotation"}],
    })

    patient_info = json.loads((RAW / "patient_info_train.json").read_text())

    rows = ["participant_id\tsource_case_id\tmachine\tsequence\tcontrast_phase"]
    for i in range(N_CASES):
        sub = f"sub-atlas{i:03d}"
        info = patient_info.get(str(i), {})
        _link(RAW / "imagesTr" / f"im{i}.nii.gz",
              BIDS_ROOT / sub / "anat" / f"{sub}_T1w.nii.gz")
        _json(BIDS_ROOT / sub / "anat" / f"{sub}_T1w.json", {
            "Modality": "MR",
            "MRAcquisitionType": "3D",
            "ScannerManufacturersModelName": info.get("machine"),
            "SequenceName": info.get("sequence"),
            "ContrastBolusIngredient": "gadolinium",
            "ContrastPhase": info.get("contrast_phase"),
            "AcquisitionDate": info.get("date"),
        })
        _link(RAW / "labelsTr" / f"lb{i}.nii.gz",
              DERIV_DIR / sub / "anat" / f"{sub}_T1w_dseg.nii.gz")
        _json(DERIV_DIR / sub / "anat" / f"{sub}_T1w_dseg.json", {
            "Manual": True,
            "Labels": {"1": "liver", "2": "tumour"},
        })
        rows.append(f"{sub}\tatlas_{i:03d}\t{info.get('machine')}\t"
                     f"{info.get('sequence')}\t{info.get('contrast_phase')}")

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"BIDSified {N_CASES} patients -> {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
