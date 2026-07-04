#!/usr/bin/env python3
"""
BIDSify open_ms_data (0_raw_open-ms/patientXX/) → 1_BIDS_open-ms/open-ms-brain/.

The raw data is already analysis-ready NIfTI (co-registered to FLAIR, 1mm iso, LPS,
N4). This step reorganises it into a BIDS-compliant tree with metadata sidecars; image
files are HARD-LINKED (no duplication on the same filesystem). Orientation is NOT touched
(already LPS, matching the chaos/sliver07 family — no reorient step needed).

Subject ids are kept as `patientXX` (→ sub-patientXX) so the splits in 4_splits_open-ms/
(which key on `patientXX`) and the nnUNet case ids stay consistent.

BIDS suffixes:  FLAIR → _FLAIR ; T2W → _T2w ; T1W → _T1w
The consensus lesion mask (FLAIR space, applies to all co-registered contrasts) is stored
once under derivatives as sub-patientXX_FLAIR_dseg.nii.gz.

Reads:   0_raw_open-ms/patientXX/{FLAIR,T2W,T1W,consensus_gt}.nii.gz
Writes:  1_BIDS_open-ms/open-ms-brain/
           dataset_description.json, participants.tsv
           sub-patientXX/anat/sub-patientXX_{FLAIR,T2w,T1w}.nii.gz (+ .json sidecars)
           derivatives/manual_masks/dataset_description.json
           derivatives/manual_masks/sub-patientXX/anat/sub-patientXX_FLAIR_dseg.nii.gz

Usage:  python 00_01_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]                 # datasets/open-ms
RAW = DATASET_ROOT / "0_raw_open-ms"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_open-ms" / "open-ms-brain"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

# raw filename -> BIDS suffix
CONTRASTS = {"FLAIR": "FLAIR", "T2W": "T2w", "T1W": "T1w"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:                      # cross-device → copy
        import shutil
        shutil.copyfile(src, dst)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "open_ms_data (cross-sectional) — brain MS lesion segmentation",
        "BIDSVersion": "1.9.0",
        "License": "CC-BY",
        "Authors": ["Lesjak et al. (2018)", "muschellij2/open_ms_data"],
        "ReferencesAndLinks": [
            "https://github.com/muschellij2/open_ms_data",
            "https://doi.org/10.1007/s12021-017-9348-7",
        ],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "open-ms consensus MS-lesion masks (multi-rater)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Multi-rater consensus (Lesjak 2018)"}],
    })

    patients = sorted(p.name for p in RAW.glob("patient*") if p.is_dir())
    rows = ["participant_id\tlesion_voxels"]
    for pid in patients:
        sub = f"sub-{pid}"
        for raw_name, suffix in CONTRASTS.items():
            _link(RAW / pid / f"{raw_name}.nii.gz",
                  BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz")
            _json(BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.json",
                  {"Modality": "MR", "Space": "FLAIR",
                   "Description": f"{raw_name}, co-registered to FLAIR, 1mm iso, N4, LPS"})
        # consensus lesion mask (FLAIR space)
        _link(RAW / pid / "consensus_gt.nii.gz",
              DERIV_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz")
        lv = int((np.asanyarray(nib.load(str(RAW / pid / "consensus_gt.nii.gz")).dataobj) > 0).sum())
        rows.append(f"{sub}\t{lv}")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"BIDSified {len(patients)} patients → {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
