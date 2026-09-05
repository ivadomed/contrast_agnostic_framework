#!/usr/bin/env python3
"""
Convert ATLAS challenge train set (0_raw) -> nnU-Net raw dataset Dataset080_AtlasLiverHCC.

Training contrast = T1w contrast-enhanced MRI (VIBE-sequence variants; contrast phase
arterial/delayed/portal varies per patient as metadata only — see
0_raw_atlas-liver-hcc/train/patient_info_train.json — NOT as separate channels; there is
exactly one modality here, unlike our other training datasets which each train two).

Single-modality dataset: unlike open-ms/chaos/brats there is no cross-contrast test set
to build — the held-out 12 patients (4_splits_atlas-liver-hcc/partition.json, from
01_create_splits/01_01_create_splits.py) ARE the generalization test, scored on the same
T1w contrast. Labels: 0=background, 1=liver, 2=tumour (kept as both, though the paper-
relevant, texture-defined target is `tumour`).

Output under 2_nnUNet_atlas-liver-hcc/raw/Dataset080_AtlasLiverHCC/:
  imagesTr/atlas_XXX_0000.nii.gz    train pool (48 cases)
  labelsTr/atlas_XXX.nii.gz         liver+tumour mask, uint8 {0,1,2}
  imagesTs_t1w/atlas_XXX_0000.nii.gz  held-out test (12 cases) — single item, "t1w"
  labelsTs_t1w/atlas_XXX.nii.gz       matching GT
  dataset.json

Run:  bash 02_nnunet/02_00_convert.sh   (or: python this file) — pure file copy/rename
over 60 small volumes, well under the login-node ~10 CPU-min/~4GB exception (no run_job
needed; see CLAUDE.md "Common gotchas" for what DOES need run_job).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/atlas-liver-hcc
RAW = DATASET_ROOT / "0_raw_atlas-liver-hcc" / "train"
SPLITS = DATASET_ROOT / "4_splits_atlas-liver-hcc"
OUT = DATASET_ROOT / "2_nnUNet_atlas-liver-hcc" / "raw" / "Dataset080_AtlasLiverHCC"


def _pid(case_id: str) -> int:
    return int(case_id.removeprefix("atlas_"))


def _copy_image(pid: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RAW / "imagesTr" / f"im{pid}.nii.gz", dst)


def _write_mask(pid: int, dst: Path) -> None:
    """Load ATLAS mask (already {0,1,2}), enforce uint8, preserve affine/header geometry."""
    img = nib.load(str(RAW / "labelsTr" / f"lb{pid}.nii.gz"))
    arr = np.asanyarray(img.dataobj).astype(np.uint8)
    out = nib.Nifti1Image(arr, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))


def main() -> None:
    partition = json.loads((SPLITS / "partition.json").read_text())
    train_pool, test = partition["train_pool"], partition["test"]

    for case_id in train_pool:
        pid = _pid(case_id)
        _copy_image(pid, OUT / "imagesTr" / f"{case_id}_0000.nii.gz")
        _write_mask(pid, OUT / "labelsTr" / f"{case_id}.nii.gz")

    # Single-modality dataset -> single test item "t1w" (no cross-contrast axis to build).
    for case_id in test:
        pid = _pid(case_id)
        _copy_image(pid, OUT / "imagesTs_t1w" / f"{case_id}_0000.nii.gz")
        _write_mask(pid, OUT / "labelsTs_t1w" / f"{case_id}.nii.gz")

    dataset_json = {
        "name": "AtlasLiverHCC",
        "description": "ATLAS challenge (Quinton et al. 2023, CC BY-NC-SA 4.0) — HCC "
                        "liver tumour segmentation on contrast-enhanced T1w MRI. Single "
                        "modality: train + cross-generalize within T1w only (no second "
                        "training contrast, unlike our other training datasets).",
        "reference": "https://atlas-challenge.u-bourgogne.fr",
        "licence": "CC BY-NC-SA 4.0",
        "release": "1.0.1",
        "channel_names": {"0": "T1w"},
        "labels": {"background": 0, "liver": 1, "tumour": 2},
        "numTraining": len(train_pool),
        "file_ending": ".nii.gz",
    }
    (OUT / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    n_tr = len(list((OUT / "imagesTr").glob("*.nii.gz")))
    n_lb = len(list((OUT / "labelsTr").glob("*.nii.gz")))
    n_ts = len(list((OUT / "imagesTs_t1w").glob("*.nii.gz")))
    n_ts_lb = len(list((OUT / "labelsTs_t1w").glob("*.nii.gz")))
    print(f"Dataset080_AtlasLiverHCC written -> {OUT}")
    print(f"  imagesTr: {n_tr}  labelsTr: {n_lb}")
    print(f"  imagesTs_t1w: {n_ts}  labelsTs_t1w: {n_ts_lb}")


if __name__ == "__main__":
    main()
