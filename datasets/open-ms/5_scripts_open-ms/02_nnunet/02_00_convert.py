#!/usr/bin/env python3
"""
Convert open_ms_data (0_raw) → nnU-Net raw dataset Dataset070_OpenMS_FLAIR.

Training contrast = FLAIR (single channel). The model is trained on the 22 train-pool
patients' FLAIR and tested CROSS-CONTRAST on the 8 held-out patients' FLAIR / T2W / T1W
(all co-registered to FLAIR, so one consensus mask serves every contrast). Mirrors the
CHAOS layout (imagesTr + per-modality imagesTs_<mod> / labelsTs_<mod>).

Requires the partition from 01_create_splits/01_01_create_splits.py (partition.json).

Output under 2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/:
  imagesTr/<patient>_0000.nii.gz        FLAIR, train pool
  labelsTr/<patient>.nii.gz             consensus lesion mask (uint8, {0,1})
  imagesTs_{flair,t2w,t1w}/<patient>_0000.nii.gz   held-out test, one dir per contrast
  labelsTs_{flair,t2w,t1w}/<patient>.nii.gz        same mask (co-registered) per contrast
  dataset.json

Run:  bash 02_nnunet/02_00_convert.sh   (or: python this file)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/open-ms
BIDS_ROOT = DATASET_ROOT / "1_BIDS_open-ms" / "ms-brain-openms"   # source = BIDS (run 00_01_bidsify.py first)
DERIV = BIDS_ROOT / "derivatives" / "labels"
SPLITS = DATASET_ROOT / "4_splits_open-ms"
OUT = DATASET_ROOT / "2_nnUNet_open-ms" / "raw" / "Dataset070_OpenMS_FLAIR"

# nnUNet test-input dir suffix -> BIDS suffix (per contrast).
CONTRASTS = {"flair": "FLAIR", "t2w": "T2w", "t1w": "T1w"}


def _img(pid: str, bids_suffix: str) -> Path:
    return BIDS_ROOT / f"sub-{pid}" / "anat" / f"sub-{pid}_{bids_suffix}.nii.gz"


def _mask(pid: str) -> Path:
    return DERIV / f"sub-{pid}" / "anat" / f"sub-{pid}_FLAIR_label-lesion_seg.nii.gz"


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _write_mask(src_gt: Path, dst: Path) -> None:
    """Load consensus mask, binarise to uint8 {0,1}, preserve affine/header geometry."""
    img = nib.load(str(src_gt))
    arr = (np.asanyarray(img.dataobj) > 0).astype(np.uint8)
    out = nib.Nifti1Image(arr, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))


def main() -> None:
    partition = json.loads((SPLITS / "partition.json").read_text())
    train_pool, test = partition["train_pool"], partition["test"]

    # --- training set: FLAIR + mask (from BIDS) ---
    for pid in train_pool:
        _copy_image(_img(pid, "FLAIR"), OUT / "imagesTr" / f"{pid}_0000.nii.gz")
        _write_mask(_mask(pid), OUT / "labelsTr" / f"{pid}.nii.gz")

    # --- held-out test set: one imagesTs_<contrast>/ + labelsTs_<contrast>/ per contrast ---
    for suffix, bids_suffix in CONTRASTS.items():
        for pid in test:
            _copy_image(_img(pid, bids_suffix),
                        OUT / f"imagesTs_{suffix}" / f"{pid}_0000.nii.gz")
            _write_mask(_mask(pid),
                        OUT / f"labelsTs_{suffix}" / f"{pid}.nii.gz")

    dataset_json = {
        "name": "OpenMS_FLAIR",
        "description": "open_ms_data (Lesjak 2018, CC-BY) — brain MS lesion segmentation. "
                       "Train on FLAIR; cross-contrast test on FLAIR/T2W/T1W (co-registered).",
        "reference": "https://github.com/muschellij2/open_ms_data",
        "licence": "CC-BY",
        "release": "1.0",
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(train_pool),
        "file_ending": ".nii.gz",
    }
    (OUT / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    n_tr = len(list((OUT / "imagesTr").glob("*.nii.gz")))
    print(f"Dataset070_OpenMS_FLAIR written → {OUT}")
    print(f"  imagesTr: {n_tr}  labelsTr: {len(list((OUT/'labelsTr').glob('*.nii.gz')))}")
    for suffix in CONTRASTS:
        n = len(list((OUT / f'imagesTs_{suffix}').glob('*.nii.gz')))
        print(f"  imagesTs_{suffix}: {n}  labelsTs_{suffix}: "
              f"{len(list((OUT/f'labelsTs_{suffix}').glob('*.nii.gz')))}")


if __name__ == "__main__":
    main()
