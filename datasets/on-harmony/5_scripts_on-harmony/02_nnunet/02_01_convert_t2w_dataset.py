#!/usr/bin/env python
"""
Convert ON-Harmony T2w data → nnUNet Dataset032_OnHarmonyT2w31 (31-class, fine-grained
bilateral FreeSurfer labels — mirrors Dataset031_OnHarmonyT1w31, the label set actually
used for T1w training).

Reads
-----
  4_splits_on-harmony/onharmony_t2w_splits.json   — produced by 01_02_create_splits_t2w.py
  1_BIDS_on-harmony/sub-*/ses-*/anat/*_T2w.nii.gz
  1_BIDS_on-harmony/derivatives/synthseg_masks/sub-*/ses-*/anat/*_T2w_synthseg.nii.gz

Writes
------
  2_nnUNet_on-harmony/raw/Dataset032_OnHarmonyT2w31/
    imagesTr/{case_id}_0000.nii.gz   — T2w image (copied, not modified)
    labelsTr/{case_id}.nii.gz        — 31-class label map (remapped from FreeSurfer IDs)
    dataset.json
    splits_final.json                — matches onharmony_t2w_splits.json, nnUNet format

Validation
----------
  - Affine matrix comparison between T2w and SynthSeg mask (max diff < 1e-3)
  - Voxel count sanity check (label volume ≥ 1% of image volume)
  - Raises on mismatch; prints summary on success

Usage: .venv/bin/python 02_01_convert_t2w_dataset.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

N_WORKERS = 64

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parents[1]
BIDS_ROOT  = Path(os.environ.get("BIDS_ROOT", DATASET_ROOT / "1_BIDS_on-harmony"))
MASKS_ROOT = BIDS_ROOT / "derivatives" / "synthseg_masks"
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_on-harmony"))
SPLITS_JSON = SPLITS_DIR / "onharmony_t2w_splits.json"
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", DATASET_ROOT / "2_nnUNet_on-harmony" / "raw"))
DATASET_DIR = NNUNET_RAW / "Dataset032_OnHarmonyT2w31"

# FreeSurfer ID → 31-class (fine-grained, bilateral kept separate) — identical mapping
# to 02_convert_to_nnunet/02_01_convert_dataset.py's 31class config.
_FS_IDS_31 = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28,
              41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
FREESURFER_TO_31CLASS: dict[int, int] = {0: 0, **{fs_id: i + 1 for i, fs_id in enumerate(_FS_IDS_31)}}
_FS_NAMES_31 = [
    "WM_L", "Cortex_L", "LatVent_L", "InfLatVent_L", "CerebWM_L", "CerebCtx_L",
    "Thalamus_L", "Caudate_L", "Putamen_L", "Pallidum_L", "3rdVent", "4thVent",
    "Brainstem", "Hippo_L", "Amygdala_L", "Accumbens_L", "VentralDC_L",
    "WM_R", "Cortex_R", "LatVent_R", "InfLatVent_R", "CerebWM_R", "CerebCtx_R",
    "Thalamus_R", "Caudate_R", "Putamen_R", "Pallidum_R", "Hippo_R", "Amygdala_R",
    "Accumbens_R", "VentralDC_R",
]
DATASET_JSON = {
    "channel_names": {"0": "T2w"},
    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(_FS_NAMES_31)}},
    "numTraining": 0, "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def remap_labels(arr: np.ndarray, label_map: dict) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for fs_id, cls in label_map.items():
        out[arr == fs_id] = cls
    return out


def validate_geometry(t2w_nii: nib.Nifti1Image, mask_nii: nib.Nifti1Image, case_id: str) -> None:
    if t2w_nii.shape[:3] != mask_nii.shape[:3]:
        raise ValueError(f"{case_id}: shape mismatch — T2w {t2w_nii.shape[:3]} vs mask {mask_nii.shape[:3]}")
    diff = np.abs(t2w_nii.affine - mask_nii.affine).max()
    if diff > 1e-3:
        raise ValueError(f"{case_id}: affine mismatch (max diff = {diff:.6f})")


def process_case(case_id: str, images_dir: Path, labels_dir: Path) -> None:
    sub, ses, _ = case_id.split("_", 2)   # sub-XXXX, ses-YYYY, T2w
    t2w_path  = BIDS_ROOT / sub / ses / "anat" / f"{case_id}.nii.gz"
    mask_path = MASKS_ROOT / sub / ses / "anat" / f"{case_id}_synthseg.nii.gz"

    if not t2w_path.exists():
        raise FileNotFoundError(f"T2w not found: {t2w_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"SynthSeg mask not found: {mask_path}")

    t2w_nii  = nib.load(t2w_path)
    mask_nii = nib.load(mask_path)
    validate_geometry(t2w_nii, mask_nii, case_id)

    mask_arr  = np.asarray(mask_nii.dataobj, dtype=np.int32)
    label_vox = int((mask_arr > 0).sum())
    total_vox = int(np.prod(mask_arr.shape))
    if label_vox < 0.01 * total_vox:
        raise ValueError(f"{case_id}: suspiciously few labeled voxels ({label_vox}/{total_vox})")

    shutil.copy2(t2w_path, images_dir / f"{case_id}_0000.nii.gz")

    remapped = remap_labels(mask_arr, FREESURFER_TO_31CLASS)
    label_nii = nib.Nifti1Image(remapped, mask_nii.affine, mask_nii.header)
    label_nii.set_data_dtype(np.uint8)
    nib.save(label_nii, labels_dir / f"{case_id}.nii.gz")


def main() -> None:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLITS_JSON}\nRun 01_02_create_splits_t2w.py first.")

    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    assert len(splits) == 4, f"Expected 4 folds, got {len(splits)}"

    all_case_ids: set[str] = set()
    for fold in splits:
        all_case_ids.update(fold["train"])
        all_case_ids.update(fold["val"])
    all_case_ids_sorted = sorted(all_case_ids)
    print(f"Total train/val cases: {len(all_case_ids_sorted)}")

    images_dir = DATASET_DIR / "imagesTr"
    labels_dir = DATASET_DIR / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    n = len(all_case_ids_sorted)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(process_case, cid, images_dir, labels_dir): cid for cid in all_case_ids_sorted}
        completed = 0
        for fut in as_completed(futures):
            cid = futures[fut]
            completed += 1
            try:
                fut.result()
                print(f"  [{completed:3d}/{n}] OK  {cid}")
            except Exception as e:
                errors.append((cid, str(e)))
                print(f"  [{completed:3d}/{n}] ERR {cid}: {e}")

    if errors:
        raise RuntimeError(f"{len(errors)} case(s) failed:\n" + "\n".join(f"  {cid}: {err}" for cid, err in errors))

    dataset_json = dict(DATASET_JSON)
    dataset_json["numTraining"] = len(all_case_ids_sorted)
    with open(DATASET_DIR / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    print(f"\nWritten: {DATASET_DIR}/dataset.json")

    splits_final = [{"train": fold["train"], "val": fold["val"]} for fold in splits]
    with open(DATASET_DIR / "splits_final.json", "w") as f:
        json.dump(splits_final, f, indent=2)
    print(f"Written: {DATASET_DIR}/splits_final.json")

    print(f"\n{DATASET_DIR.name} created at {DATASET_DIR}")
    print(f"  imagesTr/: {len(list(images_dir.glob('*.nii.gz')))} files")
    print(f"  labelsTr/: {len(list(labels_dir.glob('*.nii.gz')))} files")


if __name__ == "__main__":
    main()
