#!/usr/bin/env python
"""
Convert ON-Harmony T1w data → nnUNet Dataset030_OnHarmonyT1w.

Reads
-----
  data/splits/onharmony_splits.json   — produced by 00_create_splits.py
  data/ON-Harmony/sub-*/ses-*/anat/*_T1w.nii.gz
  data/ON-Harmony/derivatives/synthseg_masks/sub-*/ses-*/anat/*_T1w_synthseg.nii.gz

Writes
------
  data/nnUNet_raw/Dataset030_OnHarmonyT1w/
    imagesTr/{case_id}_0000.nii.gz   — T1w image (copied, not modified)
    labelsTr/{case_id}.nii.gz        — 7-class label map (remapped from FreeSurfer IDs)
    dataset.json
    splits_final.json                — matches onharmony_splits.json, nnUNet format

Validation
----------
  - Affine matrix comparison between T1w and SynthSeg mask (max diff < 1e-3)
  - Voxel count sanity check (label volume ≥ 95% of image volume)
  - Raises on mismatch; prints summary on success
"""
from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

N_WORKERS = 256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = PROJECT_ROOT / "data" / "ON-Harmony"
MASKS_ROOT   = BIDS_ROOT / "derivatives" / "synthseg_masks"
SPLITS_JSON  = PROJECT_ROOT / "data" / "splits" / "onharmony_splits.json"
import argparse
import sys

# ── Label set selection ────────────────────────────────────────────────────────
# Set via --label-set CLI arg or NNUNET_LABEL_SET env var.
# 7class  → Dataset030_OnHarmonyT1w
# 31class → Dataset031_OnHarmonyT1w31
_LABEL_SET_DEFAULT = "7class"

# FreeSurfer ID → 7-class (coarse, bilateral structures merged)
FREESURFER_TO_7CLASS: dict[int, int] = {
    0:  0,   # Background
    2:  2,   # WM left            → White Matter
    3:  1,   # Cortex left        → Cortical GM
    4:  3,   # Left lat. vent.    → CSF/Ventricles
    5:  3,   # Left inf. lat. vent.
    7:  6,   # Cerebellum WM left → Cerebellum
    8:  6,   # Cerebellum ctx left
    10: 4,   # Left thalamus      → Subcortical GM
    11: 4,   # Left caudate
    12: 4,   # Left putamen
    13: 4,   # Left pallidum
    14: 3,   # 3rd ventricle
    15: 3,   # 4th ventricle
    16: 5,   # Brainstem
    17: 4,   # Left hippocampus
    18: 4,   # Left amygdala
    26: 4,   # Left accumbens
    28: 4,   # Left ventral DC
    41: 2,   # WM right
    42: 1,   # Cortex right
    43: 3,   # Right lat. vent.
    44: 3,   # Right inf. lat. vent.
    46: 6,   # Cerebellum WM right
    47: 6,   # Cerebellum ctx right
    49: 4,   # Right thalamus
    50: 4,   # Right caudate
    51: 4,   # Right putamen
    52: 4,   # Right pallidum
    53: 4,   # Right hippocampus
    54: 4,   # Right amygdala
    58: 4,   # Right accumbens
    60: 4,   # Right ventral DC
}

DATASET_JSON_7CLASS = {
    "channel_names": {"0": "T1w"},
    "labels": {
        "background":     0, "cortical_gm":    1, "white_matter":   2,
        "csf_ventricles": 3, "subcortical_gm": 4, "brainstem":      5,
        "cerebellum":     6,
    },
    "numTraining": 0, "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

# FreeSurfer ID → 31-class (fine-grained, bilateral kept separate)
# IDs ordered by anatomy; class 1-31 match this ordering.
_FS_IDS_31 = [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,42,43,44,46,47,49,50,51,52,53,54,58,60]
FREESURFER_TO_31CLASS: dict[int, int] = {0: 0, **{fs_id: i+1 for i, fs_id in enumerate(_FS_IDS_31)}}
_FS_NAMES_31 = [
    "WM_L","Cortex_L","LatVent_L","InfLatVent_L","CerebWM_L","CerebCtx_L",
    "Thalamus_L","Caudate_L","Putamen_L","Pallidum_L","3rdVent","4thVent",
    "Brainstem","Hippo_L","Amygdala_L","Accumbens_L","VentralDC_L",
    "WM_R","Cortex_R","LatVent_R","InfLatVent_R","CerebWM_R","CerebCtx_R",
    "Thalamus_R","Caudate_R","Putamen_R","Pallidum_R","Hippo_R","Amygdala_R",
    "Accumbens_R","VentralDC_R",
]

DATASET_JSON_31CLASS = {
    "channel_names": {"0": "T1w"},
    "labels": {"background": 0, **{name: i+1 for i, name in enumerate(_FS_NAMES_31)}},
    "numTraining": 0, "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}

LABEL_CONFIGS = {
    "7class":  ("Dataset030_OnHarmonyT1w",    FREESURFER_TO_7CLASS,  DATASET_JSON_7CLASS),
    "31class": ("Dataset031_OnHarmonyT1w31",   FREESURFER_TO_31CLASS, DATASET_JSON_31CLASS),
}


def remap_labels(arr: np.ndarray, label_map: dict) -> np.ndarray:
    """Remap FreeSurfer integer labels using the given id→class dict."""
    out = np.zeros_like(arr, dtype=np.uint8)
    for fs_id, cls in label_map.items():
        out[arr == fs_id] = cls
    return out


def validate_geometry(t1w_nii: nib.Nifti1Image, mask_nii: nib.Nifti1Image, case_id: str) -> None:
    """Raise if affine or shape doesn't match within tolerance."""
    if t1w_nii.shape[:3] != mask_nii.shape[:3]:
        raise ValueError(
            f"{case_id}: shape mismatch — T1w {t1w_nii.shape[:3]} vs mask {mask_nii.shape[:3]}"
        )
    diff = np.abs(t1w_nii.affine - mask_nii.affine).max()
    if diff > 1e-3:
        raise ValueError(
            f"{case_id}: affine mismatch (max diff = {diff:.6f})"
        )


def process_case(case_id: str, images_dir: Path, labels_dir: Path) -> None:
    """Copy T1w image and write remapped label for one case."""
    sub, ses, _ = case_id.split("_", 2)   # sub-XXXX, ses-YYYY, T1w
    t1w_path  = BIDS_ROOT / sub / ses / "anat" / f"{case_id}.nii.gz"
    mask_path = MASKS_ROOT / sub / ses / "anat" / f"{case_id}_synthseg.nii.gz"

    if not t1w_path.exists():
        raise FileNotFoundError(f"T1w not found: {t1w_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"SynthSeg mask not found: {mask_path}")

    t1w_nii  = nib.load(t1w_path)
    mask_nii = nib.load(mask_path)

    validate_geometry(t1w_nii, mask_nii, case_id)

    # Sanity check: at least 95% of image volume is covered by labels
    mask_arr   = np.asarray(mask_nii.dataobj, dtype=np.int32)
    label_vox  = int((mask_arr > 0).sum())
    total_vox  = int(np.prod(mask_arr.shape))
    if label_vox < 0.01 * total_vox:
        raise ValueError(
            f"{case_id}: suspiciously few labeled voxels ({label_vox}/{total_vox})"
        )

    # Copy T1w
    dst_image = images_dir / f"{case_id}_0000.nii.gz"
    shutil.copy2(t1w_path, dst_image)

    # Remap and write labels
    remapped = remap_labels(mask_arr, _ACTIVE_LABEL_MAP)
    label_nii = nib.Nifti1Image(remapped, mask_nii.affine, mask_nii.header)
    label_nii.set_data_dtype(np.uint8)
    nib.save(label_nii, labels_dir / f"{case_id}.nii.gz")


# Globals set by __main__ before calling main(); defaults to 7class
_ACTIVE_LABEL_SET = _LABEL_SET_DEFAULT
_ACTIVE_LABEL_MAP, _ACTIVE_DATASET_JSON = FREESURFER_TO_7CLASS, DATASET_JSON_7CLASS
DATASET_DIR = PROJECT_ROOT / "data" / "nnUNet_raw" / "Dataset030_OnHarmonyT1w"


def main() -> None:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(
            f"Splits file not found: {SPLITS_JSON}\n"
            "Run 00_create_splits.py first."
        )

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    assert len(splits) == 4, f"Expected 4 folds, got {len(splits)}"

    # Collect all train/val cases (union across folds)
    all_case_ids: set[str] = set()
    for fold in splits:
        all_case_ids.update(fold["train"])
        all_case_ids.update(fold["val"])
    all_case_ids_sorted = sorted(all_case_ids)
    print(f"Total train/val cases: {len(all_case_ids_sorted)}")

    # Create dataset directory structure
    images_dir = DATASET_DIR / "imagesTr"
    labels_dir = DATASET_DIR / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Process cases in parallel (64 workers — CPU/IO bound, benefits from concurrency)
    errors = []
    n = len(all_case_ids_sorted)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(process_case, cid, images_dir, labels_dir): cid
            for cid in all_case_ids_sorted
        }
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
        raise RuntimeError(
            f"{len(errors)} case(s) failed:\n"
            + "\n".join(f"  {cid}: {err}" for cid, err in errors)
        )

    # Write dataset.json
    dataset_json = dict(_ACTIVE_DATASET_JSON)
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
    print(f"  label_set: {_ACTIVE_LABEL_SET} ({len(_ACTIVE_DATASET_JSON['labels'])-1} classes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ON-Harmony to nnUNet format.")
    parser.add_argument(
        "--label-set", choices=list(LABEL_CONFIGS.keys()), default=_LABEL_SET_DEFAULT,
        help="Label granularity: 7class (default, coarse) or 31class (fine-grained bilateral)"
    )
    args = parser.parse_args()

    _ACTIVE_LABEL_SET = args.label_set
    _dataset_name, _ACTIVE_LABEL_MAP, _ACTIVE_DATASET_JSON = LABEL_CONFIGS[_ACTIVE_LABEL_SET]
    DATASET_DIR = PROJECT_ROOT / "data" / "nnUNet_raw" / _dataset_name

    print(f"Label set: {_ACTIVE_LABEL_SET} → {_dataset_name}")
    main()
