#!/usr/bin/env python
"""
Convert ON-Harmony dwi_ap (b0-equivalent volume) → nnUNet Dataset033_OnHarmonyDWI31
(31-class, matching Dataset031_OnHarmonyT1w31 / Dataset032_OnHarmonyT2w31).

NOTE ON WHAT THIS CONTRAST IS: volume 0 of dwi_ap is a b~0 (b=0 or b=5 in all 84
training series) T2*-weighted single-shot-EPI image with distortion — NOT a diffusion-weighted
image and not 'diffusion contrast'. (1 TEST series, sub-14230 ses-NOT1ACH001, starts at b=1000;
left as-is so existing T1w/T2w results stay comparable.) Appearance-wise it sits in an EPI
cluster with bold/epi_ap (|Spearman| 0.42/0.33 vs a within-roster median of 0.20) — see
00_utils/check_dwi_vs_epi_similarity.py.

dwi_ap is 4D (multiple diffusion volumes) and natively LAS — UNLIKE T1w/T2w, which
are 3D and already RAS. Two things must be done consistently to avoid training a
mirrored/wrong-volume model (see 06_01_evaluate_testset.sh's own comment on this
exact hazard for the *prediction* path — this script is the *training*-side
counterpart):
  1. Extract the SAME volume index the eval pipeline uses (index 0 — arr[...,0],
     matching 06_01_evaluate_testset.sh's CONTRASTS["bold"]/generic 4D handling).
  2. Reorient to RAS (nib.as_closest_canonical) — applied identically to the
     extracted image AND its label mask, so the pair stays aligned. Dice/HD95 are
     rigid-transform-invariant as long as image and label get the SAME transform
     (see project memory project_hanseg_pddca_orientation_check_20260917).

Reads
-----
  4_splits_on-harmony/onharmony_dwi_splits.json   — produced by 01_03_create_splits_dwi.py
  4_splits_on-harmony/dwi_trainval_cases.json     — image/mask paths per case
  <BIDS_ROOT>/sub-*/ses-*/dwi/*_dir-AP_dwi.nii.gz
  <BIDS_ROOT>/derivatives/labels/sub-*/ses-*/dwi/*_dir-AP_dwi_label-synthseg_dseg.nii.gz

Writes
------
  <nnUNet_raw>/Dataset033_OnHarmonyDWI31/
    imagesTr/{case_id}_0000.nii.gz   — dwi_ap volume 0, RAS-canonical
    labelsTr/{case_id}.nii.gz        — 31-class label map (FreeSurfer remap), RAS-canonical
    dataset.json
    splits_final.json

Usage: .venv/bin/python 02_05_convert_dataset_dwi.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

N_WORKERS = 32

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
DATASET_ROOT = PROJECT_ROOT / "datasets" / "on-harmony"
SPLITS_DIR   = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_on-harmony"))
SPLITS_JSON  = SPLITS_DIR / "onharmony_dwi_splits.json"
CASES_JSON   = SPLITS_DIR / "dwi_trainval_cases.json"
DATASET_DIR  = Path(os.environ["nnUNet_raw"]) / "Dataset033_OnHarmonyDWI31"

# Same 31-class FreeSurfer remap as Dataset031/032 (identical labels list — required
# for cross-modality/combined_modality_summary pooling, which assumes matching class
# semantics across training modalities).
_FS_IDS_31 = [2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,42,43,44,46,47,49,50,51,52,53,54,58,60]
FREESURFER_TO_31CLASS: dict[int, int] = {0: 0, **{fs_id: i + 1 for i, fs_id in enumerate(_FS_IDS_31)}}
_FS_NAMES_31 = [
    "WM_L","Cortex_L","LatVent_L","InfLatVent_L","CerebWM_L","CerebCtx_L",
    "Thalamus_L","Caudate_L","Putamen_L","Pallidum_L","3rdVent","4thVent",
    "Brainstem","Hippo_L","Amygdala_L","Accumbens_L","VentralDC_L",
    "WM_R","Cortex_R","LatVent_R","InfLatVent_R","CerebWM_R","CerebCtx_R",
    "Thalamus_R","Caudate_R","Putamen_R","Pallidum_R","Hippo_R","Amygdala_R",
    "Accumbens_R","VentralDC_R",
]
DATASET_JSON_31CLASS = {
    "channel_names": {"0": "dwi_ap"},
    "labels": {"background": 0, **{name: i + 1 for i, name in enumerate(_FS_NAMES_31)}},
    "numTraining": 0, "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}


def remap_labels(arr: np.ndarray, label_map: dict) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.uint8)
    for fs_id, cls in label_map.items():
        out[arr == fs_id] = cls
    return out


def process_case(case: dict, images_dir: Path, labels_dir: Path) -> None:
    case_id = case["case_id"]
    dwi_path = Path(case["dwi_ap"])
    mask_path = Path(case["mask"])

    dwi_nii = nib.load(dwi_path)
    mask_nii = nib.load(mask_path)

    if dwi_nii.shape[:3] != mask_nii.shape[:3]:
        raise ValueError(f"{case_id}: shape mismatch — dwi {dwi_nii.shape[:3]} vs mask {mask_nii.shape[:3]}")
    if np.abs(np.asarray(dwi_nii.affine) - np.asarray(mask_nii.affine)).max() > 1e-3:
        raise ValueError(f"{case_id}: affine mismatch between dwi volume and mask")

    # Extract volume 0 (matches 06_01_evaluate_testset.sh's convention for
    # non-bold 4D contrasts: arr[...,0]) as its own 3D image, same affine/header.
    dwi_arr4d = np.asarray(dwi_nii.dataobj)
    if dwi_arr4d.ndim != 4:
        raise ValueError(f"{case_id}: expected 4D dwi_ap, got shape {dwi_arr4d.shape}")
    vol0 = dwi_arr4d[..., 0].astype(np.float32)
    vol0_nii = nib.Nifti1Image(vol0, dwi_nii.affine, dwi_nii.header)

    # Reorient image AND mask with the SAME canonical transform (derived from
    # each's own affine — both start from the identical native affine, so
    # as_closest_canonical produces the identical permutation/flip for both).
    vol0_ras = nib.as_closest_canonical(vol0_nii)
    mask_ras = nib.as_closest_canonical(mask_nii)
    if nib.aff2axcodes(vol0_ras.affine) != nib.aff2axcodes(mask_ras.affine):
        raise ValueError(f"{case_id}: post-canonicalization axcode mismatch (image vs mask)")

    mask_arr = np.asarray(mask_ras.dataobj, dtype=np.int32)
    label_vox = int((mask_arr > 0).sum())
    total_vox = int(np.prod(mask_arr.shape))
    if label_vox < 0.01 * total_vox:
        raise ValueError(f"{case_id}: suspiciously few labeled voxels ({label_vox}/{total_vox})")

    nib.save(nib.Nifti1Image(np.asarray(vol0_ras.dataobj, dtype=np.float32),
                              vol0_ras.affine, vol0_ras.header),
              images_dir / f"{case_id}_0000.nii.gz")

    remapped = remap_labels(mask_arr, FREESURFER_TO_31CLASS)
    label_nii = nib.Nifti1Image(remapped, mask_ras.affine, mask_ras.header)
    label_nii.set_data_dtype(np.uint8)
    nib.save(label_nii, labels_dir / f"{case_id}.nii.gz")


def main() -> None:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(f"Splits file not found: {SPLITS_JSON}\nRun 01_03_create_splits_dwi.py first.")
    if not CASES_JSON.exists():
        raise FileNotFoundError(f"Cases file not found: {CASES_JSON}\nRun 01_03_create_splits_dwi.py first.")

    splits = json.loads(SPLITS_JSON.read_text())
    assert len(splits) == 4, f"Expected 4 folds, got {len(splits)}"
    all_cases = {c["case_id"]: c for c in json.loads(CASES_JSON.read_text())}

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
        futures = {pool.submit(process_case, all_cases[cid], images_dir, labels_dir): cid
                   for cid in all_case_ids_sorted}
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
        raise RuntimeError(f"{len(errors)} case(s) failed:\n" + "\n".join(f"  {c}: {e}" for c, e in errors))

    dataset_json = dict(DATASET_JSON_31CLASS)
    dataset_json["numTraining"] = len(all_case_ids_sorted)
    (DATASET_DIR / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\nWritten: {DATASET_DIR}/dataset.json")

    splits_final = [{"train": fold["train"], "val": fold["val"]} for fold in splits]
    (DATASET_DIR / "splits_final.json").write_text(json.dumps(splits_final, indent=2))
    print(f"Written: {DATASET_DIR}/splits_final.json")

    print(f"\n{DATASET_DIR.name} created at {DATASET_DIR}")
    print(f"  imagesTr/: {len(list(images_dir.glob('*.nii.gz')))} files")
    print(f"  labelsTr/: {len(list(labels_dir.glob('*.nii.gz')))} files")


if __name__ == "__main__":
    main()
