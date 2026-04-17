#!/usr/bin/env python3
"""Convert SPIDER dataset (.mha format) directly to nnUNetv2 raw format (NIfTI)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_registry import get_dataset_nnunet_id

# nnUNet env vars
nnUNet_raw = Path(os.environ.get("nnUNet_raw", str(PROJECT_ROOT / "data" / "nnUNet_raw")))
nnUNet_preprocessed = Path(
    os.environ.get("nnUNet_preprocessed", str(PROJECT_ROOT / "data" / "nnUNet_preprocessed"))
)
nnUNet_results = Path(os.environ.get("nnUNet_results", str(PROJECT_ROOT / "results" / "nnUNet")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SPIDER .mha dataset to nnUNetv2 format (NIfTI).")
    parser.add_argument(
        "--spider-dir",
        "--spider_dir",
        type=str,
        default="/tmp/spider",
        help="Path to SPIDER dataset extraction (contains 'images' and 'masks' folders).",
    )
    parser.add_argument(
        "--split-file",
        "--split_file",
        type=str,
        default="data/splits/spider_spine_split.json",
        help="Path to split JSON file (train/val keys).",
    )
    parser.add_argument(
        "--dataset-id",
        "--dataset_id",
        type=str,
        default="102",
        help="nnUNet dataset ID.",
    )
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        type=str,
        default="Dataset102_SpiderSpine",
        help="Output dataset folder name.",
    )
    parser.add_argument(
        "--contrast",
        type=str,
        default="t1_sag",
        choices=["t1_sag", "t2_sag", "t2_space"],
        help="Which contrast to extract (SPIDER has multi-contrast per subject).",
    )
    parser.add_argument(
        "--stack-contrasts",
        action="store_true",
        help="If set, stack all available contrasts into a single multi-channel NIfTI (channels as last dim).",
    )
    return parser.parse_args()


def mha_to_nifti(mha_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load .mha file using SimpleITK, convert to NIfTI."""
    sitk_img = sitk.ReadImage(str(mha_path))
    # Get array and affine
    arr = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X) in SimpleITK convention
    arr = np.transpose(arr, (2, 1, 0))  # Convert to (X, Y, Z)
    # Build identity affine (SimpleITK origin/spacing ignored for now)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(arr, affine)
    return arr, nifti_img


def load_split(split_file: Path) -> tuple[list[str], list[str]]:
    """Load train/val subject IDs from split JSON."""
    with split_file.open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    # Support both 'train'/'val' and 'train_subjects'/'val_subjects' formats
    train = split.get("train", split.get("train_subjects", []))
    val = split.get("val", split.get("val_subjects", []))
    return train, val


def convert_spider_subject(
    subject_id: str,
    contrast: str,
    spider_dir: Path,
    output_images: Path,
    output_labels: Path,
    stack_contrasts: bool = False,
) -> bool:
    """Convert single SPIDER subject from .mha to NIfTI format."""
    # Map contrast name to SPIDER file suffix
    contrast_map = {
        "t1_sag": "t1",
        "t2_sag": "t2",
        "t2_space": "t2_SPACE",
    }

    # Accept subject identifiers that may include a trailing '_0000' suffix
    raw_id = subject_id[:-5] if isinstance(subject_id, str) and subject_id.endswith("_0000") else subject_id

    label_candidates = [
        spider_dir / "masks" / f"{raw_id}.mha",
        spider_dir / "masks" / f"{raw_id}_t1.mha",
        spider_dir / "masks" / f"{raw_id}_t2.mha",
        spider_dir / "masks" / f"{raw_id}_t2_SPACE.mha",
        spider_dir / "masks" / f"{raw_id}_t1_SPACE.mha",
    ]
    lbl_mha = None
    for candidate in label_candidates:
        if candidate.exists():
            lbl_mha = candidate
            break

    if stack_contrasts:
        # Stack all contrasts (in the order of contrast_map keys)
        arrs = []
        affines = []
        for key in contrast_map:
            suffix = contrast_map[key]
            img_mha = spider_dir / "images" / f"{raw_id}_{suffix}.mha"
            if not img_mha.exists():
                print(f"  [WARN] {subject_id}: missing contrast {suffix} at {img_mha}")
                continue
            img_arr, img_nib = mha_to_nifti(img_mha)
            arrs.append(img_arr)
            affines.append(img_nib.affine)

        if not arrs:
            print(f"  [SKIP] {subject_id}: no contrasts found to stack")
            return False

        # Ensure all arrays share the same spatial shape
        shapes = {a.shape for a in arrs}
        if len(shapes) != 1:
            print(f"  [WARN] {subject_id}: contrast shapes differ {shapes}, skipping subject")
            return False

        # Stack as last axis (X, Y, Z, C) so MONAI's LoadImaged + EnsureChannelFirstd
        # will convert to (C, X, Y, Z)
        stacked = np.stack(arrs, axis=-1)
        out_img_path = output_images / f"{raw_id}_0000.nii.gz"
        nif = nib.Nifti1Image(stacked, affines[0] if affines else np.eye(4))
        nib.save(nif, str(out_img_path))

    else:
        # Single-contrast behavior (backwards compatible)
        suffix = contrast_map.get(contrast, "t1")
        img_mha = spider_dir / "images" / f"{raw_id}_{suffix}.mha"
        if not img_mha.exists():
            print(f"  [SKIP] {subject_id}: image not found at {img_mha}")
            return False
        img_arr, img_nib = mha_to_nifti(img_mha)
        out_img_path = output_images / f"{raw_id}_0000.nii.gz"
        nib.save(img_nib, str(out_img_path))

    # Convert label if available (write with _0000 suffix to pair with image name)
    if lbl_mha is not None:
        lbl_arr, lbl_nib = mha_to_nifti(lbl_mha)
        lbl_arr = np.round(lbl_arr).astype(np.uint8)
        lbl_nib_out = nib.Nifti1Image(lbl_arr, lbl_nib.affine)
        out_lbl_path = output_labels / f"{raw_id}_0000.nii.gz"
        nib.save(lbl_nib_out, str(out_lbl_path))
    else:
        print(f"  [WARN] {subject_id}: label not found (checked {len(label_candidates)} candidates)")
    
    return True


def write_dataset_json(
    dataset_dir: Path,
    dataset_name: str,
    contrast: str,
    n_training: int,
) -> None:
    """Write nnUNet dataset.json config file."""
    # Determine channel names
    contrast_names = {
        "t1_sag": "T1 Sagittal",
        "t2_sag": "T2 Sagittal",
        "t2_space": "T2 SPACE",
    }

    # If contrast is a comma-separated list (stacked), build mapping
    if isinstance(contrast, str) and "," in contrast:
        parts = [c.strip() for c in contrast.split(",") if c.strip()]
        channel_map = {str(i): contrast_names.get(p, p) for i, p in enumerate(parts)}
        desc = ", ".join(channel_map.values())
    else:
        channel_map = {"0": contrast_names.get(contrast, contrast.upper())}
        desc = channel_map["0"]

    payload = {
        "channel_names": channel_map,
        "labels": {
            "background": 0,
            "vertebra": 1,
            "disc": 2,
            "canal": 3,
        },
        "numTraining": n_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"SPIDER spine segmentation dataset ({desc}).",
        "reference": "Zenodo 8009679",
        "licence": "CC-BY-4.0",
        "release": "2026-04-16",
    }
    
    out = dataset_dir / "dataset.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Wrote {out}  (numTraining={n_training})")


def main() -> None:
    args = parse_args()
    
    spider_dir = Path(args.spider_dir)
    dataset_name = str(args.dataset_name)
    dataset_id = str(args.dataset_id)
    contrast = str(args.contrast)
    split_file = PROJECT_ROOT / args.split_file
    
    if not spider_dir.exists():
        raise FileNotFoundError(f"SPIDER directory not found: {spider_dir}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    # Load split
    train_subjects, val_subjects = load_split(split_file)
    all_subjects = train_subjects + val_subjects
    
    print("=" * 60)
    print(f"SPIDER Conversion to nnUNetv2")
    print(f"Dataset ID  : {dataset_id}")
    print(f"Dataset name: {dataset_name}")
    print(f"Contrast    : {contrast}")
    print(f"SPIDER dir  : {spider_dir}")
    print(f"Split file  : {split_file}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects  : {len(val_subjects)}")
    print(f"Total labeled : {len(all_subjects)}")
    print("=" * 60)
    
    # Create output directories
    for d in [nnUNet_raw, nnUNet_preprocessed, nnUNet_results]:
        d.mkdir(parents=True, exist_ok=True)
    
    dataset_dir = nnUNet_raw / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    
    for d in [images_tr, labels_tr, images_ts]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Convert subjects
    n_ok = 0
    print(f"\nConverting {len(all_subjects)} subjects -> imagesTr / labelsTr ...")
    for i, subject_id in enumerate(all_subjects):
        ok = convert_spider_subject(
            subject_id,
            contrast,
            spider_dir,
            images_tr,
            labels_tr,
            stack_contrasts=args.stack_contrasts,
        )
        if ok:
            n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_subjects)} subjects processed, {n_ok} OK")
    
    print(f"  Finished: {n_ok}/{len(all_subjects)} OK")
    
    # Write dataset.json
    print("\nWriting dataset.json ...")
    # If stacking contrasts, pass a comma-separated contrast list for dataset.json
    contrast_for_json = ",".join(["t1_sag", "t2_sag", "t2_space"]) if args.stack_contrasts else contrast
    write_dataset_json(dataset_dir, dataset_name, contrast_for_json, n_ok)
    
    # Copy split file to nnUNet location
    split_final = dataset_dir / "splits_final.json"
    with open(split_file) as src:
        split_data = json.load(src)
    # Support both split key formats and strip possible '_0000' suffixes
    raw_train = split_data.get("train", split_data.get("train_subjects", []))
    raw_val = split_data.get("val", split_data.get("val_subjects", []))

    def subj_to_int(s: str) -> int:
        if isinstance(s, str) and s.endswith("_0000"):
            s = s[:-5]
        return int(s)

    nnunet_split = [{
        "train": [subj_to_int(s) for s in raw_train],
        "val": [subj_to_int(s) for s in raw_val],
    }]
    with open(split_final, "w") as dst:
        json.dump(nnunet_split, dst, indent=2)
    print(f"  Wrote {split_final}")
    
    print("\nConversion complete.")
    print(f"\nNext step:\n  set_slot 0 .venv/bin/nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()
