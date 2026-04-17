#!/usr/bin/env python3
"""Regenerate SPIDER labels from original .mha masks and write nnUNet-style labelsTr files.

This script reads the split file and for each subject looks for
`{spider_dir}/masks/{subject}.mha`. It converts the MHA mask to a NIfTI
array (matching the project's `convert_spider_to_nnunet.py` orientation
convention) and writes a uint8 NIfTI file named `{subject}_0000.nii.gz`
into the provided task/nnUNet dataset `labelsTr` directories.

Usage:
  scripts/nnunet_scripts/fix_spider_labels.py \
    --spider-dir /tmp/spider \
    --split-file data/splits/spider_spine_split.json \
    --task-dirs data/Task102_SpiderSpine,data/nnUNet_raw/Dataset102_SpiderSpine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", default="/tmp/spider", help="Extracted SPIDER dir (contains images/ and masks/)")
    p.add_argument("--split-file", default="data/splits/spider_spine_split.json", help="Split json with train/val lists")
    p.add_argument(
        "--task-dirs",
        default="data/Task102_SpiderSpine,data/nnUNet_raw/Dataset102_SpiderSpine",
        help="Comma-separated list of task dataset dirs where labelsTr will be written",
    )
    return p.parse_args()


def load_split(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    if isinstance(split, list):
        if not split:
            return []
        first = split[0] or {}
        train = list(first.get("train", []))
        val = list(first.get("val", []))
        return train + val
    # legacy format
    if "train" in split and "val" in split:
        return list(split.get("train", [])) + list(split.get("val", []))
    if "train_subjects" in split:
        return list(split.get("train_subjects", [])) + list(split.get("val_subjects", []))
    # fallback: empty
    return []


def mha_to_nifti_arr(mha_path: Path) -> np.ndarray:
    sitk_img = sitk.ReadImage(str(mha_path))
    arr = sitk.GetArrayFromImage(sitk_img)  # (Z, Y, X)
    arr = np.transpose(arr, (2, 1, 0))  # -> (X, Y, Z)
    return arr


def write_label(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr8 = np.round(arr).astype(np.uint8)
    img = nib.Nifti1Image(arr8, np.eye(4))
    nib.save(img, str(out_path))


def main():
    args = parse_args()
    spider_dir = Path(args.spider_dir)
    split_file = Path(args.split_file)
    task_dirs = [Path(p) for p in args.task_dirs.split(",") if p]

    if not spider_dir.exists():
        raise FileNotFoundError(f"SPIDER dir not found: {spider_dir}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    subjects = load_split(split_file)
    if not subjects:
        print("No subjects found in split file; nothing to do.")
        return

    masks_dir = spider_dir / "masks"
    written = 0
    missing = 0
    for subj_raw in subjects:
        # normalize subject id (some split files include the _0000 suffix)
        subj = subj_raw
        if isinstance(subj, str) and subj.endswith("_0000"):
            subj = subj[:-5]
        # candidate mask filenames (try several common SPIDER patterns)
        candidates = [
            masks_dir / f"{subj}.mha",
            masks_dir / f"{subj}_t1.mha",
            masks_dir / f"{subj}_t2.mha",
            masks_dir / f"{subj}_t2_SPACE.mha",
            masks_dir / f"{subj}_t1_SPACE.mha",
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            tried = ", ".join([p.name for p in candidates])
            print(f"[WARN] mask not found for subject {subj_raw} (tried: {tried})")
            missing += 1
            continue

        arr = mha_to_nifti_arr(found)
        # write to each task dir's labelsTr as {subject}_0000.nii.gz
        for td in task_dirs:
            out = td / "labelsTr" / f"{subj}_0000.nii.gz"
            write_label(arr, out)
        written += 1

    print(f"Wrote labels for {written} subjects, {missing} missing")


if __name__ == "__main__":
    main()
