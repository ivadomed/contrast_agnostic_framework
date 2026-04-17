#!/usr/bin/env python3
"""Repair nnUNet label geometry mismatches by resampling labels to image grids.

For each case in imagesTr/<id>_0000.nii.gz, this script expects labelsTr/<id>.nii.gz.
If geometry differs (size/spacing/origin/direction), label is resampled to image space
with nearest-neighbor interpolation and written back in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True, help="Path to nnUNet dataset folder")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def same_geometry(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetDirection() == b.GetDirection()
    )


def main() -> None:
    args = parse_args()
    ds = Path(args.dataset_dir)
    images_tr = ds / "imagesTr"
    labels_tr = ds / "labelsTr"

    fixed = 0
    checked = 0

    for img_path in sorted(images_tr.glob("*_0000.nii.gz")):
        case_id = img_path.name.replace("_0000.nii.gz", "")
        lbl_path = labels_tr / f"{case_id}.nii.gz"
        if not lbl_path.exists():
            continue

        img = sitk.ReadImage(str(img_path))
        lbl = sitk.ReadImage(str(lbl_path))
        checked += 1

        if same_geometry(img, lbl):
            continue

        print(f"[repair] {case_id}: resampling label to image geometry")
        res = sitk.Resample(
            lbl,
            img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        if not args.dry_run:
            sitk.WriteImage(res, str(lbl_path))
        fixed += 1

    print(f"[repair] checked={checked}, fixed={fixed}")


if __name__ == "__main__":
    main()
