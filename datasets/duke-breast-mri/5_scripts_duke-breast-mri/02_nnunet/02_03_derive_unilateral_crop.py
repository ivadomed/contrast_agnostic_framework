#!/usr/bin/env python3
"""
Derive a unilateral (single-breast, half-width) crop of duke-breast-mri's
existing bilateral test volumes -- t1wce AND precontrast (identical geometry,
identical mask, confirmed by direct comparison before writing this) -- so
duke's cross-dataset eval can be scored on a FOV that matches the training
distribution I-SPY2's t1wce direction mostly saw (82% of I-SPY2's own t1wce
training pool is unilateral-cropped; duke is ~100% bilateral-width -- see
CLAUDE.md's "Breast task" section). Applied uniformly to every eval source
with bilateral volumes (Paul's explicit condition to avoid result-hacking) --
this is duke's half of that; I-SPY2's own bilateral test cases already have a
matching derived-unilateral case in the test set today (see
4_splits_ispy2/fov_variants.json / test_cases.json), no new code needed there.

Method -- deliberately SIMPLER than ispy2's own
01_create_splits/01_02_derive_fov_variants.py, not a copy of it:
  - ispy2's own crop derivation fits an empirical anterior-posterior (axis 1)
    window measured from ITS OWN site-specific VOLSER pipeline geometry, and
    cross-maps between two different grids (T1wce vs T2w). Duke has no T2w
    counterpart and no VOLSER-derived empirical AP window of its own, so
    reusing ispy2's specific numeric window here would import a geometry
    calibrated to a different acquisition protocol without any evidence it
    transfers -- that would be worse than not cropping AP at all.
  - This script therefore crops ONLY axis 0 (L-R, patient laterality) to the
    half containing the lesion, and leaves axis 1 (A-P) and axis 2 (S-I) at
    each case's native extent. This still achieves the actual goal (halving
    the L-R FOV to match the dominant training shape) via the one crop
    principle that IS dataset-agnostic and well-supported: a real unilateral
    breast MRI images one side, so cropping to the lesion's own side is
    anatomically correct regardless of site/protocol.

Safety rule (same principle as ispy2's own script): a crop that clips the
tumour is a silent disaster, never allowed to happen quietly. Every case is
checked for full lesion containment in the chosen half; a case whose lesion
crosses the L-R midline (would require clipping real tumour to crop) is
EXCLUDED from the unilateral set and recorded, not force-cropped.

Output (nnU-Net flat test-set layout, same convention as the existing
imagesTs_t1wce / imagesTs_precontrast):
  2_nnUNet_duke-breast-mri/raw/imagesTs_t1wce_uni/duke_XXX_0000.nii.gz
  2_nnUNet_duke-breast-mri/raw/labelsTs_t1wce_uni/duke_XXX.nii.gz
  2_nnUNet_duke-breast-mri/raw/imagesTs_precontrast_uni/duke_XXX_0000.nii.gz
  2_nnUNet_duke-breast-mri/raw/labelsTs_precontrast_uni/duke_XXX.nii.gz
  2_nnUNet_duke-breast-mri/raw/duke_test_unilateral_crop_manifest.json

Existing imagesTs_t1wce / imagesTs_precontrast / their labels are NEVER
modified -- this only adds new sibling directories, so the original bilateral
evaluation stays intact and comparable side-by-side.

Run:  bash 02_nnunet/02_03_derive_unilateral_crop.sh   (submits via run_job --
      ~580 volumes decompressed/cropped/re-gzipped, over the login-node
      ~10 CPU-min allowance).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]   # datasets/duke-breast-mri
RAW = DATASET_ROOT / "2_nnUNet_duke-breast-mri" / "raw"

ITEMS = ["t1wce", "precontrast"]   # identical geometry/mask -- one crop box, both channels


def bbox_1d(mask: np.ndarray, axis: int):
    other = tuple(a for a in range(mask.ndim) if a != axis)
    prof = mask.any(axis=other)
    nz = np.flatnonzero(prof)
    return (int(nz[0]), int(nz[-1]) + 1) if nz.size else None


def crop_axis0(img: nib.Nifti1Image, lo: int, hi: int) -> nib.Nifti1Image:
    arr = np.asanyarray(img.dataobj)
    sub = arr[lo:hi]
    aff = np.asarray(img.affine).copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([lo, 0, 0], dtype=float)
    out = nib.Nifti1Image(np.ascontiguousarray(sub), aff)
    out.header.set_zooms(img.header.get_zooms()[:3])
    out.set_data_dtype(img.get_data_dtype())
    return out


def case_ids() -> list[str]:
    lbl_dir = RAW / "labelsTs_t1wce"
    return sorted(p.stem.replace(".nii", "") for p in lbl_dir.glob("*.nii.gz"))


def main() -> None:
    ids = case_ids()
    if not ids:
        raise SystemExit(f"No cases found under {RAW / 'labelsTs_t1wce'} -- "
                          f"has 02_01_convert_test_t1wce.py been run?")

    for item in ITEMS:
        (RAW / f"imagesTs_{item}_uni").mkdir(parents=True, exist_ok=True)
        (RAW / f"labelsTs_{item}_uni").mkdir(parents=True, exist_ok=True)

    manifest = {"cases": [], "excluded_midline_crossing": []}
    n_left, n_right = 0, 0

    for case_id in ids:
        lab_path = RAW / "labelsTs_t1wce" / f"{case_id}.nii.gz"
        lab_nib = nib.load(str(lab_path))
        assert nib.aff2axcodes(lab_nib.affine) == ("L", "P", "S"), \
            f"{case_id}: expected LPS, got {nib.aff2axcodes(lab_nib.affine)}"
        mask = np.asanyarray(lab_nib.dataobj) > 0
        n0 = lab_nib.shape[0]
        mid = n0 // 2

        if not mask.any():
            manifest["cases"].append({"case_id": case_id, "status": "skipped_empty_mask"})
            continue

        bb = bbox_1d(mask, axis=0)
        lo_lesion, hi_lesion = bb
        # side by lesion centroid on axis 0 (LPS: low index = patient RIGHT)
        centroid0 = float(np.argwhere(mask)[:, 0].mean())
        side = "right" if centroid0 < mid else "left"
        lo, hi = (0, mid) if side == "right" else (mid, n0)

        if lo_lesion < lo or hi_lesion > hi:
            manifest["excluded_midline_crossing"].append({
                "case_id": case_id, "lesion_extent_axis0": [lo_lesion, hi_lesion],
                "chosen_half": [lo, hi], "centroid0": centroid0,
            })
            manifest["cases"].append({"case_id": case_id, "status": "excluded_midline_crossing"})
            continue

        if side == "right":
            n_right += 1
        else:
            n_left += 1

        for item in ITEMS:
            img_src = RAW / f"imagesTs_{item}" / f"{case_id}_0000.nii.gz"
            lab_src = RAW / f"labelsTs_{item}" / f"{case_id}.nii.gz"
            img_dst = RAW / f"imagesTs_{item}_uni" / f"{case_id}_0000.nii.gz"
            lab_dst = RAW / f"labelsTs_{item}_uni" / f"{case_id}.nii.gz"

            img_nib = nib.load(str(img_src))
            cropped_img = crop_axis0(img_nib, lo, hi)
            nib.save(cropped_img, str(img_dst))

            this_lab_nib = nib.load(str(lab_src)) if item != "t1wce" else lab_nib
            cropped_lab = crop_axis0(this_lab_nib, lo, hi)
            nib.save(cropped_lab, str(lab_dst))

        manifest["cases"].append({
            "case_id": case_id, "status": "cropped", "side": side,
            "crop_axis0": [lo, hi], "native_shape0": n0,
            "lesion_extent_axis0": [lo_lesion, hi_lesion],
        })

    manifest["n_total"] = len(ids)
    manifest["n_cropped"] = n_left + n_right
    manifest["n_left"] = n_left
    manifest["n_right"] = n_right
    manifest["n_excluded_midline_crossing"] = len(manifest["excluded_midline_crossing"])
    manifest["n_excluded_empty_mask"] = sum(1 for c in manifest["cases"] if c["status"] == "skipped_empty_mask")
    manifest["method"] = ("axis-0-only (L-R) lesion-side half crop; axis 1 (A-P) and axis 2 (S-I) "
                          "left at native extent -- see module docstring for why ispy2's own "
                          "empirical AP window was NOT reused here")
    (RAW / "duke_test_unilateral_crop_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Duke unilateral-crop test set written -> {RAW}")
    print(f"  total cases: {manifest['n_total']}")
    print(f"  cropped: {manifest['n_cropped']}  (left: {n_left}, right: {n_right})")
    print(f"  excluded (lesion crosses midline): {manifest['n_excluded_midline_crossing']}")
    print(f"  excluded (empty mask): {manifest['n_excluded_empty_mask']}")
    if manifest["excluded_midline_crossing"]:
        print("  midline-crossing cases (not cropped, not in imagesTs_*_uni):")
        for c in manifest["excluded_midline_crossing"]:
            print(f"    {c['case_id']}: lesion {c['lesion_extent_axis0']} vs chosen half {c['chosen_half']}")


if __name__ == "__main__":
    main()
