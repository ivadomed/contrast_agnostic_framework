#!/usr/bin/env python3
"""
BIDSify MS3SEG (0_raw_ms3seg/{images,masks}/) -> 1_BIDS_ms3seg/ms3seg-brain/.

Source: MS3SEG (Bawil et al., figshare 10.6084/m9.figshare.30393475.v6, CC-BY-4.0).
100 patients, Toshiba scanner, FLAIR/T1WI_reg/T2WI_reg (already co-registered),
256x256x20 grid (0.898x0.898x6.8mm -- thick-slice clinical acquisition, NOT the
~1mm isotropic research grids open-ms/mslesseg use). Ground truth is a combined
"4-label" mask (man_4L_masks_new): background + 3 anatomically distinct classes.

GOTCHA -- broken NIfTI header (verified empirically, not assumed): every source
file has scl_slope=NaN/scl_inter=NaN. nibabel's default scaled read then returns
garbage values for the mask (e.g. {0, 16448, 49087, 65535} instead of the true
raw uint8 {0, 64, 191, 255}). Every read here uses img.dataobj.get_unscaled()
and rewrites a clean header (slope=1, inter=0) so no downstream tool re-triggers
this.

Label identity verified empirically (not assumed) by reorienting the combined
mask and the three separate per-class binary masks (abWMH/nWMH/Vent, from the
archive's separate masks.rar) to a common orientation and checking spatial
overlap: raw value 64 = Vent (ventricles), 191 = nWMH (normal white-matter
hyperintensities), 255 = abWMH (MS lesions -- "abnormal" WMH). Only 255 is
open-ms-comparable; see 06_evaluate's --label_map.

Reads:   0_raw_ms3seg/images/<id>_{FLAIR,T1WI_reg,T2WI_reg}.nii.gz
         0_raw_ms3seg/masks/<id>.nii.gz           (combined 4-label mask)
Writes:  1_BIDS_ms3seg/ms3seg-brain/
           dataset_description.json, participants.tsv
           sub-<id>/anat/sub-<id>_{FLAIR,T1w,T2w}.nii.gz (+ .json)
           derivatives/manual_masks/sub-<id>/anat/sub-<id>_FLAIR_dseg.nii.gz

Usage:  python 00_00_ingest_and_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW_IMAGES = DATASET_ROOT / "0_raw_ms3seg" / "images"
RAW_MASKS = DATASET_ROOT / "0_raw_ms3seg" / "masks"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_ms3seg" / "ms3seg-brain"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

CONTRASTS = {"FLAIR": "FLAIR", "T1WI_reg": "T1w", "T2WI_reg": "T2w"}


def _write_clean(src: Path, dst: Path) -> None:
    """Read unscaled (bypass broken scl_slope=NaN), write with a clean header."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    img = nib.load(str(src))
    raw = np.asarray(img.dataobj.get_unscaled())
    out = nib.Nifti1Image(raw, img.affine)
    nib.save(out, str(dst))


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    if not RAW_IMAGES.exists() or not RAW_MASKS.exists():
        raise SystemExit(f"Raw archive not found under {DATASET_ROOT / '0_raw_ms3seg'}")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "MS3SEG -- Multiple Sclerosis tri-mask MRI segmentation dataset",
        "BIDSVersion": "1.9.0",
        "License": "CC-BY-4.0",
        "Authors": ["Mahdi Bashiri Bawil"],
        "ReferencesAndLinks": ["https://doi.org/10.6084/m9.figshare.30393475.v6"],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "MS3SEG expert tri-mask annotations (ventricles/normal-WMH/MS-lesion)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Manual annotation (Bawil et al.)"}],
    })

    ids = sorted(p.stem.replace(".nii", "") for p in RAW_MASKS.glob("*.nii.gz"))
    rows = ["participant_id\tlesion_voxels"]
    n = 0
    for pid in ids:
        sub = f"sub-{pid}"
        anat_dir = BIDS_ROOT / sub / "anat"
        for raw_suffix, bids_suffix in CONTRASTS.items():
            src = RAW_IMAGES / f"{pid}_{raw_suffix}.nii.gz"
            if not src.exists():
                raise SystemExit(f"Missing expected file: {src}")
            _write_clean(src, anat_dir / f"{sub}_{bids_suffix}.nii.gz")
            _json(anat_dir / f"{sub}_{bids_suffix}.json",
                  {"Modality": "MR", "Description": f"{raw_suffix} (MS3SEG, Toshiba scanner)"})

        mask_src = RAW_MASKS / f"{pid}.nii.gz"
        mask_dst = DERIV_DIR / sub / "anat" / f"{sub}_FLAIR_dseg.nii.gz"
        _write_clean(mask_src, mask_dst)

        arr = np.asarray(nib.load(str(mask_dst)).dataobj)
        lesion_voxels = int((arr == 255).sum())
        rows.append(f"{sub}\t{lesion_voxels}")
        n += 1

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"BIDSified {n} patients -> {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
