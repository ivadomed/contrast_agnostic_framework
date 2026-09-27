#!/usr/bin/env python3
"""
Superior-inferior flip an already-prepared nnU-Net eval arm (images + labels) so it matches
ToothFairy2's INVERTED training convention.

WHY THIS EXISTS (2026-09-17)
----------------------------
ToothFairy2's released .mha files are superior-inferior FLIPPED relative to their own
headers: identity direction cosines, inverted voxel content. Verified on the raw release
(maxilla sits 36-41 mm BELOW the mandible; condyles 35 mm below the chin; A-P and L-R are
fine, so it is a pure z inversion). `sitk.DICOMOrient(img,"RAS")` trusts the header, so the
flip propagated into every ToothFairy2 BIDS/nnUNet volume and hence into TRAINING.

The external eval arms (hanseg CT, hanseg MR-in-CT-frame, pddca CT) are correctly oriented.
So models trained on inverted anatomy are being tested on upright anatomy — an unintended
train/test convention mismatch that also hits methods UNEQUALLY (nnU-Net baseline keeps
`mirror_axes=(0,1,2)` + mirror TTA; every AugLab method sets `mirror_axes=None`).

Rather than retrain 11 runs on corrected data, flip the EVAL data into the training
convention. This is exact for the metrics we report:
  * Dice is a voxel-count ratio -> invariant under any bijective voxel remapping applied
    identically to prediction and GT.
  * HD95 is a physical distance -> a reflection is an isometry, so pairwise distances are
    preserved. (Spacing here is isotropic, so no anisotropy subtlety either.)
Flipping image AND label together therefore cannot change any reported number by itself; it
only changes what the network sees, putting it in-convention.

NOT identical to a retrain, and do not claim it is: a model trained on corrected data is not
exactly the flip-conjugate of one trained on flipped data (CNNs are not perfectly
flip-equivariant, and training is stochastic). That difference is seed-level, not
systematic.

HOW THE FLIP IS APPLIED — this detail matters
---------------------------------------------
We flip the VOXEL ARRAY along the superior axis and LEAVE THE AFFINE UNCHANGED. That
reproduces ToothFairy2's exact (self-inconsistent) convention: header says RAS, content is
inverted. The alternative — negating the affine's z column — would leave world-space anatomy
untouched and be silently canonicalised away by nnU-Net's own reorientation, changing
nothing.

Inputs are assumed already RAS (all our prepared arms are, verified per case), so the
superior axis is nibabel array axis 2.

ORDERING: run this AFTER the FOV crop and AFTER the MR->CT-frame registration. Both of those
derive from real anatomical geometry and must operate on correctly-oriented data; this is a
final presentation-layer step on the finished image/label pair.

Writes to sibling `<dir>_sif` directories, leaving the originals untouched so the two
conventions stay comparable.

Usage:
  si_flip_eval_inputs.py --img_dir <imagesTs_x> [--lab_dir <labelsTs_x>] [--suffix _sif]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SUP_AXIS = 2  # nibabel array axis for S in an RAS volume


def flip_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.nii.gz")):
        im = nib.load(str(f))
        ax = "".join(nib.aff2axcodes(im.affine))
        if ax != "RAS":
            raise SystemExit(f"{f} is {ax}, expected RAS — flip axis would be wrong")
        arr = np.asanyarray(im.dataobj)
        out = np.flip(arr, axis=SUP_AXIS).copy()
        # affine deliberately UNCHANGED — see module docstring
        nib.save(nib.Nifti1Image(out, im.affine, im.header), str(dst / f.name))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lab_dir", default=None)
    ap.add_argument("--suffix", default="_sif")
    a = ap.parse_args()

    img = Path(a.img_dir)
    n_i = flip_dir(img, img.parent / (img.name + a.suffix))
    print(f"[sif] images {img.name} -> {img.name}{a.suffix}: {n_i}")
    if a.lab_dir:
        lab = Path(a.lab_dir)
        n_l = flip_dir(lab, lab.parent / (lab.name + a.suffix))
        print(f"[sif] labels {lab.name} -> {lab.name}{a.suffix}: {n_l}")
        if n_i != n_l:
            sys.exit(f"[sif] COUNT MISMATCH images={n_i} labels={n_l}")


if __name__ == "__main__":
    main()
