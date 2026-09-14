#!/usr/bin/env python3
"""
Independent post-conversion audit of the two ispy2 nnU-Net raw datasets
(Dataset100_ISPY2T1wce, Dataset101_ISPY2T2w) written by 02_01/02_02.

Deliberately re-derives everything from partition.json rather than trusting the
converter's own prints, and reports REAL COUNTS rather than asserting success
(this project has been bitten by scripts that "succeeded" while writing empty or
partial dirs -- see CLAUDE.md's eval stale-skip / truncated-pack gotchas).

Per Dataset, per subdir (imagesTr/labelsTr, imagesTs_t1wce, imagesTs_t2w):
  * case count vs. the expected list from partition.json (exact set equality --
    no extras, no missing)
  * every image has exactly one matching label and vice versa
  * label values are exactly a subset of {0,1}; dtype is uint8
  * no empty masks (foreground voxel count > 0)
  * orientation is project-canonical LPS via nib.aff2axcodes, for EVERY image
    and EVERY label (not a sample)
  * image and label share shape + affine

Run through:  bash 02_03_verify_raw.sh   (run_job -- it opens ~3.6k volumes).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ispy2_convert_lib", Path(__file__).with_name("02_00_convert_lib.py")
)
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

EXPECTED = lib.EXPECTED_AXCODES


def audit_dir(out: Path, images_sub: str, labels_sub: str, expected_cases: list[str], problems: list[str]) -> None:
    idir, ldir = out / images_sub, out / labels_sub
    got_img = sorted(p.name[: -len("_0000.nii.gz")] for p in idir.glob("*_0000.nii.gz"))
    got_lab = sorted(p.name[: -len(".nii.gz")] for p in ldir.glob("*.nii.gz"))
    exp = sorted(expected_cases)
    tag = f"{out.name}/{images_sub}"
    if got_img != exp:
        problems.append(f"{tag}: image set mismatch (+{sorted(set(got_img)-set(exp))[:5]} / -{sorted(set(exp)-set(got_img))[:5]})")
    if got_lab != exp:
        problems.append(f"{tag}: label set mismatch (+{sorted(set(got_lab)-set(exp))[:5]} / -{sorted(set(exp)-set(got_lab))[:5]})")

    vals: set[int] = set()
    dtypes: set[str] = set()
    empty: list[str] = []
    n_fg = 0
    for case in exp:
        ip, lp = idir / f"{case}_0000.nii.gz", ldir / f"{case}.nii.gz"
        if not (ip.exists() and lp.exists()):
            problems.append(f"{tag}: missing file for {case}")
            continue
        iimg, limg = nib.load(str(ip)), nib.load(str(lp))
        for p, im in ((ip, iimg), (lp, limg)):
            ax = tuple(nib.aff2axcodes(im.affine))
            if ax != EXPECTED:
                problems.append(f"{tag}: non-LPS {ax} in {p.name}")
        if iimg.shape != limg.shape:
            problems.append(f"{tag}: shape mismatch {case} img{iimg.shape} lab{limg.shape}")
        if not np.allclose(iimg.affine, limg.affine, atol=1e-4):
            problems.append(f"{tag}: affine mismatch {case}")
        arr = np.asanyarray(limg.dataobj)
        dtypes.add(str(arr.dtype))
        u = np.unique(arr)
        vals.update(int(v) for v in u.tolist())
        s = int((arr > 0).sum())
        n_fg += s
        if s == 0:
            empty.append(case)
    if not vals <= {0, 1}:
        problems.append(f"{tag}: label values {sorted(vals)} outside {{0,1}}")
    if empty:
        problems.append(f"{tag}: {len(empty)} EMPTY masks {empty[:10]}")
    print(f"  {images_sub:16s} n={len(exp):5d}  label values={sorted(vals)}  dtypes={sorted(dtypes)}  "
          f"empty={len(empty)}  total_fg_voxels={n_fg}")


def main() -> int:
    partition, _ = lib.load_inputs()
    problems: list[str] = []
    for modality in lib.MODALITIES:
        out = lib.RAW / lib.DATASET_DIRNAME[modality]
        print(f"\n=== {out.name}  (train modality {modality}) ===")
        if not (out / "dataset.json").exists():
            problems.append(f"{out.name}: dataset.json missing")
        audit_dir(out, "imagesTr", "labelsTr", partition["cases"][modality]["train_pool"], problems)
        for m in lib.MODALITIES:
            audit_dir(out, f"imagesTs_{m}", f"labelsTs_{m}", partition["cases"][m]["test"], problems)

    print("\n=== RESULT ===")
    if problems:
        print(f"FAILED with {len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: counts, pairing, label values {0,1}, no empty masks, LPS orientation, geometry match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
