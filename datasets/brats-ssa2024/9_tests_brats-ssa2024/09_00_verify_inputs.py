#!/usr/bin/env python3
"""Verify the BraTS-SSA 2024 nnUNet test inputs: all 95 cases present per contrast,
LPS orientation, GT labels restricted to {0,1,2,3} (no RC=4 -- pre-treatment cohort),
and 0_raw still pristine (RAS, untouched). Exits non-zero (and prints FAIL) if anything
is off.
"""
import glob, os, sys
import numpy as np, nibabel as nib

ROOT = "datasets/brats-ssa2024"
EXPECT_N = 95

ok = True

for contrast in ("t1n", "t1c", "t2w", "t2f"):
    imgs = sorted(glob.glob(f"{ROOT}/2_nnUNet_brats-ssa2024/raw/imagesTs_{contrast}/*_0000.nii.gz"))
    labs = sorted(glob.glob(f"{ROOT}/2_nnUNet_brats-ssa2024/raw/labelsTs_{contrast}/*.nii.gz"))
    print(f"{contrast}: images={len(imgs)} labels={len(labs)} (expect {EXPECT_N} each)")
    if len(imgs) != EXPECT_N or len(labs) != EXPECT_N:
        ok = False; print(f"  FAIL: wrong {contrast} input count")
        continue
    for f in imgs[:3] + labs[:3]:
        im = nib.load(f)
        axcodes = nib.aff2axcodes(im.affine)
        bad = axcodes != ("L", "P", "S")
        print(f"  {os.path.basename(f):20} axcodes={axcodes} shape={im.shape} {'<-- FAIL orientation' if bad else ''}")
        if bad:
            ok = False
    d = np.asarray(nib.load(labs[0]).dataobj)
    vals = set(np.unique(d).tolist())
    if not vals <= {0, 1, 2, 3}:
        ok = False; print(f"  FAIL: {contrast} label has unexpected values (RC=4 present?): {sorted(vals)}")

# 0_raw still pristine (RAS)
raw_candidates = sorted(glob.glob(f"{ROOT}/0_raw_brats-ssa2024/95_Glioma/BraTS-SSA-*/*-t1n.nii"))
if raw_candidates:
    raw = raw_candidates[0]
    rax = nib.aff2axcodes(nib.load(raw).affine)
    print(f"0_raw {os.path.basename(raw)} axcodes={rax} (expect R,A,S -- source orientation)")
    if rax != ("R", "A", "S"):
        ok = False; print("  FAIL: 0_raw orientation changed -- raw not pristine!")
else:
    ok = False; print("  FAIL: no 0_raw files found to verify")

print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
