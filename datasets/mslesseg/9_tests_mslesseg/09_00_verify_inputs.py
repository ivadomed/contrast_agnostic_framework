#!/usr/bin/env python3
"""Verify the MSLesSeg nnUNet test inputs: all 115 cases present per contrast, LPS
orientation, binary lesion masks, and 0_raw still pristine (LAS, untouched by the
BIDS reorient step -- see 03_preprocess/03_00_reorient_to_lps.py). Exits non-zero
(and prints FAIL) if anything is off.
"""
import glob, os, sys
import numpy as np, nibabel as nib

ROOT = "datasets/mslesseg"
EXPECT_N = 115

ok = True

for contrast in ("flair", "t1w", "t2w"):
    imgs = sorted(glob.glob(f"{ROOT}/2_nnUNet_mslesseg/raw/imagesTs_{contrast}/*_0000.nii.gz"))
    labs = sorted(glob.glob(f"{ROOT}/2_nnUNet_mslesseg/raw/labelsTs_{contrast}/*.nii.gz"))
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
    if not set(np.unique(d).tolist()) <= {0, 1}:
        ok = False; print(f"  FAIL: {contrast} label not binary: {np.unique(d).tolist()}")

# 0_raw still pristine (LAS, untouched -- BIDS files are hard-links broken by reorient,
# raw inode survives at the original orientation)
raw_candidates = sorted(glob.glob(f"{ROOT}/0_raw_mslesseg/MSLesSeg_Dataset/test/P*/P*_FLAIR.nii.gz"))
if raw_candidates:
    raw = raw_candidates[0]
    rax = nib.aff2axcodes(nib.load(raw).affine)
    print(f"0_raw {os.path.basename(raw)} axcodes={rax} (expect L,A,S -- source orientation)")
    if rax != ("L", "A", "S"):
        ok = False; print("  FAIL: 0_raw orientation changed -- raw not pristine!")
else:
    ok = False; print("  FAIL: no 0_raw files found to verify")

print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
