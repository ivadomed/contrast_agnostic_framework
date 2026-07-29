#!/usr/bin/env python3
"""Verify MS3SEG nnUNet test inputs: 100 cases per contrast, LPS orientation, GT
labels restricted to {0,64,191,255} (raw tri-mask values -- NOT small indices, see
00_utils/00_00_ingest_and_bidsify.py), 0_raw pristine."""
import glob, os, sys
import numpy as np, nibabel as nib

ROOT = "datasets/ms3seg"
EXPECT_N = 100
ok = True

for contrast in ("flair", "t1w", "t2w"):
    imgs = sorted(glob.glob(f"{ROOT}/2_nnUNet_ms3seg/raw/imagesTs_{contrast}/*_0000.nii.gz"))
    labs = sorted(glob.glob(f"{ROOT}/2_nnUNet_ms3seg/raw/labelsTs_{contrast}/*.nii.gz"))
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
    if not vals <= {0, 64, 191, 255}:
        ok = False; print(f"  FAIL: {contrast} label has unexpected values: {sorted(vals)}")

raw_candidates = sorted(glob.glob(f"{ROOT}/0_raw_ms3seg/images/*_FLAIR.nii.gz"))
if raw_candidates:
    rax = nib.aff2axcodes(nib.load(raw_candidates[0]).affine)
    print(f"0_raw {os.path.basename(raw_candidates[0])} axcodes={rax} (expect L,A,S)")
    if rax != ("L", "A", "S"):
        ok = False; print("  FAIL: 0_raw image orientation changed!")
else:
    ok = False; print("  FAIL: no 0_raw image files found")

print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
