#!/usr/bin/env python3
"""Verify the US nnUNet test inputs were resampled to the eval resolution and that
0_raw stayed pristine. Exits non-zero (and prints FAIL) if anything is off, so a
watcher can gate the prediction re-run on it. See 05_predict/05_00_build_test_inputs.py.
"""
import glob, os, sys
import numpy as np, nibabel as nib

ROOT = "datasets/trusted"
TARGET = 1.5            # expected US eval spacing (mm, iso)

ok = True

# 1) US nnUNet inputs at ~TARGET mm, small, labels binary
imgs = sorted(glob.glob(f"{ROOT}/2_nnUNet_trusted/raw/imagesTs_us/*_0000.nii.gz"))
labs = sorted(glob.glob(f"{ROOT}/2_nnUNet_trusted/raw/labelsTs_us/*.nii.gz"))
print(f"US images={len(imgs)} labels={len(labs)} (expect 59 each)")
if len(imgs) != 59 or len(labs) != 59:
    ok = False; print("  FAIL: wrong US input count")
for f in imgs[:3] + labs[:3]:
    im = nib.load(f); sp = [round(float(z), 3) for z in im.header.get_zooms()]
    vox = np.prod(im.shape) / 1e6
    bad = any(abs(s - TARGET) > 0.05 for s in sp)
    print(f"  {os.path.basename(f):16} spacing={sp} voxels={vox:.1f}M {'<-- FAIL spacing' if bad else ''}")
    if bad:
        ok = False
# label values binary
d = np.asarray(nib.load(labs[0]).dataobj)
if not set(np.unique(d).tolist()) <= {0, 1}:
    ok = False; print(f"  FAIL: US label not binary: {np.unique(d).tolist()}")

# 2) 0_raw still pristine (native 0.3 mm, untouched)
raw = sorted(glob.glob(f"{ROOT}/0_raw_trusted/US_DATA/US_images/*_imgUS.nii.gz"))[0]
rsp = [round(float(z), 3) for z in nib.load(raw).header.get_zooms()]
print(f"0_raw {os.path.basename(raw)} spacing={rsp} (expect 0.3 iso)")
if any(abs(s - 0.3) > 0.01 for s in rsp):
    ok = False; print("  FAIL: 0_raw US spacing changed — raw not pristine!")

print("RESULT:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
