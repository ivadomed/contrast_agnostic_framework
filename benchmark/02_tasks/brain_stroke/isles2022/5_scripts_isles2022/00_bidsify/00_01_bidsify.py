"""ISLES-2022 raw (already BIDS-shaped, 0_raw_isles2022/ISLES-2022) -> lab BIDS leaf
1_BIDS_isles2022/stroke-brain-isles2022. Faithful copy: no reorientation/resampling/exclusions
(3 empty-mask cases are kept here; they are excluded at nnUNet conversion).
  raw:  sub-X/ses-0001/dwi/sub-X_ses-0001_dwi.nii.gz (+json)   FLAIR: anat/..._FLAIR.nii.gz (+json)
  deriv ADC:   derivatives/adc/sub-X/ses-0001/dwi/sub-X_ses-0001_desc-adc_dwi.nii.gz (+json)
  deriv label: derivatives/labels/sub-X/ses-0001/dwi/sub-X_ses-0001_label-lesion_seg.nii.gz (binary uint8)
Masks are stored as non-integer floats in some cases -> verified to be {0,1} within 1e-3 before casting."""
import json, os, shutil
from pathlib import Path
import numpy as np, nibabel as nib

HERE = Path(__file__).resolve()
DS = HERE.parents[2]
SRC = DS / "0_raw_isles2022" / "ISLES-2022"
OUT = DS / "1_BIDS_isles2022" / "stroke-brain-isles2022"

def put(src, dst):
    if not src.exists() and src.suffix == ".json":  # scanner JSON is 'if available' in the source release
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp); os.replace(tmp, dst)

for sd in sorted(SRC.glob("sub-*")):
    s = sd.name; b = f"{s}_ses-0001"
    put(sd/"ses-0001/dwi"/f"{b}_dwi.nii.gz", OUT/s/"ses-0001/dwi"/f"{b}_dwi.nii.gz")
    put(sd/"ses-0001/dwi"/f"{b}_dwi.json", OUT/s/"ses-0001/dwi"/f"{b}_dwi.json")
    put(sd/"ses-0001/anat"/f"{b}_FLAIR.nii.gz", OUT/s/"ses-0001/anat"/f"{b}_FLAIR.nii.gz")
    put(sd/"ses-0001/anat"/f"{b}_FLAIR.json", OUT/s/"ses-0001/anat"/f"{b}_FLAIR.json")
    a = OUT/"derivatives/adc"/s/"ses-0001/dwi"
    put(sd/"ses-0001/dwi"/f"{b}_adc.nii.gz", a/f"{b}_desc-adc_dwi.nii.gz")
    put(sd/"ses-0001/dwi"/f"{b}_adc.json", a/f"{b}_desc-adc_dwi.json")
    dst = OUT/"derivatives/labels"/s/"ses-0001/dwi"/f"{b}_label-lesion_seg.nii.gz"
    if not dst.exists():
        im = nib.load(SRC/"derivatives"/s/"ses-0001"/f"{b}_msk.nii.gz")
        d = np.asanyarray(im.dataobj)
        u = np.unique(d)
        assert np.all(np.minimum(np.abs(u), np.abs(u - 1)) < 1e-3), f"{s}: non-binary mask values {u[:8]}"
        o = nib.Nifti1Image((d > 0.5).astype(np.uint8), im.affine); o.set_qform(im.affine, 1); o.set_sform(im.affine, 1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name("tmp_" + dst.name); nib.save(o, tmp); os.replace(tmp, dst)
    sc = dst.with_name(dst.name.replace(".nii.gz", ".json"))
    if not sc.exists():
        sc.write_text(json.dumps({"Description": "Acute/subacute ischemic stroke lesion mask, drawn on DWI (ADC and FLAIR as support), expert-approved (ISLES'22 hybrid annotation); binary 0/1, may be empty (3 cases).",
            "SpatialReference": f"{s}/ses-0001/dwi/{b}_dwi.nii.gz", "Sources": [f"bids::{s}/ses-0001/dwi/{b}_dwi.nii.gz"]}, indent=2))
print("done", len(list((OUT/"derivatives/labels").glob("sub-*"))))
