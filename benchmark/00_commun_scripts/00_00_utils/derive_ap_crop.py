#!/usr/bin/env python3
"""
Derive skin-anchored anterior-posterior (axis 1) crops of already-unilateral
breast test items -- shared by duke-breast-mri and acrin6698 (added 2026-10-01).

Why: I-SPY2's unilateral TRAINING volumes cover breast + a little chest wall
(A-P ~174 mm), while full-chest acquisitions extend ~330-350 mm A-P through the
whole thorax. See unilateral_crop.py's AP_* constants for the measured training
geometry; the window comes from ap_skin_window(): ~27 mm air in front of each
case's OWN anterior skin line + 147 mm behind it, extended (never clipped) to
contain the lesion with a 5 mm margin.

One window per case, computed once from --ref-dir (or the first item's image)
and the first item's label, applied identically to every listed item (all items
of a case share one voxel grid -- checked). Existing input dirs are never
modified; outputs go to sibling {imagesTs,labelsTs}_<item><suffix>/.

Usage:
  derive_ap_crop.py --raw <2_nnUNet_*/raw> --items t1wce_uni precontrast_uni \
      --suffix ap --out-names t1wce_uniap precontrast_uniap [--ref-dir <dir>]
Writes <raw>/ap_crop_manifest_<first out name>.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from unilateral_crop import (AP_ANTERIOR_AIR_MM, AP_DEPTH_FROM_SKIN_MM,
                             ap_skin_window, crop_axis1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out-names", nargs="+", required=True)
    ap.add_argument("--ref-dir", type=Path, default=None,
                    help="dir of <case>_0000.nii.gz used ONLY for skin detection (default: first item's images)")
    a = ap.parse_args()
    assert len(a.items) == len(a.out_names)
    for o in a.out_names:
        for kind in ("imagesTs", "labelsTs"):
            d = a.raw / f"{kind}_{o}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
    ref_dir = a.ref_dir or (a.raw / f"imagesTs_{a.items[0]}")
    cases = sorted(p.name[:-7] for p in (a.raw / f"labelsTs_{a.items[0]}").glob("*.nii.gz"))
    rows = []
    for cid in cases:
        lab0 = nib.load(str(a.raw / f"labelsTs_{a.items[0]}" / f"{cid}.nii.gz"))
        assert nib.aff2axcodes(lab0.affine) == ("L", "P", "S"), cid
        ref = nib.load(str(ref_dir / f"{cid}_0000.nii.gz"))
        if ref.shape != lab0.shape or not np.allclose(ref.affine, lab0.affine, atol=1e-3):
            raise RuntimeError(f"{cid}: reference grid != label grid")
        sp = float(abs(lab0.header.get_zooms()[1]))
        w = ap_skin_window(np.asanyarray(ref.dataobj), np.asanyarray(lab0.dataobj) > 0, sp)
        for it, o in zip(a.items, a.out_names):
            img = nib.load(str(a.raw / f"imagesTs_{it}" / f"{cid}_0000.nii.gz"))
            lab = nib.load(str(a.raw / f"labelsTs_{it}" / f"{cid}.nii.gz"))
            if img.shape != lab0.shape or not np.allclose(img.affine, lab0.affine, atol=1e-3):
                raise RuntimeError(f"{cid}/{it}: grid differs from {a.items[0]}")
            ci, cl = crop_axis1(img, w["lo"], w["hi"]), crop_axis1(lab, w["lo"], w["hi"])
            n_in, n_out = int((np.asanyarray(lab.dataobj) > 0).sum()), int((np.asanyarray(cl.dataobj) > 0).sum())
            if n_in != n_out:
                raise RuntimeError(f"{cid}/{it}: crop clipped the lesion ({n_out}/{n_in} voxels kept)")
            nib.save(ci, str(a.raw / f"imagesTs_{o}" / f"{cid}_0000.nii.gz"))
            nib.save(cl, str(a.raw / f"labelsTs_{o}" / f"{cid}.nii.gz"))
        rows.append({"case_id": cid, **w, "ap_spacing_mm": round(sp, 3),
                     "native_ap_mm": round(lab0.shape[1] * sp, 1),
                     "cropped_ap_mm": round((w["hi"] - w["lo"]) * sp, 1),
                     "skin_offset_mm": round(w["skin"] * sp, 1)})
    q = lambda k: {p: round(float(np.percentile([r[k] for r in rows], p)), 1) for p in (5, 50, 95)}
    man = {"items": dict(zip(a.items, a.out_names)), "n_cases": len(rows),
           "method": f"skin-anchored A-P window: {AP_ANTERIOR_AIR_MM} mm anterior air + "
                     f"{AP_DEPTH_FROM_SKIN_MM} mm behind skin (I-SPY2 unilateral training geometry), "
                     f"extended to contain lesion +5 mm; L-R / S-I untouched; lesion voxels verified 100% kept",
           "n_extended_for_lesion": sum(r["extended_for_lesion"] for r in rows),
           "native_ap_mm": q("native_ap_mm"), "cropped_ap_mm": q("cropped_ap_mm"),
           "skin_offset_mm": q("skin_offset_mm"), "ref_dir": str(ref_dir), "cases": rows}
    (a.raw / f"ap_crop_manifest_{a.out_names[0]}.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: v for k, v in man.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
