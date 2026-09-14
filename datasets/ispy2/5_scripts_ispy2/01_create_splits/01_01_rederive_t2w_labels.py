#!/usr/bin/env python3
"""
Re-derive the T2w-space lesion masks using the SHARED DICOM FRAME OF REFERENCE
(identity world transform) instead of image registration.

WHY — see 01_00_audit_t2w_label_registration.py and its report
(4_splits_ispy2/t2w_label_registration_audit.md). Summary of the audit over all 560
BIDS patients:

  natively-unilateral (n=438): centroid distance between the T1wce mask and the
      existing registration-derived T2w mask is median 101.2 mm (p90 135, max 194);
      91.8% of them have ZERO overlap; 44.1% put the lesion in the opposite breast.
  natively-bilateral  (n=122): median 6.3 mm and median Dice 0.32 — mostly fine, but
      27.9% still have zero overlap and 34.4% are >20 mm off.

The existing T2w masks were made by rigid Mattes-MI registration of the T1wce volume
onto the T2w volume (see derivatives/labels/README.md). That registration is both
unnecessary and unreliable here:
  * unnecessary — the DCE and T2w series carry the SAME DICOM FrameOfReferenceUID,
    and the BIDS affines reproduce the DICOM ImagePositionPatient/ImageOrientation
    geometry EXACTLY (max corner deviation 0.00 mm, verified on 12 patients spanning
    both FOV groups against the staged raw DICOM). The correct T1wce->T2w map is
    therefore the identity in world space.
  * unreliable — for a unilateral patient the moving image is a half-FOV single-breast
    crop and the fixed image is a full-FOV bilateral scan, so mutual information has a
    strong spurious optimum over the contralateral breast.

This script writes CORRECTED masks under NEW filenames and DOES NOT TOUCH the existing
ones — another effort's ambl->ispy2 cross-evaluation currently reads the old files, and
silently changing data under it would be worse than leaving both versions on disk and
flagging the problem. (Those existing T2w cross-eval numbers ARE affected and should be
treated as invalid until re-run; that is called out in the report this writes.)

  in : derivatives/labels/sub-X/anat/sub-X_acq-firstpost_T1w_label-lesion_seg.nii.gz
       1_BIDS_ispy2/breast-ispy2/sub-X/anat/sub-X_T2w.nii.gz          (target grid)
  out: derivatives/labels/sub-X/anat/sub-X_desc-sharedframe_T2w_label-lesion_seg.nii.gz

Method: for every T2w voxel centre, map to world, map into T1wce index space with
inv(A_T1) @ A_T2, round to the nearest T1wce voxel, take its label (0 outside). This is
standard nearest-neighbour label resampling; no interpolation of label values ever
happens. The T2w slice slab is much coarser than the DCE's (typically 30-70 slices vs
70-260), so the resampled voxel COUNT is much smaller — the meaningful check is
therefore PHYSICAL VOLUME (mm^3), which this script reports per patient, not voxel
count. Orientation is re-verified with nib.aff2axcodes on every written mask.

Writes 4_splits_ispy2/t2w_label_rederivation.{json,md}.

Run:  bash 01_create_splits/01_01_rederive_t2w_labels.sh
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT = DATASET_ROOT / "1_BIDS_ispy2" / "breast-ispy2"
DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
SPLITS = DATASET_ROOT / "4_splits_ispy2"
EXCLUSIONS = SPLITS / "unilateral_fov_exclusions.json"
N_WORKERS = int(os.environ.get("ISPY2_FOV_WORKERS", "16"))

T1_LABEL_SUFFIX = "acq-firstpost_T1w"
OUT_LABEL_SUFFIX = "desc-sharedframe_T2w"
EXPECTED_AXCODES = ("L", "P", "S")


def voxel_volume_mm3(aff: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(aff)[:3, :3])))


def resample_nn_identity(src: nib.Nifti1Image, ref: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbour resample src's binary mask onto ref's grid, identity world
    transform (shared DICOM frame of reference)."""
    M = np.linalg.inv(np.asarray(src.affine)) @ np.asarray(ref.affine)
    g = np.indices(ref.shape).reshape(3, -1)
    idx = np.rint(M[:3, :3] @ g + M[:3, 3:4]).astype(np.int64)
    ok = np.ones(idx.shape[1], dtype=bool)
    for k in range(3):
        ok &= (idx[k] >= 0) & (idx[k] < src.shape[k])
    src_arr = np.asanyarray(src.dataobj) > 0
    out = np.zeros(idx.shape[1], dtype=np.uint8)
    out[ok] = src_arr[idx[0, ok], idx[1, ok], idx[2, ok]]
    return out.reshape(ref.shape)


def _one(args) -> dict:
    sub, fov = args
    try:
        t2 = nib.load(str(BIDS_ROOT / sub / "anat" / f"{sub}_T2w.nii.gz"))
        l1 = nib.load(str(DERIV_ROOT / sub / "anat" / f"{sub}_{T1_LABEL_SUFFIX}_label-lesion_seg.nii.gz"))
        m1 = np.asanyarray(l1.dataobj) > 0
        new = resample_nn_identity(l1, t2)

        v1 = voxel_volume_mm3(l1.affine); v2 = voxel_volume_mm3(t2.affine)
        vol1 = float(m1.sum()) * v1
        vol2 = float(new.sum()) * v2
        img = nib.Nifti1Image(new, t2.affine)
        img.header.set_zooms(t2.header.get_zooms()[:3])
        img.set_data_dtype(np.uint8)
        dst = DERIV_ROOT / sub / "anat" / f"{sub}_{OUT_LABEL_SUFFIX}_label-lesion_seg.nii.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(dst))

        ax = "".join(nib.aff2axcodes(nib.load(str(dst)).affine))
        r = {"sub": sub, "fov": fov,
             "vox_t1": int(m1.sum()), "vox_t2_new": int(new.sum()),
             "vol_t1_mm3": round(vol1, 1), "vol_t2_new_mm3": round(vol2, 1),
             "vol_ratio": round(vol2 / vol1, 4) if vol1 else None,
             "t2_voxel_mm3": round(v2, 4), "t1_voxel_mm3": round(v1, 4),
             "axcodes": ax, "shape": list(new.shape)}
        w = []
        if ax != "".join(EXPECTED_AXCODES):
            w.append(f"axcodes {ax} != LPS")
        if int(new.sum()) == 0 and int(m1.sum()) > 0:
            w.append("EMPTY after resampling — lesion falls outside the T2w slab/FOV")
        if r["vol_ratio"] is not None and not (0.5 <= r["vol_ratio"] <= 2.0):
            w.append(f"volume ratio {r['vol_ratio']} outside [0.5, 2.0]")
        r["warnings"] = w
        return r
    except Exception as exc:                      # noqa: BLE001
        return {"sub": sub, "fov": fov, "failed": repr(exc), "warnings": ["FAILED"]}


def load_laterality() -> dict:
    excl = json.loads(EXCLUSIONS.read_text())
    lat = {c["sub"]: "uni" for c in excl["excluded_unilateral_cases"]}
    for c in excl["manual_overrides_reclassified_as_bilateral"]:
        lat[c["sub"]] = "bil"
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-ispy2*") if p.is_dir())
    for s in subs:
        lat.setdefault(s, "bil")
    return {s: lat[s] for s in subs}


def main() -> None:
    lat = load_laterality()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        recs = list(ex.map(_one, sorted(lat.items()), chunksize=4))
    recs.sort(key=lambda r: r["sub"])
    (SPLITS / "t2w_label_rederivation.json").write_text(json.dumps(recs, indent=1))

    ok = [r for r in recs if "failed" not in r]
    empty = [r for r in ok if r["vox_t2_new"] == 0]
    badax = [r for r in ok if r["axcodes"] != "LPS"]
    ratios = np.array([r["vol_ratio"] for r in ok if r["vol_ratio"] is not None])
    L = ["# ispy2 — T2w lesion masks re-derived through the shared DICOM frame\n",
         f"Written as `sub-*_{OUT_LABEL_SUFFIX}_label-lesion_seg.nii.gz`. The old, "
         "registration-derived `sub-*_T2w_label-lesion_seg.nii.gz` files are left "
         "untouched on disk.\n",
         "> **The old T2w masks are wrong for most patients** (see "
         "`t2w_label_registration_audit.md`: 91.8% zero overlap on the 438 unilateral "
         "patients, 27.9% on the 122 bilateral ones). Any existing ambl->ispy2 "
         "cross-evaluation **T2w** result computed against them is invalid and needs "
         "re-running against these corrected masks. T1wce results are unaffected — "
         "the T1wce mask is the collection's own DICOM-SEG and was never registered.\n",
         f"Patients: {len(ok)} written, {len(recs)-len(ok)} failed\n",
         f"- masks empty after resampling (lesion outside the T2w slab): **{len(empty)}**"
         + (f" — {[r['sub'] for r in empty]}" if empty else ""),
         f"- orientation != LPS: **{len(badax)}**" + (f" — {[r['sub'] for r in badax]}" if badax else ""),
         f"- lesion VOLUME preserved (mm^3 ratio new-T2w / T1wce): median "
         f"{np.median(ratios):.3f}, p5 {np.percentile(ratios,5):.3f}, "
         f"p95 {np.percentile(ratios,95):.3f}",
         f"- volume ratio outside [0.5, 2.0]: **{int(((ratios<0.5)|(ratios>2.0)).sum())}**"
         " (expected to be non-zero: the T2w slab is much coarser than the DCE's, so a"
         " small lesion can be under- or over-represented by whole-voxel quantisation)\n"]
    warned = [r for r in ok if r["warnings"]]
    L.append(f"## Patients with warnings ({len(warned)})\n")
    for r in warned[:400]:
        L.append(f"- `{r['sub']}` ({r['fov']}): " + "; ".join(r["warnings"]))
    if len(warned) > 400:
        L.append(f"- ... and {len(warned)-400} more (see t2w_label_rederivation.json)")
    (SPLITS / "t2w_label_rederivation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
