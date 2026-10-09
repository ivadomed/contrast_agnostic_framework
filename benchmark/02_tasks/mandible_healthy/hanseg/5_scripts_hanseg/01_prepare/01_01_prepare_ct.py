#!/usr/bin/env python3
"""
HaN-Seg -> FOV-matched, CBCT-comparable CT test set for toothfairy2 models.

Produces, per case:
  BIDS  sub-<id>/anat/sub-<id>_ct.nii.gz  + derivatives .../_label-mandible_seg.nii.gz
  nnUNet <nnUNet_raw>/imagesTs_ct/hanseg_<id>_0000.nii.gz  (+ labelsTs_ct/)

     ⚠️ FLAT layout, deliberately — no Dataset<id>/ subdir. The shared cross-dataset
     predict driver (00_commun_scripts/00_02_predict/predict_common.sh) resolves
     cross-mode inputs as "${nnUNet_raw}/imagesTs_<item>", and a nested layout makes
     it skip every item with "input dir missing/empty" rather than fail loudly. Every
     other cross-dataset-only consumer in this repo (mslesseg, lld-mmri-hcc,
     liverhccseg, ispy2's old role) uses the same flat layout.

WHY ONLY CT (the MR arm is deliberately NOT built here)
--------------------------------------------------------
HaN-Seg ships a planning CT and a T1 MR for the same 42 patients, which is why it
was chosen — same cohort, same annotator, so CT-vs-MR would be a clean modality
contrast. But the two are stored in DIFFERENT, UNREGISTERED frames (case_01: CT
origin z=-759.0 with 0.558x0.558x2.0 spacing; MR origin z=-98.6 with
0.703x0.703x3.0), and every OAR mask — Bone_Mandible included — is defined on the
CT grid only. Producing MR ground truth would mean registering 42 CT/MR pairs here
and trusting the result unvalidated. A registration error would silently displace
the GT and show up as a "method failure" on the MR arm — precisely the class of bug
that has twice produced a wrong conclusion in this project (see CLAUDE.md's
orientation-verification note). So the MR arm is left unbuilt and recorded as
follow-up work rather than fabricated. The CT arm alone is still a genuine
cross-MODALITY test: CBCT -> conventional CT.

⚠️ FIELD-OF-VIEW MATCHING — THE POINT OF THIS SCRIPT
-----------------------------------------------------
A CBCT-trained model has never seen a whole head-and-neck volume. Raw HaN-Seg CT is
571 x 571 x 404 mm; toothfairy2's training volumes are 112 x 104 x 51 mm (median;
p10-p90 x 111-139, y 88-123, z 50-80). Predicting on the raw volume would test the
model on a field of view outside anything it was trained on, and the resulting
collapse would be a FOV artefact reported as a modality-generalization result.

Each case is therefore cropped to a FIXED-SIZE box centred on the mandible, sized
from the training FOV distribution — i.e. "what a dental CBCT of this patient would
have covered".

LEAKAGE, STATED PLAINLY: the box is CENTRED using the ground-truth mandible, so it
leaks the structure's POSITION (not its extent — the size is fixed, not a tight
bounding box). Two things make this acceptable, and both must stay true:
  1. It is applied IDENTICALLY to every method, so it cannot bias the between-method
     comparison, which is the only thing this dataset is used to measure.
  2. It inflates ABSOLUTE Dice relative to a fully automatic localize-then-segment
     pipeline, so these numbers are not comparable to a clinical end-to-end system
     and must never be presented as such.
This mirrors the FOV-restricted / hard-crop eval already established for chaos.

If a patient's mandible does not fit the fixed box, the box is EXPANDED for that
case up to the largest training FOV, and the expansion is recorded in the audit.
Beyond that cap the crop wins, deliberately: growing further would put the input
back outside the FOV range the model was trained on, which is the failure this whole
script exists to prevent.

MEASURED RESULT (42/42 cases, 2026-09-07): 41 cases needed some expansion and 34 hit
the training-FOV cap — HaN-Seg's Bone_Mandible is a COMPLETE mandible (condyle to
condyle), whereas a dental CBCT frequently truncates the rami/condyles, so
toothfairy2's own training labels are themselves FOV-truncated. That is a real
property of the training distribution, not a defect here. Crops land at
144-153 x 128-133 x 80-91 mm. GT retention: median 100%, p10 98.3%, minimum 95.3%,
none below 95%. Because the IMAGE and the LABEL are cropped identically, a "lost" GT
voxel is simply not present in the evaluated volume — it is never scored as a false
negative — so the comparison stays self-consistent and mirrors what a real dental
CBCT of that patient would have covered.

OTHER REAL DOMAIN SHIFTS, for the record (not bugs, do not "fix" them): HaN-Seg CT is
strongly anisotropic natively (0.558 x 0.558 x 2.0 or 3.0 mm) against toothfairy2's
0.6 mm isotropic, so fine mandibular detail is genuinely unrecoverable after
resampling; and CT carries calibrated Hounsfield units while CBCT does not.

Run via 01_01_prepare_ct.sh.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

HANSEG_ZIP = Path(os.environ["HANSEG_ZIP"])
BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
N_WORKERS = int(os.environ.get("HANSEG_WORKERS", "8"))
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "hanseg_stage"

TARGET_SPACING = float(os.environ.get("HANSEG_TARGET_SPACING", "0.6"))
# Fixed crop, in mm. Chosen near the p90 of toothfairy2's own training FOV
# (x 139 / y 123 / z 80) rather than the median, so a typical mandible — ~120 mm
# bicondylar width, ~100 mm A-P, ~65 mm tall — fits without expansion while the box
# still sits inside the range the model was trained on.
BOX_MM = np.array([float(os.environ.get("HANSEG_BOX_X", "144")),
                   float(os.environ.get("HANSEG_BOX_Y", "128")),
                   float(os.environ.get("HANSEG_BOX_Z", "80"))])
# Never exceed the largest training FOV — beyond this we are back to testing on an
# unseen field of view, which is the whole thing this script exists to prevent.
MAX_BOX_MM = np.array([154.0, 154.0, 89.0])
MARGIN_MM = 8.0

DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
_ZIP: zipfile.ZipFile | None = None


def _zip() -> zipfile.ZipFile:
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(HANSEG_ZIP)
    return _ZIP


def _extract(member: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _zip().open(member) as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=8 << 20)
    return dst


def _resample(img: sitk.Image, is_label: bool) -> sitk.Image:
    img = sitk.DICOMOrient(img, "RAS")
    sp = np.array(img.GetSpacing(), float)
    sz = np.array(img.GetSize(), int)
    dst = np.ceil(sz * sp / TARGET_SPACING).astype(int)
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([TARGET_SPACING] * 3)
    rs.SetSize([int(v) for v in dst])
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetTransform(sitk.Transform())
    if is_label:
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
    else:
        rs.SetInterpolator(sitk.sitkBSpline)
        rs.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return rs.Execute(img)


def prepare_one(case: str) -> dict:
    """Crop FIRST (in the native grid), resample SECOND.

    Order matters enormously here and the naive order does not merely run slow, it
    dies: a HaN-Seg CT is 571 x 571 x 404 mm, so resampling the whole volume to
    0.6 mm isotropic BEFORE cropping produces a ~950 x 950 x 673 array — 610 M
    voxels, ~2.4 GB per image in float32, before B-spline scratch space — and a
    handful of parallel workers exhaust the node's memory with no Python traceback
    (the first attempt was killed exactly this way). Cropping first in the native
    grid reduces the work to a ~260 x 230 x 40 native box, which resamples to the
    target ~240 x 213 x 133 in a fraction of a second.
    """
    stage = TMPDIR / case
    try:
        base = f"HaN-Seg/set_1/{case}/{case}"
        ct_p = _extract(f"{base}_IMG_CT.nrrd", stage / "ct.nrrd")
        sg_p = _extract(f"{base}_OAR_Bone_Mandible.seg.nrrd", stage / "seg.nrrd")
        ct0 = sitk.DICOMOrient(sitk.ReadImage(str(ct_p)), "RAS")
        sg0 = sitk.DICOMOrient(sitk.ReadImage(str(sg_p)), "RAS")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    seg0 = (sitk.GetArrayFromImage(sg0) > 0).astype(np.uint8)   # (z, y, x)
    if seg0.sum() == 0:
        raise RuntimeError("empty Bone_Mandible mask")
    if seg0.shape != sitk.GetArrayFromImage(ct0).shape:
        raise RuntimeError(f"CT/seg grid mismatch: {sitk.GetArrayFromImage(ct0).shape} vs {seg0.shape}")

    sp = np.array(ct0.GetSpacing(), float)                       # (x, y, z) mm
    sp_zyx = sp[::-1]
    idx = np.array(np.nonzero(seg0))
    lo_v, hi_v = idx.min(1), idx.max(1) + 1
    centre_v = (lo_v + hi_v) / 2.0

    # Box sizes in mm -> native voxels, per numpy axis (z, y, x).
    box_mm_zyx = np.array([BOX_MM[2], BOX_MM[1], BOX_MM[0]])
    max_mm_zyx = np.array([MAX_BOX_MM[2], MAX_BOX_MM[1], MAX_BOX_MM[0]])
    gt_mm = (hi_v - lo_v) * sp_zyx
    need_mm = gt_mm + 2 * MARGIN_MM
    expanded = bool(np.any(need_mm > box_mm_zyx))
    clipped = bool(np.any(need_mm > max_mm_zyx))
    use_mm = np.minimum(np.maximum(box_mm_zyx, need_mm), max_mm_zyx)
    box_v = use_mm / sp_zyx

    start = np.round(centre_v - box_v / 2).astype(int)
    shape = np.array(seg0.shape)
    start = np.clip(start, 0, np.maximum(shape - 1, 0))
    end = np.clip(start + np.round(box_v).astype(int), 1, shape)
    start = np.clip(end - np.round(box_v).astype(int), 0, None)   # keep full size at edges

    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    img_c = sitk.GetArrayFromImage(ct0)[sl]
    seg_c = seg0[sl]
    kept = float(seg_c.sum()) / float(seg0.sum())

    origin = ct0.TransformContinuousIndexToPhysicalPoint(
        [float(start[2]), float(start[1]), float(start[0])])
    img_i = sitk.GetImageFromArray(img_c.astype(np.float32))
    seg_i = sitk.GetImageFromArray(seg_c.astype(np.uint8))
    for o in (img_i, seg_i):
        o.SetSpacing(ct0.GetSpacing()); o.SetDirection(ct0.GetDirection()); o.SetOrigin(origin)

    # Now resample the (small) crop to the training resolution.
    img_r = _resample(img_i, is_label=False)
    seg_r = _resample(seg_i, is_label=True)

    sub = f"sub-{case.replace('_', '')}"
    a_dir = BIDS_ROOT / sub / "anat"; a_dir.mkdir(parents=True, exist_ok=True)
    d_dir = DERIV_ROOT / sub / "anat"; d_dir.mkdir(parents=True, exist_ok=True)
    img_p = a_dir / f"{sub}_ct.nii.gz"
    seg_p = d_dir / f"{sub}_ct_label-mandible_seg.nii.gz"
    sitk.WriteImage(img_r, str(img_p), useCompression=True)
    sitk.WriteImage(sitk.Cast(seg_r, sitk.sitkUInt8), str(seg_p), useCompression=True)

    size = np.array(img_r.GetSize(), float)
    return {"case": case, "sub": sub,
            "native_spacing": [round(float(v), 3) for v in sp],
            "crop_shape": [int(v) for v in img_r.GetSize()],
            "crop_mm": [round(float(v * TARGET_SPACING), 1) for v in size],
            "box_expanded": expanded, "gt_exceeds_max_training_fov": clipped,
            "gt_voxels_kept_frac": round(kept, 5),
            "orientation": "".join(nib.aff2axcodes(nib.load(str(img_p)).affine)),
            "label_orientation": "".join(nib.aff2axcodes(nib.load(str(seg_p)).affine))}


def main() -> None:
    with zipfile.ZipFile(HANSEG_ZIP) as z:
        cases = sorted({m.split("/")[2] for m in z.namelist()
                        if m.startswith("HaN-Seg/set_1/") and m.count("/") > 2 and m.split("/")[2]})
    print(f"{len(cases)} HaN-Seg cases", flush=True)

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(prepare_one, c): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                failed.append({"case": futs[f], "error": repr(e)})
            if i % 10 == 0:
                print(f"  {i}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: r["case"])

    # nnU-Net test dirs (hard links from BIDS) — FLAT, see the module docstring.
    img_ts, lab_ts = NNUNET_RAW / "imagesTs_ct", NNUNET_RAW / "labelsTs_ct"
    img_ts.mkdir(parents=True, exist_ok=True); lab_ts.mkdir(parents=True, exist_ok=True)
    for r in rows:
        sub = r["sub"]; cid = f"hanseg_{sub.removeprefix('sub-')}"
        for src, dst in ((BIDS_ROOT / sub / "anat" / f"{sub}_ct.nii.gz", img_ts / f"{cid}_0000.nii.gz"),
                         (DERIV_ROOT / sub / "anat" / f"{sub}_ct_label-mandible_seg.nii.gz",
                          lab_ts / f"{cid}.nii.gz")):
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    bad = [r["case"] for r in rows if r["orientation"] != "RAS" or r["label_orientation"] != "RAS"]
    lost = [r["case"] for r in rows if r["gt_voxels_kept_frac"] < 0.999]
    audit = {"n": len(rows), "n_failed": len(failed), "failed": failed,
             "target_spacing_mm": TARGET_SPACING, "box_mm": BOX_MM.tolist(),
             "n_box_expanded": sum(r["box_expanded"] for r in rows),
             "n_gt_exceeds_max_training_fov": sum(r["gt_exceeds_max_training_fov"] for r in rows),
             "cases_losing_gt_voxels": lost, "not_ras": bad, "cases": rows}
    (BIDS_ROOT / "prepare_audit.json").write_text(json.dumps(audit, indent=2))
    print(f"prepared {len(rows)}, failed {len(failed)}")
    print(f"  box expanded for {audit['n_box_expanded']} case(s); "
          f"{audit['n_gt_exceeds_max_training_fov']} exceed the largest training FOV")
    if lost:
        print(f"  ⚠️ {len(lost)} case(s) lost GT voxels to the crop: {lost[:10]}", file=sys.stderr)
    if bad:
        print(f"  ⚠️ not RAS: {bad}", file=sys.stderr)
    if failed or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
