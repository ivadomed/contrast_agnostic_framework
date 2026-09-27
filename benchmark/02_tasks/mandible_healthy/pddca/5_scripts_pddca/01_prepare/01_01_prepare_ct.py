#!/usr/bin/env python3
"""
Build the FOV-matched PDDCA CT test set for toothfairy2 CBCT models.

Outputs
  BIDS      <BIDS_ROOT>/sub-<case>/anat/sub-<case>_ct.nii.gz
            <BIDS_ROOT>/derivatives/labels/sub-<case>/anat/sub-<case>_ct_label-mandible_seg.nii.gz
  nnUNet    <nnUNet_raw>/imagesTs_ct/pddca_<case>_0000.nii.gz  (+ labelsTs_ct/)

WHY THIS DATASET AND WHY IT LOOKS LIKE THE HANSEG ONE
-----------------------------------------------------
PDDCA is the SECOND, INDEPENDENT CT arm for the mandible task. HaN-Seg is one
42-patient Ljubljana cohort, so every CT conclusion currently rests on a single
institution; PDDCA is RTOG 0522 (US, 370 participating sites). Its specific job is to
test whether the CT-arm findings replicate — above all the +AugLab rung's CT-specific
-7 Dice cost.

This file is deliberately a close sibling of hanseg's 01_01_prepare_ct.py and uses the
IDENTICAL box and target spacing. That is the point: the two CT arms must be
FOV-matched to each other as well as to training, or a pddca-vs-hanseg_ct difference
would confound FOV with cohort.

LABEL SEMANTICS (verified, not inherited)
-----------------------------------------
PDDCA's own protocol doc, verbatim figure caption: "Example of mandible segmentation.
Only the bone is segmented, while the teeth are excluded." Confirmed empirically:
inside the mask p99 = 1726 HU (cortical bone, no enamel population), 99.5% of
enamel-range (>2000 HU) voxels in the mandible bounding box lie OUTSIDE the mask, and
the per-slice hole-fill ratio is 1.0000 — a solid bone envelope with sockets NOT
carved. That is the IDENTICAL signature to hanseg's Bone_Mandible, so both CT arms
share one label convention.

Scoring is therefore MANDIBLE-ONLY (toothfairy2 label 1 vs this mask, one-to-one) —
never the union, which was disproven 2026-09-17. Both CT arms carry the same ~2% root
under-coverage against toothfairy2's `mandible` (which DOES carve the sockets, fill
ratio 1.073), so the residual bias is consistent rather than a new confound.

N: 40 usable of 48. Eight cases have no Mandible.nrrd and are skipped by existence
check, not by hardcoded list (0522c0329/0330/0427/0433/0441/0455/0457/0479 at v1.4.1).

CROP FIRST, RESAMPLE SECOND — same reason as hanseg: resampling a whole head-and-neck
CT to 0.6 mm before cropping builds a ~600M-voxel array per worker and gets the job
OOM-killed with no Python traceback.

Run via 01_01_prepare_ct.sh
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

ZIP_DIR = Path(os.environ["PDDCA_ZIP_DIR"])
BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
N_WORKERS = int(os.environ.get("PDDCA_WORKERS", "8"))
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "pddca_stage"

# MUST match hanseg's values — the two CT arms are FOV-matched to each other.
TARGET_SPACING = float(os.environ.get("PDDCA_TARGET_SPACING", "0.6"))
BOX_MM = np.array([float(os.environ.get("PDDCA_BOX_X", "144")),
                   float(os.environ.get("PDDCA_BOX_Y", "128")),
                   float(os.environ.get("PDDCA_BOX_Z", "80"))])
MAX_BOX_MM = np.array([154.0, 154.0, 89.0])
MARGIN_MM = 8.0

DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
ZIPS = sorted(ZIP_DIR.glob("part*.zip"))


def index_cases() -> dict[str, tuple[Path, str, str]]:
    """case -> (zip path, img member, mandible member). Only cases WITH a mandible."""
    found: dict[str, dict[str, tuple[Path, str]]] = {}
    for z in ZIPS:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith("/"):
                    continue
                parts = m.split("/")
                case = parts[0]
                leaf = parts[-1]
                if leaf == "img.nrrd":
                    found.setdefault(case, {})["img"] = (z, m)
                elif leaf == "Mandible.nrrd":
                    found.setdefault(case, {})["seg"] = (z, m)
    out = {}
    for case, d in found.items():
        if "img" in d and "seg" in d:
            out[case] = (d["img"][0], d["img"][1], d["seg"][1])
    return dict(sorted(out.items()))


def _extract(zpath: Path, member: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf, zf.open(member) as f_in, open(dst, "wb") as f_out:
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


def prepare_one(case: str, zpath_s: str, img_m: str, seg_m: str) -> dict:
    zpath = Path(zpath_s)
    stage = TMPDIR / case
    try:
        img_p = _extract(zpath, img_m, stage / "img.nrrd")
        seg_p = _extract(zpath, seg_m, stage / "Mandible.nrrd")
        ct0 = sitk.DICOMOrient(sitk.ReadImage(str(img_p)), "RAS")
        sg0 = sitk.DICOMOrient(sitk.ReadImage(str(seg_p)), "RAS")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    seg0 = (sitk.GetArrayFromImage(sg0) > 0).astype(np.uint8)   # (z, y, x)
    if seg0.sum() == 0:
        raise RuntimeError("empty Mandible mask")
    if seg0.shape != sitk.GetArrayFromImage(ct0).shape:
        raise RuntimeError(f"CT/seg grid mismatch: {sitk.GetArrayFromImage(ct0).shape} vs {seg0.shape}")

    sp = np.array(ct0.GetSpacing(), float)
    sp_zyx = sp[::-1]
    idx = np.array(np.nonzero(seg0))
    lo_v, hi_v = idx.min(1), idx.max(1) + 1
    centre_v = (lo_v + hi_v) / 2.0

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
    start = np.clip(end - np.round(box_v).astype(int), 0, None)

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

    img_r = _resample(img_i, is_label=False)
    seg_r = _resample(seg_i, is_label=True)

    sub = f"sub-{case}"
    a_dir = BIDS_ROOT / sub / "anat"; a_dir.mkdir(parents=True, exist_ok=True)
    d_dir = DERIV_ROOT / sub / "anat"; d_dir.mkdir(parents=True, exist_ok=True)
    out_img = a_dir / f"{sub}_ct.nii.gz"
    out_seg = d_dir / f"{sub}_ct_label-mandible_seg.nii.gz"
    sitk.WriteImage(img_r, str(out_img), useCompression=True)
    sitk.WriteImage(sitk.Cast(seg_r, sitk.sitkUInt8), str(out_seg), useCompression=True)

    size = np.array(img_r.GetSize(), float)
    return {"case": case, "sub": sub,
            "native_spacing": [round(float(v), 3) for v in sp],
            "crop_shape": [int(v) for v in img_r.GetSize()],
            "crop_mm": [round(float(v * TARGET_SPACING), 1) for v in size],
            "box_expanded": expanded, "gt_exceeds_max_training_fov": clipped,
            "gt_voxels_kept_frac": round(kept, 5),
            "orientation": "".join(nib.aff2axcodes(nib.load(str(out_img)).affine)),
            "label_orientation": "".join(nib.aff2axcodes(nib.load(str(out_seg)).affine))}


def main() -> None:
    if not ZIPS:
        sys.exit(f"no part*.zip under {ZIP_DIR}")
    cases = index_cases()
    print(f"[pddca] {len(ZIPS)} archives, {len(cases)} cases WITH a mandible label", flush=True)
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    TMPDIR.mkdir(parents=True, exist_ok=True)

    rows, errs = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(prepare_one, c, str(z), im, sm): c for c, (z, im, sm) in cases.items()}
        for f in as_completed(futs):
            c = futs[f]
            try:
                rows.append(f.result()); print(f"  {c} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                errs.append({"case": c, "error": str(e)}); print(f"  {c} FAILED: {e}", file=sys.stderr, flush=True)

    rows.sort(key=lambda r: r["case"])
    img_ts, lab_ts = NNUNET_RAW / "imagesTs_ct", NNUNET_RAW / "labelsTs_ct"
    img_ts.mkdir(parents=True, exist_ok=True); lab_ts.mkdir(parents=True, exist_ok=True)
    for r in rows:
        sub = r["sub"]
        shutil.copyfile(BIDS_ROOT / sub / "anat" / f"{sub}_ct.nii.gz",
                        img_ts / f"pddca_{r['case']}_0000.nii.gz")
        shutil.copyfile(DERIV_ROOT / sub / "anat" / f"{sub}_ct_label-mandible_seg.nii.gz",
                        lab_ts / f"pddca_{r['case']}.nii.gz")

    bad = [r["case"] for r in rows if r["orientation"] != "RAS" or r["label_orientation"] != "RAS"]
    audit = {"n_cases_with_mandible": len(cases), "n_ok": len(rows), "n_failed": len(errs),
             "target_spacing_mm": TARGET_SPACING, "box_mm": BOX_MM.tolist(),
             "max_box_mm": MAX_BOX_MM.tolist(), "margin_mm": MARGIN_MM,
             "non_RAS": bad, "errors": errs, "cases": rows}
    (BIDS_ROOT / "prepare_audit.json").write_text(json.dumps(audit, indent=2))

    kept = [r["gt_voxels_kept_frac"] for r in rows]
    print(f"\n[pddca] {len(rows)} ok / {len(errs)} failed")
    if kept:
        print(f"[pddca] GT retention: min {min(kept):.4f} mean {sum(kept)/len(kept):.4f}")
    print(f"[pddca] expanded box: {sum(r['box_expanded'] for r in rows)}; "
          f"clipped at max training FOV: {sum(r['gt_exceeds_max_training_fov'] for r in rows)}")
    if bad:
        print(f"  ⚠️ not RAS: {bad}", file=sys.stderr)
        sys.exit(1)
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
