#!/usr/bin/env python3
"""
ToothFairy2: unpacked challenge release (nnU-Net-format .mha) -> BIDS NIfTI.

Run via 00_00_extract_and_bidsify.sh (never bare — that wrapper dispatches through
run_job, per CLAUDE.md's "no heavy compute on a login node").

WHAT THIS DOES, PER CASE
  1. stream imagesTr/<case>_0000.mha + labelsTr/<case>.mha OUT OF THE RELEASE ZIP
     into $SLURM_TMPDIR, read with SimpleITK, delete immediately
  2. REORIENT both to RAS
  3. RESAMPLE both to TARGET_SPACING mm isotropic (image: B-spline; label: nearest)
  4. REMAP the 42-class label volume to this project's 5 classes
     (00_utils/toothfairy2_labels.py is the single source of truth)
  5. write BIDS: sub-<id>/anat/sub-<id>_acq-cbct_ct.nii.gz
     + derivatives/labels/sub-<id>/anat/sub-<id>_acq-cbct_ct_label-maxillofacial_seg.nii.gz
  6. append a per-case audit row (orientation BEFORE and AFTER, shape, spacing,
     per-class voxel counts) to conversion_audit.json

WHY RESAMPLE (0.3 -> 0.6 mm isotropic)
--------------------------------------
The release is 0.3 mm isotropic, which nnU-Net's planner would faithfully adopt —
the DKFZ ToothFairy2 entry trains at 0.3 mm with a 160x320x320 patch, i.e. a
config built to win a segmentation challenge on 42 classes including sub-mm
canals. That is not this project's question. Here CBCT is one arm of a 6-method
augmentation comparison that must run 10 training configurations x 3 folds
alongside five other datasets, and the reduced task's smallest structure
(pharynx, maxillary sinus, whole teeth) is centimetre-scale. Halving the
resolution cuts voxel count 8x, which is what makes the full suite + ablation
ladder affordable at all. It is applied identically to every method, so it cannot
bias the comparison — it only means absolute Dice here is not comparable to the
ToothFairy2 leaderboard, which we never claim.

WHY NEVER FULLY UNZIP
---------------------
The 26 GB release expands to 109.5 GB of uncompressed .mha. Materialising that on
$SCRATCH would be ~20% of the whole 1 TB quota for a staging artifact nothing reads
twice, on a filesystem that purges on inactivity. Instead each worker opens its own
handle on the zip and extracts ONE case at a time into $SLURM_TMPDIR (node-local
disk, job lifetime — exactly what CLAUDE.md prescribes for many-small-files work),
converts it, and deletes it. Peak extra disk is a few GB, and the network
filesystem sees only the sequential zip read plus the compressed NIfTI writes.

⚠️ ORIENTATION IS CHECKED FOR EVERY CASE, NOT SAMPLED. Twice in this project a
wrong "method failure" conclusion has been traced to an unverified orientation
(see CLAUDE.md / the liverhccseg bugfix). The audit records aff2axcodes before
and after; 00_00_verify_bidsify.py fails loudly if any post-conversion case is
not RAS.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toothfairy2_labels import build_remap, NNUNET_LABELS, TARGET_ID_TO_NAME  # noqa: E402

TARGET_SPACING = float(os.environ.get("TF2_TARGET_SPACING", "0.6"))
TF2_ZIP = Path(os.environ["TF2_ZIP"])                 # the release zip, read in place
BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
N_WORKERS = int(os.environ.get("TF2_WORKERS", "8"))
LIMIT = int(os.environ.get("TF2_LIMIT", "0"))         # 0 = all (debug aid only)
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "tf2_stage"

DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
REMAP = build_remap()

# One ZipFile handle per worker process. A single shared handle is NOT thread- or
# process-safe (it seeks), so it is created lazily inside each worker instead.
_ZIP: zipfile.ZipFile | None = None
_PREFIX: str = ""


def _zip() -> zipfile.ZipFile:
    global _ZIP, _PREFIX
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(TF2_ZIP)
        _PREFIX = zip_prefix(_ZIP)
    return _ZIP


def zip_prefix(z: zipfile.ZipFile) -> str:
    """Leading path inside the archive that contains imagesTr/ (e.g.
    'Dataset112_ToothFairy2/'). Derived, never assumed — the mirror could be
    repacked flat."""
    for n in z.namelist():
        if "imagesTr/" in n:
            return n[: n.index("imagesTr/")]
    raise RuntimeError(f"no imagesTr/ inside {TF2_ZIP}")


def _extract(member: str, dst_dir: Path) -> Path:
    """Extract one member to dst_dir (flat), returning its path."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    out = dst_dir / Path(member).name
    with _zip().open(member) as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=8 << 20)
    return out


def _resample(img: sitk.Image, is_label: bool) -> sitk.Image:
    """Reorient to RAS, then resample to TARGET_SPACING mm isotropic."""
    img = sitk.DICOMOrient(img, "RAS")
    src_spacing = np.array(img.GetSpacing(), dtype=float)
    src_size = np.array(img.GetSize(), dtype=int)
    # ceil so the resampled grid always covers the full physical extent; a floor
    # here would silently crop a slice off the superior/anterior edge.
    dst_size = np.ceil(src_size * src_spacing / TARGET_SPACING).astype(int)

    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([TARGET_SPACING] * 3)
    rs.SetSize([int(v) for v in dst_size])
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetTransform(sitk.Transform())
    if is_label:
        rs.SetInterpolator(sitk.sitkNearestNeighbor)
        rs.SetDefaultPixelValue(0)
    else:
        rs.SetInterpolator(sitk.sitkBSpline)
        # CBCT air is strongly negative; pad with the volume's own minimum rather
        # than 0, which would read as soft tissue and invent a bright rim.
        rs.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return rs.Execute(img)


# ITK/SimpleITK direction cosines are expressed in LPS, while nibabel's
# aff2axcodes — the convention this project states its orientations in, and the one
# every downstream reader sees — is RAS. Converting between them is a sign flip on
# the first two axes. Getting this backwards does not corrupt data, but it makes the
# audit REPORT the opposite of the truth, and a mislabelled orientation audit is
# worse than none: this project has twice traced a wrong "method failure" conclusion
# to an unverified/misread orientation.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def _axcodes(img: sitk.Image) -> str:
    """RAS-convention anatomical axis codes of a SimpleITK image."""
    d = _LPS_TO_RAS @ np.array(img.GetDirection()).reshape(3, 3)
    codes = []
    for col in range(3):
        axis = int(np.argmax(np.abs(d[:, col])))
        pos = int(d[axis, col] > 0)   # int(), not the numpy.bool_ — it indexes a str
        codes.append(("LR"[pos], "PA"[pos], "IS"[pos])[axis])
    return "".join(codes)


def _nib_axcodes(path: Path) -> str:
    """Ground-truth orientation of a WRITTEN file, read back the way every
    downstream consumer reads it. Authoritative over the SimpleITK-side codes."""
    return "".join(nib.aff2axcodes(nib.load(str(path)).affine))


def convert_one(case: str) -> dict:
    sub = f"sub-{case.replace('_', '')}"
    stage = TMPDIR / case
    _zip()          # MUST precede the _PREFIX reads below — it is what sets _PREFIX
    try:
        img_p = _extract(f"{_PREFIX}imagesTr/{case}_0000.mha", stage)
        lab_p = _extract(f"{_PREFIX}labelsTr/{case}.mha", stage)
        img = sitk.ReadImage(str(img_p))
        lab = sitk.ReadImage(str(lab_p))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    before = {"image": _axcodes(img), "label": _axcodes(lab),
              "shape": list(img.GetSize()), "spacing": [round(s, 4) for s in img.GetSpacing()]}

    img_r = _resample(img, is_label=False)
    lab_r = _resample(lab, is_label=True)

    arr = sitk.GetArrayFromImage(lab_r)
    out = np.zeros_like(arr, dtype=np.uint8)
    for src_id, tgt_id in REMAP.items():
        out[arr == src_id] = tgt_id
    counts = {TARGET_ID_TO_NAME[i]: int((out == i).sum()) for i in TARGET_ID_TO_NAME}

    lab_out = sitk.GetImageFromArray(out)
    lab_out.CopyInformation(lab_r)

    a_dir = BIDS_ROOT / sub / "anat"; a_dir.mkdir(parents=True, exist_ok=True)
    d_dir = DERIV_ROOT / sub / "anat"; d_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.Cast(img_r, sitk.sitkFloat32),
                    str(a_dir / f"{sub}_acq-cbct_ct.nii.gz"), useCompression=True)
    sitk.WriteImage(lab_out,
                    str(d_dir / f"{sub}_acq-cbct_ct_label-maxillofacial_seg.nii.gz"),
                    useCompression=True)

    img_out_p = a_dir / f"{sub}_acq-cbct_ct.nii.gz"
    lab_out_p = d_dir / f"{sub}_acq-cbct_ct_label-maxillofacial_seg.nii.gz"
    return {"case": case, "sub": sub, "before": before,
            "after": {"image": _axcodes(img_r), "label": _axcodes(lab_r),
                      "image_nib": _nib_axcodes(img_out_p),
                      "label_nib": _nib_axcodes(lab_out_p),
                      "shape": list(img_r.GetSize()),
                      "spacing": [round(s, 4) for s in img_r.GetSpacing()]},
            "label_voxels": counts,
            "empty_classes": sorted(k for k, v in counts.items() if v == 0)}


def main() -> None:
    with zipfile.ZipFile(TF2_ZIP) as z:
        pre = zip_prefix(z)
        cases = sorted(Path(n).name[: -len("_0000.mha")] for n in z.namelist()
                       if n.startswith(f"{pre}imagesTr/") and n.endswith("_0000.mha"))
    if LIMIT:
        cases = cases[:LIMIT]
    print(f"{len(cases)} cases -> {BIDS_ROOT} @ {TARGET_SPACING}mm iso", flush=True)

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(convert_one, c): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:                       # keep going; report at the end
                failed.append({"case": futs[f], "error": repr(e)})
            if i % 25 == 0:
                print(f"  {i}/{len(cases)}", flush=True)

    rows.sort(key=lambda r: r["case"])
    bad_orient = [r["case"] for r in rows
                  if r["after"]["image_nib"] != "RAS" or r["after"]["label_nib"] != "RAS"]
    mismatched = [r["case"] for r in rows
                  if r["after"]["image_nib"] != r["after"]["label_nib"]]
    audit = BIDS_ROOT / "conversion_audit.json"
    audit.write_text(json.dumps(
        {"target_spacing_mm": TARGET_SPACING, "nnunet_labels": NNUNET_LABELS,
         "n_converted": len(rows), "n_failed": len(failed),
         "all_ras": not bad_orient, "n_orientation_mismatch": len(mismatched),
         "failed": failed, "cases": rows}, indent=2))
    print(f"converted {len(rows)}, failed {len(failed)} -> {audit}")
    if bad_orient:
        print(f"NOT RAS after conversion ({len(bad_orient)}): {bad_orient[:10]}", file=sys.stderr)
    if mismatched:
        print(f"image/label orientation mismatch ({len(mismatched)}): {mismatched[:10]}",
              file=sys.stderr)
    if bad_orient or mismatched:
        sys.exit(2)
    if failed:
        print("FAILURES:", json.dumps(failed[:10], indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
