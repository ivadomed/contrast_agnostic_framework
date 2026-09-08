#!/usr/bin/env python3
"""
STATUS: BUILT AND WORKING, BUT DISABLED BY DECISION (2026-09-08). NOT USED FOR RESULTS.
--------------------------------------------------------------------------------------
Kept because the machinery is correct and documented, not because the arm is usable.
The ground truth here is propagated from CT by our own registration, and
01_03_validate_mr_registration.py showed the QC gate that accepted all 42 cases
(tissue_frac) is ANTI-CORRELATED with accuracy: displacing a mask by 12 mm RAISES
tissue_frac from 0.943 to 0.967, because sliding the mask off bone onto soft tissue
increases the "on tissue" fraction. It would have passed a centimetre-scale error.
The registration itself does have support — edge_score (mean MR gradient on the mask
boundary) peaks sharply at zero displacement (1.969) and falls monotonically to 1.420
at 12 mm — but "ground truth we generated ourselves, validated post-hoc by a metric we
also designed" cannot carry a cross-modality claim. Do not re-enable without an
INDEPENDENT validation (e.g. agreement with a second, independently-implemented
registration, or real human annotations on the MR).
Searched for a real-GT replacement and found none compatible: no public MRI dataset
annotates mandible or teeth; the one open manually-annotated airway MRI database
(53 vocal-tract volumes, 10 French speakers) labels the airway INCLUDING the open oral
cavity during phonation, which is a different structure from this task's `pharynx` in
occlusion; and public head/neck PET datasets label tumours, not anatomy.

HaN-Seg MR-T1 -> FOV-matched MRI test set for toothfairy2 models, with the mandible
label brought over from CT by registration.

THIRD MODALITY ARM. With this, a CBCT-trained model is tested on CBCT (in-domain),
conventional CT, and T1 MRI — on the SAME 42 patients, with the SAME annotator and
the SAME Bone_Mandible structure. Cohort, anatomy and ground-truth definition are
held fixed and only the imaging physics changes, which is a much cleaner modality
ablation than bolting on an unrelated MRI cohort.

⚠️⚠️ THE GROUND TRUTH HERE IS PROPAGATED, NOT DRAWN. Read this before using any
number this arm produces.
HaN-Seg delineates every OAR on the CT grid only, and its MR sits in a DIFFERENT,
UNREGISTERED frame — verified, not assumed: for case_01 the CT mandible spans
z in [-567, -483] mm while the MR volume spans z in [-99, +150] mm, i.e. the two
do not overlap at all in world coordinates. So the MR mandible mask cannot be read
off the file; it has to be transported from CT by registration, and every resulting
Dice carries that registration's error on top of the model's. Numbers from this arm
are therefore NOT directly comparable to the CT arm's, and any table showing them
must say the GT is registration-propagated.

WHY THE REGISTRATION IS MANDIBLE-LOCAL, NOT WHOLE-HEAD
------------------------------------------------------
The mandible is a MOBILE bone. A whole-head rigid registration is driven by the
skull, and if the jaw sat differently between the CT and MR sessions the mandible
will be systematically mis-seated even when the skull aligns perfectly — which
would look exactly like a segmentation failure. So registration is run twice:
  1. a coarse whole-volume rigid pass, to get the two disjoint frames roughly onto
     each other (multi-resolution Mattes MI, geometry-initialised);
  2. a refinement pass whose metric is sampled ONLY inside a dilated box around the
     mandible, so the mandible itself — not the cranium — drives the final fit.

MEASURED BEHAVIOUR (42 cases, verified by running the whole pipeline TWICE and
diffing the audits case-by-case)
  * The mandible-local refinement moves the mandible by a median of 1.37 mm, p90
    3.76 mm, and **up to 20.62 mm** (case_02). That maximum is the justification for
    the whole two-pass design: jaw position genuinely differs between the CT and MR
    sessions, so a whole-head registration would have mis-seated that patient's
    mandible by 2 cm and the resulting Dice would have been read as a model failure.
  * Run-to-run reproducibility, after switching the metric to REGULAR sampling:
    max |delta tissue_frac| = 0.0015, max |delta centroid| = 1.4 mm, and ZERO cases
    change QC verdict. That residual is floating-point jitter from the metric's
    multi-threaded reduction, not a different solution. It is NOT bit-reproducible;
    set HANSEG_DETERMINISTIC=1 to force single-threaded ITK if exact reproducibility
    is ever required (much slower).
    For contrast, with the previous RANDOM sampler the SAME script on the SAME inputs
    put case_15's mandible on tissue in one run (tissue_frac 0.968) and in mid-air in
    the next (0.058) — a QC flip. When the registration IS the ground truth, that
    class of irreproducibility is disqualifying, which is why sampling is deterministic.

PER-CASE QC, AND FAILURES ARE EXCLUDED
--------------------------------------
Registration silently going wrong is the single most likely way this arm produces a
bogus result, so every case is checked and failures are dropped rather than shipped:
  * `tissue_frac` — fraction of propagated-mask voxels landing on actual tissue
    (MR intensity above an Otsu threshold). A mask that has slipped out of the head
    lands in air and this collapses toward 0.
  * `final_metric` — the registration's own converged Mattes MI value.
  * `centroid_shift_mm` — how far the mandible centroid moved between the coarse and
    the mandible-local pass; a large value means the two passes disagree.
Cases below QC_TISSUE_FRAC are written to the audit as excluded and never enter the
nnU-Net test dirs. (An earlier version of this docstring claimed QC overlay PNGs were
written for spot-checking — they never were; no visual check was ever performed.)

Run via 01_02_prepare_mr.sh.
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
N_WORKERS = int(os.environ.get("HANSEG_WORKERS", "6"))
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "hanseg_mr"
QC_DIR = BIDS_ROOT / "qc_mr"

TARGET_SPACING = float(os.environ.get("HANSEG_TARGET_SPACING", "0.6"))
BOX_MM = np.array([144.0, 128.0, 80.0])          # same box as the CT arm
MAX_BOX_MM = np.array([154.0, 154.0, 89.0])
MARGIN_MM = 8.0
QC_TISSUE_FRAC = float(os.environ.get("HANSEG_QC_TISSUE_FRAC", "0.80"))
# Opt-in bit-reproducibility: ITK's metric reduction is multi-threaded, so even with
# deterministic sampling the optimizer path varies at float precision. Single-threading
# removes that at a large speed cost; the default is off because the measured residual
# changes no QC verdict (see the module docstring).
if os.environ.get("HANSEG_DETERMINISTIC", "0") == "1":
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

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


def _rigid(fixed: sitk.Image, moving: sitk.Image, init: sitk.Transform | None = None,
           fixed_mask: sitk.Image | None = None) -> tuple[sitk.Transform, float]:
    """Rigid Mattes-MI registration. Returns (transform fixed->moving, final metric)."""
    f = sitk.Cast(fixed, sitk.sitkFloat32)
    m = sitk.Cast(moving, sitk.sitkFloat32)
    if init is None:
        init = sitk.CenteredTransformInitializer(
            f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    # REGULAR (deterministic grid) sampling, NOT RANDOM. ITK's random sampler is
    # multi-threaded and the seed does not pin which sample lands on which thread, so
    # RANDOM makes the whole registration irreproducible run-to-run. That is
    # unacceptable when the registration IS the ground truth: two runs of this script
    # on identical inputs disagreed on case_15 (tissue_frac 0.058 vs 0.968 — i.e. one
    # run put the mandible in mid-air and the other got it right), which is how this
    # was found. Deterministic sampling makes the propagated GT reproducible.
    r.SetMetricSamplingStrategy(r.REGULAR)
    r.SetMetricSamplingPercentage(0.20)
    if fixed_mask is not None:
        r.SetMetricFixedMask(fixed_mask)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Parameter-copy rather than sitk.Euler3DTransform(init): pass 2 hands in the
    # COMPOSITE Transform that Execute() returns, and the downcast constructor is not
    # reliable for that. When it raised, the caller's except-block silently reverted to
    # the coarse fit and the mandible-local refinement never ran at all — visible only
    # as a centroid shift of exactly 0.00 mm on all 42 cases.
    init_euler = sitk.Euler3DTransform()
    try:
        init_euler.SetFixedParameters(init.GetFixedParameters())
        init_euler.SetParameters(init.GetParameters())
    except Exception as e:
        raise RuntimeError(f"could not seed Euler3D from initial transform: {e!r}")
    r.SetInitialTransform(init_euler, inPlace=False)
    out = r.Execute(f, m)
    return out, float(r.GetMetricValue())


REG_SPACING = float(os.environ.get("HANSEG_REG_SPACING", "2.0"))


def _downsample(img: sitk.Image, spacing: float) -> sitk.Image:
    """Isotropic downsample used ONLY to drive registration.

    A HaN-Seg CT is 1024x1024x202 (~212 M voxels); running a multi-resolution
    Mattes-MI registration at native resolution would spend minutes per case for no
    benefit. A rigid transform is defined in PHYSICAL space, so it can be estimated on
    a 2 mm downsample and then applied to resample the FULL-resolution mask onto the
    full-resolution MR grid — the transform does not carry the grid it was fitted on.
    """
    sp = np.array(img.GetSpacing(), float)
    sz = np.array(img.GetSize(), int)
    dst = np.maximum(np.round(sz * sp / spacing).astype(int), 1)
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([spacing] * 3)
    rs.SetSize([int(v) for v in dst])
    rs.SetOutputDirection(img.GetDirection())
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return rs.Execute(img)


def _centroid_mm(mask: sitk.Image) -> np.ndarray:
    a = sitk.GetArrayFromImage(mask) > 0
    if a.sum() == 0:
        return np.array([np.nan] * 3)
    idx = np.array(np.nonzero(a)).mean(1)[::-1]          # x,y,z index
    return np.array(mask.TransformContinuousIndexToPhysicalPoint([float(v) for v in idx]))


def _resample_iso(img: sitk.Image, is_label: bool) -> sitk.Image:
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
        rs.SetInterpolator(sitk.sitkNearestNeighbor); rs.SetDefaultPixelValue(0)
    else:
        rs.SetInterpolator(sitk.sitkBSpline)
        rs.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return rs.Execute(img)


def prepare_one(case: str) -> dict:
    stage = TMPDIR / case
    try:
        base = f"HaN-Seg/set_1/{case}/{case}"
        ct = sitk.DICOMOrient(sitk.ReadImage(str(_extract(f"{base}_IMG_CT.nrrd", stage / "ct.nrrd"))), "RAS")
        mr = sitk.DICOMOrient(sitk.ReadImage(str(_extract(f"{base}_IMG_MR_T1.nrrd", stage / "mr.nrrd"))), "RAS")
        sg = sitk.DICOMOrient(sitk.ReadImage(str(_extract(f"{base}_OAR_Bone_Mandible.seg.nrrd", stage / "seg.nrrd"))), "RAS")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    seg = sitk.Cast(sg > 0, sitk.sitkUInt8)
    if sitk.GetArrayFromImage(seg).sum() == 0:
        raise RuntimeError("empty Bone_Mandible mask")

    # Registration is driven by 2 mm downsamples (see _downsample); the resulting
    # physical-space transform is then applied at full resolution.
    ct_lo, mr_lo = _downsample(ct, REG_SPACING), _downsample(mr, REG_SPACING)

    # Pass 1 — coarse whole-volume rigid. fixed=MR, moving=CT: the transform this
    # yields maps MR-grid points into CT space, which is exactly what resampling the
    # CT-space mask onto the MR grid requires.
    t_coarse, m_coarse = _rigid(mr_lo, ct_lo)
    seg_mr_coarse = sitk.Resample(seg, mr, t_coarse, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    c_coarse = _centroid_mm(seg_mr_coarse)

    # Pass 2 — refine with the metric sampled only around the mandible, so the mobile
    # mandible drives the fit rather than the cranium.
    seg_lo = sitk.Resample(seg, mr_lo, t_coarse, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    roi = sitk.BinaryDilate(seg_lo, [max(1, int(round(12.0 / s))) for s in mr_lo.GetSpacing()])
    t_fine, m_fine = (t_coarse, m_coarse)
    refine_status = "ok"
    if sitk.GetArrayFromImage(roi).sum() == 0:
        refine_status = "skipped: empty mandible ROI after coarse pass"
    else:
        try:
            t_fine, m_fine = _rigid(mr_lo, ct_lo, init=t_coarse, fixed_mask=roi)
        except Exception as e:
            # NOT silent: this refinement is the safeguard against jaw-position
            # mismatch, so any fallback to the coarse fit must appear in the audit.
            refine_status = f"FAILED, fell back to coarse: {e!r}"
    seg_mr = sitk.Resample(seg, mr, t_fine, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    c_fine = _centroid_mm(seg_mr)
    shift = float(np.linalg.norm(c_fine - c_coarse)) if np.all(np.isfinite(c_fine)) else float("nan")

    m_arr = sitk.GetArrayFromImage(seg_mr) > 0
    if m_arr.sum() == 0:
        raise RuntimeError("propagated mandible mask is empty")

    # QC: does the propagated mask sit on tissue, or has it slipped into air?
    mr_arr = sitk.GetArrayFromImage(sitk.Cast(mr, sitk.sitkFloat32))
    try:
        thr = float(sitk.GetArrayFromImage(sitk.OtsuThreshold(mr, 0, 1, 128)).astype(bool).sum() and
                    np.percentile(mr_arr[mr_arr > mr_arr.min()], 25))
    except Exception:
        thr = float(np.percentile(mr_arr, 25))
    tissue_frac = float((mr_arr[m_arr] > thr).mean())

    # Crop to the same mandible-centred, FOV-matched box the CT arm uses.
    idx = np.array(np.nonzero(m_arr))
    lo_v, hi_v = idx.min(1), idx.max(1) + 1
    centre_v = (lo_v + hi_v) / 2.0
    sp_zyx = np.array(mr.GetSpacing(), float)[::-1]
    box_mm = np.array([BOX_MM[2], BOX_MM[1], BOX_MM[0]])
    max_mm = np.array([MAX_BOX_MM[2], MAX_BOX_MM[1], MAX_BOX_MM[0]])
    need_mm = (hi_v - lo_v) * sp_zyx + 2 * MARGIN_MM
    use_mm = np.minimum(np.maximum(box_mm, need_mm), max_mm)
    box_v = use_mm / sp_zyx
    shape = np.array(m_arr.shape)
    start = np.clip(np.round(centre_v - box_v / 2).astype(int), 0, np.maximum(shape - 1, 0))
    end = np.clip(start + np.round(box_v).astype(int), 1, shape)
    start = np.clip(end - np.round(box_v).astype(int), 0, None)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    img_c, seg_c = mr_arr[sl], m_arr[sl].astype(np.uint8)
    kept = float(seg_c.sum()) / float(m_arr.sum())

    origin = mr.TransformContinuousIndexToPhysicalPoint(
        [float(start[2]), float(start[1]), float(start[0])])
    img_i = sitk.GetImageFromArray(img_c.astype(np.float32))
    seg_i = sitk.GetImageFromArray(seg_c)
    for o in (img_i, seg_i):
        o.SetSpacing(mr.GetSpacing()); o.SetDirection(mr.GetDirection()); o.SetOrigin(origin)
    img_r, seg_r = _resample_iso(img_i, False), _resample_iso(seg_i, True)

    sub = f"sub-{case.replace('_', '')}"
    a_dir = BIDS_ROOT / sub / "anat"; a_dir.mkdir(parents=True, exist_ok=True)
    d_dir = DERIV_ROOT / sub / "anat"; d_dir.mkdir(parents=True, exist_ok=True)
    img_p = a_dir / f"{sub}_T1w.nii.gz"
    seg_p = d_dir / f"{sub}_T1w_label-mandible_seg.nii.gz"
    sitk.WriteImage(img_r, str(img_p), useCompression=True)
    sitk.WriteImage(sitk.Cast(seg_r, sitk.sitkUInt8), str(seg_p), useCompression=True)

    return {"case": case, "sub": sub,
            "refine_status": refine_status,
            "final_metric": round(m_fine, 5), "coarse_metric": round(m_coarse, 5),
            "centroid_shift_mm": None if not np.isfinite(shift) else round(shift, 2),
            "tissue_frac": round(tissue_frac, 4),
            "qc_pass": bool(tissue_frac >= QC_TISSUE_FRAC),
            "crop_mm": [round(float(v * TARGET_SPACING), 1) for v in np.array(img_r.GetSize())],
            "gt_voxels_kept_frac": round(kept, 5),
            "orientation": "".join(nib.aff2axcodes(nib.load(str(img_p)).affine)),
            "label_orientation": "".join(nib.aff2axcodes(nib.load(str(seg_p)).affine))}


def main() -> None:
    with zipfile.ZipFile(HANSEG_ZIP) as z:
        cases = sorted({m.split("/")[2] for m in z.namelist()
                        if m.startswith("HaN-Seg/set_1/") and m.count("/") > 2 and m.split("/")[2]})
    print(f"{len(cases)} cases; registering CT->MR (mandible-local)", flush=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(prepare_one, c): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                failed.append({"case": futs[f], "error": repr(e)})
            if i % 5 == 0:
                print(f"  {i}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: r["case"])

    passed = [r for r in rows if r["qc_pass"]]
    excluded = [r["case"] for r in rows if not r["qc_pass"]]

    # nnU-Net test dirs — FLAT (cross-dataset predict layout), QC-PASSING CASES ONLY.
    img_ts, lab_ts = NNUNET_RAW / "imagesTs_mrt1", NNUNET_RAW / "labelsTs_mrt1"
    img_ts.mkdir(parents=True, exist_ok=True); lab_ts.mkdir(parents=True, exist_ok=True)
    for r in passed:
        sub = r["sub"]; cid = f"hanseg_{sub.removeprefix('sub-')}"
        for src, dst in ((BIDS_ROOT / sub / "anat" / f"{sub}_T1w.nii.gz", img_ts / f"{cid}_0000.nii.gz"),
                         (DERIV_ROOT / sub / "anat" / f"{sub}_T1w_label-mandible_seg.nii.gz",
                          lab_ts / f"{cid}.nii.gz")):
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    n_refined = sum(1 for r in rows if r.get("refine_status") == "ok")
    audit = {"n": len(rows), "n_failed": len(failed), "failed": failed,
             "n_mandible_local_refinement_ok": n_refined,
             "gt_provenance": "registration-propagated from CT (NOT natively drawn)",
             "qc_tissue_frac_threshold": QC_TISSUE_FRAC,
             "n_qc_pass": len(passed), "qc_excluded": excluded, "cases": rows}
    (BIDS_ROOT / "prepare_mr_audit.json").write_text(json.dumps(audit, indent=2))
    tf = [r["tissue_frac"] for r in rows]
    print(f"prepared {len(rows)}, failed {len(failed)}; QC pass {len(passed)}/{len(rows)}")
    print(f"  mandible-local refinement OK on {n_refined}/{len(rows)}")
    for b in sorted({r["refine_status"] for r in rows if r.get("refine_status") != "ok"})[:3]:
        print(f"  refinement issue: {b}", file=sys.stderr)
    if tf:
        print(f"  tissue_frac: min={min(tf):.3f} median={sorted(tf)[len(tf)//2]:.3f} max={max(tf):.3f}")
    if excluded:
        print(f"  QC-EXCLUDED (not written to test dirs): {excluded}", file=sys.stderr)
    if failed:
        print(f"  FAILED: {failed[:5]}", file=sys.stderr)


if __name__ == "__main__":
    main()
