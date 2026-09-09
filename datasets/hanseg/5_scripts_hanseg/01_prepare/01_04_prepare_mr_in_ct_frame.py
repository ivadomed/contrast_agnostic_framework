#!/usr/bin/env python3
"""
HaN-Seg MR-T1 resampled INTO THE CT FRAME — a second OOD modality that reuses the
CT arm's ground truth untouched.

THE DESIGN, AND WHY IT IS DEFENSIBLE
------------------------------------
An earlier attempt (01_02_prepare_mr.py, now disabled) transported the LABELS into MR
space. That created a new ground-truth artifact of our own making, and its QC gate
turned out to be anti-correlated with accuracy. This script does the opposite, and it
is what the HaN-Seg challenge participants actually did:

    register MR -> CT, resample the MR IMAGE into CT space,
    and evaluate against the ORIGINAL, HUMAN-DRAWN CT LABELS.

Consequences, all of them good:
  * The ground truth is never modified. `labelsTs_ct` is byte-identical for both the
    CT arm and this MR arm, so the two columns are scored against literally the same
    files in the same frame with the same FOV crop. The ONLY difference between them
    is the image modality — a properly controlled cross-modality comparison.
  * It matches the dataset's documented intended use. The HaN-Seg challenge released
    CT and MR deliberately NON-registered ("simulated a real-world clinical scenario",
    challenge report), and 3 of the 5 reporting teams registered MR->CT themselves:
    the winner (eli1) with rigid SimpleElastix, CHB-QuantIF with rigid ANTsPy, and
    Mamaa with a pure translation. The organisers likewise registered MR to CT during
    annotation ("each MR image was first registered to the CT image of the same
    patient, and then OARs were annotated in the reference coordinate system of the
    CT image" — dataset paper §2.2).
  * The CT arm needs no recomputation. The crop written by 01_01_prepare_ct.py
    preserves world coordinates, so the cropped CT shares the ORIGINAL CT's frame and
    a CT->MR transform applies to it directly. We resample the MR straight onto the
    existing cropped-CT grid; existing CT predictions and metrics stay valid.

WHY THE TRANSFORM IS NEARLY TRIVIAL HERE (measured, not assumed)
----------------------------------------------------------------
Probing 5 cases: rotations 0.25-3.27 deg (mostly <1.6), and the y-translation is
near-CONSTANT at ~99-105 mm across every case while z varies wildly (-597 to +165).
That signature is a coordinate-origin convention difference plus per-scan DICOM table
position — not patient movement. These are RT-planning scans in an immobilization
mask, so the anatomy really is in the same pose. This is also why one challenge team
succeeded with a translation alone.

DEVIATION FROM THE WINNER'S EXACT TOOLING, STATED PLAINLY
---------------------------------------------------------
They used SimpleElastix. Neither `itk-elastix` nor `SimpleITK-SimpleElastix` exists in
the Alliance wheelhouse, and pulling ITK from PyPI into this project's shared venv is
a documented fragility (CLAUDE.md: the venv has already been rebuilt once with
permanent losses). We therefore use SimpleITK's own rigid registration with Mattes
mutual information — the SAME rigid+MI method class elastix implements. Given the
measured transform is a near-pure translation with sub-degree-to-3-degree rotation,
the optimizer implementation is not the load-bearing part. REGULAR (deterministic)
sampling is used so the result is reproducible run to run.

THE ONE LIMITATION THAT SURVIVES, AND IT MUST BE REPORTED
----------------------------------------------------------
Residual registration error means a perfect model cannot score 100% on this arm: the
resampled MR content sits slightly off the GT. That is unavoidable and symmetric — you
cannot escape it by choosing which volume to move — and it is the situation every
challenge participant was in. Crucially it does NOT bias the between-method
comparison: every method sees the same resampled images and the same GT.

Outputs: <nnUNet_raw>/imagesTs_mrt1/hanseg_<id>_0000.nii.gz   (NO labels — labelsTs_ct is reused)
         <BIDS_ROOT>/prepare_mr_ctframe_audit.json

Run via 01_04_prepare_mr_in_ct_frame.sh
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HANSEG_ZIP = Path(os.environ["HANSEG_ZIP"])
BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
N_WORKERS = int(os.environ.get("HANSEG_WORKERS", "8"))
TMPDIR = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / "hanseg_mrct"
REG_SPACING = float(os.environ.get("HANSEG_REG_SPACING", "2.0"))
MAX_ROT_DEG = float(os.environ.get("HANSEG_MAX_ROT_DEG", "10.0"))

_ZIP: zipfile.ZipFile | None = None


def _zip() -> zipfile.ZipFile:
    global _ZIP
    if _ZIP is None:
        _ZIP = zipfile.ZipFile(HANSEG_ZIP)
    return _ZIP


def _extract(member: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _zip().open(member) as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, 8 << 20)
    return dst


def _downsample(img: sitk.Image, spacing: float) -> sitk.Image:
    """Isotropic downsample used ONLY to drive registration; a rigid transform is
    defined in physical space so it applies unchanged at full resolution."""
    sp = np.array(img.GetSpacing(), float)
    sz = np.array(img.GetSize(), int)
    dst = np.maximum(np.round(sz * sp / spacing).astype(int), 1)
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([spacing] * 3)
    r.SetSize([int(v) for v in dst])
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(sitk.sitkLinear)
    r.SetDefaultPixelValue(float(sitk.GetArrayViewFromImage(img).min()))
    return r.Execute(img)


def register_ct_to_mr(ct: sitk.Image, mr: sitk.Image) -> tuple[sitk.Transform, float]:
    """Rigid Mattes-MI. fixed=CT, moving=MR -> transform maps CT-space points into
    MR space, which is exactly what sitk.Resample needs to pull the MR onto a
    CT-frame grid."""
    f = sitk.Cast(_downsample(ct, REG_SPACING), sitk.sitkFloat32)
    m = sitk.Cast(_downsample(mr, REG_SPACING), sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(
        f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetMetricSamplingStrategy(r.REGULAR)          # deterministic; see module docstring
    r.SetMetricSamplingPercentage(0.20)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    e = sitk.Euler3DTransform()
    e.SetFixedParameters(init.GetFixedParameters())
    e.SetParameters(init.GetParameters())
    r.SetInitialTransform(e, inPlace=False)
    out = r.Execute(f, m)
    return out, float(r.GetMetricValue())


def prepare_one(case: str) -> dict:
    cid = f"hanseg_sub-{case.replace('_', '')}".replace("sub-", "")
    ref_p = NNUNET_RAW / "imagesTs_ct" / f"hanseg_{case.replace('_','')}_0000.nii.gz"
    if not ref_p.exists():
        raise FileNotFoundError(f"reference cropped CT missing: {ref_p}")
    ref = sitk.ReadImage(str(ref_p))          # the EXISTING cropped CT grid — GT frame

    stage = TMPDIR / case
    try:
        base = f"HaN-Seg/set_1/{case}/{case}"
        ct = sitk.DICOMOrient(sitk.ReadImage(str(_extract(f"{base}_IMG_CT.nrrd", stage / "ct.nrrd"))), "RAS")
        mr = sitk.DICOMOrient(sitk.ReadImage(str(_extract(f"{base}_IMG_MR_T1.nrrd", stage / "mr.nrrd"))), "RAS")
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    T, metric = register_ct_to_mr(ct, mr)
    p = np.array(T.GetParameters())
    rot_deg = np.degrees(p[:3])
    trans = p[3:]

    # Pull the MR onto the cropped-CT grid. Background = MR minimum (air), not 0.
    mr_in_ct = sitk.Resample(mr, ref, T, sitk.sitkBSpline,
                             float(sitk.GetArrayViewFromImage(mr).min()), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(mr_in_ct)
    # A registration that lands entirely outside the MR FOV yields a constant volume.
    frac_bg = float((arr <= arr.min() + 1e-6).mean())

    out_dir = NNUNET_RAW / "imagesTs_mrt1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"hanseg_{case.replace('_','')}_0000.nii.gz"
    sitk.WriteImage(mr_in_ct, str(out_p), useCompression=True)

    ok = bool(np.abs(rot_deg).max() <= MAX_ROT_DEG and frac_bg < 0.60)
    return {"case": case, "out": out_p.name,
            "rot_deg": [round(float(v), 3) for v in rot_deg],
            "translation_mm": [round(float(v), 2) for v in trans],
            "max_rot_deg": round(float(np.abs(rot_deg).max()), 3),
            "final_metric": round(metric, 5),
            "frac_background": round(frac_bg, 4),
            "shape": list(mr_in_ct.GetSize()),
            "spacing": [round(float(s), 3) for s in mr_in_ct.GetSpacing()],
            "qc_pass": ok}


def main() -> None:
    with zipfile.ZipFile(HANSEG_ZIP) as z:
        cases = sorted({m.split("/")[2] for m in z.namelist()
                        if m.startswith("HaN-Seg/set_1/") and m.count("/") > 2 and m.split("/")[2]})
    print(f"{len(cases)} cases; rigid MR->CT, resampling MR onto the existing cropped-CT grid",
          flush=True)

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

    bad = [r["case"] for r in rows if not r["qc_pass"]]
    rot = [r["max_rot_deg"] for r in rows]
    ty = [r["translation_mm"][1] for r in rows]
    audit = {"n": len(rows), "n_failed": len(failed), "failed": failed,
             "gt_provenance": "UNCHANGED — evaluated against the original labelsTs_ct; "
                              "only the MR IMAGE was resampled into the CT frame",
             "registration": "rigid, Mattes MI, SimpleITK (SimpleElastix unavailable in "
                             "the Alliance wheelhouse; same rigid+MI class)",
             "qc_failed": bad, "cases": rows}
    (BIDS_ROOT / "prepare_mr_ctframe_audit.json").write_text(json.dumps(audit, indent=2))

    print(f"prepared {len(rows)}, failed {len(failed)}, QC-failed {len(bad)}")
    if rot:
        print(f"  max rotation deg: min={min(rot):.2f} median={sorted(rot)[len(rot)//2]:.2f} max={max(rot):.2f}")
        print(f"  y-translation mm: min={min(ty):.1f} median={sorted(ty)[len(ty)//2]:.1f} max={max(ty):.1f}"
              f"   (near-constant => coordinate-convention offset, not patient motion)")
    if bad:
        print(f"  QC-FAILED (excluded): {bad}", file=sys.stderr)
        for b in bad:
            r = next(x for x in rows if x["case"] == b)
            (NNUNET_RAW / "imagesTs_mrt1" / r["out"]).unlink(missing_ok=True)
    if failed:
        print(f"  FAILED: {failed[:5]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
