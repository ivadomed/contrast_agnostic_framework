#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the TRUSTED evaluation set (CT + US).

TRUSTED has TWO test modalities. The chaos models are single-channel, so each
volume is fed as channel _0000. This materialises image dirs + matching GT label
dirs (one pair per modality) so 06_evaluate can score predictions.

US is RESAMPLED to the model's resolution (see WHY below); CT is left native.

WHY US is resampled (eval at model resolution — decision 2026-06-29)
--------------------------------------------------------------------
US ships at 0.3 mm isotropic (~570 M voxels). The chaos MR models work at
~1.6 mm in-plane / 5.5–7.8 mm through-plane (see chaos nnUNetPlans target_spacing),
so nnU-Net downsamples the US ~5× for inference — the prediction genuinely has no
detail finer than ~1.6 mm. If we predicted at native 0.3 mm, nnU-Net would then
resample the 5-class softmax BACK up to 570 M voxels (~11 GB/case) purely to match
the stored GT grid — heavy, and it adds no real information.

Instead we score at the model's resolution: resample the US IMAGE (linear) and its
GT MASK (nearest) to US_EVAL_SPACING before prediction. nnU-Net's output then lands
on that small grid and the GT already matches — trivial memory, and Dice/HD95 are
computed at the resolution the model actually operates at. CT (~0.74 mm, already
close to model res, ~210 M-voxel output is manageable) stays native and is hardlinked.

US_EVAL_SPACING = 1.5 mm isotropic — ≈ the chaos in-plane resolution (1.58–1.70 mm),
chosen isotropic (US has no meaningful slice axis) and a hair finer so we don't
discard in-plane detail the model can resolve. One grid for both contrasts → a
single labelsTs_us serves t1in and t2spir. Tunable here.

Reads:  ../../1_BIDS_trusted/trusted-kidney/sub-<id>/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_trusted/raw/imagesTs_ct/{case}_0000.nii.gz   labelsTs_ct/{case}.nii.gz   (native hardlink)
                                  /imagesTs_us/{case}_0000.nii.gz   labelsTs_us/{case}.nii.gz   (resampled to 1.5mm)

Case ids (match the original TRUSTED names so GT/pred line up):
  CT: "<id>"        (e.g. 220)          US: "<id><R|L>"   (e.g. 263R, 263L)

Heavy for US (reads/resamples ~570 M-voxel volumes) → dispatch via
05_00_build_test_inputs.sh (run_job). Idempotent: re-run overwrites US, skips CT
links that already exist.

    bash 05_00_build_test_inputs.sh        # (preferred — runs on a compute node)
    python 05_00_build_test_inputs.py      # direct (CT-only is light; US is heavy)
"""
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Parallel workers for US resampling (independent per volume). Backfill-friendly:
# more workers → shorter wall time → fits smaller scheduler gaps. From Slurm, else 4.
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("BUILD_WORKERS", "4")))

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_trusted" / "trusted-kidney"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_trusted" / "raw"

# Eval resolution for US (mm, isotropic). See module docstring for rationale.
US_EVAL_SPACING = (1.5, 1.5, 1.5)


def link_or_copy(src: Path, dst: Path) -> None:
    """CT: native, lossless — hardlink (fall back to copy across filesystems)."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _resample(img: sitk.Image, spacing, is_label: bool) -> sitk.Image:
    """Resample to isotropic `spacing` (mm). Nearest for labels, B-spline for images."""
    in_sp = np.array(img.GetSpacing(), float)
    in_sz = np.array(img.GetSize(), int)
    out_sp = np.array(spacing, float)
    out_sz = np.maximum(1, np.round(in_sz * in_sp / out_sp).astype(int)).tolist()
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing([float(s) for s in out_sp])
    rs.SetSize([int(s) for s in out_sz])
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    # Linear for images (downsampling ~5×: fast and near-equivalent to B-spline here,
    # which cost ~50 s/volume on the 568 M-voxel input → timeouts); nearest for labels.
    rs.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    rs.SetDefaultPixelValue(0)
    return rs.Execute(img)


def resample_to(src: Path, dst: Path, spacing, is_label: bool) -> None:
    """Resample src → dst at `spacing`. Overwrites (US is re-derived, not pristine).

    IMPORTANT: unlink dst first. A previous (native) build may have HARDLINKED dst to
    the pristine BIDS/0_raw volume; writing through that path in place would corrupt
    0_raw. Unlinking drops only this directory entry (raw inode survives) so WriteImage
    creates a fresh, independent file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    out = _resample(sitk.ReadImage(str(src)), spacing, is_label)
    if is_label:
        out = sitk.Cast(out, sitk.sitkUInt8)
    sitk.WriteImage(out, str(dst))


def _resample_case(args) -> str:
    """Top-level (picklable) worker: resample one US case's image + mask. Returns ''
    on success or the case id if its source files are missing."""
    case, img, seg, img_dst, lab_dst, spacing = args
    if not Path(img).exists() or not Path(seg).exists():
        return case
    resample_to(Path(img), Path(img_dst), spacing, is_label=False)
    resample_to(Path(seg), Path(lab_dst), spacing, is_label=True)
    return ""


def build_modality(items, modality: str, resample_spacing=None) -> None:
    """items = list of (case_id, image_path, seg_path).

    resample_spacing=None → native hardlink (CT, serial/instant); else resample
    image+mask (US) in parallel across WORKERS processes (each volume independent)."""
    img_dir = NNUNET_RAW / f"imagesTs_{modality}"
    lab_dir = NNUNET_RAW / f"labelsTs_{modality}"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    if resample_spacing is None:                       # CT: hardlink, instant
        n_ok = 0
        for case, img, seg in items:
            if not img.exists() or not seg.exists():
                missing.append(case); continue
            link_or_copy(img, img_dir / f"{case}_0000.nii.gz")
            link_or_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1
        extra = " (native hardlink)"
    else:                                              # US: resample in parallel
        work = [(case, str(img), str(seg),
                 str(img_dir / f"{case}_0000.nii.gz"), str(lab_dir / f"{case}.nii.gz"),
                 resample_spacing) for case, img, seg in items]
        print(f"    {modality}: resampling {len(work)} volumes with {WORKERS} workers …")
        done = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for miss in pool.map(_resample_case, work):
                if miss:
                    missing.append(miss)
                else:
                    done += 1
                    if done % 10 == 0:
                        print(f"      {done}/{len(work)} resampled")
        n_ok = done
        extra = f" @ {resample_spacing[0]}mm iso ({WORKERS} workers)"

    status = f"{n_ok}/{len(items)} → {img_dir.name}{extra} (+labels)"
    if missing:
        status += f"  [MISSING {len(missing)}: {missing[:5]}]"
    print(f"  {modality:3s}: {status}")


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    if not subs:
        raise SystemExit(f"No sub-* under {BIDS_ROOT} — run 00_00_ingest_and_bidsify.py first.")

    ct_items, us_items = [], []
    for sub in subs:
        pid = sub[len("sub-"):]
        anat = BIDS_ROOT / sub / "anat"
        deriv = DERIV_DIR / sub / "anat"
        ct_img = anat / f"{sub}_CT.nii.gz"
        if ct_img.exists():
            ct_items.append((pid, ct_img, deriv / f"{sub}_CT_dseg.nii.gz"))
        for side in ("R", "L"):
            us_img = anat / f"{sub}_acq-{side}_US.nii.gz"
            if us_img.exists():
                us_items.append((f"{pid}{side}", us_img,
                                 deriv / f"{sub}_acq-{side}_US_dseg.nii.gz"))

    build_modality(ct_items, "ct")                                   # native
    build_modality(us_items, "us", resample_spacing=US_EVAL_SPACING)  # → model resolution


if __name__ == "__main__":
    main()
