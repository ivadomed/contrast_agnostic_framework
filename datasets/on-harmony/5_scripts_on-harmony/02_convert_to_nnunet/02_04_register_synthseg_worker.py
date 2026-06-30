#!/usr/bin/env python3
"""
Register T1w SynthSeg masks into the native space of other modalities using
ANTs rigid registration (command-line, not antspyx).

Registration direction: fixed = target modality, moving = T1w.
The 0GenericAffine.mat produced maps T1w → target, so applying it to the
T1w synthseg mask resamples it into the target modality's native space.

No class remapping — output keeps the original 32-label SynthSeg format,
matching derivatives/synthseg_masks/*/anat/*_T1w_synthseg.nii.gz.

Expects antsRegistration and antsApplyTransforms on PATH (load the ants
module before launching this script).

Usage (via 02_04_register_synthseg_masks.sh):
    python register_synthseg_worker.py \\
        --modality T2w --rank 0 --world-size 4 \\
        --bids-root /path/to/1_BIDS_on-harmony
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Modality descriptors
#   subdir  : BIDS subdir under subject/session/
#   glob    : filename glob (may match multiple files — bold has mb4/mb6/plain)
#   vol_fn  : how to collapse a 4D array to a 3D reference (identity if 3D)
# ---------------------------------------------------------------------------
MODALITIES = {
    "T2w": {
        "subdir": "anat",
        "glob":   "*_T2w.nii.gz",
        "vol_fn": lambda d: d[..., 0] if d.ndim == 4 else d,
    },
    "dwi": {
        "subdir": "dwi",
        "glob":   "*_dir-AP_dwi.nii.gz",
        "vol_fn": lambda d: d[..., 0] if d.ndim == 4 else d,  # first vol ≈ b0
    },
    "epi": {
        "subdir": "fmap",
        "glob":   "*_dir-AP_epi.nii.gz",
        "vol_fn": lambda d: d[..., 0] if d.ndim == 4 else d,
    },
    "bold": {
        "subdir": "func",
        "glob":   "*_task-rest*bold.nii.gz",   # matches mb4, mb6, and plain variants
        "vol_fn": lambda d: d.mean(-1).astype(np.float32) if d.ndim == 4 else d,
    },
    "GRE": {
        "subdir": "swi",
        "glob":   "*_echo-1_part-mag_GRE.nii.gz",
        "vol_fn": lambda d: d[..., 0] if d.ndim == 4 else d,
    },
}


def check_ants():
    if not shutil.which("antsRegistration"):
        print(
            "ERROR: antsRegistration not found on PATH.\n"
            "       Make sure you ran: module load ants/2.6.5",
            flush=True,
        )
        sys.exit(1)


def extract_reference(img_path: Path, vol_fn, tmp_dir: Path) -> Path:
    """Return a 3D reference image path, extracting from 4D if needed."""
    img = nib.load(img_path)
    data = img.get_fdata(dtype=np.float32)
    vol3d = vol_fn(data)
    if vol3d.ndim != 3:
        raise RuntimeError(f"vol_fn produced shape {vol3d.shape} for {img_path.name}")
    if data.ndim == 3:
        return img_path  # no copy needed
    out = tmp_dir / "ref.nii.gz"
    nib.save(nib.Nifti1Image(vol3d, img.affine), out)
    return out


def ants_register(fixed: Path, moving: Path, tmp_dir: Path) -> Path:
    """
    Rigid registration. Fixed = target modality (native space we want).
    Moving = T1w. Returns path to the 0GenericAffine.mat transform.
    """
    prefix = str(tmp_dir / "reg_")
    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{prefix}]",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--use-histogram-matching", "0",
        "--initial-moving-transform", f"[{fixed},{moving},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{fixed},{moving},1,32,Regular,0.25]",
        "--convergence", "[500x250x100,1e-6,10]",
        "--shrink-factors", "4x2x1",
        "--smoothing-sigmas", "2x1x0vox",
        "--verbose", "0",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"antsRegistration failed (rc={r.returncode}):\n"
            f"stderr: {r.stderr[-3000:]}"
        )
    mat = Path(prefix + "0GenericAffine.mat")
    if not mat.exists():
        raise RuntimeError(f"Expected transform not produced: {mat}")
    return mat


def ants_apply(mask: Path, reference: Path, transform: Path, output: Path) -> None:
    """Warp mask into reference space with nearest-neighbour interpolation."""
    cmd = [
        "antsApplyTransforms",
        "--dimensionality", "3",
        "--input", str(mask),
        "--reference-image", str(reference),
        "--output", str(output),
        "--interpolation", "NearestNeighbor",
        "--transform", str(transform),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms failed (rc={r.returncode}):\n{r.stderr[-2000:]}"
        )


def process_session(
    bids_root: Path, sub: str, ses: str, modality: str
) -> tuple[int, int]:
    """
    Register T1w synthseg mask to all modality images found in this session.
    Returns (n_done, n_failed).  n_done includes pre-existing outputs.
    """
    cfg = MODALITIES[modality]
    ses_dir = bids_root / sub / ses
    mask_root = bids_root / "derivatives" / "synthseg_masks"

    t1w_img = ses_dir / "anat" / f"{sub}_{ses}_T1w.nii.gz"
    t1w_mask = mask_root / sub / ses / "anat" / f"{sub}_{ses}_T1w_synthseg.nii.gz"

    if not t1w_img.exists():
        print(f"  SKIP {sub}/{ses}: T1w image missing", flush=True)
        return 0, 0
    if not t1w_mask.exists():
        print(f"  SKIP {sub}/{ses}: T1w synthseg mask missing", flush=True)
        return 0, 0

    target_dir = ses_dir / cfg["subdir"]
    targets = sorted(target_dir.glob(cfg["glob"])) if target_dir.exists() else []
    if not targets:
        print(f"  SKIP {sub}/{ses}/{modality}: no matching images", flush=True)
        return 0, 0

    n_done, n_failed = 0, 0
    for target_img in targets:
        out_name = target_img.name.replace(".nii.gz", "_synthseg.nii.gz")
        out_dir = mask_root / sub / ses / cfg["subdir"]
        out_path = out_dir / out_name

        if out_path.exists():
            print(f"  EXISTS {sub}/{ses}/{out_name}", flush=True)
            n_done += 1
            continue

        print(f"  REG   {sub}/{ses}/{target_img.name}", flush=True)
        tmp = Path(tempfile.mkdtemp(prefix="reg_synthseg_"))
        try:
            fixed = extract_reference(target_img, cfg["vol_fn"], tmp)
            transform = ants_register(fixed, t1w_img, tmp)
            out_dir.mkdir(parents=True, exist_ok=True)
            ants_apply(t1w_mask, fixed, transform, out_path)
            print(f"  OK    → {out_path.relative_to(bids_root)}", flush=True)
            n_done += 1
        except Exception as exc:
            print(f"  FAIL  {sub}/{ses}/{target_img.name}: {exc}", flush=True)
            n_failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    return n_done, n_failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality",   required=True, choices=list(MODALITIES))
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--bids-root",  required=True, type=Path)
    args = p.parse_args()

    check_ants()

    mask_root = args.bids_root / "derivatives" / "synthseg_masks"
    sessions = sorted(
        (sub_dir.name, ses_dir.name)
        for sub_dir in sorted(mask_root.iterdir())
        if sub_dir.is_dir() and sub_dir.name.startswith("sub-")
        for ses_dir in sorted(sub_dir.iterdir())
        if ses_dir.is_dir() and ses_dir.name.startswith("ses-")
    )
    my_sessions = sessions[args.rank :: args.world_size]
    print(
        f"[rank {args.rank}/{args.world_size}] modality={args.modality} "
        f"sessions={len(my_sessions)}/{len(sessions)}",
        flush=True,
    )

    total_done, total_failed = 0, 0
    for sub, ses in my_sessions:
        done, failed = process_session(args.bids_root, sub, ses, args.modality)
        total_done += done
        total_failed += failed

    print(
        f"[rank {args.rank}] DONE — {total_done} registered, {total_failed} failed",
        flush=True,
    )
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()
