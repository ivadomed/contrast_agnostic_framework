#!/usr/bin/env python3
"""
AUDIT: is the derived T2w-space lesion mask actually in the right place?

WHY THIS EXISTS
---------------
While deriving the FOV variants (01_00_derive_fov_variants.py) the T2w lesion mask
was found to sit ~95 mm away — in the OPPOSITE BREAST — from where the T1wce lesion
mask sits, for the natively-unilateral patients spot-checked. That is not a cropping
bug; it is a property of how the T2w mask was made.

Per 1_BIDS_ispy2/breast-ispy2/derivatives/labels/README.md, I-SPY2 ships exactly ONE
DICOM-SEG per patient, defined in the VOLSER-cropped DCE (T1wce) geometry. The
T2w-space mask is NOT a second annotation: it was produced by rigid (6-DOF Euler3D,
Mattes mutual information) registration of the T1wce volume onto the T2w volume,
followed by nearest-neighbour resampling.

That registration is unnecessary AND failure-prone here:
  * UNNECESSARY — the DCE and T2w series share the same DICOM FrameOfReferenceUID
    (verified directly on the staged raw DICOM at
    /scratch/paulh/ispy2_staging/raw_dicom/ISPY2-*/{dce_cropped,t2}/), and their
    ImagePositionPatient/ImageOrientationPatient values place them consistently in
    that one scanner frame. The BIDS affines faithfully reproduce those DICOM
    positions (checked to <0.5 mm). So the correct T1wce->T2w mapping is simply the
    IDENTITY in world space.
  * FAILURE-PRONE — for the 438 natively-unilateral patients the moving image is a
    half-FOV single-breast crop and the fixed image is a full-FOV bilateral scan.
    Mutual information has a strong secondary optimum on the CONTRALATERAL breast,
    so the registration can (and demonstrably does) land the mask on the wrong side.

This script measures the disagreement for every patient, using the shared scanner
frame as ground truth: it maps the T1wce mask's world centroid into the T2w volume
and compares it with the T2w mask's own world centroid.

METRICS PER PATIENT
  d_centroid_mm     distance between the T1wce-mask and T2w-mask world centroids
  same_side         do both masks fall on the same side of the patient midline
                    (world x sign, RAS: x<0 = patient RIGHT)?
  dice_shared_frame Dice between the T2w mask and the T1wce mask resampled into T2w
                    space by the IDENTITY world transform (nearest neighbour)
  t1_in_t2_fov      fraction of the identity-resampled T1wce mask that lands inside
                    the T2w grid at all

A correct derived mask should have d_centroid_mm of a few mm (residual inter-sequence
patient motion) and a clearly non-zero Dice. A wrong-breast registration shows up as
d_centroid_mm ~ 60-150 mm, same_side False, and Dice 0.

Writes 4_splits_ispy2/t2w_label_registration_audit.{json,md}. Read-only w.r.t. all
image and label data — it changes nothing.

Run:  bash 01_create_splits/01_00_audit_t2w_label_registration.sh
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


def resample_identity(src: nib.Nifti1Image, ref: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbour resample a binary mask onto ref's grid using the IDENTITY
    world transform (the two volumes share a DICOM frame of reference)."""
    ref_shape = ref.shape
    M = np.linalg.inv(np.asarray(src.affine)) @ np.asarray(ref.affine)
    g = np.indices(ref_shape).reshape(3, -1)
    idx = (M[:3, :3] @ g + M[:3, 3:4])
    idx = np.rint(idx).astype(np.int64)
    ok = np.ones(idx.shape[1], dtype=bool)
    for k in range(3):
        ok &= (idx[k] >= 0) & (idx[k] < src.shape[k])
    src_arr = np.asanyarray(src.dataobj) > 0
    out = np.zeros(idx.shape[1], dtype=bool)
    out[ok] = src_arr[idx[0, ok], idx[1, ok], idx[2, ok]]
    return out.reshape(ref_shape)


def world_centroid(mask: np.ndarray, aff: np.ndarray):
    idx = np.argwhere(mask)
    if not len(idx):
        return None
    c = idx.mean(0)
    return (aff[:3, :3] @ c + aff[:3, 3])


def _one(args) -> dict:
    sub, fov = args
    try:
        t2 = nib.load(str(BIDS_ROOT / sub / "anat" / f"{sub}_T2w.nii.gz"))
        l1 = nib.load(str(DERIV_ROOT / sub / "anat" / f"{sub}_acq-firstpost_T1w_label-lesion_seg.nii.gz"))
        l2 = nib.load(str(DERIV_ROOT / sub / "anat" / f"{sub}_T2w_label-lesion_seg.nii.gz"))
        m1 = np.asanyarray(l1.dataobj) > 0
        m2 = np.asanyarray(l2.dataobj) > 0
        r = {"sub": sub, "fov": fov, "n1": int(m1.sum()), "n2": int(m2.sum())}
        c1 = world_centroid(m1, np.asarray(l1.affine))
        c2 = world_centroid(m2, np.asarray(l2.affine))
        if c1 is None or c2 is None:
            r["note"] = "empty mask"; return r
        r["c1_world"] = [round(float(v), 1) for v in c1]
        r["c2_world"] = [round(float(v), 1) for v in c2]
        r["d_centroid_mm"] = round(float(np.linalg.norm(c1 - c2)), 1)
        r["same_side"] = bool(np.sign(c1[0]) == np.sign(c2[0]))
        m1_in_t2 = resample_identity(l1, t2)
        r["t1_in_t2_fov"] = round(float(m1_in_t2.sum() / max(1, m1.sum())), 4)
        inter = int((m1_in_t2 & m2).sum())
        r["dice_shared_frame"] = round(2 * inter / max(1, int(m1_in_t2.sum()) + int(m2.sum())), 4)
        r["n1_resampled"] = int(m1_in_t2.sum())
        return r
    except Exception as exc:                      # noqa: BLE001
        return {"sub": sub, "fov": fov, "failed": repr(exc)}


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
    (SPLITS / "t2w_label_registration_audit.json").write_text(json.dumps(recs, indent=1))

    ok = [r for r in recs if "d_centroid_mm" in r]
    L = ["# ispy2 — audit of the registration-derived T2w lesion masks\n",
         "Ground truth for this audit is the shared DICOM FrameOfReferenceUID: the DCE",
         "and T2w series of a patient are in ONE scanner frame, so the T1wce mask mapped",
         "into T2w space by the IDENTITY world transform is where the lesion really is.\n",
         f"Patients audited: {len(ok)} (of {len(recs)}; {len(recs)-len(ok)} skipped/empty/failed)\n"]
    for grp in ("uni", "bil"):
        g = [r for r in ok if r["fov"] == grp]
        if not g:
            continue
        d = np.array([r["d_centroid_mm"] for r in g])
        dice = np.array([r["dice_shared_frame"] for r in g])
        wrong_side = [r for r in g if not r["same_side"]]
        zero_dice = [r for r in g if r["dice_shared_frame"] == 0.0]
        far = [r for r in g if r["d_centroid_mm"] > 20]
        L += [f"## Natively-{'unilateral' if grp=='uni' else 'bilateral'} patients (n={len(g)})\n",
              f"- centroid distance T1wce-mask vs T2w-mask (mm): "
              f"median **{np.median(d):.1f}**, p90 {np.percentile(d,90):.1f}, max {d.max():.1f}",
              f"- Dice(T2w mask, identity-resampled T1wce mask): median **{np.median(dice):.3f}**, "
              f"mean {dice.mean():.3f}",
              f"- masks on OPPOSITE sides of the midline: **{len(wrong_side)}/{len(g)}** "
              f"({100*len(wrong_side)/len(g):.1f}%)",
              f"- Dice exactly 0 (no overlap at all): **{len(zero_dice)}/{len(g)}** "
              f"({100*len(zero_dice)/len(g):.1f}%)",
              f"- centroid distance > 20 mm: **{len(far)}/{len(g)}** "
              f"({100*len(far)/len(g):.1f}%)\n"]
    (SPLITS / "t2w_label_registration_audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
