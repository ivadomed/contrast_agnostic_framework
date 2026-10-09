"""
Lesion-side unilateral (single-breast) crop -- the breast task's standard eval
FOV since 2026-09-17 (the project notes "Breast task" section: unilateral-crop is the
only reported breast evaluation).

Same rule duke-breast-mri's 02_nnunet/02_03_derive_unilateral_crop.py applies
(that script predates this helper and keeps its own copy): on an LPS-oriented
volume, keep the L-R half (axis 0) containing the lesion centroid; axes 1/2
untouched; a lesion whose axis-0 extent crosses into the other half is NOT
force-cropped -- the case is reported as midline-crossing and the caller
excludes it. Used by ispy1 and acrin6698's nnU-Net conversions.
"""
from __future__ import annotations

import nibabel as nib
import numpy as np


def crop_axis0(img: nib.Nifti1Image, lo: int, hi: int) -> nib.Nifti1Image:
    arr = np.asanyarray(img.dataobj)
    aff = np.asarray(img.affine).copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([lo, 0, 0], dtype=float)
    out = nib.Nifti1Image(np.ascontiguousarray(arr[lo:hi]), aff, header=img.header)
    out.set_qform(aff)
    out.set_sform(aff)
    return out


def lesion_side_half(mask: np.ndarray) -> dict:
    """mask: LPS-oriented boolean array. Returns {'side','lo','hi','lesion_axis0',
    'midline_crossing'} (side by lesion centroid; LPS low index = patient right)."""
    n0 = mask.shape[0]
    mid = n0 // 2
    nz = np.flatnonzero(mask.any(axis=(1, 2)))
    lo_l, hi_l = int(nz[0]), int(nz[-1]) + 1
    centroid0 = float(np.argwhere(mask)[:, 0].mean())
    side = "right" if centroid0 < mid else "left"
    lo, hi = (0, mid) if side == "right" else (mid, n0)
    return {"side": side, "lo": lo, "hi": hi, "lesion_axis0": [lo_l, hi_l],
            "midline_crossing": bool(lo_l < lo or hi_l > hi)}


# ── Anterior-posterior (axis 1) crop anchored on the anterior skin (added 2026-10-01) ──
# I-SPY2's unilateral TRAINING volumes are cropped in-plane on BOTH axes (VOLSER),
# keeping only the breast + a little chest wall, while full-chest acquisitions
# (duke-breast-mri, acrin6698 DWI) extend ~330-350 mm A-P through the whole thorax.
# Measured 2026-10-01 over I-SPY2's 476 unilateral training cases with
# anterior_skin_index() below: A-P FOV median 174.5 mm, of which skin->posterior edge
# median 147.4 mm (=> ~27 mm of air in front of the skin). The window reproduces that
# geometry from each case's OWN skin line (anatomy-anchored, not I-SPY2's index window,
# which is calibrated to its own reconstruction grid) and is never allowed to clip the
# lesion (extended posteriorly instead; recorded).
AP_ANTERIOR_AIR_MM = 27.0
AP_DEPTH_FROM_SKIN_MM = 147.4
AP_LESION_MARGIN_MM = 5.0


def anterior_skin_index(arr: np.ndarray, frac: float = 0.05) -> int:
    """First axis-1 (A->P on LPS) slab where >`frac` of voxels are tissue.
    Tissue = above p30 + 10% of the (p30..p99) range -- the same detector used for
    the I-SPY2 training measurement, so test and training windows are comparable."""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 30), np.percentile(a, 99)
    body = a > lo + 0.1 * (hi - lo)
    prof = body.mean(axis=(0, 2))
    idx = np.flatnonzero(prof > frac)
    return int(idx[0]) if idx.size else 0


def ap_skin_window(ref_arr: np.ndarray, mask: np.ndarray, ap_spacing_mm: float) -> dict:
    """{'lo','hi','skin','extended_for_lesion','lesion_axis1'} on axis 1 (LPS: low=anterior)."""
    n1 = ref_arr.shape[1]
    skin = anterior_skin_index(ref_arr)
    lo = max(0, skin - int(round(AP_ANTERIOR_AIR_MM / ap_spacing_mm)))
    hi = min(n1, skin + int(round(AP_DEPTH_FROM_SKIN_MM / ap_spacing_mm)))
    nz = np.flatnonzero(mask.any(axis=(0, 2)))
    l_lo, l_hi = int(nz[0]), int(nz[-1]) + 1
    m = int(round(AP_LESION_MARGIN_MM / ap_spacing_mm))
    ext = False
    if l_lo - m < lo:
        lo, ext = max(0, l_lo - m), True
    if l_hi + m > hi:
        hi, ext = min(n1, l_hi + m), True
    return {"lo": lo, "hi": hi, "skin": skin, "extended_for_lesion": ext, "lesion_axis1": [l_lo, l_hi]}


def crop_axis1(img: nib.Nifti1Image, lo: int, hi: int) -> nib.Nifti1Image:
    arr = np.asanyarray(img.dataobj)
    aff = np.asarray(img.affine).copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([0, lo, 0], dtype=float)
    out = nib.Nifti1Image(np.ascontiguousarray(arr[:, lo:hi]), aff, header=img.header)
    out.set_qform(aff)
    out.set_sform(aff)
    return out
