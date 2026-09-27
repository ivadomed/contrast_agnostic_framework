#!/usr/bin/env python3
"""
Field-of-view (FOV) geometry helpers, shared across all datasets' evaluation.

WHY THIS EXISTS
---------------
CHAOS (the training set) has a RESTRICTED axial field of view: each MR acquisition
covers only a ~2-3 dm superior-inferior slab of the upper abdomen, and it even
TRUNCATES the top of the liver at its superior FOV edge. Models trained on CHAOS
therefore never saw anatomy outside that slab. When such a model is applied to a
full-torso CT (AMOS / SLIVER07 / TRUSTED), predictions it makes far outside the
CHAOS FOV are out-of-distribution and should NOT be scored — but genuine false
positives *inside* the FOV must still count.

The fix is to score only within the CHAOS-equivalent slab of each test volume,
anchored on a stable landmark (the kidneys; the liver for kidney-free datasets):

    keep slab = [ anchor_inferior - inferior_margin , anchor_superior + superior_margin ]

where the margins are the *median* CHAOS distance from that anchor's bounding box
to the image FOV edge (measured by 06_30_measure_chaos_fov.py, in mm). Both the
prediction AND the GT are zeroed outside the slab before Dice/HD95, so out-of-FOV
predictions are dropped (not counted as FP) and out-of-FOV GT is dropped (not
counted as FN) — exactly the anatomy the model was never trained to see.

This module is pure geometry: it knows nothing about organs, datasets or margins.
It only resolves the superior-inferior axis from an image's direction cosines
(robust to any orientation, not just LPS-identity) and builds/applies the slab.
"""
import numpy as np


def si_axis_sign(img):
    """Resolve the superior-inferior axis of a SimpleITK image.

    Returns (numpy_axis, sup_dir, spacing_si_mm):
      numpy_axis    axis in sitk.GetArrayFromImage(img) (z,y,x order) that runs S-I.
      sup_dir       +1 if INCREASING index along numpy_axis is MORE SUPERIOR, else -1.
      spacing_si_mm voxel size (mm) along that axis.

    Works for any axis-aligned orientation: it reads the world-S (LPS row 2)
    component of the direction cosine matrix and picks the dominant index axis.
    """
    D = np.array(img.GetDirection()).reshape(3, 3)   # maps itk index (x,y,z) -> world LPS
    itk_axis = int(np.argmax(np.abs(D[2, :])))       # world Superior is LPS row 2
    sup_dir = 1 if D[2, itk_axis] >= 0 else -1
    numpy_axis = 2 - itk_axis                         # GetArrayFromImage reverses axis order
    spacing_si = float(img.GetSpacing()[itk_axis])
    return numpy_axis, sup_dir, spacing_si


def bbox_si(mask, axis):
    """(imin, imax) inclusive index extent of the nonzero `mask` along `axis`,
    or None if the mask is empty."""
    if not mask.any():
        return None
    other = tuple(a for a in range(mask.ndim) if a != axis)
    prof = np.any(mask, axis=other)
    nz = np.nonzero(prof)[0]
    return int(nz.min()), int(nz.max())


def edge_margins_mm(shape, axis, sup_dir, spacing_si, anchor_imin, anchor_imax):
    """CHAOS-side measurement: distance (mm) from the anchor's S/I extent to the
    image's superior / inferior FOV edge. Returns (sup_mm, inf_mm)."""
    n = shape[axis]
    if sup_dir > 0:                      # superior = high index
        sup_mm = (n - 1 - anchor_imax) * spacing_si
        inf_mm = anchor_imin * spacing_si
    else:                                # superior = low index
        sup_mm = anchor_imin * spacing_si
        inf_mm = (n - 1 - anchor_imax) * spacing_si
    return float(sup_mm), float(inf_mm)


def slab_keep_range(shape, axis, sup_dir, spacing_si, anchor_imin, anchor_imax,
                    sup_mm, inf_mm):
    """Test-side crop: expand the anchor's S/I extent by sup_mm toward superior and
    inf_mm toward inferior (mm -> voxels), clamp to the volume. Returns the inclusive
    index range (lo, hi) to KEEP along `axis`."""
    n = shape[axis]
    sup_vox = int(round(sup_mm / spacing_si))
    inf_vox = int(round(inf_mm / spacing_si))
    if sup_dir > 0:
        sup_idx = anchor_imax + sup_vox
        inf_idx = anchor_imin - inf_vox
    else:
        sup_idx = anchor_imin - sup_vox
        inf_idx = anchor_imax + inf_vox
    lo, hi = sorted((int(sup_idx), int(inf_idx)))
    return max(0, lo), min(n - 1, hi)


def zero_outside_slab(arr, axis, lo, hi):
    """Return a copy of `arr` with everything outside the inclusive index range
    [lo, hi] along `axis` set to 0."""
    keep = np.zeros(arr.shape[axis], bool)
    keep[lo:hi + 1] = True
    shape = [1] * arr.ndim
    shape[axis] = arr.shape[axis]
    return arr * keep.reshape(shape)
