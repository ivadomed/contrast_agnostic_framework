#!/usr/bin/env python3
"""
BraTS illustration: one central axial slice of one case, shown on T1w (no
overlay), T2w (ground-truth region), T2-FLAIR (a DEGRADED region), and the
same T1w slice with PALETTE applied.

Two output forms, from the same slice and the same crop so they line up:
  * one combined 3-panel PDF/PNG (labelled, for dropping into a document);
  * individual undecorated PNGs in figures/FRQ_illustration/ (no titles, no
    axes, overlay burnt into the pixels) for composing slides by hand --
    same convention as figures/method_panels/.

⚠️ THE DEGRADED MASK IS SYNTHETIC. It is the ground truth eroded by this
script, NOT a prediction from any model in this project. Nothing about it is a
measurement, and it must never be presented or captioned as one. The figure
exists to illustrate the SHAPE of an argument -- "a region recovered well on
one contrast and poorly on another" -- for a talk or an explanatory figure,
which is why the panel label says so on the figure itself. If a real
failure case is ever wanted instead, use an actual prediction directory under
8_results_brats2024-glioma/01_predictions/ (see make_snfh_et_zoom_figure.py,
which does exactly that with real noise-fill vs real-fill predictions).

Case/slice were chosen by scanning for the largest single-label region lying
on a genuinely central slice (0.3-0.7 of the S-I extent), so the region reads
clearly without hunting: sub-BraTSGLI00078101, label 2 (SNFH / peritumoural
oedema, the largest of the BraTS labels), z=110, 2537 px in plane.

The PALETTE panels are the REAL transform, not a mock-up: they call
run_pipeline() from generate_method_figure_panels.py, which is built on
src/synthesis/v26_6_synthesis.py's own _kmeans_1d/_voronoi_region_ids and
Eq. (1) remap. Two consequences worth knowing. PALETTE is stochastic, so
N_PALETTE_VARIANTS seeded draws are written and you pick one -- that
variability IS the method, not noise in the figure. And the transform runs on
the cropped slice, so its k-means clusters come from this field of view
rather than the whole volume; for an illustration that is what you want (the
panels then match pixel-for-pixel), but it is not identical to a training-time
draw over a full volume.

Usage:
  .venv/bin/python make_brats_contrast_overlay_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent.parent
BIDS = (REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma"
             / "1_BIDS_brats2024-glioma/glioma-brain-brats2024")
OUT_DIR = REPO / "paper" / "cvpr_format_latex" / "figures"
PANEL_DIR = OUT_DIR / "FRQ_illustration"
STEM = "brats_contrast_overlay_illustration"

# Reuse the real PALETTE pass rather than reimplementing it — CLAUDE.md's
# shared-layer rule. generate_method_figure_panels.py is a sibling script, so
# it is imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_method_figure_panels import (  # noqa: E402
    normalize01, run_pipeline,
)
import torch  # noqa: E402  (only needed to seed the stochastic transform)

N_PALETTE_VARIANTS = 4

# Export scale for the standalone panels. The DATA is 1 mm isotropic
# (182x218x182, the same grid for all four contrasts and the segmentation --
# BraTS resamples to an atlas upstream and does not ship the native
# acquisition), so the crop below is 140x168 real voxels and that is the hard
# information ceiling. SCALE only adds pixels so the panel does not look
# blocky when a slide blows it up; it invents no detail. Set SCALE = 1 for a
# strictly 1:1 export.
SCALE = 8

CASE = "sub-BraTSGLI00078101"
Z = 110
LABEL = 2                 # SNFH / peritumoural oedema — the largest BraTS label

# Degradation strength. r=3 keeps all three of this slice's GT components
# (so a viewer can still match the degraded region to the ground truth panel
# and see that it is the SAME structure done badly) while shrinking it to
# ~40% of its area; the ragged-boundary step then stops it reading as a tidy
# geometric shrink, which no real segmenter produces. Larger radii look more
# broken but delete whole components, which reads as a different structure
# rather than a worse delineation of this one.
EROSION_RADIUS = 3
RAGGED_SIGMA = 2.5        # correlation length of the boundary roughening
RAGGED_KEEP = 0.38        # lower = more bites taken out of the region
RNG_SEED = 0              # fixed so the figure is reproducible

# The paper's own ladder palette (ladder_ood_common.py), so a "good" and a
# "bad" region read the same here as they do in fig:ladder.
GOOD, BAD = "#2f7d6b", "#c0392b"

# Says on the figure that the third panel is not a measurement. Turn off only
# for a context where the caption already makes that unmistakable.
ANNOTATE_ILLUSTRATIVE = True


def load(rel: str) -> np.ndarray:
    return np.asarray(nib.load(str(BIDS / rel)).dataobj)


def axial(vol: np.ndarray, z: int) -> np.ndarray:
    """One axial slice, displayed anterior-up / patient-left-on-image-right.

    BraTS volumes are LAS (verified with nib.aff2axcodes), so axis 0 runs
    towards patient-left and axis 1 towards anterior; rot90 puts anterior at
    the top and leaves left on the image's right (radiological convention)."""
    return np.rot90(vol[:, :, z])


def window(sl: np.ndarray) -> np.ndarray:
    """1-99 percentile window over the BRAIN only -- the same normalisation
    make_snfh_et_zoom_figure.py uses, except that including the large black
    background would drag the low percentile to 0 and wash the tissue out."""
    fg = sl[sl > 0]
    lo, hi = np.percentile(fg, [1, 99]) if fg.size else (0.0, 1.0)
    return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)


def outline(mask: np.ndarray) -> np.ndarray:
    """1-px boundary of a binary mask, for drawing a crisp edge over anatomy."""
    return mask & ~ndimage.binary_erosion(mask, np.ones((3, 3), bool))


def overlay(ax, base: np.ndarray, mask: np.ndarray, color: str, alpha=0.38):
    rgb = matplotlib.colors.to_rgb(color)
    fill = np.zeros((*mask.shape, 4))
    fill[mask] = (*rgb, alpha)
    edge = np.zeros((*mask.shape, 4))
    edge[outline(mask)] = (*rgb, 1.0)
    ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(fill, interpolation="nearest")
    ax.imshow(edge, interpolation="nearest")


def upscale(arr: np.ndarray, order: int) -> np.ndarray:
    """Enlarge by SCALE.

    order=3 for continuous anatomy; order=0 (nearest) for MASKS, so a region
    boundary lands exactly where the voxels put it instead of feathering.

    PALETTE output gets order=1, not 0. It is tempting to treat it as
    piecewise-constant and keep hard edges, but it is not: the affine remap
    preserves within-region texture (that is the entire point of the method),
    so it carries real voxel-level detail just like the anatomy. Nearest
    leaves it in visible SCALE-sized blocks while the neighbouring anatomy
    panels are smooth -- a display artifact that reads as PALETTE being
    coarse, which would misrepresent it. Bilinear rather than cubic because
    the region boundaries are genuine step edges and cubic rings at them."""
    if SCALE == 1:
        return arr
    return ndimage.zoom(arr.astype(float), SCALE, order=order,
                        mode="nearest", grid_mode=False)


def burn_overlay(base: np.ndarray, mask: np.ndarray | None, color: str,
                 alpha=0.38) -> np.ndarray:
    """Grey base + translucent mask + solid edge, flattened to RGB, at SCALE.

    The standalone panels carry no axes or titles, so the overlay has to live
    in the pixels rather than in a matplotlib artist drawn on top. Anatomy is
    interpolated smoothly but the mask is not, so the region boundary stays
    exactly where the voxels put it instead of feathering."""
    big = np.clip(upscale(base, order=3), 0, 1)
    rgb = np.repeat(big[:, :, None], 3, axis=2).astype(float)
    if mask is None or not mask.any():
        return np.clip(rgb, 0, 1)
    m = upscale(mask.astype(float), order=0) > 0.5
    c = np.array(matplotlib.colors.to_rgb(color))
    rgb[m] = (1 - alpha) * rgb[m] + alpha * c
    # Edge thickness tracks SCALE, or an upscaled 1-px line vanishes.
    w = max(1, SCALE // 3)
    edge = m & ~ndimage.binary_erosion(m, np.ones((2 * w + 1, 2 * w + 1), bool))
    rgb[edge] = c
    return np.clip(rgb, 0, 1)


def write_panels(t1_disp, t2_disp, fl_disp, gt, degraded, t1_raw, lbl_slice,
                 fl_full):
    """Individual undecorated PNGs, all on the same crop so they overlay."""
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for name, arr in (
        ("t1w", burn_overlay(t1_disp, None, GOOD)),
        ("t2w_gt", burn_overlay(t2_disp, gt, GOOD)),
        ("t2flair_degraded", burn_overlay(fl_disp, degraded, BAD)),
        # Same FLAIR image with the CORRECT ground truth instead, so the
        # good/bad pair differs only in the mask, not in the underlying
        # contrast. GOOD/BAD are the set's existing colours.
        ("t2flair_gt", burn_overlay(fl_disp, gt, GOOD)),
    ):
        out = PANEL_DIR / f"{name}.png"
        plt.imsave(out, arr)
        written.append(out)

    # Plain T2-FLAIR, no overlay, written at 1:1 with the data (SCALE
    # deliberately bypassed) — one pixel per voxel, nothing interpolated.
    # Two framings: the crop the rest of the set uses, so it overlays them,
    # and the whole uncropped slice.
    for name, arr in (("t2flair", fl_disp), ("t2flair_fullfov", fl_full)):
        out = PANEL_DIR / f"{name}.png"
        plt.imsave(out, arr, cmap="gray", vmin=0.0, vmax=1.0)
        written.append(out)
        print(f"  {name}.png is 1:1 native: {arr.shape[1]}x{arr.shape[0]} px")

    # PALETTE on the same T1w slice. normalize01 (not the display window) is
    # what the transform's DARK_THRESHOLD and k-means expect as input.
    img01 = normalize01(t1_raw)
    for k in range(1, N_PALETTE_VARIANTS + 1):
        torch.manual_seed(k)
        out_arr = run_pipeline(img01, lbl_slice)["e"]
        out = PANEL_DIR / f"t1w_palette_{k}.png"
        # PALETTE runs at native voxel resolution (the real transform on real
        # voxels); only its output is enlarged, for display.
        plt.imsave(out, np.clip(upscale(out_arr, order=1), 0, 1),
                   cmap="gray", vmin=0.0, vmax=1.0)
        written.append(out)

    h, w_ = gt.shape
    print(f"  panels: {w_}x{h} voxels -> {w_ * SCALE}x{h * SCALE} px (SCALE={SCALE})")
    for f in written:
        print("wrote", f)


def main():
    t1 = load(f"{CASE}/anat/{CASE}_T1w.nii")
    t2 = load(f"{CASE}/anat/{CASE}_T2w.nii")
    fl = load(f"{CASE}/anat/{CASE}_FLAIR.nii")
    seg = load(f"derivatives/manual_masks/{CASE}/anat/{CASE}_dseg.nii")

    gt = axial(seg, Z) == LABEL
    if not gt.any():
        raise SystemExit(f"label {LABEL} absent on slice {Z} of {CASE}")

    # Synthetic degradation, in two steps. Erode so the region is clearly
    # under-covered, then roughen the boundary with a smoothed random field so
    # it does not read as a tidy morphological shrink -- a real segmenter
    # fails with ragged edges and bites out of the interior, not with a
    # uniformly inset copy of the truth.
    r = EROSION_RADIUS
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r ** 2
    degraded = ndimage.binary_erosion(gt, disk)

    rng = np.random.default_rng(RNG_SEED)
    field = ndimage.gaussian_filter(rng.random(gt.shape), sigma=RAGGED_SIGMA)
    field = (field - field.min()) / max(np.ptp(field), 1e-9)
    degraded &= field > RAGGED_KEEP
    degraded = ndimage.binary_opening(degraded, np.ones((3, 3), bool))

    lab, n = ndimage.label(degraded)
    if n:
        sizes = ndimage.sum(degraded, lab, range(1, n + 1))
        keep = {i + 1 for i, s in enumerate(sizes) if s >= 25}
        degraded = np.isin(lab, list(keep)) if keep else degraded

    inter = (gt & degraded).sum()
    dice = 2 * inter / max(gt.sum() + degraded.sum(), 1)
    print(f"{CASE}  z={Z}  label={LABEL}")
    print(f"  GT area       {gt.sum():5d} px")
    print(f"  degraded area {degraded.sum():5d} px  "
          f"({degraded.sum() / gt.sum():.0%} retained, Dice vs GT {dice:.2f})")

    # Crop to the brain so the slice is not mostly background.
    brain = axial(t1, Z) > 0
    ys, xs = np.where(brain)
    m = 4
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, brain.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, brain.shape[1])
    crop = (slice(y0, y1), slice(x0, x1))

    t1_disp = window(axial(t1, Z))[crop]
    t2_disp = window(axial(t2, Z))[crop]
    fl_disp = window(axial(fl, Z))[crop]

    # Standalone panels first, so a failure in the combined figure's layout
    # doesn't cost the images that are the actual deliverable here.
    write_panels(t1_disp, t2_disp, fl_disp, gt[crop], degraded[crop],
                 axial(t1, Z)[crop], axial(seg, Z)[crop].astype(np.int64),
                 window(axial(fl, Z)))

    panels = [
        (t1_disp, None,           None, "T1w",      "no overlay"),
        (t2_disp, gt[crop],       GOOD, "T2w",      "ground truth"),
        (fl_disp, degraded[crop], BAD,  "T2-FLAIR", "degraded delineation"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1))
    for ax, (base, mask, color, title, sub) in zip(axes, panels):
        if mask is None:
            ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            overlay(ax, base, mask, color)
        ax.set_title(title, fontsize=13, pad=7)
        ax.text(0.5, -0.045, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=(color or "#555555"))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.14, wspace=0.04)
    if ANNOTATE_ILLUSTRATIVE:
        fig.text(0.5, 0.025,
                 "Illustrative: the degraded region is the ground truth eroded for display, "
                 "not a model prediction.",
                 ha="center", fontsize=8.5, color="#8a8a8a")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("pdf", 300), ("png", 200)):
        p = OUT_DIR / f"{STEM}.{ext}"
        fig.savefig(p, dpi=dpi)
        print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
