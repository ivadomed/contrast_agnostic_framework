#!/usr/bin/env python3
"""
Three-panel BraTS illustration: one central axial slice of one case, shown on
T1w (no overlay), T2w (ground-truth region), and T2-FLAIR (a DEGRADED region).

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

Usage:
  .venv/bin/python make_brats_contrast_overlay_figure.py
"""
from __future__ import annotations

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
STEM = "brats_contrast_overlay_illustration"

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

    panels = [
        (window(axial(t1, Z))[crop], None,            None, "T1w",      "no overlay"),
        (window(axial(t2, Z))[crop], gt[crop],        GOOD, "T2w",      "ground truth"),
        (window(axial(fl, Z))[crop], degraded[crop],  BAD,  "T2-FLAIR", "degraded delineation"),
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
