#!/usr/bin/env python3
"""
Small figure for the causal ablation (fig:fill-swap-example) on a BraTS glioma T1c training slice: the input with the
zoom box, then the same zoom under the three fills of one partition (shared k-means classes, Voronoi sub-regions, region
target means and label-remap draws): noise fill (the ablation's rung 4), flat fill (each region painted with its target
mean, the partition alone, as in fig:method-pipeline b-d) and real fill (PALETTE). One row:
    [input, box] [noise zoom] [flat zoom] [real zoom]
Panels come from generate_method_figure_panels.py (brats_t1c_{a,n,d,e}.png, brats_t1c_lbl.npy).

Usage:  .venv/bin/python paper/scripts/make_fill_swap_example_figure.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from make_method_figure import FIG, PANELS, load  # noqa: E402

OUT = FIG / "fill_swap_example"
SLUG = "brats_t1c"                 # BraTS glioma T1c training case (generate_method_figure_panels.py EXAMPLE_CASES)
ORIENT = lambda a: a.T[::-1]       # noqa: E731  LAS array (rows x=L, cols y=A) -> anterior up, patient left on image right
ZOOM_LABELS = (3,)                 # BraTS-GLI ET: the enhancing ring (NCR = 1, SNFH = 2 the edema, RC = 4)
MARGIN = 0.12                      # zoom = ring bounding box grown by this fraction per side, then squared
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})


def main():
    src, noise, flat, real = (load(SLUG, k, ORIENT) for k in "ande")
    h, w = src.shape
    ring = np.isin(ORIENT(np.load(PANELS / f"{SLUG}_lbl.npy")), ZOOM_LABELS)
    comp, n = ndimage.label(ring)                                # largest connected piece: the ring itself, not stray ET voxels
    ys, xs = np.where(comp == 1 + int(np.argmax(ndimage.sum(ring, comp, range(1, n + 1)))))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) * (0.5 + MARGIN)
    r0, r1 = int(max(cy - half, 0)), int(min(cy + half, h))
    c0, c1 = int(max(cx - half, 0)), int(min(cx + half, w))
    zw = (c1 - c0) / (r1 - r0)                                   # zoom aspect (w/h)

    ph = 1.25                                                    # panel height, inches
    widths = [ph * w / h, ph * zw, ph * zw, ph * zw]
    gap, head = 0.05, 0.20
    W = sum(widths) + gap * 3
    fig = plt.figure(figsize=(W, ph + head))
    axes, x = [], 0.0
    for wd in widths:
        axes.append(fig.add_axes([x / W, 0, wd / W, ph / (ph + head)])); x += wd + gap
    ax_src, zooms = axes[0], axes[1:]
    ax_src.imshow(src, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
    ax_src.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="#f2c14e", linewidth=1.0))
    ax_src.set_axis_off()
    for ax, im in zip(zooms, (noise, flat, real)):
        ax.imshow(im[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#f2c14e"); sp.set_linewidth(1.2)
    for ry, zy in ((r0, 0), (r1, r1 - r0)):                      # magnifier lines from the box to the first zoom
        fig.add_artist(ConnectionPatch(xyA=(c1, ry), coordsA=ax_src.transData, xyB=(-0.5, zy - 0.5),
                                       coordsB=zooms[0].transData, color="#f2c14e", linewidth=0.7))
    top = (ph + head - 0.02) / (ph + head)
    titles = ("Input", "Noise fill", "Flat fill", "Real fill")
    x = 0.0
    for wd, t in zip(widths, titles):
        fig.text((x + wd / 2) / W, top, t, ha="center", va="top", fontsize=7); x += wd + gap
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=400)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
