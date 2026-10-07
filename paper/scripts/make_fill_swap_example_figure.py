#!/usr/bin/env python3
"""
Small figure for the causal ablation (fig:fill-swap-example): the ablation's NOISE fill next to PALETTE's REAL fill on
the same partition and the same region target means, on the ON-Harmony T1w slice of fig:method-pipeline, each with a
magnified crop so the texture difference is visible at column width. One row:
    [noise fill, box] [noise zoom] [real zoom] [real fill, box]
Panels come from generate_method_figure_panels.py (onharmony_n.png, onharmony_e.png); the noise fill is not part of the
method, so it is kept out of fig:method-pipeline.

Usage:  .venv/bin/python paper/scripts/make_fill_swap_example_figure.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle  # noqa: E402

from make_method_figure import FIG, ROWS, load  # noqa: E402

OUT = FIG / "fill_swap_example"
# zoom window as fractions of the (radiologically oriented) slice: (row0, row1, col0, col1)
ZOOM = (0.36, 0.64, 0.30, 0.58)   # deep grey matter around the ventricles (caudate, thalamus, internal capsule)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})


def main():
    slug, _, orient = next(r for r in ROWS if r[0] == "onharmony")
    noise, real = load(slug, "n", orient), load(slug, "e", orient)
    h, w = noise.shape
    r0, r1, c0, c1 = int(ZOOM[0] * h), int(ZOOM[1] * h), int(ZOOM[2] * w), int(ZOOM[3] * w)
    zw = (c1 - c0) / (r1 - r0)                                   # zoom aspect (w/h)

    ph = 1.25                                                    # panel height, inches
    widths = [ph * w / h, ph * zw, ph * zw, ph * w / h]
    gap, head = 0.05, 0.20
    W = sum(widths) + gap * 3
    fig = plt.figure(figsize=(W, ph + head))
    axes, x = [], 0.0
    for wd in widths:
        axes.append(fig.add_axes([x / W, 0, wd / W, ph / (ph + head)])); x += wd + gap
    ax_n, ax_nz, ax_rz, ax_r = axes
    for ax, im in ((ax_n, noise), (ax_r, real)):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor="#f2c14e", linewidth=1.0))
    for ax, im in ((ax_nz, noise), (ax_rz, real)):
        ax.imshow(im[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for sp in ax.spines.values():
            sp.set_edgecolor("#f2c14e"); sp.set_linewidth(1.2)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    for ax in (ax_n, ax_r):
        ax.set_axis_off()
    # magnifier lines from the box to its zoom
    for src, dst, side in ((ax_n, ax_nz, "right"), (ax_r, ax_rz, "left")):
        xc = c1 if side == "right" else c0
        zx = 0 if side == "right" else (c1 - c0)
        for ry, zy in ((r0, 0), (r1, r1 - r0)):
            fig.add_artist(ConnectionPatch(xyA=(xc, ry), coordsA=src.transData, xyB=(zx - 0.5, zy - 0.5), coordsB=dst.transData,
                                           color="#f2c14e", linewidth=0.7))
    top = (ph + head - 0.02) / (ph + head)
    fig.text((widths[0] + gap / 2) / W, top, "Noise fill (texture removed)", ha="center", va="top", fontsize=7)
    fig.text((sum(widths[:3]) + 2.5 * gap) / W, top, "Real fill (texture kept)", ha="center", va="top", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=400)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
