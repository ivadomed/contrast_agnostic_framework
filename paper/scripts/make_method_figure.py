#!/usr/bin/env python3
"""
Compose fig:method-pipeline (2 rows x 6 panels) from the panels written by
generate_method_figure_panels.py, in one matplotlib figure so every cell has the
same width, the headers sit on a common baseline and the slices are shown in
radiological orientation (anterior up, patient right on the image left; the
panel PNGs keep the raw array axes, which put the CHAOS slice sideways).

Usage:  .venv/bin/python paper/scripts/make_method_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "cvpr_format_latex/figures"
PANELS = FIG / "method_panels"
OUT = FIG / "method_pipeline"
LETTERS = "abcdne"   # n = noise fill (rung 4), shown before the real fill (rung 5)
HEADERS = ["(a) Input", "(b) + $k$-means\n(flat fill)", "(c) + label remap\n(flat fill)",
           "(d) + Voronoi\n(flat fill)", "(e) Noise fill\n(texture removed)",
           "(f) Real fill\n(PALETTE output)"]
# slug, row label, raw-array -> radiological display (array axes from nib.aff2axcodes of the source volume)
ROWS = [("chaos", "Abdomen\nCHAOS T2SPIR", lambda a: a.T),                 # LPS: rows=x(L), cols=y(P)
        ("onharmony", "Brain\nON-Harmony T1w", lambda a: a.T[::-1, ::-1])]  # RAS: rows=x(R), cols=y(A)
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})


def load(slug: str, letter: str, orient) -> np.ndarray:
    a = plt.imread(PANELS / f"{slug}_{letter}.png")
    a = a[..., 0] if a.ndim == 3 else a          # grayscale PNG saved via a colormap -> any channel
    return orient(a)


def main():
    imgs = [[load(s, l, o) for l in LETTERS] for s, _, o in ROWS]
    aspect = [im[0].shape[0] / im[0].shape[1] for im in imgs]       # h / w per row
    cell_w = 1.07                                                    # inches
    lab_w, head_h, gap = 0.32, 0.34, 0.04
    heights = [cell_w * a for a in aspect]
    W = lab_w + len(LETTERS) * cell_w + (len(LETTERS) - 1) * gap
    H = head_h + sum(heights) + gap * (len(ROWS) - 1)
    fig = plt.figure(figsize=(W, H))
    y = H - head_h
    for r, ((slug, label, _), row) in enumerate(zip(ROWS, imgs)):
        h = heights[r]
        y -= h
        for c, im in enumerate(row):
            x = lab_w + c * (cell_w + gap)
            ax = fig.add_axes([x / W, y / H, cell_w / W, h / H])
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            ax.set_axis_off()
            if r == 0:
                fig.text((x + cell_w / 2) / W, (H - 0.03) / H, HEADERS[c], ha="center", va="top",
                         fontsize=7.2, linespacing=1.1)
        fig.text(0.12 / W, (y + h / 2) / H, label, rotation=90, ha="center", va="center",
                 fontsize=7.2, linespacing=1.1)
        y -= gap
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=400)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
