#!/usr/bin/env python3
"""
Small figure for the causal ablation (fig:fill-swap-example): the ablation's NOISE fill next to
PALETTE's REAL fill on the same partition and the same region target means, for the two slices of
fig:method-pipeline. Panels come from generate_method_figure_panels.py ({slug}_n.png, {slug}_e.png);
the noise fill is not part of the method, so it is kept out of fig:method-pipeline.

Usage:  .venv/bin/python paper/scripts/make_fill_swap_example_figure.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_method_figure import FIG, ROWS, load  # noqa: E402

OUT = FIG / "fill_swap_example"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})


def main():
    # one row: [abdomen noise | abdomen real] [brain noise | brain real], each image at its own aspect
    imgs = [(label.split("\n")[0], [load(s, l, o) for l in "ne"]) for s, label, o in ROWS]
    h, gap, pair_gap, head_h = 0.95, 0.03, 0.10, 0.30
    widths = [[h * im.shape[1] / im.shape[0] for im in row] for _, row in imgs]
    W = sum(sum(w) + gap for w in widths) - gap + pair_gap
    H = h + head_h
    fig = plt.figure(figsize=(W, H))
    x = 0.0
    for (label, row), ws in zip(imgs, widths):
        x0 = x
        for c, (im, w) in enumerate(zip(row, ws)):
            ax = fig.add_axes([x / W, 0, w / W, h / H])
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            ax.set_axis_off()
            fig.text((x + w / 2) / W, (h + 0.02) / H, ("noise fill", "real fill")[c], ha="center", va="bottom", fontsize=6.5)
            x += w + gap
        fig.text((x0 + (x - gap - x0) / 2) / W, (H - 0.02) / H, label, ha="center", va="top", fontsize=7, fontweight="bold")
        x += pair_gap - gap
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=400)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
