#!/usr/bin/env python
"""Illustration of the proposed fill arms on one real BraTS T2w slice (simplified partition: intensity
quantiles within brain + the edema label as its own region; NOT the real K-means/Voronoi transform)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cross_contrast_ngf import load_patient, region_masks  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "plots" / "viz_fill_arms.png"
PID, SIGMA, K = "BraTSGLI00512101", 2.0, 4
rng = np.random.default_rng(0)

vols, lab = load_patient(PID, "cpu")
m = {k: v.numpy() for k, v in region_masks(lab, vols["t1n"]).items()}
x3 = vols["t2w"].numpy().astype(float)
brain3 = m["healthy"] | m["whole_tumor"]
z = int(np.argmax(m["SNFH"].sum((0, 1))))
x, brain, edema = x3[:, :, z], brain3[:, :, z], m["SNFH"][:, :, z]
x = (x - x[brain].mean()) / x[brain].std()
rows, cols = np.where(brain)
sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))

part = np.full(x.shape, -1)
qs = np.quantile(x[brain & ~edema], np.linspace(0, 1, K + 1)[1:-1])
part[brain & ~edema] = np.searchsorted(qs, x[brain & ~edema])
part[edema] = K
smooth = np.zeros_like(x)
for c in range(K + 1):  # smooth WITHIN each region (normalized convolution), so no cross-border edges leak in
    r = (part == c).astype(float)
    smooth += r * gaussian_filter(x * r, SIGMA) / np.maximum(gaussian_filter(r, SIGMA), 1e-6)
fine = np.where(brain, x - smooth, 0)


def corr_noise(mask, ref):
    n = gaussian_filter(rng.normal(size=x.shape), 1.0)
    n = (n - n[mask].mean()) / n[mask].std()
    return n * ref[mask].std()


arms = {"original T2w": None, "rung 4: noise-fill": "noise", "rung 5: real-fill": "real",
        "A1 smooth-only": "smooth", "A2 fine-texture-only": "fine", "A3 correlated noise": "corr",
        "A4 histogram noise": "hist"}
mus = {c: rng.uniform(-1.5, 1.5) for c in range(K + 1)}
alphas = {c: rng.uniform(0.5, 2) * rng.choice([-1, 1]) for c in range(K + 1)}
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for ax, (title, mode) in zip(axes.flat, arms.items()):
    if mode is None:
        img = x.copy()
    else:
        img = np.zeros_like(x)
        for c in range(K + 1):
            r = part == c
            if not r.any():
                continue
            mu, a, xr = mus[c], alphas[c], x[r]
            if mode == "noise":
                v = mu + 0.15 * rng.normal(size=r.sum())
            elif mode == "real":
                v = mu + a * (xr - xr.mean())
            elif mode == "smooth":
                s = smooth[r]
                v = mu + a * (s - s.mean()) + rng.normal(size=r.sum()) * fine[r].std()
            elif mode == "fine":
                v = mu + a * fine[r]
            elif mode == "corr":
                v = mu + corr_noise(r, x)[r]
            else:
                v = mu + rng.choice(xr - xr.mean(), size=r.sum())
            img[r] = v
    vals = img[brain]
    lo, hi = np.percentile(vals, [1, 99])
    ax.imshow(np.rot90(np.where(brain, img, np.nan)[sl]), cmap="gray", vmin=lo, vmax=hi)
    ax.contour(np.rot90(edema[sl]), levels=[0.5], colors="tab:orange", linewidths=0.8)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
ax = axes.flat[-1]
ax.axis("off")
ax.text(0, 0.95, "What each arm keeps inside a region\n(same partition, same random means μ):\n\n"
        "noise-fill: only the mean (flat + iid noise)\nreal-fill: all real structure (random scale/sign)\n"
        "A1: smooth shading only (+ matched iid noise)\nA2: fine texture only (shading removed)\n"
        "A3: noise with real-like grain, no anatomy\nA4: real intensity spread, no spatial order\n\n"
        "orange = GT edema (its own region here)\nSimplified illustration, not the real transform.",
        va="top", fontsize=10, family="monospace")
fig.suptitle(f"Proposed fill arms on one T2w slice ({PID}, axial {z})", fontsize=13)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(OUT)
