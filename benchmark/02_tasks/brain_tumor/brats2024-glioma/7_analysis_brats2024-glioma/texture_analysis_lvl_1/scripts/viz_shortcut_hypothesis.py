#!/usr/bin/env python
"""Illustration of Paul's shortcut hypothesis (label remap -> flat target patch with a sharp step; real-fill
buries that cue), with the existing sham-flatten evidence and the T1n-edema puzzle."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cross_contrast_ngf import load_patient, region_masks  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs"
PID, K = "BraTSGLI00512101", 4
rng = np.random.default_rng(1)

vols, lab = load_patient(PID, "cpu")
m = {k: v.numpy() for k, v in region_masks(lab, vols["t1n"]).items()}
brain3 = m["healthy"] | m["whole_tumor"]
z = int(np.argmax(m["SNFH"].sum((0, 1))))
brain, edema = brain3[:, :, z], m["SNFH"][:, :, z]
rows, cols = np.where(brain)
sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))


def zs(c):
    x = vols[c].numpy()[:, :, z].astype(float)
    return (x - x[brain].mean()) / x[brain].std()


x = zs("t2w")
part = np.full(x.shape, -1)
qs = np.quantile(x[brain & ~edema], np.linspace(0, 1, K + 1)[1:-1])
part[brain & ~edema] = np.searchsorted(qs, x[brain & ~edema])
part[edema] = K
mus = {c: rng.uniform(-1.5, 1.5) for c in range(K + 1)}
mus[K] = 1.2
alph = {c: rng.uniform(0.5, 2) * rng.choice([-1, 1]) for c in range(K + 1)}
noise, real = np.zeros_like(x), np.zeros_like(x)
for c in range(K + 1):
    r = part == c
    noise[r] = mus[c] + 0.15 * rng.normal(size=r.sum())
    real[r] = mus[c] + alph[c] * (x[r] - x[r].mean())


def show(ax, img, title, contour=True):
    lo, hi = np.percentile(img[brain], [1, 99])
    ax.imshow(np.rot90(np.where(brain, img, np.nan)[sl]), cmap="gray", vmin=lo, vmax=hi)
    if contour:
        ax.contour(np.rot90(edema[sl]), levels=[0.5], colors="tab:orange", linewidths=0.8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def profile(img):
    from scipy.ndimage import distance_transform_edt
    d = np.where(edema, -distance_transform_edt(edema), distance_transform_edt(~edema))
    bins = np.arange(-6, 7)
    return bins, [img[brain & (np.round(d) == b)].mean() for b in bins]


fig = plt.figure(figsize=(16, 13))
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.35)
ax = fig.add_subplot(gs[0, 0]); show(ax, x, "training image (T2w)")
ax = fig.add_subplot(gs[0, 1]); show(ax, noise, "noise-fill training sample\nedema = FLAT patch + sharp step")
ax = fig.add_subplot(gs[0, 2]); show(ax, real, "real-fill training sample\nstep buried in real structure")
ax = fig.add_subplot(gs[0, 3])
for img, lab_, c in ((noise, "noise-fill", "#1f77b4"), (real, "real-fill", "#d62728")):
    b, p = profile(img); ax.plot(b, p, "-o", color=c, ms=3, label=lab_)
ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("distance to edema border (vox, inside < 0)", fontsize=8)
ax.set_ylabel("mean intensity", fontsize=8); ax.legend(fontsize=8)
ax.set_title("the shortcut: one clean step at the label\n(noise-fill) vs the same step + texture", fontsize=10)

for j, c in enumerate(("t1n", "t1c", "t2f")):
    ax = fig.add_subplot(gs[1, j]); show(ax, zs(c), f"eval contrast {c} (real image)")
ax = fig.add_subplot(gs[1, 3])
for c, col in (("t1n", "#9467bd"), ("t1c", "#8c564b"), ("t2f", "#2ca02c"), ("t2w", "#7f7f7f")):
    b, p = profile(zs(c)); ax.plot(b, p, "-o", color=col, ms=3, label=c)
ax.axvline(0, color="k", lw=0.8); ax.legend(fontsize=8); ax.set_xlabel("distance to edema border (vox)", fontsize=8)
ax.set_title("how edema looks per contrast (this patient)\nT1n: no step, flat, low texture", fontsize=10)

D = OUT / "data"
cells = pd.read_csv(D / "ranking_within_train_cells.csv")
cc = cells[(cells["train"] == "t2w") & (cells["region"] == "SNFH")].set_index("eval")
ax = fig.add_subplot(gs[2, 0:2])
evs = ["t1n", "t1c", "t2f"]; w = 0.38; xs = np.arange(len(evs))
ax.bar(xs - w / 2, [100 * cc.loc[e, "dice_voronoi"] for e in evs], w, color="#1f77b4", label="noise-fill")
ax.bar(xs + w / 2, [100 * cc.loc[e, "dice_realfill"] for e in evs], w, color="#d62728", label="real-fill")
for i, e in enumerate(evs):
    d = 100 * cc.loc[e, "delta"]
    ax.text(i, 100 * max(cc.loc[e, "dice_voronoi"], cc.loc[e, "dice_realfill"]) + 2, f"Δ {d:+.1f}", ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([f"T2w → {e}" for e in evs]); ax.set_ylabel("edema Dice (%)")
ax.set_title("the T1n puzzle: noise-fill does WELL on T1n edema (no step there),\nreal-fill loses; on T1c/FLAIR real-fill wins", fontsize=10)
ax.legend(fontsize=8)

s = pd.read_csv(OUT / "tables" / "intervention_summary.csv")
fp = s[s["metric"].str.startswith("new_fp_voxels_in_sham")]
ax = fig.add_subplot(gs[2, 2:4])
exps = [("E1_flatten_t2f_t1n_trained", "T1n-trained on FLAIR"), ("E2_flatten_t2f_t2w_trained_cross", "T2w-trained on FLAIR")]
for i, (e, lbl) in enumerate(exps):
    g = fp[fp["experiment"] == e].set_index("metric")["median_delta_noise"]
    ax.bar(i - w / 2, g["new_fp_voxels_in_sham_noise"], w, color="#1f77b4", label="noise-fill" if i == 0 else None)
    ax.bar(i + w / 2, g["new_fp_voxels_in_sham_real"], w, color="#d62728", label="real-fill" if i == 0 else None)
ax.set_xticks(range(len(exps))); ax.set_xticklabels([l for _, l in exps]); ax.set_ylabel("new false-positive edema voxels")
ax.set_title("existing evidence (no GT leak): flatten a HEALTHY patch\nnoise-fill calls it edema far more often", fontsize=10)
ax.legend(fontsize=8)
fig.suptitle(f"Shortcut hypothesis — label remap makes the target a flat patch with a sharp step; real-fill buries it ({PID})", fontsize=12)
fig.savefig(OUT / "plots" / "viz_shortcut_hypothesis.png", dpi=130, bbox_inches="tight")
print("ok")
