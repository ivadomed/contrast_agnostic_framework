"""Visual preview of boundary partial-volume (PV) simulation in PALETTE (V26_6_2) on BraTS t1n.

Renders the FINAL rung-6 implementation (pv_levels API). Same seed with and without PV, the legacy blur_sigmas forced to 0 in both, so every visible
difference is the PV step alone. Columns: source | PALETTE | PV weak/medium/strong | |strong - PALETTE|,
plus a zoom row around the tumour. CPU-only, a few cases. Output: outputs/pv_preview_<case>_s<seed>.png
"""
import random, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "sub-workspaces/auglab_workspace/AugLab"))
from auglab.transforms.gpu.fromSeg import RandomV26_6_2ContrastGPU  # noqa: E402

RAW = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n"
CASES = ["BraTSGLI00005100", "BraTSGLI00006100", "BraTSGLI00005101"]
SEEDS = [0, 1, 2]
CROP = 128
# pv_levels entries of the rung-6 config: (spatial sigma, spatial weight, K-means intensity softness tau)
LEVELS = {"none": None, "weak": (0.7, 0.4, 0.02), "medium": (1.0, 0.7, 0.04), "strong": (1.5, 1.0, 0.08)}
OUT = HERE / "outputs"; OUT.mkdir(exist_ok=True)


def make(level):
    kw = dict(blur_sigmas=[0.0], p=1.0)
    if LEVELS[level] is not None:
        s, w, t = LEVELS[level]
        kw.update(pv_prob=1.0, pv_levels=[[s, w, t]])
    return RandomV26_6_2ContrastGPU(**kw)


def load(case):
    img = nib.load(RAW / "imagesTr" / f"{case}_0000.nii.gz").get_fdata().astype(np.float32)
    seg = nib.load(RAW / "labelsTr" / f"{case}.nii.gz").get_fdata().astype(np.int64)
    c = np.array(np.nonzero(seg)).mean(1).astype(int)
    sl = tuple(slice(max(0, ci - CROP // 2), max(0, ci - CROP // 2) + CROP) for ci in c)
    return img[sl], seg[sl]


for case in CASES:
    img, seg = load(case)
    x = torch.from_numpy(img)[None, None]
    s = torch.from_numpy(seg)[None, None]
    z = int(np.argmax((seg > 0).sum((0, 1))))  # slice with most tumour (last array axis)
    outs = {}
    for seed in SEEDS:
        for lvl in LEVELS:
            random.seed(seed); torch.manual_seed(seed)
            outs[(seed, lvl)] = make(lvl).apply_transform(x, {"seg": s}, {})[0, 0, :, :, z].numpy()
        ys, xs = np.nonzero(seg[:, :, z]); cy, cx = int(ys.mean()), int(xs.mean())
        zoom = (slice(max(0, cy - 24), cy + 24), slice(max(0, cx - 24), cx + 24))
        cols = [("source t1n", img[:, :, z])] + [(f"PV {l}" if LEVELS[l] else "PALETTE (no PV)", outs[(seed, l)]) for l in LEVELS]
        cols.append(("|strong - PALETTE|", np.abs(outs[(seed, "strong")] - outs[(seed, "none")])))
        fig, ax = plt.subplots(2, len(cols), figsize=(3 * len(cols), 6.4))
        for j, (t, a) in enumerate(cols):
            fg = img[:, :, z] > 0
            lo, hi = (np.percentile(a[fg], [1, 99]) if fg.any() and j < len(cols) - 1 else (0, a.max() + 1e-6))
            for r, aa in enumerate([a, a[zoom]]):
                ax[r, j].imshow(aa.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
                if r == 1: ax[r, j].contour(seg[:, :, z][zoom].T > 0, [0.5], colors="r", linewidths=0.5, origin="lower")
                ax[r, j].axis("off")
            ax[0, j].set_title(t, fontsize=9)
        fig.suptitle(f"{case} seed {seed} — K-means PV in intensity space, Voronoi + label PV spatial (rung-6 levels); red = GT tumour", fontsize=10)
        fig.tight_layout(); p = OUT / f"pv_preview_final_{case}_s{seed}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        d = outs[(seed, "strong")] - outs[(seed, "none")]
        print(f"{p.name}: frac fg voxels |Δz|>0.05 = {(np.abs(d) > 0.05)[img[:, :, z] > 0].mean():.3f}")
print("DONE")
