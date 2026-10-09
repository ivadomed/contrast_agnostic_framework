"""PV preview, ToothFairy2 CBCT vs BraTS t1n, side by side (rung-6 PV implementation, PALETTE V26_6_2).

Each case is resampled to its task's training spacing (toothfairy2 0.6 mm, BraTS 1.0 mm), then PALETTE is run
with the legacy blur OFF (blur_sigmas=[0]) with and without PV on the SAME seed, so every difference is PV alone.
PV levels shown: the strongest (sigma 1.5 vox, w 1.0, tau 0.08) and the typical medium one (1.0, 0.7, 0.04).
Rows: full crop | zoom on the segmented structure's edge. Red contour = ground truth (toothfairy2: mandible).
Usage: pv_compare_tasks.py <out_dir> [toothfairy2|brats]   (CPU; toothfairy2 data lives on TamIA, the BraTS training case on Vulcan,
so run the two halves where their data is; no 2nd arg = both)
"""
import random, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import zoom
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "sub-workspaces/auglab_workspace/AugLab"))
from auglab.transforms.gpu.fromSeg import RandomV26_6_2ContrastGPU  # noqa: E402

S = Path("/scratch/p/paulh")
BR = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n"   # Vulcan
TASKS = {
    "ToothFairy2 CBCT (0.6 mm)": dict(img=S / "toothfairy2/2_nnUNet/raw/Dataset110_ToothFairy2CBCT/imagesTr/toothfairy2_ToothFairy2F002_0000.nii.gz",
                                      seg=S / "toothfairy2/2_nnUNet/raw/Dataset110_ToothFairy2CBCT/labelsTr/toothfairy2_ToothFairy2F002.nii.gz",
                                      spacing=0.6, target=1, crop=(112, 112, 96)),
    "BraTS t1n (1.0 mm)": dict(img=BR / "imagesTr/BraTSGLI00005100_0000.nii.gz",
                               seg=BR / "labelsTr/BraTSGLI00005100.nii.gz",
                               spacing=1.0, target=None, crop=(128, 128, 96)),
}
LEVELS = {"PALETTE (no PV)": None, "PV medium": (1.0, 0.7, 0.04), "PV strong": (1.5, 1.0, 0.08)}
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)


def load(cfg):
    im, sg = nib.load(cfg["img"]), nib.load(cfg["seg"])
    x, y = im.get_fdata().astype(np.float32), np.rint(sg.get_fdata()).astype(np.int64)
    f = [z / cfg["spacing"] for z in im.header.get_zooms()[:3]]
    print(cfg["img"].name, "native zooms", [round(float(z), 2) for z in im.header.get_zooms()[:3]], "-> resample factor", [round(a, 2) for a in f])
    if max(abs(a - 1) for a in f) > 0.02:
        x, y = zoom(x, f, order=1), zoom(y, f, order=0)
    t = (y == cfg["target"]) if cfg["target"] else (y > 0)
    c = np.array(np.nonzero(t)).mean(1).astype(int)
    sl = tuple(slice(max(0, ci - n // 2), max(0, ci - n // 2) + n) for ci, n in zip(c, cfg["crop"]))
    x, y, t = x[sl], y[sl], t[sl]
    pad = [(0, max(0, n - s)) for n, s in zip(cfg["crop"], x.shape)]
    return np.pad(x, pad), np.pad(y, pad), np.pad(t, pad)


for name, cfg in TASKS.items():
    if len(sys.argv) > 2 and sys.argv[2].lower() not in name.lower():
        continue
    x, y, t = load(cfg)
    xt, yt = torch.from_numpy(x)[None, None], torch.from_numpy(y)[None, None]
    z = int(np.argmax(t.sum((0, 1))))
    for seed in (0, 1):
        res = {}
        for lv, p in LEVELS.items():
            random.seed(seed); torch.manual_seed(seed)
            kw = dict(blur_sigmas=[0.0], p=1.0) | ({} if p is None else dict(pv_prob=1.0, pv_levels=[list(p)]))
            res[lv] = RandomV26_6_2ContrastGPU(**kw).apply_transform(xt, {"seg": yt}, {})[0, 0, :, :, z].numpy()
        cols = [("source", x[:, :, z])] + list(res.items()) + [("|strong - PALETTE|", np.abs(res["PV strong"] - res["PALETTE (no PV)"]))]
        ys, xs = np.nonzero(t[:, :, z]); cy, cx = int(ys.mean()), int(xs.mean()); h = 22
        zm = (slice(max(0, cy - h), cy + h), slice(max(0, cx - h), cx + h))
        fig, ax = plt.subplots(2, len(cols), figsize=(3.1 * len(cols), 6.6))
        for j, (tt, a) in enumerate(cols):
            fg = x[:, :, z] > x[:, :, z].min() + 1e-3
            lo, hi = (np.percentile(a[fg], [1, 99]) if j < len(cols) - 1 else (0, a.max() + 1e-6))
            for r, aa in enumerate([a, a[zm]]):
                ax[r, j].imshow(aa.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
                ax[r, j].contour((t[:, :, z][zm] if r else t[:, :, z]).T, [0.5], colors="r", linewidths=0.6, origin="lower"); ax[r, j].axis("off")
            ax[0, j].set_title(tt, fontsize=9)
        fig.suptitle(f"{name} - seed {seed}: boundary PV on the same synthesis (legacy blur off); red = GT" + (" mandible" if cfg["target"] else " tumour"), fontsize=10)
        fig.tight_layout(); fn = out / f"pv_compare_{name.split()[0].lower()}_s{seed}.png"; fig.savefig(fn, dpi=110); plt.close(fig); print("wrote", fn)
