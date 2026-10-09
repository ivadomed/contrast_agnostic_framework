#!/usr/bin/env python
"""What the noise-fill and real-fill models SAW, side by side, for Open-MS FLAIR training patches (2026-10-07).
Columns: original FLAIR patch (nnU-Net z-score), rung-4 noise-fill synthesis, rung-5 real-fill synthesis, same seed for both
arms (the two pipelines share the partition sampler; the synthesis probability is forced to 1 for this illustration; training uses p = 0.5). Lesion GT as an orange contour; random-offset 96-vox zoom on the largest lesion section + full-slice inset.
Usage (GPU job): .venv/bin/python viz_fill_training_samples.py [--n 6]   -> outputs/plots/viz_fill_training_samples_openms.png"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(THIS))
from auglab.transforms.gpu.transforms import AugTransformsGPU  # noqa: E402
from trainme_build import zscore, patch_box, zoom_box, CFG, ARMS  # noqa: E402
OUT = THIS.parent / "outputs/plots"
R = REPO / "benchmark/02_tasks/brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
P = REPO / "benchmark/02_tasks/brain_ms/open-ms/2_nnUNet_open-ms/preprocessed/Dataset070_OpenMS_FLAIR"


def show(ax, g, m, title):
    fgm = g != 0; lo, hi = np.percentile(g[fgm], [1, 99]) if fgm.any() else (0, 1)
    ax.imshow(g.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    if m.any():
        ax.contour(m.T.astype(float), levels=[0.5], colors=["#eb6834"], linewidths=1.0)
    ax.set_title(title, fontsize=9, loc="left"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    dev = torch.device(a.device); rng = np.random.default_rng(1)
    size = json.load(open(P / "nnUNetPlans.json"))["configurations"]["3d_fullres"]["patch_size"]
    cases = json.load(open(P / "splits_final.json"))[0]["train"]
    pipes = {k: AugTransformsGPU(json_path=str(CFG / v), num_labels=2).to(dev).eval() for k, v in ARMS.items()}
    for pipe in pipes.values():   # illustration only: make the synthesis fire every time (training uses p = 0.5)
        for m in pipe.modules():
            if "V26_6_2" in type(m).__name__ and hasattr(m, "p"):
                m.p = 1.0; print("forced p=1 on", type(m).__name__)
    picked = [c for c in rng.permutation(cases) if (np.asarray(nib.load(str(R / "labelsTr" / f"{c}.nii.gz")).dataobj) > 0).sum() > 500][: a.n]
    fig, axes = plt.subplots(len(picked), 3, figsize=(10.5, 3.4 * len(picked)), facecolor="#fcfcfb")
    for i, case in enumerate(picked):
        img = zscore(np.asarray(nib.load(str(R / "imagesTr" / f"{case}_0000.nii.gz")).dataobj).astype(np.float32))
        seg = np.asarray(nib.load(str(R / "labelsTr" / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
        box = patch_box(seg, size); xi = img[box]; si = seg[box]; z = int(np.argmax((si > 0).sum((0, 1)))); zb = zoom_box(si[:, :, z] > 0, rng)
        x = torch.from_numpy(xi)[None, None].to(dev); y = torch.from_numpy(si.astype(np.float32))[None, None].to(dev)
        outs = {}
        for k, pipe in pipes.items():
            seed = 100 * i; torch.manual_seed(seed); np.random.seed(seed)
            with torch.no_grad():
                xo, yo = pipe(x.clone(), y.clone())
            outs[k] = (xo[0, 0].float().cpu().numpy(), yo[0, 0].round().cpu().numpy(), seed)
        show(axes[i, 0], xi[:, :, z][zb], si[:, :, z][zb] > 0, f"{case}: FLAIR training patch")
        for j, k in enumerate(ARMS):
            xo, yo, seed = outs[k]; show(axes[i, j + 1], xo[:, :, z][zb], yo[:, :, z][zb] > 0, f"{k} synthesis (seed {seed})")
        print(case, flush=True)
    fig.suptitle("Open-MS FLAIR training patches as seen by the two models (orange = lesion GT; random-offset 96-vox zoom)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); out = OUT / "viz_fill_training_samples_openms.png"; fig.savefig(out, dpi=130, facecolor="#fcfcfb"); print("wrote", out)


if __name__ == "__main__":
    main()
