#!/usr/bin/env python
"""Concrete examples for the visibility story (2026-10-07): the lesion/edema becomes LESS visible from the training contrast
to the test contrast in both panels, yet real-fill HELPS on Open-MS FLAIR->T1w and HURTS on BraTS T2w->T1n edema.
Rows = test cases, columns = training-contrast image of the same patient, test image, GT, noise-fill prediction, real-fill
prediction (fold 0, val000 rungs), slice through the largest target section, random-offset 96-vox zoom + full-slice inset.
Output: outputs/plots/viz_visibility_examples_{openms,brats}.png. CPU, light."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"; OUT = THIS.parent / "outputs/plots"
rng = np.random.default_rng(3)
WIN = 96


def zoom_box(m, win=WIN):
    H, W = m.shape; idx = np.argwhere(m); out = []
    for lo, hi, n in ((idx[:, 0].min(), idx[:, 0].max() + 1, H), (idx[:, 1].min(), idx[:, 1].max() + 1, W)):
        a, b = max(0, hi - win), min(lo, n - win)
        if a > b:
            a, b = max(0, lo - win // 2), min(hi - win // 2, n - win)
        s = int(rng.integers(min(a, b), max(a, b) + 1)); out.append(slice(s, s + win))
    return tuple(out)


def show(ax, g, m=None, col="#eb6834", title=""):
    fgm = g != 0; lo, hi = np.percentile(g[fgm], [1, 99]) if fgm.any() else (0, 1)
    ax.imshow(g.T, cmap="gray", vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
    if m is not None and m.any():
        rgba = np.zeros(m.T.shape + (4,)); rgba[m.T] = matplotlib.colors.to_rgba(col, 0.45); ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=9, loc="left"); ax.axis("off")


def inset(ax, full, zb):
    ia = ax.inset_axes([0.68, 0.68, 0.3, 0.3]); fgm = full != 0; lo, hi = np.percentile(full[fgm], [1, 99])
    ia.imshow(full.T, cmap="gray", vmin=lo, vmax=hi, origin="lower"); ia.add_patch(plt.Rectangle((zb[0].start, zb[1].start), WIN, WIN, fill=False, ec="#eda100", lw=1)); ia.axis("off")


def dice(a, b):
    s = a.sum() + b.sum(); return 2 * (a & b).sum() / s if s else np.nan


def panel(name, train_img, test_img, gt, pred_noise, pred_real, lid, cases, train_lab, test_lab, target, out):
    fig, axes = plt.subplots(len(cases), 5, figsize=(16, 3.3 * len(cases)), facecolor="#fcfcfb")
    for r, case in enumerate(cases):
        g = np.asarray(nib.load(str(gt(case))).dataobj).round().astype(int) == lid
        ti = np.asarray(nib.load(str(test_img(case))).dataobj).astype(np.float32); tr = np.asarray(nib.load(str(train_img(case))).dataobj).astype(np.float32)
        pn = np.asarray(nib.load(str(pred_noise(case))).dataobj).round().astype(int) == lid; pr = np.asarray(nib.load(str(pred_real(case))).dataobj).round().astype(int) == lid
        z = int(np.argmax(g.sum((0, 1)))); zb = zoom_box(g[:, :, z]); same = tr.shape == ti.shape
        show(axes[r, 0], tr[:, :, z][zb] if same else tr[:, :, int(np.argmax((tr > 0).sum((0, 1))))], None, title=f"{case}: {train_lab} (training contrast{'' if same else ', own grid'})")
        show(axes[r, 1], ti[:, :, z][zb], None, title=f"{test_lab} (test image)")
        show(axes[r, 2], ti[:, :, z][zb], g[:, :, z][zb], title=f"GT {target}")
        show(axes[r, 3], ti[:, :, z][zb], pn[:, :, z][zb], "#2a78d6", title=f"noise-fill pred, case Dice {dice(pn, g):.2f}")
        show(axes[r, 4], ti[:, :, z][zb], pr[:, :, z][zb], "#2a78d6", title=f"real-fill pred, case Dice {dice(pr, g):.2f}")
        inset(axes[r, 1], ti[:, :, z], zb)
    fig.suptitle(name, fontsize=11, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=130, facecolor="#fcfcfb"); plt.close(fig); print("wrote", out)


def main():
    OUT.mkdir(exist_ok=True)
    # --- Open-MS FLAIR-trained -> T1w (real-fill HELPS, +6.2 pooled)
    O = T2 / "brain_ms/open-ms"; R = O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"; P = O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet"
    d = json.load(open(O / "8_results_open-ms/02_metrics/open_ms_model/flair/ablations/ladder_series.json"))
    rn, rr = (Path(d["run_keys"][k]).name.replace("nnUNet_", "", 1) for k in (3, 4))
    cases = sorted(p.name.replace("_0000.nii.gz", "") for p in (R / "imagesTs_t1w").glob("*_0000.nii.gz"))
    # pick the 3 cases with the largest real-noise Dice gain on t1w from the metrics
    M = O / "8_results_open-ms/02_metrics/open_ms_model/flair/ablations"
    ev = {}
    for tag, run in (("noise", rn), ("real", rr)):
        f = next(M.glob(f"*{run}/fold0/eval_all.csv")); df = pd.read_csv(f); ev[tag] = df[df.group == "t1w"].set_index("case").dice
    delta = (ev["real"] - ev["noise"]).dropna().sort_values(ascending=False)
    pick = list(delta.index[:3]); print("open-ms picks", delta.head(3).round(3).to_dict())
    panel("Open-MS, FLAIR-trained model tested on T1w: lesion visibility drops (AUC 0.89 -> 0.62) and real-fill HELPS (+6.2 Dice pooled)",
          lambda c: R / "imagesTs_flair" / f"{c}_0000.nii.gz", lambda c: R / "imagesTs_t1w" / f"{c}_0000.nii.gz", lambda c: R / "labelsTs_t1w" / f"{c}.nii.gz",
          lambda c: P / rn / "fold0/t1w" / f"{c}.nii.gz", lambda c: P / rr / "fold0/t1w" / f"{c}.nii.gz", 1, pick, "FLAIR", "T1w", "lesion", OUT / "viz_visibility_examples_openms.png")
    # --- BraTS T2w-trained -> T1n edema (real-fill HURTS, -11.9)
    B = T2 / "brain_tumor/brats2024-glioma"; RB = B / "2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"; PB = B / "8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab"
    runs = {"noise": "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}
    dd = pd.read_csv(THIS.parent / "outputs/data/patient_region_deltas.csv"); dd = dd[(dd.train == "t2w") & (dd["eval"] == "t1n") & (dd.region == "SNFH")].sort_values("delta_dice")
    pick = list(dd.case.head(3)); print("brats picks", dd.head(3)[["case", "delta_dice"]].round(3).to_dict("records"))
    panel("BraTS, T2w-trained model tested on T1n: edema visibility drops (AUC 0.67 -> 0.56) and real-fill HURTS (-11.9 Dice)",
          lambda c: RB / "imagesTs_t2w" / f"{c}_0000.nii.gz", lambda c: RB / "imagesTs_t1n" / f"{c}_0000.nii.gz", lambda c: RB / "labelsTr" / f"{c}.nii.gz",
          lambda c: PB / runs["noise"] / "fold0/t1n" / f"{c}.nii.gz", lambda c: PB / runs["real"] / "fold0/t1n" / f"{c}.nii.gz", 2, pick, "T2w", "T1n", "edema", OUT / "viz_visibility_examples_brats.png")


if __name__ == "__main__":
    main()
