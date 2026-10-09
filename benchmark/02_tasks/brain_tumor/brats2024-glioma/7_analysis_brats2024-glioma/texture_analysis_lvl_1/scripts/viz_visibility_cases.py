#!/usr/bin/env python
"""
Case sheets for the visibility story (Paul, 2026-10-07): one PNG per (train -> eval) cell, 3 test patients, 7 panels each:
  training-contrast image of the SAME patient | noise-fill synthesis of it | real-fill synthesis of it |
  eval image | eval + GT | noise-fill prediction | real-fill prediction   (fold 0, val000 rungs; slice Dice in titles)
Synthesis = the real AugLab rung-4 / rung-5 pipelines (configs of the training wrappers) on the nnU-Net-normalised
training-contrast volume (plans patch around the target), synthesis probability forced to 1, same seed for both arms.
Zoom: random-offset 96-vox window containing the target (eval slice); the training-contrast panels use the same window
when the two contrasts share the grid (BraTS, Open-MS), otherwise their own target-based window.
Cells: open-ms flair->t1w (exception), brats t2w->t1c (exception), ispy2 t1wce->t2w (exception), brats t2w->t1n (rule holds).
Output: outputs/plots/visibility_cases/<cell>.png.   GPU job: .venv/bin/python viz_visibility_cases.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS = Path(__file__).resolve().parent; REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(THIS))
from auglab.transforms.gpu.transforms import AugTransformsGPU  # noqa: E402
from trainme_build import zscore, patch_box, zoom_box, CFG, ARMS  # noqa: E402
from viz_visibility_examples import show, inset, dice, WIN  # noqa: E402
OUT = THIS.parent / "outputs/plots/visibility_cases"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

B = T2 / "brain_tumor/brats2024-glioma"; RB = B / "2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"
O = T2 / "brain_ms/open-ms"; RO = O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
I = T2 / "breast_cancer/ispy2"; RI = I / "2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce"
CELLS = [
    dict(name="open-ms_flair_to_t1w_EXCEPTION_helps", title="Open-MS, FLAIR-trained -> T1w: lesion less visible (0.89 -> 0.62) yet real-fill HELPS (+6.2)", direction="help",
         plans=O / "2_nnUNet_open-ms/preprocessed/Dataset070_OpenMS_FLAIR/nnUNetPlans.json", lid=1, target="lesion", train_lab="FLAIR", eval_lab="T1w",
         train_img=lambda c: RO / "imagesTs_flair" / f"{c}_0000.nii.gz", train_gt=lambda c: RO / "labelsTs_flair" / f"{c}.nii.gz",
         eval_img=lambda c: RO / "imagesTs_t1w" / f"{c}_0000.nii.gz", eval_gt=lambda c: RO / "labelsTs_t1w" / f"{c}.nii.gz",
         pred=lambda arm, c: O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet" / {"noise": "open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_20260711_062857", "real": "open-ms_flair_v26_6_2_train050_val000_20260711_062946"}[arm] / "fold0/t1w" / f"{c}.nii.gz",
         metrics=lambda arm: O / "8_results_open-ms/02_metrics/open_ms_model/flair/ablations" / {"noise": "nnUNet_open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_20260711_062857", "real": "nnUNet_open-ms_flair_v26_6_2_train050_val000_20260711_062946"}[arm] / "fold0/eval_all.csv", group="t1w", label="lesion"),
    dict(name="brats_t2w_to_t1c_EXCEPTION_helps", title="BraTS, T2w-trained -> T1c edema: less visible (0.67 -> 0.62) yet real-fill HELPS (+8.6)", direction="help",
         plans=B / "2_nnUNet_brats2024-glioma/preprocessed/Dataset052_BraTS2024GliomaT2w/nnUNetPlans.json", lid=2, target="edema", train_lab="T2w", eval_lab="T1c",
         train_img=lambda c: RB / "imagesTs_t2w" / f"{c}_0000.nii.gz", train_gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz",
         eval_img=lambda c: RB / "imagesTs_t1c" / f"{c}_0000.nii.gz", eval_gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz",
         pred=lambda arm, c: B / "8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab" / {"noise": "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}[arm] / "fold0/t1c" / f"{c}.nii.gz",
         metrics=lambda arm: B / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w/ablations" / {"noise": "auglab_brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "auglab_brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}[arm] / "fold0/eval_all.csv", group="t1c", label="SNFH"),
    dict(name="ispy2_t1wce_to_t2w_EXCEPTION_helps", title="I-SPY2, T1wce-trained -> T2w: tumour less visible (0.94 -> 0.79) yet real-fill HELPS (+1.2 pooled)", direction="help",
         plans=I / "2_nnUNet_ispy2/preprocessed/Dataset100_ISPY2T1wce/nnUNetPlans.json", lid=1, target="tumour", train_lab="T1wce", eval_lab="T2w",
         train_img=lambda c: RI / "imagesTs_t1wce" / f"{c}_0000.nii.gz", train_gt=lambda c: RI / "labelsTs_t1wce" / f"{c}.nii.gz",
         eval_img=lambda c: RI / "imagesTs_t2w" / f"{c}_0000.nii.gz", eval_gt=lambda c: RI / "labelsTs_t2w" / f"{c}.nii.gz",
         pred=lambda arm, c: I / "8_results_ispy2/01_predictions/ispy2_model/t1wce/auglab" / {"noise": "ispy2_t1wce_baseline_kmeans_label_remap_voronoi_20260905_163655", "real": "ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"}[arm] / "fold0/t2w" / f"{c}.nii.gz",
         metrics=lambda arm: I / "8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations" / {"noise": "auglab_ispy2_t1wce_baseline_kmeans_label_remap_voronoi_20260905_163655", "real": "auglab_ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"}[arm] / "fold0/eval_all.csv", group="t2w", label="tumour"),
    dict(name="brats_t2w_to_t1n_RULE_hurts", title="BraTS, T2w-trained -> T1n edema: less visible (0.67 -> 0.56) and real-fill HURTS (-11.9): the rule holds", direction="hurt",
         plans=B / "2_nnUNet_brats2024-glioma/preprocessed/Dataset052_BraTS2024GliomaT2w/nnUNetPlans.json", lid=2, target="edema", train_lab="T2w", eval_lab="T1n",
         train_img=lambda c: RB / "imagesTs_t2w" / f"{c}_0000.nii.gz", train_gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz",
         eval_img=lambda c: RB / "imagesTs_t1n" / f"{c}_0000.nii.gz", eval_gt=lambda c: RB / "labelsTr" / f"{c}.nii.gz",
         pred=lambda arm, c: B / "8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab" / {"noise": "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}[arm] / "fold0/t1n" / f"{c}.nii.gz",
         metrics=lambda arm: B / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w/ablations" / {"noise": "auglab_brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "auglab_brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}[arm] / "fold0/t1n" / f"{c}.nii.gz" if False else B / "8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w/ablations" / {"noise": "auglab_brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "auglab_brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}[arm] / "fold0/eval_all.csv", group="t1n", label="SNFH"),
]


def L(p):
    return np.asarray(nib.load(str(p)).dataobj)


def pick_cases(cell, k=3):
    ev = {}
    for arm in ("noise", "real"):
        df = pd.read_csv(cell["metrics"](arm)); df = df[(df.group == cell["group"]) & (df.label == cell["label"])]; ev[arm] = df.set_index("case").dice
    d = (ev["real"] - ev["noise"]).dropna().sort_values(ascending=(cell["direction"] == "hurt"))
    return list(d.index[:k]), d


def main():
    dev = torch.device("cuda")
    pipes = {k: AugTransformsGPU(json_path=str(CFG / v), num_labels=5).to(dev).eval() for k, v in ARMS.items()}
    for pipe in pipes.values():
        for m in pipe.modules():
            if "V26_6_2" in type(m).__name__ and hasattr(m, "p"):
                m.p = 1.0
    for cell in CELLS:
        size = json.load(open(cell["plans"]))["configurations"]["3d_fullres"]["patch_size"]
        cases, d = pick_cases(cell); print(cell["name"], {c: round(float(d[c]), 3) for c in cases}, flush=True)
        fig, axes = plt.subplots(len(cases), 7, figsize=(22, 3.3 * len(cases)), facecolor="#fcfcfb")
        for r, case in enumerate(cases):
            ti = L(cell["eval_img"](case)).astype(np.float32); g = L(cell["eval_gt"](case)).round().astype(int) == cell["lid"]
            pn = L(cell["pred"]("noise", case)).round().astype(int) == cell["lid"]; pr = L(cell["pred"]("real", case)).round().astype(int) == cell["lid"]
            z = int(np.argmax(g.sum((0, 1)))); zb = zoom_box(g[:, :, z], rng)
            tr = zscore(L(cell["train_img"](case)).astype(np.float32)); tg = L(cell["train_gt"](case)).round().astype(int)
            same = tr.shape == ti.shape
            # synthesis of the training-contrast volume around the target (plans patch)
            box = patch_box(tg == cell["lid"], size) if (tg == cell["lid"]).any() else patch_box(tg > 0, size)
            x = torch.from_numpy(tr[box])[None, None].to(dev); y = torch.from_numpy(tg[box].astype(np.float32))[None, None].to(dev)
            syn = {}
            for k, pipe in pipes.items():
                torch.manual_seed(11 * r + 1); np.random.seed(11 * r + 1)
                with torch.no_grad():
                    xo, yo = pipe(x.clone(), y.clone())
                syn[k] = (xo[0, 0].float().cpu().numpy(), yo[0, 0].round().cpu().numpy() == cell["lid"])
            tgm = tg[box] == cell["lid"]
            if same:
                zt = z - box[2].start; zbt = tuple(slice(s.start - b.start, s.stop - b.start) for s, b in zip(zb, box[:2]))
                if not (0 <= zt < tgm.shape[2]) or min(zbt[0].start, zbt[1].start) < 0 or zbt[0].stop > tgm.shape[0] or zbt[1].stop > tgm.shape[1]:
                    same = False
            if not same:
                zt = int(np.argmax(tgm.sum((0, 1)))); zbt = zoom_box(tgm[:, :, zt], rng)
            show(axes[r, 0], tr[box][:, :, zt][zbt], None, title=f"{case}: {cell['train_lab']} (training contrast{'' if same else ', own grid'})")
            axes[r, 0].contour((tgm[:, :, zt][zbt]).T.astype(float), levels=[0.5], colors=["#eb6834"], linewidths=0.8)
            for j, k in enumerate(ARMS):
                xo, yo = syn[k]; show(axes[r, 1 + j], xo[:, :, zt][zbt], None, title=f"{k.split(' ')[0]} synthesis of it")
                axes[r, 1 + j].contour((yo[:, :, zt][zbt]).T.astype(float), levels=[0.5], colors=["#eb6834"], linewidths=0.8)
            show(axes[r, 3], ti[:, :, z][zb], None, title=f"{cell['eval_lab']} (eval image)"); inset(axes[r, 3], ti[:, :, z], zb)
            show(axes[r, 4], ti[:, :, z][zb], g[:, :, z][zb], title=f"GT {cell['target']}")
            show(axes[r, 5], ti[:, :, z][zb], pn[:, :, z][zb], "#2a78d6", title=f"noise-fill pred, case Dice {dice(pn, g):.2f}")
            show(axes[r, 6], ti[:, :, z][zb], pr[:, :, z][zb], "#2a78d6", title=f"real-fill pred, case Dice {dice(pr, g):.2f}")
        fig.suptitle(cell["title"] + "   (orange contour = target label in the synthesis panels)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"{cell['name']}.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig); print("wrote", cell["name"], flush=True)


if __name__ == "__main__":
    main()
