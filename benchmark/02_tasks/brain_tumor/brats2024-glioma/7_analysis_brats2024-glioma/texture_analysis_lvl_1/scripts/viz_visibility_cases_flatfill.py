#!/usr/bin/env python
"""
Case sheets for the cells where the visibility sign rule is WRONG when the noise arm is rung 4.5 (FLAT fill, 2026-10-09):
same 7 panels as viz_visibility_cases.py, with the flat-fill pipeline / predictions in place of the noise-fill ones:
  training-contrast image | flat-fill synthesis of it | real-fill synthesis of it |
  eval image | eval + GT | flat-fill prediction | real-fill prediction   (fold 0; case Dice in titles)
Cells: open-ms flair->t2w (gap -0.11, real fill HELPS +10.6), ispy2 t1wce->t2w (gap -0.15, HELPS +0.9),
       pelvis mri->ct hip_left (gap +0.23, real fill HURTS -3.9), pelvis mri->ct gluteus_maximus_left (gap +0.08, HURTS -5.2).
Pelvis CT and MRI are DIFFERENT subjects, so the training-contrast panels show a random MRI training case (own grid).
Output: outputs/plots/visibility_cases_flatfill/<cell>.png.   GPU job: .venv/bin/python viz_visibility_cases_flatfill.py
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
from trainme_build import zscore, patch_box, zoom_box, CFG  # noqa: E402
from viz_visibility_examples import show, inset, dice  # noqa: E402
from viz_visibility_cases import L, pick_cases  # noqa: E402
OUT = THIS.parent / "outputs/plots/visibility_cases_flatfill"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
ARMS = {"flat-fill (rung 4.5)": "transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json",
        "real-fill (rung 5)": "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"}

O = T2 / "brain_ms/open-ms"; RO = O / "2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR"
I = T2 / "breast_cancer/ispy2"; RI = I / "2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce"
P = T2 / "pelvis_healthy/totalseg-pelvic"; RPM = P / "2_nnUNet_totalseg-pelvic/raw/Dataset131_TotalsegPelvic_MRI"; RPC = P / "2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT"
def _iso(p, ratio=1.6):   # near-isotropic training MRIs only (TotalSeg MRI mixes thick-slice stacks)
    z = nib.load(str(p)).header.get_zooms()[:3]; return max(z) / min(z) <= ratio
_VOL = {}
def _big_train(lid):
    if lid not in _VOL:
        v = {c: int((np.asarray(nib.load(str(RPM / "labelsTr" / f"{c}.nii.gz")).dataobj) == lid).sum()) for c in PELVIS_TRAIN}
        _VOL[lid] = sorted(v, key=v.get, reverse=True)
    return _VOL[lid]
PELVIS_TRAIN = sorted(p.name.replace("_0000.nii.gz", "") for p in (RPM / "imagesTr").glob("*_0000.nii.gz") if _iso(p))
_pel = dict(plans=P / "2_nnUNet_totalseg-pelvic/preprocessed/Dataset131_TotalsegPelvic_MRI/nnUNetPlans.json", num_labels=12, train_lab="MRI", eval_lab="CT",
            train_case=lambda r, c, lid: _big_train(lid)[int(rng.integers(20))],   # random among the 20 training MRIs with the most of that label (small muscles sit at the MRI FOV edge)
            train_img=lambda c: RPM / "imagesTr" / f"{c}_0000.nii.gz", train_gt=lambda c: RPM / "labelsTr" / f"{c}.nii.gz",
            eval_img=lambda c: RPC / "imagesTs_ct" / f"{c}_0000.nii.gz", eval_gt=lambda c: RPC / "labelsTs_ct" / f"{c}.nii.gz",
            pred=lambda arm, c: P / "8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab" / {"noise": "totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257", "real": "totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058"}[arm] / "fold0/ct" / f"{c}.nii.gz",
            metrics=lambda arm: P / "8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri" / {"noise": "auglab_totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257", "real": "auglab_totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058"}[arm] / "fold0/eval_all.csv", group="ct")
CELLS = [
    dict(name="open-ms_flair_to_t2w_EXCEPTION_helps", title="Open-MS, FLAIR-trained -> T2w: lesion less visible (0.89 -> 0.78) yet real fill HELPS vs flat fill (+10.6)", direction="help",
         plans=O / "2_nnUNet_open-ms/preprocessed/Dataset070_OpenMS_FLAIR/nnUNetPlans.json", num_labels=5, lid=1, target="lesion", train_lab="FLAIR", eval_lab="T2w",
         train_img=lambda c: RO / "imagesTs_flair" / f"{c}_0000.nii.gz", train_gt=lambda c: RO / "labelsTs_flair" / f"{c}.nii.gz",
         eval_img=lambda c: RO / "imagesTs_t2w" / f"{c}_0000.nii.gz", eval_gt=lambda c: RO / "labelsTs_t2w" / f"{c}.nii.gz",
         pred=lambda arm, c: O / "8_results_open-ms/01_predictions/open_ms_model/flair/nnUNet" / {"noise": "open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831", "real": "open-ms_flair_v26_6_2_train050_val000_20260711_062946"}[arm] / "fold0/t2w" / f"{c}.nii.gz",
         metrics=lambda arm: O / "8_results_open-ms/02_metrics/open_ms_model/flair/ablations" / {"noise": "nnUNet_open-ms_flair_baseline_kmeans_label_remap_voronoi_train050_val000_flatfill_20261008_190831", "real": "nnUNet_open-ms_flair_v26_6_2_train050_val000_20260711_062946"}[arm] / "fold0/eval_all.csv", group="t2w", label="lesion"),
    dict(name="ispy2_t1wce_to_t2w_EXCEPTION_helps", title="I-SPY2, T1wce-trained -> T2w: tumour less visible (0.94 -> 0.79) yet real fill HELPS vs flat fill (+0.9)", direction="help",
         plans=I / "2_nnUNet_ispy2/preprocessed/Dataset100_ISPY2T1wce/nnUNetPlans.json", num_labels=5, lid=1, target="tumour", train_lab="T1wce", eval_lab="T2w",
         train_case=lambda r, c, lid: c if (RI / "imagesTs_t1wce" / f"{c}_0000.nii.gz").exists() else c.replace("_bil", "_uni"),   # T1wce test set has the uni crop for most patients
         train_img=lambda c: RI / "imagesTs_t1wce" / f"{c}_0000.nii.gz", train_gt=lambda c: RI / "labelsTs_t1wce" / f"{c}.nii.gz",
         eval_img=lambda c: RI / "imagesTs_t2w" / f"{c}_0000.nii.gz", eval_gt=lambda c: RI / "labelsTs_t2w" / f"{c}.nii.gz",
         pred=lambda arm, c: I / "8_results_ispy2/01_predictions/ispy2_model/t1wce/auglab" / {"noise": "ispy2_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191117", "real": "ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"}[arm] / "fold0/t2w" / f"{c}.nii.gz",
         metrics=lambda arm: I / "8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations" / {"noise": "auglab_ispy2_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191117", "real": "auglab_ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"}[arm] / "fold0/eval_all.csv", group="t2w", label="tumour"),
    dict(name="pelvis_mri_to_ct_hip_left_EXCEPTION_hurts", title="Pelvis, MRI-trained -> CT hip (left): MORE visible on CT (0.67 -> 0.91) yet real fill HURTS vs flat fill (-3.9)", direction="hurt",
         lid=1, target="hip_left", label="hip_left", **_pel),
    dict(name="pelvis_mri_to_ct_gluteus_maximus_left_EXCEPTION_hurts", title="Pelvis, MRI-trained -> CT gluteus maximus (left): more visible on CT (0.71 -> 0.80) yet real fill HURTS vs flat fill (-5.2)", direction="hurt",
         lid=4, target="gluteus_maximus_left", label="gluteus_maximus_left", **_pel),
]


def main():
    dev = torch.device("cuda")
    pipes_by_n = {}
    for cell in CELLS:
        n = cell["num_labels"]
        if n not in pipes_by_n:
            pipes_by_n[n] = {k: AugTransformsGPU(json_path=str(CFG / v), num_labels=n).to(dev).eval() for k, v in ARMS.items()}
            for pipe in pipes_by_n[n].values():
                for m in pipe.modules():
                    if "V26_6_2" in type(m).__name__ and hasattr(m, "p"):
                        m.p = 1.0
    for cell in CELLS:
        pipes = pipes_by_n[cell["num_labels"]]
        plans = json.load(open(cell["plans"])); ps = plans["configurations"]["3d_fullres"]["patch_size"]
        size = [0, 0, 0]   # plans patch is in nnU-Net's transposed axes -> back to raw array axes
        for i, ax in enumerate(plans["transpose_forward"]):
            size[ax] = ps[i]
        cases, d = pick_cases(cell); print(cell["name"], {c: round(float(d[c]), 3) for c in cases}, flush=True)
        fig, axes = plt.subplots(len(cases), 7, figsize=(22, 3.3 * len(cases)), facecolor="#fcfcfb")
        for r, case in enumerate(cases):
            ti = L(cell["eval_img"](case)).astype(np.float32); g = L(cell["eval_gt"](case)).round().astype(int) == cell["lid"]
            pn = L(cell["pred"]("noise", case)).round().astype(int) == cell["lid"]; pr = L(cell["pred"]("real", case)).round().astype(int) == cell["lid"]
            z = int(np.argmax(g.sum((0, 1)))); zb = zoom_box(g[:, :, z], rng)
            tcase = cell["train_case"](r, case, cell["lid"]) if "train_case" in cell else case
            tr = zscore(L(cell["train_img"](tcase)).astype(np.float32)); tg = L(cell["train_gt"](tcase)).round().astype(int)
            same = tcase == case and tr.shape == ti.shape
            who = "" if tcase == case else f"\n(training case {tcase}, different subject)"
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
            show(axes[r, 0], tr[box][:, :, zt][zbt], None, title=f"{case}: {cell['train_lab']} (training contrast{'' if same else ', own grid'}){who}")
            axes[r, 0].contour((tgm[:, :, zt][zbt]).T.astype(float), levels=[0.5], colors=["#eb6834"], linewidths=0.8)
            for j, k in enumerate(ARMS):
                xo, yo = syn[k]; show(axes[r, 1 + j], xo[:, :, zt][zbt], None, title=f"{k.split(' ')[0]} synthesis of it")
                axes[r, 1 + j].contour((yo[:, :, zt][zbt]).T.astype(float), levels=[0.5], colors=["#eb6834"], linewidths=0.8)
            show(axes[r, 3], ti[:, :, z][zb], None, title=f"{cell['eval_lab']} (eval image)"); inset(axes[r, 3], ti[:, :, z], zb)
            show(axes[r, 4], ti[:, :, z][zb], g[:, :, z][zb], title=f"GT {cell['target']}")
            show(axes[r, 5], ti[:, :, z][zb], pn[:, :, z][zb], "#2a78d6", title=f"flat-fill pred, case Dice {dice(pn, g):.2f}")
            show(axes[r, 6], ti[:, :, z][zb], pr[:, :, z][zb], "#2a78d6", title=f"real-fill pred, case Dice {dice(pr, g):.2f}")
        fig.suptitle(cell["title"] + "   (orange contour = target label in the synthesis panels)", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(OUT / f"{cell['name']}.png", dpi=120, facecolor="#fcfcfb"); plt.close(fig); print("wrote", cell["name"], flush=True)


if __name__ == "__main__":
    main()
