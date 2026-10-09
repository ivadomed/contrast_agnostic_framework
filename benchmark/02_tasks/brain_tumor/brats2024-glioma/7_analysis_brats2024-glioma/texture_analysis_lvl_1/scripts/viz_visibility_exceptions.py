#!/usr/bin/env python
"""Exceptions to the visibility rule (eval contrast harder, yet real-fill helps): BraTS T2w-trained -> T1c edema (+8.6,
gap -0.05; the SAME model that loses on T1n) and I-SPY2 T1wce-trained -> T2w (+1.2 pooled, gap -0.15). Same panel layout as
viz_visibility_examples.py (imported). Output: outputs/plots/viz_visibility_exceptions_{brats_t1c,ispy2_t2w}.png"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
from viz_visibility_examples import panel, OUT, T2  # noqa: E402


def main():
    B = T2 / "brain_tumor/brats2024-glioma"; RB = B / "2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w"; PB = B / "8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab"
    runs = {"noise": "brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659", "real": "brats2024-glioma_t2w_v26_6_2_train050_val000_20261005_222300"}
    dd = pd.read_csv(THIS.parent / "outputs/data/patient_region_deltas.csv"); dd = dd[(dd.train == "t2w") & (dd["eval"] == "t1c") & (dd.region == "SNFH")].sort_values("delta_dice", ascending=False)
    pick = list(dd.case.head(3)); print("brats t1c picks", dd.head(3)[["case", "delta_dice"]].round(3).to_dict("records"))
    panel("EXCEPTION: BraTS, the same T2w-trained model tested on T1c: edema visibility drops (0.67 -> 0.62) yet real-fill HELPS (+8.6 Dice)",
          lambda c: RB / "imagesTs_t2w" / f"{c}_0000.nii.gz", lambda c: RB / "imagesTs_t1c" / f"{c}_0000.nii.gz", lambda c: RB / "labelsTr" / f"{c}.nii.gz",
          lambda c: PB / runs["noise"] / "fold0/t1c" / f"{c}.nii.gz", lambda c: PB / runs["real"] / "fold0/t1c" / f"{c}.nii.gz", 2, pick, "T2w", "T1c", "edema", OUT / "viz_visibility_exceptions_brats_t1c.png")
    I = T2 / "breast_cancer/ispy2"; RI = I / "2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce"; PI = I / "8_results_ispy2/01_predictions/ispy2_model/t1wce/auglab"
    M = I / "8_results_ispy2/02_metrics/ispy2_model/t1wce/ablations"
    rn, rr = "ispy2_t1wce_baseline_kmeans_label_remap_voronoi_20260905_163655", "ispy2_t1wce_v26_6_2_train050_val000_20261005_222919"
    ev = {}
    for tag, run in (("noise", rn), ("real", rr)):
        df = pd.read_csv(M / f"auglab_{run}/fold0/eval_all.csv"); ev[tag] = df[df.group == "t2w"].set_index("case").dice
    d = (ev["real"] - ev["noise"]).dropna().sort_values(ascending=False); pick = list(d.index[:3]); print("ispy2 picks", d.head(3).round(3).to_dict())
    panel("EXCEPTION: I-SPY2, T1wce-trained model tested on T2w: tumour visibility drops (0.94 -> 0.79) yet real-fill HELPS (+1.2 Dice pooled; 3 largest per-case gains shown)",
          lambda c: RI / "imagesTs_t1wce" / f"{c}_0000.nii.gz", lambda c: RI / "imagesTs_t2w" / f"{c}_0000.nii.gz", lambda c: RI / "labelsTs_t2w" / f"{c}.nii.gz",
          lambda c: PI / rn / "fold0/t2w" / f"{c}.nii.gz", lambda c: PI / rr / "fold0/t2w" / f"{c}.nii.gz", 1, pick, "T1wce", "T2w", "tumour", OUT / "viz_visibility_exceptions_ispy2_t2w.png")


if __name__ == "__main__":
    main()
