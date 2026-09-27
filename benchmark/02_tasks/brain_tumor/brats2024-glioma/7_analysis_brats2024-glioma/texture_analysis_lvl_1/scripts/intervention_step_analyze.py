#!/usr/bin/env python
"""
Analysis for the visibility-step intervention (V1 S-ADD dose-response, V2/V3 S-REMOVE, sham).
See intervention_step_build.py's docstring for the pre-registration.

DiD = delta_real - delta_noise (same convention as the transplant round).

Writes:
  outputs/tables/intervention_step.md
  outputs/plots/intervention_step.png (dose-response lines, noise vs real, over k)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

PROJECT = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
DS = PROJECT / "benchmark" / "02_tasks" / "brain_tumor" / "brats2024-glioma"
LABELS_DIR = DS / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset051_BraTS2024GliomaT1n" / "labelsTr"

ANALYSIS_DIR = DS / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1"
OUT_DATA = ANALYSIS_DIR / "outputs" / "data"
OUT_TABLES = ANALYSIS_DIR / "outputs" / "tables"
OUT_PLOTS = ANALYSIS_DIR / "outputs" / "plots"
sys.path.insert(0, str(PROJECT / "benchmark/00_commun_scripts/00_00_utils"))
from stat_tests import wilcoxon_p  # noqa: E402

SCRATCH_ROOT = Path("/scratch/paulh/brats_intervention")
PREDS_ROOT = SCRATCH_ROOT / "preds"
INPUTS_ROOT = SCRATCH_ROOT / "inputs"

SNFH = 2
MODEL_KEYS = {("t1n", "noise"): "t1n_noise", ("t1n", "real"): "t1n_real",
              ("t2w", "noise"): "t2w_noise", ("t2w", "real"): "t2w_real"}
K_TAGS = {0.5: "p050", 1.0: "p100", -0.5: "m050", -1.0: "m100"}

V2_EXPS = {
    "V2_sremove_t2w_indomain": ("t2w", "orig_t2w", "v2_sremove_t2w", "sham_v2_sremove_t2w"),
    "V2_sremove_t2f_cross": ("t2w", "orig_t2f", "v2_sremove_t2f", "sham_v2_sremove_t2f"),
    "V3_sremove_t2f_t1n_trained": ("t1n", "orig_t2f", "v2_sremove_t2f", None),
}


def dice_recall(gt, pred, lid):
    G, P = gt == lid, pred == lid
    ng, npred = int(G.sum()), int(P.sum())
    if ng == 0:
        return float("nan"), float("nan")
    tp = int((G & P).sum())
    return (2 * tp / (ng + npred) if (ng + npred) else float("nan")), tp / ng


def load_pred(model_key, set_name, case):
    f = PREDS_ROOT / model_key / set_name / f"{case}.nii.gz"
    return np.asarray(nib.load(str(f)).dataobj).round().astype(np.int16) if f.exists() else None


def main() -> None:
    manifest = pd.read_csv(OUT_DATA / "intervention_step_manifest.csv")
    ok_cases = set(manifest[manifest["ok"]]["case"])
    sham_ok_cases = set(manifest[manifest["sham_status"] != "failed"]["case"]) if "sham_status" in manifest else set()
    all_cases = sorted(pd.read_csv(OUT_DATA / "patient_region_deltas.csv")["case"].unique())[:30]
    valid = ok_cases & set(all_cases)
    sham_valid = sham_ok_cases & set(all_cases)

    rows = []
    # V1 dose-response (t2w-trained models on S-ADD(t1n, k)), + sham
    for k, tag in K_TAGS.items():
        set_name = f"v1_sadd_t1n_k{tag}"
        sham_set = f"sham_v1_sadd_t1n_k{tag}"
        for rung in ("noise", "real"):
            model_key = MODEL_KEYS[("t2w", rung)]
            for case in valid:
                gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                pred_o = load_pred(model_key, "orig_t1n", case)
                pred_v = load_pred(model_key, set_name, case)
                if pred_o is None or pred_v is None:
                    continue
                d_o, r_o = dice_recall(gt, pred_o, SNFH)
                d_v, r_v = dice_recall(gt, pred_v, SNFH)
                rows.append(dict(experiment="V1_sadd_t1n", k=k, train="t2w", rung=rung, case=case,
                                  dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                  delta_dice=d_v - d_o, delta_recall=r_v - r_o, new_fp=np.nan))
            for case in sham_valid:
                gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                pred_o = load_pred(model_key, "orig_t1n", case)
                pred_v = load_pred(model_key, sham_set, case)
                if pred_o is None or pred_v is None:
                    continue
                d_o, r_o = dice_recall(gt, pred_o, SNFH)
                d_v, r_v = dice_recall(gt, pred_v, SNFH)
                new_fp = int(((pred_v == SNFH) & (pred_o != SNFH) & (gt == 0)).sum())
                rows.append(dict(experiment="V1_sadd_t1n_SHAM", k=k, train="t2w", rung=rung, case=case,
                                  dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                  delta_dice=d_v - d_o, delta_recall=r_v - r_o, new_fp=new_fp))

    # V2/V3
    for exp_name, (train, orig_set, var_set, sham_set) in V2_EXPS.items():
        for rung in ("noise", "real"):
            model_key = MODEL_KEYS[(train, rung)]
            for case in valid:
                gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                pred_o = load_pred(model_key, orig_set, case)
                pred_v = load_pred(model_key, var_set, case)
                if pred_o is None or pred_v is None:
                    continue
                d_o, r_o = dice_recall(gt, pred_o, SNFH)
                d_v, r_v = dice_recall(gt, pred_v, SNFH)
                rows.append(dict(experiment=exp_name, k=np.nan, train=train, rung=rung, case=case,
                                  dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                  delta_dice=d_v - d_o, delta_recall=r_v - r_o, new_fp=np.nan))
            if sham_set is not None:
                for case in sham_valid:
                    gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                    pred_o = load_pred(model_key, orig_set, case)
                    pred_v = load_pred(model_key, sham_set, case)
                    if pred_o is None or pred_v is None:
                        continue
                    d_o, r_o = dice_recall(gt, pred_o, SNFH)
                    d_v, r_v = dice_recall(gt, pred_v, SNFH)
                    new_fp = int(((pred_v == SNFH) & (pred_o != SNFH) & (gt == 0)).sum())
                    rows.append(dict(experiment=exp_name + "_SHAM", k=np.nan, train=train, rung=rung, case=case,
                                      dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                      delta_dice=d_v - d_o, delta_recall=r_v - r_o, new_fp=new_fp))

    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(OUT_DATA / "intervention_step_per_patient.csv", index=False)

    summary_rows = []
    for (exp_name, k), sub in per_patient.groupby(["experiment", "k"], dropna=False):
        noise = sub[sub["rung"] == "noise"].set_index("case")
        real = sub[sub["rung"] == "real"].set_index("case")
        common = noise.index.intersection(real.index)
        noise, real = noise.loc[common], real.loc[common]
        for metric in ("dice", "recall"):
            dn = noise[f"delta_{metric}"].to_numpy()
            dr = real[f"delta_{metric}"].to_numpy()
            v = ~np.isnan(dn) & ~np.isnan(dr)
            dn_v, dr_v = dn[v], dr[v]
            p = wilcoxon_p(dr_v, dn_v) if v.sum() >= 1 else float("nan")
            summary_rows.append(dict(
                experiment=exp_name, k=k, metric=metric, n=int(v.sum()),
                abs_dice_orig_noise=float(noise["dice_orig"].median()), abs_dice_var_noise=float(noise["dice_var"].median()),
                abs_recall_orig_noise=float(noise["recall_orig"].median()), abs_recall_var_noise=float(noise["recall_var"].median()),
                abs_dice_orig_real=float(real["dice_orig"].median()), abs_dice_var_real=float(real["dice_var"].median()),
                abs_recall_orig_real=float(real["recall_orig"].median()), abs_recall_var_real=float(real["recall_var"].median()),
                median_delta_noise=float(np.nanmedian(dn_v)) if v.sum() else np.nan,
                median_delta_real=float(np.nanmedian(dr_v)) if v.sum() else np.nan,
                did_real_minus_noise=float(np.nanmedian(dr_v - dn_v)) if v.sum() else np.nan,
                wilcoxon_p=p,
                median_new_fp_noise=float(noise["new_fp"].median()) if not noise["new_fp"].isna().all() else np.nan,
                median_new_fp_real=float(real["new_fp"].median()) if not real["new_fp"].isna().all() else np.nan,
            ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / "intervention_step_raw.csv", index=False)

    # ---- markdown report ----
    lines = ["# Visibility-step intervention (S-ADD / S-REMOVE)\n",
             "Fold 0, same 30 patients. t2w-trained models unless noted (V3 = t1n-trained). "
             "DiD = delta_real - delta_noise. Absolute Dice/recall are MEDIANS per condition.\n"]

    lines.append("## V1: S-ADD(t1n, k) dose-response, t2w-trained models\n")
    lines.append("| k | metric | n | noise: orig->var (abs) | real: orig->var (abs) | delta_noise | delta_real | DiD | wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    v1 = summary[summary.experiment == "V1_sadd_t1n"].sort_values("k")
    for _, r in v1.iterrows():
        lines.append(f"| {r['k']:+.1f} | {r['metric']} | {r['n']} | "
                     f"{r['abs_dice_orig_noise' if r['metric']=='dice' else 'abs_recall_orig_noise']:.3f} -> "
                     f"{r['abs_dice_var_noise' if r['metric']=='dice' else 'abs_recall_var_noise']:.3f} | "
                     f"{r['abs_dice_orig_real' if r['metric']=='dice' else 'abs_recall_orig_real']:.3f} -> "
                     f"{r['abs_dice_var_real' if r['metric']=='dice' else 'abs_recall_var_real']:.3f} | "
                     f"{r['median_delta_noise']:.4f} | {r['median_delta_real']:.4f} | {r['did_real_minus_noise']:.4f} | {r['wilcoxon_p']:.3g} |")
    lines.append("")

    lines.append("## V1 SHAM: new false-positive SNFH voxels created (median)\n")
    lines.append("| k | metric | noise median new FP vox | real median new FP vox |")
    lines.append("|---|---|---|---|")
    v1s = summary[summary.experiment == "V1_sadd_t1n_SHAM"].sort_values("k")
    for _, r in v1s[v1s.metric == "dice"].iterrows():
        lines.append(f"| {r['k']:+.1f} | fp_count | {r['median_new_fp_noise']:.0f} | {r['median_new_fp_real']:.0f} |")
    lines.append("")

    lines.append("## V2/V3: S-REMOVE results\n")
    lines.append("| experiment | metric | n | noise: orig->var (abs) | real: orig->var (abs) | delta_noise | delta_real | DiD | wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for exp_name in ["V2_sremove_t2w_indomain", "V2_sremove_t2w_indomain_SHAM",
                     "V2_sremove_t2f_cross", "V2_sremove_t2f_cross_SHAM",
                     "V3_sremove_t2f_t1n_trained"]:
        sub = summary[summary.experiment == exp_name]
        for _, r in sub.iterrows():
            mkey = "dice" if r["metric"] == "dice" else "recall"
            lines.append(f"| {exp_name} | {r['metric']} | {r['n']} | "
                         f"{r[f'abs_{mkey}_orig_noise']:.3f} -> {r[f'abs_{mkey}_var_noise']:.3f} | "
                         f"{r[f'abs_{mkey}_orig_real']:.3f} -> {r[f'abs_{mkey}_var_real']:.3f} | "
                         f"{r['median_delta_noise']:.4f} | {r['median_delta_real']:.4f} | {r['did_real_minus_noise']:.4f} | {r['wilcoxon_p']:.3g} |")
    lines.append("")

    lines.append("## Verdict\n")
    for _, r in v1.iterrows():
        held = (r["did_real_minus_noise"] > 0) and (r["wilcoxon_p"] < 0.05)
        verdict = "HELD" if held else ("direction correct, not significant" if r["did_real_minus_noise"] > 0 else "DID NOT HOLD / reversed")
        lines.append(f"- V1 k={r['k']:+.1f} / {r['metric']}: DiD={r['did_real_minus_noise']:.4f}, p={r['wilcoxon_p']:.3g} -> {verdict}")
    for exp_name in ["V2_sremove_t2w_indomain", "V2_sremove_t2f_cross"]:
        sub = summary[summary.experiment == exp_name]
        for _, r in sub.iterrows():
            held = (r["did_real_minus_noise"] < 0) and (r["wilcoxon_p"] < 0.05)
            verdict = "HELD" if held else ("direction correct, not significant" if r["did_real_minus_noise"] < 0 else "DID NOT HOLD / reversed")
            lines.append(f"- {exp_name} / {r['metric']}: DiD={r['did_real_minus_noise']:.4f}, p={r['wilcoxon_p']:.3g} -> {verdict}")
    lines.append("\nV3 (t1n-trained models) has no pre-registered direction — reported as-is above.")
    lines.append("SHAM Dice/recall rows should be ~0; the false-positive-voxel table is the specificity/hallucination check.")

    (OUT_TABLES / "intervention_step.md").write_text("\n".join(lines))
    print("wrote", OUT_TABLES / "intervention_step.md")

    # ---- dose-response plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("dice", "recall")):
        sub = summary[summary.experiment == "V1_sadd_t1n"].sort_values("k")
        sub_m = sub[sub.metric == metric]
        ax.plot(sub_m["k"], sub_m["median_delta_noise"], "o-", label="noise-fill", color="#d95f02")
        ax.plot(sub_m["k"], sub_m["median_delta_real"], "o-", label="real-fill", color="#1b9e77")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("k (S-ADD dose, x ring std)")
        ax.set_ylabel(f"median delta {metric} vs original t1n")
        ax.set_title(f"V1 dose-response: {metric}")
        ax.legend()
    fig.suptitle("S-ADD(t1n, k) dose-response, t2w-trained models")
    fig.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOTS / "intervention_step.png", dpi=140)
    print("wrote", OUT_PLOTS / "intervention_step.png")


if __name__ == "__main__":
    main()
