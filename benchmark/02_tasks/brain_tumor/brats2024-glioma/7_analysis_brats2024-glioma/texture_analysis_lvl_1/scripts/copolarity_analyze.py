#!/usr/bin/env python
"""
Combined Part A + Part B analysis/report for the co-polarity investigation. See
copolarity_partA_build.py and copolarity_partB_build.py for the pre-registrations.

Writes:
  outputs/tables/copolarity.md
  outputs/plots/copolarity.png (Part A cluster-membership sweep + Part B DiD comparison)
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
DS = PROJECT / "datasets" / "brats2024-glioma"
LABELS_DIR = DS / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset051_BraTS2024GliomaT1n" / "labelsTr"
ANALYSIS_DIR = DS / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1"
OUT_DATA = ANALYSIS_DIR / "outputs" / "data"
OUT_TABLES = ANALYSIS_DIR / "outputs" / "tables"
OUT_PLOTS = ANALYSIS_DIR / "outputs" / "plots"
sys.path.insert(0, str(PROJECT / "benchmark/00_commun_scripts/00_00_utils"))
from stat_tests import wilcoxon_p  # noqa: E402

SCRATCH_ROOT = Path("/scratch/paulh/brats_intervention")
PREDS_ROOT = SCRATCH_ROOT / "preds"
SNFH = 2


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


def compute_did(exp_name, orig_set, var_set, valid_cases, sham=False):
    rows = []
    for rung in ("noise", "real"):
        model_key = f"t2w_{rung}"
        for case in valid_cases:
            gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
            pred_o = load_pred(model_key, orig_set, case)
            pred_v = load_pred(model_key, var_set, case)
            if pred_o is None or pred_v is None:
                continue
            d_o, r_o = dice_recall(gt, pred_o, SNFH)
            d_v, r_v = dice_recall(gt, pred_v, SNFH)
            new_fp = np.nan
            if sham:
                new_fp = int(((pred_v == SNFH) & (pred_o != SNFH) & (gt == 0)).sum())
            rows.append(dict(rung=rung, case=case, dice_orig=d_o, recall_orig=r_o,
                              dice_var=d_v, recall_var=r_v,
                              delta_dice=d_v - d_o, delta_recall=r_v - r_o, new_fp=new_fp))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    noise = df[df.rung == "noise"].set_index("case")
    real = df[df.rung == "real"].set_index("case")
    common = noise.index.intersection(real.index)
    noise, real = noise.loc[common], real.loc[common]
    summary = {}
    for metric in ("dice", "recall"):
        dn = noise[f"delta_{metric}"].to_numpy()
        dr = real[f"delta_{metric}"].to_numpy()
        v = ~np.isnan(dn) & ~np.isnan(dr)
        p = wilcoxon_p(dr[v], dn[v]) if v.sum() >= 1 else np.nan
        summary[metric] = dict(
            n=int(v.sum()),
            abs_dice_orig_noise=float(noise["dice_orig"].median()), abs_dice_var_noise=float(noise["dice_var"].median()),
            abs_recall_orig_noise=float(noise["recall_orig"].median()), abs_recall_var_noise=float(noise["recall_var"].median()),
            abs_dice_orig_real=float(real["dice_orig"].median()), abs_dice_var_real=float(real["dice_var"].median()),
            abs_recall_orig_real=float(real["recall_orig"].median()), abs_recall_var_real=float(real["recall_var"].median()),
            median_delta_noise=float(np.nanmedian(dn[v])) if v.sum() else np.nan,
            median_delta_real=float(np.nanmedian(dr[v])) if v.sum() else np.nan,
            did=float(np.nanmedian(dr[v] - dn[v])) if v.sum() else np.nan,
            p=p,
            median_new_fp_noise=float(noise["new_fp"].median()) if sham else np.nan,
            median_new_fp_real=float(real["new_fp"].median()) if sham else np.nan,
        )
    return df, summary


def main() -> None:
    # ---------- Part A load ----------
    cluster_df = pd.read_csv(OUT_DATA / "copolarity_cluster_membership.csv")
    sign_df = pd.read_csv(OUT_DATA / "copolarity_sign_simulation.csv")
    cluster_summary = cluster_df.groupby("c")[[
        "frac_edema_same_cluster_as_ncr", "frac_edema_same_cluster_as_et",
        "frac_edema_same_cluster_as_normal"]].mean().reset_index()

    # ---------- Part B ----------
    manifest = pd.read_csv(OUT_DATA / "copolarity_partB_manifest.csv")
    ok_cases = set(manifest[manifest["ok"]]["case"])
    sham_ok_cases = set(manifest[manifest.get("sham_ok", False)]["case"]) if "sham_ok" in manifest else set()
    all_cases = sorted(pd.read_csv(OUT_DATA / "patient_region_deltas.csv")["case"].unique())[:30]
    valid = ok_cases & set(all_cases)
    sham_valid = sham_ok_cases & set(all_cases)

    exps = {
        "S-ADD(+1) alone": ("orig_t1n", "v1_sadd_t1n_kp100", valid, False),
        "S-ADD(+1) alone SHAM": ("orig_t1n", "sham_v1_sadd_t1n_kp100", sham_valid, True),
        "C-FLIP alone": ("orig_t1n", "corefeather_t1n", valid, False),
        "C-FLIP alone SHAM": ("orig_t1n", "sham_corefeather_t1n", sham_valid, True),
        "C-FLIP + S-ADD(+1) combined": ("orig_t1n", "corefeather_sadd_t1n", valid, False),
        "C-FLIP + S-ADD(+1) SHAM": ("orig_t1n", "sham_corefeather_sadd_t1n", sham_valid, True),
    }
    all_summaries = {}
    for name, (orig_set, var_set, cases, is_sham) in exps.items():
        _, summ = compute_did(name, orig_set, var_set, cases, sham=is_sham)
        all_summaries[name] = summ

    # DiD-of-DiD: combined minus S-ADD alone
    dod = {}
    for metric in ("dice", "recall"):
        if "S-ADD(+1) alone" in all_summaries and "C-FLIP + S-ADD(+1) combined" in all_summaries:
            a = all_summaries["S-ADD(+1) alone"].get(metric, {})
            b = all_summaries["C-FLIP + S-ADD(+1) combined"].get(metric, {})
            if a and b:
                dod[metric] = b["did"] - a["did"]

    # ---------- markdown ----------
    lines = ["# Co-polarity investigation (Part A: training-data analysis; Part B: intervention)\n"]

    lines.append("## Part A -- K-means cluster co-membership (mean over ~40 T2w training cases, "
                 "30 subsample repeats/case/c)\n")
    lines.append("| c (K-means clusters) | edema same cluster as NCR | edema same cluster as ET | edema same cluster as normal tissue |")
    lines.append("|---|---|---|---|")
    for _, r in cluster_summary.iterrows():
        lines.append(f"| {int(r['c'])} | {r['frac_edema_same_cluster_as_ncr']:.3f} | "
                     f"{r['frac_edema_same_cluster_as_et']:.3f} | {r['frac_edema_same_cluster_as_normal']:.3f} |")
    lines.append("")

    lines.append("## Part A -- end-to-end same-sign fraction (200 draws/case, full step1+step2 simulation)\n")
    lines.append(f"- edema-NCR same final sign: **{sign_df['frac_same_sign_edema_ncr'].mean():.3f}** "
                 f"(n_cases={int(sign_df['frac_same_sign_edema_ncr'].notna().sum())})")
    lines.append(f"- edema-ET same final sign: **{sign_df['frac_same_sign_edema_et'].mean():.3f}** "
                 f"(n_cases={int(sign_df['frac_same_sign_edema_et'].notna().sum())})")
    lines.append(f"- Part A prediction (same-sign fraction substantially > 0.5): "
                 f"{'HELD' if sign_df['frac_same_sign_edema_ncr'].mean() > 0.6 else 'weak/not clearly held'}\n")

    lines.append("## Part B -- absolute Dice/recall and DiD (t2w-trained models, T1n input, median per condition)\n")
    lines.append("| condition | metric | n | noise: orig->var | real: orig->var | delta_noise | delta_real | DiD (real-noise) | wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in exps:
        summ = all_summaries.get(name, {})
        for metric in ("dice", "recall"):
            r = summ.get(metric)
            if not r:
                continue
            mkey = metric
            lines.append(f"| {name} | {metric} | {r['n']} | "
                         f"{r[f'abs_{mkey}_orig_noise']:.3f} -> {r[f'abs_{mkey}_var_noise']:.3f} | "
                         f"{r[f'abs_{mkey}_orig_real']:.3f} -> {r[f'abs_{mkey}_var_real']:.3f} | "
                         f"{r['median_delta_noise']:.4f} | {r['median_delta_real']:.4f} | {r['did']:.4f} | {r['p']:.3g} |")
    lines.append("")

    lines.append("## Part B -- SHAM false-positive SNFH voxels created (median)\n")
    lines.append("| condition | noise median new FP vox | real median new FP vox |")
    lines.append("|---|---|---|")
    for name in ("S-ADD(+1) alone SHAM", "C-FLIP alone SHAM", "C-FLIP + S-ADD(+1) SHAM"):
        r = all_summaries.get(name, {}).get("dice")
        if r:
            lines.append(f"| {name} | {r['median_new_fp_noise']:.0f} | {r['median_new_fp_real']:.0f} |")
    lines.append("")

    lines.append("## Verdict\n")
    for metric in ("dice", "recall"):
        if metric in dod:
            held = dod[metric] > 0
            lines.append(f"- **DiD-of-DiD ({metric})**: combined DiD minus S-ADD-alone DiD = "
                         f"{dod[metric]:.4f} -> {'HELD (co-polarity restoration makes the bright step less damaging to real-fill)' if held else 'DID NOT HOLD (co-polarity restoration did not rescue, or made it worse)'}")
    lines.append("\nC-FLIP alone has no pre-registered direction -- reported descriptively above. "
                 "SHAM rows are the specificity check.")

    (OUT_TABLES / "copolarity.md").write_text("\n".join(lines))
    print("wrote", OUT_TABLES / "copolarity.md")

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(cluster_summary["c"], cluster_summary["frac_edema_same_cluster_as_ncr"], "o-", label="vs NCR")
    ax.plot(cluster_summary["c"], cluster_summary["frac_edema_same_cluster_as_et"], "o-", label="vs ET")
    ax.plot(cluster_summary["c"], cluster_summary["frac_edema_same_cluster_as_normal"], "o-", label="vs normal tissue")
    ax.axhline(1.0 / 3, color="gray", linestyle="--", linewidth=0.8, label="chance (1/c avg)")
    ax.set_xlabel("K-means c")
    ax.set_ylabel("fraction of edema voxels in same cluster")
    ax.set_title("Part A: edema-core K-means co-membership")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    names = ["S-ADD(+1)\nalone", "C-FLIP\nalone", "C-FLIP+S-ADD(+1)\ncombined"]
    keys = ["S-ADD(+1) alone", "C-FLIP alone", "C-FLIP + S-ADD(+1) combined"]
    for i, metric in enumerate(("dice", "recall")):
        vals = [all_summaries.get(k, {}).get(metric, {}).get("did", np.nan) for k in keys]
        x = np.arange(len(keys)) + (i - 0.5) * 0.35
        ax2.bar(x, vals, width=0.35, label=metric)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(np.arange(len(keys)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("DiD (median delta_real - delta_noise)")
    ax2.set_title("Part B: does C-FLIP rescue the bright edema step?")
    ax2.legend()
    fig.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOTS / "copolarity.png", dpi=140)
    print("wrote", OUT_PLOTS / "copolarity.png")


if __name__ == "__main__":
    main()
