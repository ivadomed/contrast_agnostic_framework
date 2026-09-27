#!/usr/bin/env python
"""
Analysis for the texture-transplant test (X1-X4 + symmetry check + sham). See
intervention_transplant_build.py's docstring for the pre-registration.

DiD reported as delta_real - delta_noise (this round's convention, per the coordinator's framing
— the opposite sign from the earlier FLATTEN/leakage rounds' delta_noise - delta_real; both are
computed here from the same per-patient deltas, just be careful reading across reports).

Writes:
  outputs/tables/intervention_transplant.md
  outputs/plots/intervention_transplant.png
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
# name -> (train, orig_set, variant_set, directional?)
EXPS = {
    "X1_tswap_t1n_donor_t2w": ("t2w", "orig_t1n", "x1_tswap_t1n_donor_t2w", "real-noise>0"),
    "X2_tswap_t2w_donor_t1n": ("t2w", "orig_t2w", "x2_tswap_t2w_donor_t1n", "real-noise<0"),
    "X3_tswap_t1n_donor_t1c": ("t2w", "orig_t1n", "x3_tswap_t1n_donor_t1c", "none"),
    "X4_permute_t1n": ("t2w", "orig_t1n", "x4_permute_t1n", "none"),
    "SYM_tswap_t2f_donor_t1n": ("t1n", "orig_t2f", "symmetry_tswap_t2f_donor_t1n", "none"),
}
SHAM_EXPS = {
    "X1_tswap_t1n_donor_t2w": "sham_tswap_t1n_donor_t2w",
    "X2_tswap_t2w_donor_t1n": "sham_tswap_t2w_donor_t1n",
}
MODEL_KEYS = {
    ("t1n", "noise"): "t1n_noise", ("t1n", "real"): "t1n_real",
    ("t2w", "noise"): "t2w_noise", ("t2w", "real"): "t2w_real",
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
    manifest = pd.read_csv(OUT_DATA / "intervention_transplant_manifest.csv")
    core_ok_cases = set(manifest[manifest["core_ok"]]["case"])
    sham_ok_cases = set(manifest[manifest.get("n_sham_core", 0) >= 50]["case"]) if "n_sham_core" in manifest else set()
    all_cases = sorted(pd.read_csv(OUT_DATA / "patient_region_deltas.csv")["case"].unique())[:30]

    rows = []
    for exp_name, (train, orig_set, set_name, _) in EXPS.items():
        for rung in ("noise", "real"):
            model_key = MODEL_KEYS[(train, rung)]
            for case in core_ok_cases & set(all_cases):
                gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                pred_orig = load_pred(model_key, orig_set, case)
                pred_v = load_pred(model_key, set_name, case)
                if pred_orig is None or pred_v is None:
                    continue
                d_o, r_o = dice_recall(gt, pred_orig, SNFH)
                d_v, r_v = dice_recall(gt, pred_v, SNFH)
                rows.append(dict(experiment=exp_name, train=train, rung=rung, case=case,
                                  dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                  delta_dice=d_v - d_o, delta_recall=r_v - r_o))
        # sham
        if exp_name in SHAM_EXPS:
            sham_set = SHAM_EXPS[exp_name]
            for rung in ("noise", "real"):
                model_key = MODEL_KEYS[(train, rung)]
                for case in sham_ok_cases & set(all_cases):
                    gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                    pred_orig = load_pred(model_key, orig_set, case)
                    pred_v = load_pred(model_key, sham_set, case)
                    if pred_orig is None or pred_v is None:
                        continue
                    d_o, r_o = dice_recall(gt, pred_orig, SNFH)
                    d_v, r_v = dice_recall(gt, pred_v, SNFH)
                    rows.append(dict(experiment=exp_name + "_SHAM", train=train, rung=rung, case=case,
                                      dice_orig=d_o, recall_orig=r_o, dice_var=d_v, recall_var=r_v,
                                      delta_dice=d_v - d_o, delta_recall=r_v - r_o))

    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(OUT_DATA / "intervention_transplant_per_patient.csv", index=False)

    summary_rows = []
    for exp_full in per_patient["experiment"].unique():
        sub = per_patient[per_patient["experiment"] == exp_full]
        noise = sub[sub["rung"] == "noise"].set_index("case")
        real = sub[sub["rung"] == "real"].set_index("case")
        common = noise.index.intersection(real.index)
        noise, real = noise.loc[common], real.loc[common]
        for metric in ("dice", "recall"):
            dn = noise[f"delta_{metric}"].to_numpy()
            dr = real[f"delta_{metric}"].to_numpy()
            valid = ~np.isnan(dn) & ~np.isnan(dr)
            dn_v, dr_v = dn[valid], dr[valid]
            p = wilcoxon_p(dr_v, dn_v) if valid.sum() >= 1 else float("nan")  # x=real, y=noise
            summary_rows.append(dict(
                experiment=exp_full, metric=metric, n=int(valid.sum()),
                median_delta_noise=float(np.nanmedian(dn_v)) if valid.sum() else float("nan"),
                median_delta_real=float(np.nanmedian(dr_v)) if valid.sum() else float("nan"),
                did_real_minus_noise=float(np.nanmedian(dr_v - dn_v)) if valid.sum() else float("nan"),
                wilcoxon_p=p))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / "intervention_transplant_raw.csv", index=False)

    # absolute Dice levels for X1
    x1 = per_patient[per_patient["experiment"] == "X1_tswap_t1n_donor_t2w"]
    abs_rows = []
    for rung in ("noise", "real"):
        r = x1[x1["rung"] == rung]
        abs_rows.append(dict(rung=rung, median_dice_orig_t1n=float(r["dice_orig"].median()),
                              median_dice_tswap=float(r["dice_var"].median())))
    abs_df = pd.DataFrame(abs_rows)

    sign_summary = manifest[["x1_sign", "x2_sign", "x3_sign", "symmetry_sign"]].apply(
        lambda c: c.value_counts(dropna=True).to_dict())

    lines = []
    lines.append("# Texture-transplant test (T-SWAP)\n")
    lines.append("Fold 0, same 30 patients, t2w-trained noise-/real-fill models unless noted "
                 "(SYM row = t1n-trained models). DiD = delta_real - delta_noise (this round's "
                 "sign convention). All interventions restricted to the depth>3 eroded core.\n")

    lines.append("## X1 absolute Dice levels (t1n, orig vs T-SWAP(t1n, donor=t2w))\n")
    lines.append("| rung | median Dice, orig t1n | median Dice, T-SWAP |")
    lines.append("|---|---|---|")
    for _, r in abs_df.iterrows():
        lines.append(f"| {r['rung']} | {r['median_dice_orig_t1n']:.3f} | {r['median_dice_tswap']:.3f} |")
    lines.append("")

    lines.append("## Sign selection (how often donor's z-score kept + vs flipped -)\n")
    lines.append(f"```\n{sign_summary.to_string()}\n```\n")

    lines.append("## Results by experiment\n")
    lines.append("| experiment | metric | n | delta_noise | delta_real | DiD (real-noise) | wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|")
    order = ["X1_tswap_t1n_donor_t2w", "X1_tswap_t1n_donor_t2w_SHAM",
             "X2_tswap_t2w_donor_t1n", "X2_tswap_t2w_donor_t1n_SHAM",
             "X3_tswap_t1n_donor_t1c", "X4_permute_t1n", "SYM_tswap_t2f_donor_t1n"]
    for exp_full in order:
        for metric in ("dice", "recall"):
            row = summary[(summary.experiment == exp_full) & (summary.metric == metric)]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(f"| {exp_full} | {metric} | {r['n']} | {r['median_delta_noise']:.4f} | "
                         f"{r['median_delta_real']:.4f} | {r['did_real_minus_noise']:.4f} | {r['wilcoxon_p']:.3g} |")
    lines.append("")

    lines.append("## Verdict\n")
    for exp_name, expect in (("X1_tswap_t1n_donor_t2w", "real-noise>0"), ("X2_tswap_t2w_donor_t1n", "real-noise<0")):
        for metric in ("dice", "recall"):
            row = summary[(summary.experiment == exp_name) & (summary.metric == metric)]
            if row.empty:
                continue
            r = row.iloc[0]
            did = r["did_real_minus_noise"]
            p = r["wilcoxon_p"]
            ok_dir = (did > 0) if expect == "real-noise>0" else (did < 0)
            held = ok_dir and p < 0.05
            verdict = "HELD" if held else ("direction correct, not significant" if ok_dir else "DID NOT HOLD / reversed")
            lines.append(f"- **{exp_name} / {metric}**: predicted {expect}, observed DiD={did:.4f}, p={p:.3g} -> {verdict}")
    lines.append("")
    lines.append("X3 (donor=t1c control), X4 (permuted core control) and the SYM (t1n-trained "
                 "symmetry check) rows above have no pre-registered direction; reported as-is.")
    lines.append("SHAM rows are the specificity check (expect ~0 DiD, no consistent direction).")

    (OUT_TABLES / "intervention_transplant.md").write_text("\n".join(lines))
    print("wrote", OUT_TABLES / "intervention_transplant.md")

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_exps = ["X1_tswap_t1n_donor_t2w", "X1_tswap_t1n_donor_t2w_SHAM", "X2_tswap_t2w_donor_t1n",
                 "X2_tswap_t2w_donor_t1n_SHAM", "X3_tswap_t1n_donor_t1c", "X4_permute_t1n",
                 "SYM_tswap_t2f_donor_t1n"]
    labels = ["X1\nt1n<-t2w", "X1 sham", "X2\nt2w<-t1n", "X2 sham", "X3\nt1n<-t1c (ctrl)",
              "X4\npermute (ctrl)", "SYM\nt2f<-t1n"]
    for i, metric in enumerate(("dice", "recall")):
        vals = []
        for e in plot_exps:
            row = summary[(summary.experiment == e) & (summary.metric == metric)]
            vals.append(float(row["did_real_minus_noise"].iloc[0]) if not row.empty else np.nan)
        x = np.arange(len(plot_exps)) + (i - 0.5) * 0.35
        ax.bar(x, vals, width=0.35, label=metric)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(plot_exps)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("DiD (median delta_real - delta_noise)")
    ax.set_title("Texture transplant: real-fill vs noise-fill DiD by experiment")
    ax.legend()
    fig.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOTS / "intervention_transplant.png", dpi=140)
    print("wrote", OUT_PLOTS / "intervention_transplant.png")


if __name__ == "__main__":
    main()
