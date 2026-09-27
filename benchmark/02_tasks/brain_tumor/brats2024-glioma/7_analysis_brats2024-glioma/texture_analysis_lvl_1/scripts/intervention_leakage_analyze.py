#!/usr/bin/env python
"""
Analysis for the label-leakage control (F-ERODE / F-SMOOTH / SHAM-ERODE) — see
intervention_leakage_build.py's docstring for the pre-registration.

Computes per-patient edema (SNFH) Dice/recall for each (train, rung, experiment, variant)
combination, the noise-vs-real difference-in-differences (paired Wilcoxon, stat_tests.wilcoxon_p)
against the ORIGINAL (unmodified) prediction baseline from the first intervention run
(outputs/data/intervention_per_patient.csv), and compares each control variant's DiD to the
full-FLATTEN DiD from that same file to check whether the effect shrinks/collapses.

Writes:
  outputs/tables/intervention_leakage_control.md
  outputs/plots/intervention_leakage_control.png
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
MODEL_KEYS = {
    ("t1n", "noise"): "t1n_noise", ("t1n", "real"): "t1n_real",
    ("t2w", "noise"): "t2w_noise", ("t2w", "real"): "t2w_real",
}
# experiment -> (train, orig_set, variant_set_by_name)
EXPERIMENTS = {
    "E1_t1n_trained_t2f": ("t1n", "orig_t2f", dict(F_ERODE="f_erode_t2f", F_SMOOTH="f_smooth_t2f",
                                                    SHAM_ERODE="sham_erode_t2f")),
    "E2_cross_t2w_trained_t2f": ("t2w", "orig_t2f", dict(F_ERODE="f_erode_t2f", F_SMOOTH="f_smooth_t2f",
                                                          SHAM_ERODE="sham_erode_t2f")),
    "E2_indomain_t2w_trained_t2w": ("t2w", "orig_t2w", dict(F_ERODE="f_erode_t2w", F_SMOOTH="f_smooth_t2w",
                                                             SHAM_ERODE="sham_erode_t2w")),
}
# map to the ORIGINAL full-FLATTEN experiment name in intervention_per_patient.csv, for comparison
FULL_FLATTEN_EXP = {
    "E1_t1n_trained_t2f": "E1_flatten_t2f_t1n_trained",
    "E2_cross_t2w_trained_t2f": "E2_flatten_t2f_t2w_trained_cross",
    "E2_indomain_t2w_trained_t2w": "E2_flatten_t2w_t2w_trained_indomain",
}


def dice_recall(gt: np.ndarray, pred: np.ndarray, lid: int) -> tuple[float, float]:
    G, P = gt == lid, pred == lid
    ng, npred = int(G.sum()), int(P.sum())
    if ng == 0:
        return float("nan"), float("nan")
    tp = int((G & P).sum())
    return (2 * tp / (ng + npred) if (ng + npred) else float("nan")), tp / ng


def load_pred(model_key: str, set_name: str, case: str):
    f = PREDS_ROOT / model_key / set_name / f"{case}.nii.gz"
    if not f.exists():
        return None
    return np.asarray(nib.load(str(f)).dataobj).round().astype(np.int16)


def main() -> None:
    leak_manifest = pd.read_csv(OUT_DATA / "intervention_leakage_manifest.csv")
    core_ok_cases = set(leak_manifest[leak_manifest["core_ok"]]["case"])
    sham_core_ok_cases = set(leak_manifest[leak_manifest.get("n_sham_core", 0) >= 50]["case"]) \
        if "n_sham_core" in leak_manifest else set()
    all_cases = sorted(pd.read_csv(OUT_DATA / "patient_region_deltas.csv")["case"].unique())[:30]

    rows = []
    for exp_name, (train, orig_set, variants) in EXPERIMENTS.items():
        for rung in ("noise", "real"):
            model_key = MODEL_KEYS[(train, rung)]
            for variant, set_name in variants.items():
                valid_cases = sham_core_ok_cases if variant == "SHAM_ERODE" else core_ok_cases
                for case in valid_cases & set(all_cases):
                    gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)
                    pred_orig = load_pred(model_key, orig_set, case)
                    pred_v = load_pred(model_key, set_name, case)
                    if pred_orig is None or pred_v is None:
                        continue
                    d_o, r_o = dice_recall(gt, pred_orig, SNFH)
                    d_v, r_v = dice_recall(gt, pred_v, SNFH)
                    rows.append(dict(experiment=exp_name, train=train, rung=rung, variant=variant,
                                      case=case, dice_orig=d_o, recall_orig=r_o,
                                      dice_var=d_v, recall_var=r_v,
                                      delta_dice=d_v - d_o, delta_recall=r_v - r_o))

    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(OUT_DATA / "intervention_leakage_per_patient.csv", index=False)

    # original full-FLATTEN per-patient deltas + absolute Dice, for comparison
    orig_full = pd.read_csv(OUT_DATA / "intervention_per_patient.csv")

    summary_rows = []
    for exp_name in EXPERIMENTS:
        full_exp = FULL_FLATTEN_EXP[exp_name]
        full_sub = orig_full[orig_full["experiment"] == full_exp]
        for metric in ("dice", "recall"):
            for variant in ("F_ERODE", "F_SMOOTH", "SHAM_ERODE"):
                sub = per_patient[(per_patient["experiment"] == exp_name) & (per_patient["variant"] == variant)]
                noise = sub[sub["rung"] == "noise"].set_index("case")
                real = sub[sub["rung"] == "real"].set_index("case")
                common = noise.index.intersection(real.index)
                dn = noise.loc[common, f"delta_{metric}"].to_numpy()
                dr = real.loc[common, f"delta_{metric}"].to_numpy()
                valid = ~np.isnan(dn) & ~np.isnan(dr)
                dn_v, dr_v = dn[valid], dr[valid]
                p = wilcoxon_p(dn_v, dr_v) if valid.sum() >= 1 else float("nan")
                did = float(np.nanmedian(dn_v - dr_v)) if valid.sum() else float("nan")

                # matching full-FLATTEN DiD for comparison (same patient subset where possible)
                fn = full_sub[full_sub["rung"] == "noise"].set_index("case")[f"delta_{metric}"]
                fr = full_sub[full_sub["rung"] == "real"].set_index("case")[f"delta_{metric}"]
                fc = fn.index.intersection(fr.index).intersection(common)
                full_did = float(np.nanmedian(fn.loc[fc].to_numpy() - fr.loc[fc].to_numpy())) if len(fc) else float("nan")

                summary_rows.append(dict(experiment=exp_name, metric=metric, variant=variant,
                                          n=int(valid.sum()),
                                          median_delta_noise=float(np.nanmedian(dn_v)) if valid.sum() else float("nan"),
                                          median_delta_real=float(np.nanmedian(dr_v)) if valid.sum() else float("nan"),
                                          did=did, wilcoxon_p=p, full_flatten_did_same_patients=full_did))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / "intervention_leakage_control_raw.csv", index=False)

    # absolute Dice levels for E1 FLATTEN (original vs flattened, noise and real)
    e1_full = orig_full[orig_full["experiment"] == "E1_flatten_t2f_t1n_trained"]
    abs_rows = []
    for rung in ("noise", "real"):
        r = e1_full[e1_full["rung"] == rung]
        abs_rows.append(dict(rung=rung, median_dice_orig=float(r["dice_orig"].median()),
                              median_dice_flatten=float(r["dice_interv"].median())))
    abs_df = pd.DataFrame(abs_rows)

    # ---- markdown report ----
    lines = []
    lines.append("# Label-leakage control for the FLATTEN intervention\n")
    lines.append("Fold 0, same 30 patients, same 4 models (t1n/t2w x noise/real-fill), "
                 "edema = SNFH. F-ERODE/F-SMOOTH touch only voxels with depth > 3 inside the "
                 "GT edema mask; the true GT border and its 0-3 voxel shell are untouched.\n")

    lines.append("## E1 FLATTEN — absolute Dice levels (not deltas), t1n-trained models on t2f\n")
    lines.append("| rung | median Dice, original t2f | median Dice, FLATTEN(t2f) |")
    lines.append("|---|---|---|")
    for _, r in abs_df.iterrows():
        lines.append(f"| {r['rung']} | {r['median_dice_orig']:.3f} | {r['median_dice_flatten']:.3f} |")
    lines.append("")

    lines.append("## Noise-vs-real difference-in-differences, by variant\n")
    lines.append("`did` = median(delta_noise - delta_real) on the variant's own valid patient "
                 "subset; `full_flatten_did_same_patients` = the ORIGINAL full-FLATTEN DiD "
                 "recomputed on that SAME patient subset, for a fair shrink/collapse comparison.\n")
    lines.append("| experiment | metric | variant | n | delta_noise | delta_real | DiD | wilcoxon p | full-FLATTEN DiD (same n) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['experiment']} | {r['metric']} | {r['variant']} | {r['n']} | "
                     f"{r['median_delta_noise']:.4f} | {r['median_delta_real']:.4f} | "
                     f"{r['did']:.4f} | {r['wilcoxon_p']:.3g} | {r['full_flatten_did_same_patients']:.4f} |")
    lines.append("")

    lines.append("## Verdict\n")
    for exp_name in EXPERIMENTS:
        for metric in ("dice", "recall"):
            for variant in ("F_ERODE", "F_SMOOTH"):
                row = summary[(summary.experiment == exp_name) & (summary.metric == metric) & (summary.variant == variant)]
                if row.empty:
                    continue
                row = row.iloc[0]
                held = (not np.isnan(row["did"])) and row["did"] > 0 and row["wilcoxon_p"] < 0.05
                shrink = ""
                if not np.isnan(row["full_flatten_did_same_patients"]) and row["full_flatten_did_same_patients"] != 0:
                    frac = row["did"] / row["full_flatten_did_same_patients"]
                    shrink = f" ({frac*100:.0f}% of full-FLATTEN DiD)"
                verdict = "HELD (direction+significant)" if held else (
                    "direction correct, not significant" if (not np.isnan(row["did"]) and row["did"] > 0) else "DID NOT HOLD / reversed")
                lines.append(f"- **{exp_name} / {metric} / {variant}**: {verdict}{shrink}")
    lines.append("")
    lines.append("SHAM-ERODE rows above show the specificity check at the eroded scale "
                 "(should be ~0 DiD / no consistent direction).")

    (OUT_TABLES / "intervention_leakage_control.md").write_text("\n".join(lines))
    print("wrote", OUT_TABLES / "intervention_leakage_control.md")

    # ---- plot: DiD comparison, full-FLATTEN vs F-ERODE vs F-SMOOTH, per experiment/metric ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, exp_name in zip(axes, EXPERIMENTS):
        variants = ["FULL_FLATTEN", "F_ERODE", "F_SMOOTH"]
        for i, metric in enumerate(("dice", "recall")):
            vals = []
            for v in variants:
                if v == "FULL_FLATTEN":
                    full_exp = FULL_FLATTEN_EXP[exp_name]
                    fs = orig_full[orig_full["experiment"] == full_exp]
                    n_ = fs[fs["rung"] == "noise"].set_index("case")[f"delta_{metric}"]
                    r_ = fs[fs["rung"] == "real"].set_index("case")[f"delta_{metric}"]
                    c_ = n_.index.intersection(r_.index)
                    vals.append(float(np.nanmedian(n_.loc[c_] - r_.loc[c_])))
                else:
                    row = summary[(summary.experiment == exp_name) & (summary.metric == metric) & (summary.variant == v)]
                    vals.append(float(row["did"].iloc[0]) if not row.empty else np.nan)
            x = np.arange(len(variants)) + (i - 0.5) * 0.35
            ax.bar(x, vals, width=0.35, label=metric)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, rotation=20)
        ax.set_title(exp_name, fontsize=10)
    axes[0].set_ylabel("noise-vs-real DiD (median delta_noise - delta_real)")
    axes[0].legend()
    fig.suptitle("Leakage control: DiD shrinkage from full-FLATTEN to border-preserving variants")
    fig.tight_layout()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOTS / "intervention_leakage_control.png", dpi=140)
    print("wrote", OUT_PLOTS / "intervention_leakage_control.png")


if __name__ == "__main__":
    main()
