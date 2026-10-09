#!/usr/bin/env python
"""
Analysis for the pre-registered FLATTEN/RAMP causal intervention (see intervention_build.py's
docstring for the full pre-registration written before any inference was run).

For each (train_contrast, rung) model and each experiment's (orig, intervened, sham) triple,
computes per-patient edema (SNFH, label 2) Dice and recall against ground truth, then the
per-patient delta (intervened - orig) and (sham - orig), and the noise-vs-real
difference-in-differences via a paired Wilcoxon signed-rank test (stat_tests.wilcoxon_p) on the
per-patient deltas — the pre-registered test.

Also reports, for the sham condition, the count of predicted-SNFH voxels that land INSIDE the
sham region itself (a false-positive check distinct from the true-edema Dice/recall change).

Usage (CPU-only, run inside a small run_job or the login-node exception since it's plain
pandas/nibabel over ~30 patients x ~2-9 sets x 2 rungs):
  .venv/bin/python intervention_analyze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

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
INPUTS_ROOT = SCRATCH_ROOT / "inputs"
PREDS_ROOT = SCRATCH_ROOT / "preds"

SNFH = 2

# (train, rung) -> model_key used under PREDS_ROOT
MODEL_KEYS = {
    ("t1n", "noise"): "t1n_noise",
    ("t1n", "real"): "t1n_real",
    ("t2w", "noise"): "t2w_noise",
    ("t2w", "real"): "t2w_real",
}

# experiment -> (train, orig_set, interv_set, sham_set)
EXPERIMENTS = {
    "E1_flatten_t2f_t1n_trained": ("t1n", "orig_t2f", "e1_flatten_t2f", "sham_flatten_t2f"),
    "E2_flatten_t2f_t2w_trained_cross": ("t2w", "orig_t2f", "e1_flatten_t2f", "sham_flatten_t2f"),
    "E2_flatten_t2w_t2w_trained_indomain": ("t2w", "orig_t2w", "e2_flatten_t2w", "sham_flatten_t2w"),
    "E3_ramp_t1n_donor_t2w_t2w_trained": ("t2w", "orig_t1n", "e3_ramp_t1n_donor_t2w", "sham_ramp_t1n"),
}


def dice_recall(gt: np.ndarray, pred: np.ndarray, lid: int) -> tuple[float, float]:
    G, P = gt == lid, pred == lid
    ng, npred = int(G.sum()), int(P.sum())
    if ng == 0:
        return float("nan"), float("nan")
    tp = int((G & P).sum())
    dice = 2 * tp / (ng + npred) if (ng + npred) else float("nan")
    recall = tp / ng
    return dice, recall


def load_pred(model_key: str, set_name: str, case: str) -> np.ndarray | None:
    f = PREDS_ROOT / model_key / set_name / f"{case}.nii.gz"
    if not f.exists():
        return None
    return np.asarray(nib.load(str(f)).dataobj).round().astype(np.int16)


def main() -> None:
    manifest = pd.read_csv(OUT_DATA / "intervention_manifest.csv")
    ok = manifest[~manifest["excluded"]].copy()
    cases = sorted(ok["case"].unique())
    sham_ok_cases = set(ok[ok["sham_status"] != "failed"]["case"])
    print(f"{len(cases)} usable patients (n_snfh >= 50); {len(sham_ok_cases)} with a valid sham mask")

    rows = []
    for exp_name, (train, orig_set, interv_set, sham_set) in EXPERIMENTS.items():
        for rung in ("noise", "real"):
            model_key = MODEL_KEYS[(train, rung)]
            for case in cases:
                gt = np.asarray(nib.load(str(LABELS_DIR / f"{case}.nii.gz")).dataobj).round().astype(np.int16)

                pred_orig = load_pred(model_key, orig_set, case)
                pred_interv = load_pred(model_key, interv_set, case)
                if pred_orig is None or pred_interv is None:
                    continue
                d_o, r_o = dice_recall(gt, pred_orig, SNFH)
                d_i, r_i = dice_recall(gt, pred_interv, SNFH)

                d_s = r_s = float("nan")
                n_pred_in_sham = float("nan")
                if case in sham_ok_cases:
                    pred_sham = load_pred(model_key, sham_set, case)
                    if pred_sham is not None:
                        d_s, r_s = dice_recall(gt, pred_sham, SNFH)
                        sham_input_f = INPUTS_ROOT / sham_set / f"{case}_0000.nii.gz"
                        # Recover sham mask indirectly is expensive; instead just count predicted
                        # SNFH voxels that are NOT also predicted SNFH in orig at the same location
                        # AND not part of true tumor -- proxy for "new false positives" (a coarser
                        # but self-contained false-positive signal, avoids re-deriving the mask).
                        new_fp = int(((pred_sham == SNFH) & (pred_orig != SNFH) & (gt == 0)).sum())
                        n_pred_in_sham = new_fp

                rows.append(dict(
                    experiment=exp_name, train=train, rung=rung, case=case,
                    dice_orig=d_o, recall_orig=r_o,
                    dice_interv=d_i, recall_interv=r_i,
                    dice_sham=d_s, recall_sham=r_s,
                    delta_dice=d_i - d_o, delta_recall=r_i - r_o,
                    delta_dice_sham=d_s - d_o if not np.isnan(d_s) else np.nan,
                    delta_recall_sham=r_s - r_o if not np.isnan(r_s) else np.nan,
                    new_fp_sham=n_pred_in_sham,
                ))

    per_patient = pd.DataFrame(rows)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    per_patient.to_csv(OUT_DATA / "intervention_per_patient.csv", index=False)
    print(f"wrote {OUT_DATA / 'intervention_per_patient.csv'} ({len(per_patient)} rows)")

    # --- summary + pre-registered diff-in-diff tests ---
    summary_rows = []
    for exp_name in EXPERIMENTS:
        sub = per_patient[per_patient["experiment"] == exp_name]
        noise = sub[sub["rung"] == "noise"].set_index("case")
        real = sub[sub["rung"] == "real"].set_index("case")
        common = noise.index.intersection(real.index)
        noise, real = noise.loc[common], real.loc[common]

        for metric in ("dice", "recall"):
            dn = noise[f"delta_{metric}"].to_numpy()
            dr = real[f"delta_{metric}"].to_numpy()
            valid = ~np.isnan(dn) & ~np.isnan(dr)
            dn_v, dr_v = dn[valid], dr[valid]
            p = wilcoxon_p(dn_v, dr_v) if valid.sum() >= 1 else float("nan")
            summary_rows.append(dict(
                experiment=exp_name, metric=metric, n=int(valid.sum()),
                median_delta_noise=float(np.nanmedian(dn_v)) if valid.sum() else float("nan"),
                median_delta_real=float(np.nanmedian(dr_v)) if valid.sum() else float("nan"),
                did_noise_minus_real=float(np.nanmedian(dn_v - dr_v)) if valid.sum() else float("nan"),
                wilcoxon_p_did=p,
            ))

        # sham specificity check (same patients that have a valid sham mask)
        for metric in ("dice", "recall"):
            for rung_name, df_r in (("noise", noise), ("real", real)):
                ds = df_r[f"delta_{metric}_sham"].to_numpy()
                ds_v = ds[~np.isnan(ds)]
                summary_rows.append(dict(
                    experiment=exp_name, metric=f"{metric}_SHAM_{rung_name}", n=len(ds_v),
                    median_delta_noise=float(np.nanmedian(ds_v)) if len(ds_v) else float("nan"),
                    median_delta_real=float("nan"), did_noise_minus_real=float("nan"),
                    wilcoxon_p_did=float("nan"),
                ))
        for rung_name, df_r in (("noise", noise), ("real", real)):
            fp = df_r["new_fp_sham"].to_numpy()
            fp_v = fp[~np.isnan(fp)]
            summary_rows.append(dict(
                experiment=exp_name, metric=f"new_fp_voxels_in_sham_{rung_name}", n=len(fp_v),
                median_delta_noise=float(np.nanmedian(fp_v)) if len(fp_v) else float("nan"),
                median_delta_real=float("nan"), did_noise_minus_real=float("nan"),
                wilcoxon_p_did=float("nan"),
            ))

    summary = pd.DataFrame(summary_rows)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLES / "intervention_summary.csv", index=False)
    print(f"wrote {OUT_TABLES / 'intervention_summary.csv'}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
