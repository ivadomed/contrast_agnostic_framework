#!/usr/bin/env python
"""
How does real-fill fail? Separates two hypotheses for the T2w-trained model on T1n:
  CONFUSION  (learned texture resembles the eval surroundings): real-fill adds false edema in the
             surroundings -> precision drops, predicted volume grows, FPs land in the ring.
  MISSING CUE (the texture/appearance cue it relies on is too weak on the eval contrast): real-fill
             misses true edema -> recall drops, predicted volume shrinks, precision holds.
Only the T2w ladder has both rung-4 (noise-fill) and rung-5 (real-fill) prediction volumes on
Vulcan, so this covers train=t2w, eval in {t1n (failure), t1c, t2f (successes), t2w (in-domain)}.

Per patient x fold x eval, for region in {SNFH, RC} (GT-present patients only):
  recall, precision, vol_ratio = |pred|/|GT|, Dice;
  FP location shares: ring (1-5 voxels outside GT region), other GT tumor labels, elsewhere;
  what GT-region voxels are predicted as (background / NCR / SNFH / ET / RC shares);
  depth of missed voxels: mean inside-distance to the GT-region border of FN voxels, relative to
  the mean depth of all GT-region voxels (< 1: misses concentrated at the border).
Per patient values are averaged over folds 0-2; real vs noise compared with the project's paired
two-sided Wilcoxon (stat_tests.wilcoxon_p), patient = unit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[4]
sys.path.insert(0, str(PROJECT / "datasets" / "00_commun_scripts" / "00_00_utils"))
sys.path.insert(0, str(THIS_DIR))
from stat_tests import wilcoxon_p  # noqa: E402
from compute_cross_contrast_ngf import FOLDS, LABELS_DIR  # noqa: E402

PRED = PROJECT / "benchmark/02_tasks/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w"
RUNS = {"noise": PRED / "auglab/brats2024-glioma_t2w_baseline_kmeans_label_remap_voronoi_20260805_020659",
        "real": PRED / "nnUNet/brats2024-glioma_t2w_v26_6_2_train050_val100_20260620_125217"}
OUT = THIS_DIR.parent / "outputs"
EVALS = ("t1n", "t1c", "t2f", "t2w")
REGIONS = {"SNFH": 2, "RC": 4}
LABELS = {0: "bg", 1: "NCR", 2: "SNFH", 3: "ET", 4: "RC"}


def load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj).round().astype(np.int16)


def metrics(pred, gt, lid, ring, depth):
    G, P = gt == lid, pred == lid
    tp, fp, fn = int((G & P).sum()), int((~G & P).sum()), int((G & ~P).sum())
    ng, npred = int(G.sum()), int(P.sum())
    fpm = ~G & P
    other_tumor = (gt > 0) & (gt != lid)
    fn_mask = G & ~P
    row = dict(recall=tp / ng, precision=tp / npred if npred else np.nan, vol_ratio=npred / ng,
               dice=2 * tp / (ng + npred),
               fp_share_ring=(fpm & ring).sum() / fp if fp else np.nan,
               fp_share_other_tumor=(fpm & other_tumor).sum() / fp if fp else np.nan,
               fp_share_elsewhere=(fpm & ~ring & ~other_tumor).sum() / fp if fp else np.nan,
               fn_depth_rel=float(depth[fn_mask].mean() / depth[G].mean()) if fn else np.nan)
    for k, name in LABELS.items():
        row[f"gt_as_{name}"] = float((pred[G] == k).mean())
    return row


def main():
    cases = sorted(pd.read_csv(OUT / "data" / "patient_region_deltas.csv")["case"].unique())
    rows = []
    for ci, case in enumerate(cases):
        gt = load(LABELS_DIR / f"{case}.nii.gz")
        brain_ok = True
        per_region = {}
        for rname, lid in REGIONS.items():
            G = gt == lid
            if not G.any():
                continue
            dout = distance_transform_edt(~G)
            per_region[rname] = (lid, (dout > 1) & (dout <= 5) & ~G, distance_transform_edt(G))
        for fold in FOLDS:
            for ev in EVALS:
                preds = {}
                for rung, d in RUNS.items():
                    f = d / fold / ev / f"{case}.nii.gz"
                    if not f.exists():
                        preds = None
                        break
                    preds[rung] = load(f)
                if preds is None:
                    continue
                if any(p.shape != gt.shape for p in preds.values()):
                    brain_ok = False
                    continue
                for rname, (lid, ring, depth) in per_region.items():
                    for rung, pred in preds.items():
                        rows.append(dict(case=case, fold=fold, eval=ev, region=rname, rung=rung,
                                         **metrics(pred, gt, lid, ring, depth)))
        if not brain_ok:
            print(f"WARNING {case}: prediction/GT shape mismatch, skipped some volumes")
        if (ci + 1) % 10 == 0:
            print(f"{ci + 1}/{len(cases)} cases", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "data" / "error_characterization_per_fold.csv", index=False)
    per_case = df.groupby(["case", "eval", "region", "rung"]).mean(numeric_only=True).reset_index()
    metric_cols = [c for c in per_case.columns if c not in ("case", "eval", "region", "rung")]
    out = []
    for (ev, region), g in per_case.groupby(["eval", "region"]):
        w = g.pivot(index="case", columns="rung", values=metric_cols)
        for m in metric_cols:
            a, b = w[(m, "real")], w[(m, "noise")]
            ok = a.notna() & b.notna()
            out.append(dict(eval=ev, region=region, metric=m, n=int(ok.sum()),
                            noise_median=float(b[ok].median()), real_median=float(a[ok].median()),
                            median_diff=float((a[ok] - b[ok]).median()),
                            frac_real_higher=float((a[ok] > b[ok]).mean()),
                            p=wilcoxon_p(a[ok].to_numpy(), b[ok].to_numpy())))
    summ = pd.DataFrame(out)
    summ.to_csv(OUT / "data" / "error_characterization_summary.csv", index=False)

    key = ["dice", "recall", "precision", "vol_ratio", "fp_share_ring", "fp_share_other_tumor",
           "fp_share_elsewhere", "fn_depth_rel", "gt_as_bg", "gt_as_NCR", "gt_as_SNFH", "gt_as_ET", "gt_as_RC"]
    L = ["# How does real-fill fail? T2w-trained noise-fill vs real-fill, error characterization", "",
         "Confusion hypothesis predicts: precision ↓, volume ↑, false positives in the ring. "
         "Missing-cue hypothesis predicts: recall ↓, volume ↓, precision holds. "
         "Medians over patients (each = mean over folds 0–2); p = paired two-sided Wilcoxon, patient unit. "
         "fn_depth_rel < 1 = missed voxels concentrated near the lesion border.", ""]
    for region in REGIONS:
        L += [f"## {region}", "", "| metric | " + " | ".join(f"{e}: noise → real (p)" for e in EVALS) + " |",
              "|---|" + "---|" * len(EVALS)]
        for m in key:
            cells = []
            for ev in EVALS:
                r = summ[(summ["eval"] == ev) & (summ["region"] == region) & (summ["metric"] == m)]
                if r.empty:
                    cells.append("—")
                    continue
                r = r.iloc[0]
                cells.append(f"{r['noise_median']:.3f} → {r['real_median']:.3f} ({r['p']:.2g})")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    (OUT / "tables" / "error_characterization.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
