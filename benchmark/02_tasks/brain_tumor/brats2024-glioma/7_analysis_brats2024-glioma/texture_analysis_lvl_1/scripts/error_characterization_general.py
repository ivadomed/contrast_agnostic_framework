#!/usr/bin/env python
"""
Generalized version of error_characterization.py: same noise-fill vs real-fill error
characterization (CONFUSION vs MISSING-CUE hypotheses), for an arbitrary --train contrast, with
explicit --noise-run/--real-run directory arguments instead of the t2w-hardcoded RUNS dict.

Motivation: the t1n- and t2f-trained ladders' rung-4 (noise-fill) / rung-5 (real-fill) prediction
volumes were produced on TamIA and staged to Vulcan under $SCRATCH/brats_ladder_preds/ (not the
usual 8_results_brats2024-glioma/01_predictions/ path — those volumes live only on TamIA's
scratch). This script accepts the run directories explicitly so it works against either location.

Usage (see the two provided wrapper invocations at the bottom of this docstring):
  .venv/bin/python error_characterization_general.py --train t1n \
      --noise-run /scratch/paulh/brats_ladder_preds/t1n/auglab/brats2024-glioma_t1n_baseline_kmeans_label_remap_voronoi_20260730_200711 \
      --real-run  /scratch/paulh/brats_ladder_preds/t1n/nnUNet/brats2024-glioma_t1n_v26_6_2_train050_val100_20260730_200711

  .venv/bin/python error_characterization_general.py --train t2f \
      --noise-run /scratch/paulh/brats_ladder_preds/t2f/auglab/brats2024-glioma_t2f_baseline_kmeans_label_remap_voronoi_20260917_094019 \
      --real-run  /scratch/paulh/brats_ladder_preds/t2f/nnUNet/brats2024-glioma_t2f_v26_6_2_train050_val100_20260917_113412

Outputs (never overwrites the original t2w-only outputs, which keep their unsuffixed names):
  outputs/data/error_characterization_<train>_per_fold.csv
  outputs/data/error_characterization_<train>_summary.csv
  outputs/tables/error_characterization_<train>.md

Reuses `load`/`metrics` and the LABELS/REGIONS constants from error_characterization.py verbatim
(imported, not copy-pasted) so the metric definitions never drift between the t2w-only original and
this generalized runner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.ndimage import distance_transform_edt

THIS_DIR = Path(__file__).resolve().parent
PROJECT = THIS_DIR.parents[4]
sys.path.insert(0, str(PROJECT / "datasets" / "00_commun_scripts" / "00_00_utils"))
sys.path.insert(0, str(THIS_DIR))
from stat_tests import wilcoxon_p  # noqa: E402
from compute_cross_contrast_ngf import FOLDS, LABELS_DIR  # noqa: E402
from error_characterization import load, metrics, LABELS, REGIONS  # noqa: E402

OUT = THIS_DIR.parent / "outputs"
EVALS = ("t1n", "t1c", "t2f", "t2w")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, choices=["t1n", "t1c", "t2w", "t2f"],
                     help="Training contrast of this ladder (the in-domain eval column).")
    ap.add_argument("--noise-run", required=True, type=Path,
                     help="Rung-4 (noise-fill / kmeans+label_remap+voronoi) prediction run dir, "
                          "containing fold0/fold1/fold2/<eval_contrast>/<case>.nii.gz.")
    ap.add_argument("--real-run", required=True, type=Path,
                     help="Rung-5 (real-fill / v26_6_2 train050_val100) prediction run dir, same layout.")
    args = ap.parse_args()

    train = args.train
    RUNS = {"noise": args.noise_run, "real": args.real_run}
    for tag, d in RUNS.items():
        if not d.is_dir():
            sys.exit(f"error: --{tag}-run does not exist or is not a directory: {d}")

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
    df.to_csv(OUT / "data" / f"error_characterization_{train}_per_fold.csv", index=False)
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
    summ.to_csv(OUT / "data" / f"error_characterization_{train}_summary.csv", index=False)

    key = ["dice", "recall", "precision", "vol_ratio", "fp_share_ring", "fp_share_other_tumor",
           "fp_share_elsewhere", "fn_depth_rel", "gt_as_bg", "gt_as_NCR", "gt_as_SNFH", "gt_as_ET", "gt_as_RC"]
    L = [f"# How does real-fill fail? {train}-trained noise-fill vs real-fill, error characterization", "",
         f"Training contrast = **{train}** (in-domain eval column, marked with a `*`). "
         "Confusion hypothesis predicts: precision ↓, volume ↑, false positives in the ring. "
         "Missing-cue hypothesis predicts: recall ↓, volume ↓, precision holds. "
         "Medians over patients (each = mean over folds 0–2); p = paired two-sided Wilcoxon, patient unit. "
         "fn_depth_rel < 1 = missed voxels concentrated near the lesion border.", ""]
    ev_headers = [f"{e}*" if e == train else e for e in EVALS]
    for region in REGIONS:
        L += [f"## {region}", "", "| metric | " + " | ".join(f"{e}: noise → real (p)" for e in ev_headers) + " |",
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
    (OUT / "tables" / f"error_characterization_{train}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
