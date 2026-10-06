#!/usr/bin/env python3
"""
Geometry check for on-harmony predictions: every prediction must sit on the SAME voxel grid
(shape + affine) as its native-space ground truth before it is scored.

Why: on-harmony predicts on RAS-canonical inputs and resamples the predictions back to native
geometry (05_predict/05_02_resample_predictions_to_native.py). When that step is skipped, the
prediction still has the GT's voxel count but a RAS grid; evaluate.py compares arrays voxel by
voxel, so a natively-LAS contrast (bold / dwi_ap / epi_ap) is scored x-mirrored -- silently
(2026-10-05: OOD Dice 58 -> 32 on one run). A RAS-canonical contrast (T1w ...) is unaffected.

Two modes:
  STRICT (guard, used by 06_evaluate/06_01_evaluate_testset.sh before scoring a contrast):
      check_pred_geometry.py --pred_dir <fold/contrast dir> --gt_dir <_test_set/<c>/gt_native>
      exit 1 (and print every offending case) if ANY prediction's shape/affine differs from its GT.
  AUDIT (report only, never modifies anything):
      check_pred_geometry.py --audit <PREDICTIONS_ROOT/<model>> [--out report.tsv]
      scans <train_contrast>/<category>/<run>/fold*/[<ckpt_tag>/]<contrast>/ and writes one TSV row per
      prediction directory: run, fold, ckpt subdir, contrast, n_pred, n_mismatch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

CONTRASTS = ["T1w", "T2w", "bold", "dwi_ap", "epi_ap", "gre_echo1_mag"]


def check_dir(pred_dir: Path, gt_dir: Path) -> tuple[int, list[str]]:
    n, bad = 0, []
    for p in sorted(pred_dir.glob("*.nii.gz")):
        g = gt_dir / p.name
        if not g.exists():
            continue
        pi, gi = nib.load(str(p)), nib.load(str(g))
        n += 1
        if pi.shape[:3] != gi.shape[:3] or not np.allclose(pi.affine, gi.affine, atol=1e-3):
            bad.append(f"{p.name}: pred {nib.aff2axcodes(pi.affine)} {pi.shape[:3]} vs gt "
                       f"{nib.aff2axcodes(gi.affine)} {gi.shape[:3]}")
    return n, bad


def audit(model_root: Path, out: Path | None) -> int:
    rows = []
    testset = model_root / "_test_set"
    for run in sorted(model_root.glob("*/*/*")):
        if run.parent.parent.name == "_test_set" or not run.is_dir():
            continue
        for fold in sorted(run.glob("fold[0-9]")):
            # flat fold{k}/<contrast> and any checkpoint subdir fold{k}/<tag>/<contrast>
            for sub in [fold] + [d for d in sorted(fold.iterdir()) if d.is_dir() and d.name not in CONTRASTS]:
                for c in CONTRASTS:
                    d = sub / c
                    if not d.is_dir():
                        continue
                    n, bad = check_dir(d, testset / c / "gt_native")
                    tag = "" if sub == fold else sub.name
                    rows.append((str(run.relative_to(model_root)), fold.name, tag, c, n, len(bad)))
    hdr = "run\tfold\tckpt_subdir\tcontrast\tn_pred\tn_mismatch"
    lines = [hdr] + ["\t".join(map(str, r)) for r in rows]
    if out:
        out.write_text("\n".join(lines) + "\n")
    bad_runs = sorted({r[0] for r in rows if r[5]})
    print(f"[geometry-audit] {len(rows)} prediction dirs, {sum(r[4] for r in rows)} predictions; "
          f"{sum(1 for r in rows if r[5])} dirs with mismatches in {len(bad_runs)} runs")
    for b in bad_runs:
        dirs = [f"{r[1]}/{r[2] + '/' if r[2] else ''}{r[3]}({r[5]}/{r[4]})" for r in rows if r[0] == b and r[5]]
        print(f"  MISMATCH {b}: {' '.join(dirs)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path)
    ap.add_argument("--gt_dir", type=Path)
    ap.add_argument("--audit", type=Path, help="PREDICTIONS_ROOT/<model> to scan (report only)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.audit:
        return audit(a.audit, a.out)
    if not (a.pred_dir and a.gt_dir):
        ap.error("--pred_dir and --gt_dir (strict mode), or --audit")
    n, bad = check_dir(a.pred_dir, a.gt_dir)
    if bad:
        print(f"[geometry] REFUSING {a.pred_dir}: {len(bad)}/{n} predictions are not on the native GT grid "
              f"(RAS->native resample missing?)", file=sys.stderr)
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"[geometry] OK {a.pred_dir}: {n} predictions on the native GT grid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
