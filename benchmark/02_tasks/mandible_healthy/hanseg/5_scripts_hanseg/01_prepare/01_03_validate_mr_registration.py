#!/usr/bin/env python3
"""
Does the MR arm's ground truth actually sit on the mandible?

This script exists because the QC used when building the MR arm (`tissue_frac`, the
fraction of propagated-mask voxels landing on non-background MR signal) is a
CATASTROPHE detector, not an accuracy measure. The mandible is surrounded by soft
tissue, so a mask displaced by a centimetre still lands almost entirely "on tissue".
Observed range across 42 cases was 0.955-1.000 with a single outlier at 0.058 — no
dynamic range in the regime that matters.

THE TEST: take each case's propagated mask and deliberately displace it by known
amounts, then recompute both metrics at each displacement.

  * tissue_frac(d)  — if this stays flat as d grows, the QC is provably blind and the
                      GT is UNVALIDATED, whatever its value at d=0.
  * edge_score(d)   — mean MR gradient magnitude on the mask's boundary shell,
                      normalised by the volume's mean gradient. This is the metric
                      that SHOULD peak at d=0 if the mandible's boundary is actually
                      visible in T1 and the registration found it.

READING THE RESULT
  edge_score peaks sharply at d=0      -> registration is locally locked onto a real
                                          image edge; GT has support.
  edge_score flat / peak away from 0   -> the mandible boundary is not a usable
                                          feature in T1 here, so the registration is
                                          unconstrained in the direction that matters
                                          and the propagated GT should NOT be trusted.

Deliberately makes no claim either way in advance — the point is to find out.

Run via 01_03_validate_mr_registration.sh
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

BIDS_ROOT = Path(os.environ["BIDS_ROOT"])
OUT = BIDS_ROOT / "mr_registration_validation.json"
SHIFTS_MM = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
N_CASES = int(os.environ.get("VAL_N_CASES", "0"))    # 0 = all


def boundary_shell(mask: np.ndarray) -> np.ndarray:
    """1-voxel shell just inside the mask surface."""
    er = ndimage.binary_erosion(mask, iterations=1)
    return mask & ~er


def metrics_at(mr: np.ndarray, grad: np.ndarray, mask: np.ndarray,
               thr: float) -> tuple[float, float]:
    if mask.sum() == 0:
        return float("nan"), float("nan")
    tissue = float((mr[mask] > thr).mean())
    shell = boundary_shell(mask)
    edge = float(grad[shell].mean()) if shell.sum() else float("nan")
    return tissue, edge


def main() -> None:
    subs = sorted(p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir())
    if N_CASES:
        subs = subs[:N_CASES]
    rows = []
    for sub in subs:
        img_p = BIDS_ROOT / sub / "anat" / f"{sub}_T1w.nii.gz"
        seg_p = BIDS_ROOT / "derivatives" / "labels" / sub / "anat" / f"{sub}_T1w_label-mandible_seg.nii.gz"
        if not (img_p.exists() and seg_p.exists()):
            continue
        mr = np.asanyarray(nib.load(str(img_p)).dataobj).astype(np.float32)
        seg = np.asanyarray(nib.load(str(seg_p)).dataobj) > 0
        if seg.sum() == 0:
            continue
        zoom = np.abs(np.diag(nib.load(str(img_p)).affine)[:3])
        thr = float(np.percentile(mr[mr > mr.min()], 25))
        gx, gy, gz = np.gradient(ndimage.gaussian_filter(mr, 1.0))
        grad = np.sqrt(gx**2 + gy**2 + gz**2)
        gmean = float(grad.mean()) or 1.0

        per_axis = {}
        for axis, ax_name in enumerate("xyz"):
            vals = []
            for d in SHIFTS_MM:
                sh = int(round(d / zoom[axis]))
                shifted = np.roll(seg, sh, axis=axis) if sh else seg
                t, e = metrics_at(mr, grad, shifted, thr)
                vals.append({"mm": d, "tissue_frac": round(t, 4),
                             "edge_score": round(e / gmean, 4)})
            per_axis[ax_name] = vals
        rows.append({"sub": sub, "per_axis": per_axis})
        print(f"  {sub} done", flush=True)

    # aggregate: mean over cases and axes at each displacement
    agg = []
    for i, d in enumerate(SHIFTS_MM):
        tf = [r["per_axis"][a][i]["tissue_frac"] for r in rows for a in "xyz"]
        es = [r["per_axis"][a][i]["edge_score"] for r in rows for a in "xyz"]
        agg.append({"mm": d,
                    "tissue_frac_mean": round(float(np.nanmean(tf)), 4),
                    "edge_score_mean": round(float(np.nanmean(es)), 4)})
    OUT.write_text(json.dumps({"shifts_mm": SHIFTS_MM, "n_cases": len(rows),
                               "aggregate": agg, "cases": rows}, indent=2))

    print("\n=== DISPLACEMENT SENSITIVITY (mean over cases x 3 axes) ===")
    print(f"{'shift mm':>9} {'tissue_frac':>12} {'edge_score':>11}")
    for a in agg:
        print(f"{a['mm']:>9.0f} {a['tissue_frac_mean']:>12.4f} {a['edge_score_mean']:>11.4f}")
    t0, t12 = agg[0]["tissue_frac_mean"], agg[-1]["tissue_frac_mean"]
    e0, e12 = agg[0]["edge_score_mean"], agg[-1]["edge_score_mean"]
    print(f"\ntissue_frac drop 0->12mm : {t0 - t12:+.4f}   "
          f"({'BLIND — cannot detect a 12mm error' if (t0 - t12) < 0.05 else 'responsive'})")
    print(f"edge_score  drop 0->12mm : {e0 - e12:+.4f}   "
          f"({'peaks at 0 (registration locked on a real edge)' if e0 > e12 else 'NOT peaked at 0 — unconstrained'})")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
