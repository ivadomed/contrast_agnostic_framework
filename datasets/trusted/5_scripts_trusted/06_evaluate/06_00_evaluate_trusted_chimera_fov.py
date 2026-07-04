#!/usr/bin/env python3
"""
TRUSTED CHIMERA evaluator — FOV-restricted, side-aware kidney Dice/HD95.

Unlike the ROI-only chimera evaluator (06_00_evaluate_trusted_chimera.py, which
scores STRICTLY inside the pasted-US footprint as a context-injection control), this
one measures the kidney segmentation the way the standard cross-dataset evals do:
regular Dice/HD95 over the CHAOS-equivalent axial slab, so a model that fires the
kidney label ELSEWHERE IN THE FOV is penalised as a false positive.

Per chimera case (a full-torso CT with US kidney(s) pasted at the real kidney site):
  * FOV slab  — anchored on the CT REAL-kidney GT (both kidneys; BIDS CT_dseg, which
                shares the chimera grid), expanded by the CHAOS kidney margins. Both
                prediction and GT are cropped to this slab.
  * GT        — the placed US kidney mask (labelsTs_chimera): only the inserted
                side(s) are present.
  * pred      — chaos labels, made SIDE-AWARE from the manifest:
                  one side inserted  → only that side's chaos label
                                       (R→right_kidney 2, L→left_kidney 3)
                  both sides         → union {2,3}
                In all cases the scored class is reported as "kidney".

So for a one-sided (say left) chimera we compare the model's LEFT-kidney prediction
across the whole FOV against the single placed US kidney — the other side's real CT
kidney is not part of the experiment and its label is ignored.

Usage:
  python 06_00_evaluate_trusted_chimera_fov.py --pred_dir D --gt_dir D
         --manifest chimera_manifest.csv --ct_gt_dir <BIDS manual_masks>
         --out_csv C [--name chimera] [--fov_sup_mm 117] [--fov_inf_mm 18] [--workers N]
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_UTILS = Path(__file__).resolve().parents[3] / "00_commun_scripts" / "00_00_utils"
sys.path.insert(0, str(_UTILS))
import eval_metrics as em  # noqa: E402
import fov as _fov         # noqa: E402

SIDE_PRED = {"R": 2, "L": 3}   # manifest side char → chaos kidney label id


def _ct_dseg_path(ct_gt_dir: Path, pid: str) -> Path:
    return ct_gt_dir / f"sub-{pid}" / "anat" / f"sub-{pid}_CT_dseg.nii.gz"


def _process(args):
    pf, gf, ct_gt_path, sides, name, sup_mm, inf_mm = args
    gt_img = sitk.ReadImage(gf)
    spacing = gt_img.GetSpacing()[::-1]
    gt = sitk.GetArrayFromImage(gt_img)
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pf))
    case = Path(pf).name[:-7]

    axis, sup_dir, sp_si = _fov.si_axis_sign(gt_img)
    # FOV anchor: the CT real-kidney GT (both kidneys). Falls back to the placed US
    # kidney if the CT dseg is missing, so we still get a slab.
    if ct_gt_path and Path(ct_gt_path).exists():
        anchor = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_gt_path))) > 0
    else:
        print(f"  WARNING: CT dseg missing for {case} — anchoring FOV on the placed US kidney")
        anchor = gt > 0
    bb = _fov.bbox_si(anchor, axis)
    if bb is None:
        return [{"group": name, "case": case, "label": "kidney",
                 "dice": float("nan"), "hd95": float("nan")}]
    lo, hi = _fov.slab_keep_range(gt.shape, axis, sup_dir, sp_si, bb[0], bb[1], sup_mm, inf_mm)

    # Crop to the slab along the S-I axis (exact for Dice/HD95, keeps memory low).
    sl = [slice(None)] * gt.ndim
    sl[axis] = slice(lo, hi + 1)
    sl = tuple(sl)
    gt, pred = gt[sl], pred[sl]

    # Side-aware prediction foreground.
    pred_ids = [SIDE_PRED[s] for s in sides if s in SIDE_PRED]
    if not pred_ids:                       # shouldn't happen for "ok" manifest rows
        pred_ids = [2, 3]
    pk = np.isin(pred, pred_ids)
    gk = gt > 0
    return [{"group": name, "case": case, "label": "kidney",
             "dice": em.dice_score(pk, gk), "hd95": em.hd95(pk, gk, spacing)}]


def _load_manifest(path):
    sides = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                sides[r["patient"]] = list(r["sides_inserted"])
    return sides


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True, help="labelsTs_chimera (placed US kidney GT)")
    ap.add_argument("--manifest", required=True, help="chimera_manifest.csv")
    ap.add_argument("--ct_gt_dir", required=True,
                    help="BIDS derivatives/manual_masks dir (holds sub-<pid>/anat/sub-<pid>_CT_dseg.nii.gz)")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--name", default="chimera")
    ap.add_argument("--fov_sup_mm", type=float, required=True)
    ap.add_argument("--fov_inf_mm", type=float, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    pred_dir, gt_dir, ct_gt_dir = Path(args.pred_dir), Path(args.gt_dir), Path(args.ct_gt_dir)
    sides_by_pid = _load_manifest(args.manifest)
    preds = sorted(pred_dir.glob("*.nii.gz"))
    if not preds:
        raise SystemExit(f"No predictions in {pred_dir}")

    work = []
    for p in preds:
        pid = p.name[:-7]
        gt_p = gt_dir / p.name
        if not gt_p.exists() or pid not in sides_by_pid:
            continue
        work.append((str(p), str(gt_p), str(_ct_dseg_path(ct_gt_dir, pid)),
                     sides_by_pid[pid], args.name, args.fov_sup_mm, args.fov_inf_mm))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_process, w) for w in work]):
            rows.extend(fut.result())

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "case", "label", "dice", "hd95"])
        w.writeheader(); w.writerows(rows)

    d = np.array([r["dice"] for r in rows if r["label"] == "kidney"], float)
    print(f"[chimera-fov] {len(work)} cases  |  slab = kidney bbox +{args.fov_sup_mm}mm sup / +{args.fov_inf_mm}mm inf")
    if len(d):
        print(f"  kidney (n={int(np.isfinite(d).sum())}): Dice {np.nanmean(d)*100:5.1f} ± {np.nanstd(d)*100:4.1f}")
    print(f"  → {out}")


if __name__ == "__main__":
    main()
