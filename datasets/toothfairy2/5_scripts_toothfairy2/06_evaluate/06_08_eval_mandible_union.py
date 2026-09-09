#!/usr/bin/env python3
"""
Score toothfairy2's OWN held-out CBCT test set on the MANDIBLE UNION, so the
cross-modality table compares like with like.

WHY THIS EXISTS
---------------
The headline cross-dataset table was averaging incommensurable things. Its `cbct`
column was a 3-class macro (mandible + lower_teeth + pharynx) while its `hanseg_ct`
column was a SINGLE class (mandible union mandible+lower_teeth, because HaN-Seg
delineates one Bone_Mandible that includes the lower dentition). The `all` column then
averaged a 3-class score with a 1-class score, and the in-domain -> OOD drop conflated
"harder modality" with "different label set".

This script removes that confound by scoring the in-domain set on exactly the structure
hanseg scores: predictions and ground truth both collapsed to
mandible ∪ lower_teeth, pharynx -> background. Same merge used on the hanseg side
(05_20_merge_mandible_union.py), applied to the GT as well so both sides are binary.

The 3-class in-domain table is NOT replaced — it stays as the characterisation of the
full task. This produces an ADDITIONAL, label-consistent view for the cross-modality
comparison. Both are reported; they answer different questions.

Note the ladder was already label-consistent (its OOD values come only from hanseg), so
the rung 4->5 result is unaffected by this.

Writes: <METRICS>/toothfairy2_model/cbct[/<subdir>]/<cat>_<run>/fold{k}/cbct_union_metrics.csv

Usage: 06_08_eval_mandible_union.py --run <RUN_ID> --category <nnUNet|auglab> [--subdir ablations]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DATASET_ROOT / "5_scripts_toothfairy2" / "00_utils"))
from toothfairy2_labels import MANDIBLE_UNION_TARGET_IDS  # noqa: E402

COMMUN = DATASET_ROOT.parent / "00_commun_scripts" / "00_03_evaluate"


def binarize(src_dir: Path, dst_dir: Path) -> int:
    """Collapse a 3-class volume dir to the mandible union {0,1}."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.nii.gz")):
        out = dst_dir / f.name
        if out.exists():
            n += 1
            continue
        img = nib.load(str(f))
        arr = np.asanyarray(img.dataobj)
        u = np.isin(arr, MANDIBLE_UNION_TARGET_IDS).astype(np.uint8)
        nib.save(nib.Nifti1Image(u, img.affine, img.header), str(out))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--category", required=True, choices=["nnUNet", "auglab"])
    ap.add_argument("--subdir", default="")
    ap.add_argument("--folds", default="0 1 2")
    a = ap.parse_args()

    preds_root = Path(os.environ["PREDICTIONS_ROOT"]) / "toothfairy2_model" / "cbct" / a.category / a.run
    metrics_root = Path(os.environ["METRICS_ROOT"]) / "toothfairy2_model" / "cbct"
    raw = Path(os.environ["nnUNet_raw"]) / os.environ.get("NNUNET_DATASET_ID", "Dataset110_ToothFairy2CBCT")
    gt_dir = raw / "labelsTs_cbct"

    # GT is binarised ONCE into a sibling dir and reused across runs.
    gt_union = raw / "labelsTs_cbct_mandible_union"
    ngt = binarize(gt_dir, gt_union)
    print(f"GT union: {ngt} cases -> {gt_union}")

    for F in a.folds.split():
        pred = preds_root / f"fold{F}" / "cbct"
        if not pred.is_dir():
            print(f"  skip fold{F}: no predictions at {pred}")
            continue
        pu = pred.parent / "cbct_mandible_union"
        npd = binarize(pred, pu)
        out = metrics_root / (a.subdir or "") / f"{a.category}_{a.run}" / f"fold{F}"
        out.mkdir(parents=True, exist_ok=True)
        csv = out / "cbct_union_metrics.csv"
        cmd = [str(DATASET_ROOT.parent.parent / ".venv" / "bin" / "python"),
               str(DATASET_ROOT / "5_scripts_toothfairy2" / "06_evaluate" / "06_00_evaluate.py"),
               "--pred_dir", str(pu), "--gt_dir", str(gt_union),
               "--label_map", '{"mandible": [1, 1]}',
               "--name", "cbct_union", "--out_csv", str(csv), "--workers", "6"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  fold{F} FAILED: {r.stderr[-400:]}", file=sys.stderr)
            continue
        rows = sum(1 for _ in open(csv)) - 1
        print(f"  fold{F}: {npd} preds -> {rows} rows -> {csv}")


if __name__ == "__main__":
    main()
