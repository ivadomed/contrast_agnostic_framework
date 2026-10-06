#!/usr/bin/env python3
"""
Resample RAS-space predictions back to native geometry, IN PLACE.

Extracted verbatim (same SimpleITK logic, unchanged) from the python heredoc that used to
run right after nnUNetv2_predict inline in 06_evaluate/06_01_evaluate_testset.sh.

Why this exists as a separate step rather than living inside predict_common.sh: on-harmony
is the only dataset that reorients its predict INPUT to RAS-canonical (see
05_00_build_test_inputs.py's docstring -- bold/dwi/epi are natively LAS and would otherwise
be mirrored relative to the RAS-trained network). Every other dataset predicts directly in
native space, so predict_common.sh (shared by all datasets) has no resample-back step and
none should be added there for one dataset's quirk. This script is on-harmony-local
post-processing, run by 05_01_predict_common.sh INSIDE each predict job (the shared driver's
PREDICT_POST_ITEM_CMD hook) right after that item's nnUNetv2_predict. Exits non-zero if a
prediction has no native reference (it would otherwise be scored on the RAS grid).

Usage: 05_02_resample_predictions_to_native.py <PRED_DIR> <REF_DIR>
  PRED_DIR: directory of *.nii.gz predictions in RAS space (overwritten in place)
  REF_DIR:  directory of native-geometry reference images (TESTSET/<contrast>/images_native)
            -- filenames are REF_DIR/<pred_stem>_0000.nii.gz
"""
from __future__ import annotations

import sys
from pathlib import Path

import SimpleITK as sitk


def main() -> None:
    pred_dir = Path(sys.argv[1])
    ref_dir = Path(sys.argv[2])
    if not pred_dir.is_dir():
        print(f"[resample_to_native] skip (no such dir): {pred_dir}")
        return
    n, missing = 0, []
    for p in sorted(pred_dir.glob("*.nii.gz")):
        cid = p.stem.replace(".nii", "")
        ref = ref_dir / f"{cid}_0000.nii.gz"
        if not ref.exists():
            missing.append(cid)
            continue
        pr = sitk.ReadImage(str(p), sitk.sitkUInt8)
        rf = sitk.ReadImage(str(ref))
        resampled = sitk.Resample(pr, rf, sitk.Transform(), sitk.sitkNearestNeighbor, 0, pr.GetPixelID())
        sitk.WriteImage(resampled, str(p))
        n += 1
    print(f"[resample_to_native] {pred_dir}: {n} predictions resampled to native geometry")
    if missing:
        sys.exit(f"[resample_to_native] ERROR {pred_dir}: no native reference in {ref_dir} for {missing}")


if __name__ == "__main__":
    main()
