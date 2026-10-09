#!/usr/bin/env python3
"""
Materialize toothfairy2's held-out CBCT test set as nnU-Net predict inputs.

Reads the sealed test list from 4_splits_toothfairy2/test_cases.json (written by
01_create_splits — the SINGLE source of truth for which cases are held out; nothing
downstream re-derives it) and hard-links those cases out of the training
Dataset110 tree into a flat pair of dirs:

    <nnUNet_raw>/Dataset110_ToothFairy2CBCT/imagesTs_cbct/<case>_0000.nii.gz
    <nnUNet_raw>/Dataset110_ToothFairy2CBCT/labelsTs_cbct/<case>.nii.gz

Hard links, not copies: same filesystem, no extra disk, and the training tree stays
the one place the voxels live.

⚠️ These cases are ALSO present in imagesTr/ (nnU-Net's raw dir holds every case);
what makes them "held out" is that 01_create_splits excluded them from every fold's
train AND val list, and the trainer's own contamination guard
(toothfairy2/trainers/base.py) re-checks that from test_cases.json at training time.
Do not "clean up" imagesTr by deleting them — nnU-Net's dataset fingerprint is built
from that dir.

Run via 05_00_build_test_inputs.sh.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW = Path(os.environ["nnUNet_raw"])
SPLITS = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_toothfairy2"))
DS_NAME = os.environ.get("NNUNET_DATASET_ID", "Dataset110_ToothFairy2CBCT")
ITEM = "cbct"


def main() -> None:
    ds = NNUNET_RAW / DS_NAME
    test = json.loads((SPLITS / "test_cases.json").read_text())["test"]
    img_ts, lab_ts = ds / f"imagesTs_{ITEM}", ds / f"labelsTs_{ITEM}"
    img_ts.mkdir(parents=True, exist_ok=True)
    lab_ts.mkdir(parents=True, exist_ok=True)

    n = 0
    missing = []
    for cid in sorted(test):
        pairs = [(ds / "imagesTr" / f"{cid}_0000.nii.gz", img_ts / f"{cid}_0000.nii.gz"),
                 (ds / "labelsTr" / f"{cid}.nii.gz",      lab_ts / f"{cid}.nii.gz")]
        if not all(src.exists() for src, _ in pairs):
            missing.append(cid)
            continue
        for src, dst in pairs:
            if dst.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)
        n += 1

    if missing:
        raise SystemExit(f"{len(missing)} test cases absent from the raw tree: {missing[:10]}")
    print(f"{n} held-out test cases -> {img_ts} / {lab_ts}")


if __name__ == "__main__":
    main()
