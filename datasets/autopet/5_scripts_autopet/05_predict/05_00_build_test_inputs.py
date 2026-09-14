#!/usr/bin/env python3
"""
Materialize autopet's held-out FDG test set as nnU-Net predict inputs, for BOTH
per-modality datasets (own-contrast AND cross-contrast — a CT-trained model still needs
imagesTs_pet to measure cross-contrast Dice, and vice versa, exactly like every other
dataset here).

Reads the sealed test list from 4_splits_autopet/test_cases.json (written by
01_create_splits — the single source of truth for which cases are held out) and
hard-links those cases into a flat pair of dirs per item — ALL consolidated under
Dataset120_AutoPET_CT's own raw tree (same convention as chaos: predict_common.sh's
"own" mode resolves imagesTs_<item> under ONE fixed PREDICT_DATASET_ID_DEFAULT
regardless of which per-modality Dataset actually trained the model, so every item a
model might be evaluated on — including cross-contrast ones it never trained on — has to
live in that one place):

    <nnUNet_raw>/Dataset120_AutoPET_CT/imagesTs_ct/<case>_0000.nii.gz
    <nnUNet_raw>/Dataset120_AutoPET_CT/labelsTs_ct/<case>.nii.gz
    <nnUNet_raw>/Dataset120_AutoPET_CT/imagesTs_pet/<case>_0000.nii.gz   (source: Dataset121's imagesTr)
    <nnUNet_raw>/Dataset120_AutoPET_CT/labelsTs_pet/<case>.nii.gz

Hard links, not copies. These cases are ALSO present in each Dataset's own imagesTr/ —
what makes them "held out" is 01_create_splits excluding them from every fold's train/val,
re-checked by the trainer's own contamination guard (autopet/trainers/base.py). Do not
delete them from imagesTr — nnU-Net's dataset fingerprint is built from that dir.

See 05_00b_build_psma_test_inputs.py for the SEPARATE cross-institution PSMA test set
(not part of this script — PSMA was never converted into Dataset120/121's imagesTr at
all, since it's eval-only and never trains).

Run via 05_00_build_test_inputs.sh.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(DATASET_ROOT / "2_nnUNet_autopet" / "raw")))
SPLITS = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_autopet")))

# item -> the Dataset it SOURCES images/labels from (its own imagesTr/labelsTr).
SOURCE_DATASETS = {"ct": "Dataset120_AutoPET_CT", "pet": "Dataset121_AutoPET_PET"}
# Every item's imagesTs_<item>/labelsTs_<item> is consolidated under this ONE dataset's
# raw tree, regardless of source — see module docstring.
TARGET_DATASET = "Dataset120_AutoPET_CT"


def _link_test_set(source_ds_name: str, item: str) -> None:
    src_ds = NNUNET_RAW / source_ds_name
    target_ds = NNUNET_RAW / TARGET_DATASET
    test = json.loads((SPLITS / "test_cases.json").read_text())["test"]
    img_ts, lab_ts = target_ds / f"imagesTs_{item}", target_ds / f"labelsTs_{item}"
    img_ts.mkdir(parents=True, exist_ok=True)
    lab_ts.mkdir(parents=True, exist_ok=True)

    n, missing = 0, []
    for cid in sorted(test):
        pairs = [(src_ds / "imagesTr" / f"{cid}_0000.nii.gz", img_ts / f"{cid}_0000.nii.gz"),
                 (src_ds / "labelsTr" / f"{cid}.nii.gz", lab_ts / f"{cid}.nii.gz")]
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

    print(f"[build_test_inputs] {TARGET_DATASET}/imagesTs_{item} (from {source_ds_name}): "
          f"{n} cases linked, {len(missing)} missing "
          f"({missing[:5]}{'...' if len(missing) > 5 else ''})")


def main() -> None:
    for item, source_ds_name in SOURCE_DATASETS.items():
        _link_test_set(source_ds_name, item)


if __name__ == "__main__":
    main()
