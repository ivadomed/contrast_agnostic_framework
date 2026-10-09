#!/usr/bin/env python3
"""
Materialize totalseg-pelvic's held-out test sets as nnU-Net predict inputs, for BOTH
per-modality datasets (own-contrast AND cross-contrast — a CT-trained model needs
imagesTs_mri to measure cross-contrast Dice, and vice versa, same convention as every
other multi-modality dataset here).

Unlike autopet (paired 2-channel archive, one consolidated raw tree), CT and MRI here are
UNPAIRED — different patients entirely — so there is no "same case, other channel"
relationship. Cross-contrast evaluation instead means: predict a CT-trained model on the
MRI test set (and a MRI-trained model on the CT test set), each already has its own real
label map in the SAME 10-class label space (see 02_nnunet/02_00_convert.py's LABEL_MAP).

Reads each modality's sealed test list from 4_splits_totalseg-pelvic/<modality>/test_cases.json
and hard-links those cases into a flat pair of dirs per item, consolidated under
Dataset130_TotalsegPelvic_CT's own raw tree (same "one consolidated target dataset"
convention as chaos/autopet's build_test_inputs, so predict_common.sh's "own" mode always
resolves imagesTs_<item> under ONE fixed PREDICT_DATASET_ID_DEFAULT regardless of which
per-modality Dataset actually trained the model):

    <nnUNet_raw>/Dataset130_TotalsegPelvic_CT/imagesTs_ct/<case>_0000.nii.gz
    <nnUNet_raw>/Dataset130_TotalsegPelvic_CT/labelsTs_ct/<case>.nii.gz
    <nnUNet_raw>/Dataset130_TotalsegPelvic_CT/imagesTs_mri/<case>_0000.nii.gz   (source: Dataset131's imagesTr)
    <nnUNet_raw>/Dataset130_TotalsegPelvic_CT/labelsTs_mri/<case>.nii.gz

Hard links, not copies. Do not delete from imagesTr — nnU-Net's dataset fingerprint is
built from that dir.

Run via 05_00_build_test_inputs.sh.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(DATASET_ROOT / "2_nnUNet_totalseg-pelvic" / "raw")))
SPLITS_ROOT = Path(os.environ.get("SPLITS_ROOT", str(DATASET_ROOT / "4_splits_totalseg-pelvic")))

SOURCE_DATASETS = {"ct": "Dataset130_TotalsegPelvic_CT", "mri": "Dataset131_TotalsegPelvic_MRI"}
TARGET_DATASET = "Dataset130_TotalsegPelvic_CT"


def _link_test_set(source_ds_name: str, item: str) -> None:
    src_ds = NNUNET_RAW / source_ds_name
    target_ds = NNUNET_RAW / TARGET_DATASET
    test = json.loads((SPLITS_ROOT / item / "test_cases.json").read_text())["test"]
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
