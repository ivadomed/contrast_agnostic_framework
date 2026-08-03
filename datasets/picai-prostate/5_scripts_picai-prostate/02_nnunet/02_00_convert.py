#!/usr/bin/env python3
"""
Convert the picai-prostate BIDS tree → an nnU-Net raw dataset, ONE per training contrast:

  --contrast t2w  →  Dataset080_PICAI_T2W   (training modality 1)
  --contrast adc  →  Dataset081_PICAI_ADC   (training modality 2)

Single input channel in both cases. Each dataset carries the SAME held-out test set in all
three contrasts, because the headline result of this project is cross-contrast
generalisation — a model trained on T2W is scored on T2W *and* ADC *and* HBV. Exactly the
open-ms layout (imagesTr + per-contrast imagesTs_<c>/labelsTs_<c>).

HBV never gets its own Dataset0xx: it is a test-only contrast, so it only ever appears as
imagesTs_hbv/labelsTs_hbv inside the two training datasets.

Requires 00_01_bidsify.sh (BIDS tree + cases.json) and 01_01_create_splits.sh
(partition.json) to have run.

Output under 2_nnUNet_picai-prostate/raw/Dataset08X_PICAI_<C>/:
  imagesTr/<case>_0000.nii.gz        training contrast, train pool
  labelsTr/<case>.nii.gz             csPCa lesion mask (uint8, {0,1})
  imagesTs_{t2w,adc,hbv}/<case>_0000.nii.gz   held-out test, one dir per contrast
  labelsTs_{t2w,adc,hbv}/<case>.nii.gz        same mask (common grid) per contrast
  dataset.json

Files are HARD-LINKED from the BIDS tree, not copied: the two datasets plus their three
test-contrast dirs would otherwise store ~5 redundant copies of every volume, and tamia's
binding quota is file count / space on $SCRATCH. Falls back to a copy across filesystems.

Run:  bash 02_nnunet/02_00_convert.sh          (t2w)
      bash 02_nnunet/02_01_convert_adc.sh      (adc)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]          # datasets/picai-prostate

# training contrast -> (nnUNet dataset dir name, nnUNet channel name)
DATASETS = {
    "t2w": ("Dataset080_PICAI_T2W", "T2W"),
    "adc": ("Dataset081_PICAI_ADC", "ADC"),
}
# nnUNet test-input dir suffix -> BIDS suffix (per contrast). All three share one grid.
CONTRASTS = {"t2w": "T2w", "adc": "ADC", "hbv": "HBV"}


def _bids_paths(bids_root: Path, case_id: str, bids_suffix: str) -> tuple[Path, Path]:
    pid, sid = case_id.split("_", 1)
    img = bids_root / f"sub-{pid}" / f"ses-{sid}" / "anat" / f"sub-{pid}_ses-{sid}_{bids_suffix}.nii.gz"
    mask = (bids_root / "derivatives" / "manual_masks" / f"sub-{pid}" / f"ses-{sid}" / "anat"
            / f"sub-{pid}_ses-{sid}_dseg.nii.gz")
    return img, mask


def _link(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True, choices=sorted(DATASETS))
    args = ap.parse_args()
    ds_name, channel_name = DATASETS[args.contrast]

    bids_root = Path(os.environ.get(
        "BIDS_ROOT", DATASET_ROOT / "1_BIDS_picai-prostate" / "picai-prostate-bpmri"))
    splits_dir = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_picai-prostate"))
    nnunet_raw = Path(os.environ.get(
        "nnUNet_raw", DATASET_ROOT / "2_nnUNet_picai-prostate" / "raw"))
    out = nnunet_raw / ds_name

    partition = json.loads((splits_dir / "partition.json").read_text())
    train_pool, test = partition["train_pool"], partition["test"]

    # --- training set: the training contrast + mask ---
    for cid in train_pool:
        img, mask = _bids_paths(bids_root, cid, CONTRASTS[args.contrast])
        _link(img, out / "imagesTr" / f"{cid}_0000.nii.gz")
        _link(mask, out / "labelsTr" / f"{cid}.nii.gz")

    # --- held-out test: one imagesTs_<contrast>/ + labelsTs_<contrast>/ per contrast ---
    for suffix, bids_suffix in CONTRASTS.items():
        for cid in test:
            img, mask = _bids_paths(bids_root, cid, bids_suffix)
            _link(img, out / f"imagesTs_{suffix}" / f"{cid}_0000.nii.gz")
            _link(mask, out / f"labelsTs_{suffix}" / f"{cid}.nii.gz")

    (out / "dataset.json").write_text(json.dumps({
        "name": ds_name.split("_", 1)[1],
        "description": (
            "PI-CAI public training/development set (Saha et al. 2022), csPCa-positive "
            f"studies only — clinically significant prostate cancer LESION segmentation. "
            f"Train on {channel_name}; cross-contrast test on T2W/ADC/HBV, all resampled "
            "onto one prostate-centred 0.5x0.5x3.0 mm grid so a single mask serves every "
            "contrast."),
        "reference": "https://pi-cai.grand-challenge.org/ ; "
                     "https://doi.org/10.5281/zenodo.6624726 ; "
                     "labels: github.com/DIAGNijmegen/picai_labels (human expert)",
        "licence": "CC-BY-NC-4.0",
        "release": "1.0",
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(train_pool),
        "file_ending": ".nii.gz",
    }, indent=2))

    print(f"{ds_name} written → {out}")
    print(f"  imagesTr: {len(list((out / 'imagesTr').glob('*.nii.gz')))}  "
          f"labelsTr: {len(list((out / 'labelsTr').glob('*.nii.gz')))}")
    for suffix in CONTRASTS:
        print(f"  imagesTs_{suffix}: {len(list((out / f'imagesTs_{suffix}').glob('*.nii.gz')))}  "
              f"labelsTs_{suffix}: {len(list((out / f'labelsTs_{suffix}').glob('*.nii.gz')))}")


if __name__ == "__main__":
    main()
