#!/usr/bin/env python3
"""
Build per-modality nnUNet test-input dirs for the CHAOS internal test set.

The segmentation model is single-channel (trained on MR T1 in-phase as channel
_0000). To measure cross-modality generalization we feed each available modality in
turn as channel _0000. This materialises, per modality, an image dir + matching GT
label dir (so 06_00_evaluate.py can score predictions).

Test groups (from 4_splits_chaos/test_cases.json):
  - MR internal-test patients → t1in, t1out, t2spir   (4 organs scoreable)
  - all CT patients           → ct                     (liver only scoreable)

Reads:  ../../1_BIDS_chaos/chaos-abdominal/sub-<case>/anat/  (+ derivatives masks)
Writes: ../../2_nnUNet_chaos/raw/<DS_NAME>/imagesTs_<modality>/{case}_0000.nii.gz
                                          /labelsTs_<modality>/{case}.nii.gz

    python 05_00_build_test_inputs.py                    # all modalities
    python 05_00_build_test_inputs.py --modalities t1in ct
"""
import argparse
import gzip
import json
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_chaos" / "chaos-abdominal"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
TEST_CASES   = DATASET_ROOT / "4_splits_chaos" / "test_cases.json"

# modality tag → BIDS anat image suffix (without sub- prefix / extension)
MODALITY_SUFFIX = {
    "t1in":   "acq-inphase_T1w",
    "t1out":  "acq-outphase_T1w",
    "t2spir": "T2w",
    "ct":     "CT",
}


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=60)
    ap.add_argument("--modalities", nargs="+", default=list(MODALITY_SUFFIX),
                    choices=list(MODALITY_SUFFIX))
    args = ap.parse_args()

    meta = json.loads(TEST_CASES.read_text())
    cases_for = {m: meta["mr_internal_test"] for m in meta["mr_test_modalities"]}
    for m in meta["ct_test_modalities"]:
        cases_for[m] = meta["ct_test"]

    ds_name  = f"Dataset{args.dataset_id:03d}_CHAOS_MR_T1in"
    out_root = NNUNET_RAW / ds_name

    for mod in args.modalities:
        suffix = MODALITY_SUFFIX[mod]
        cases  = cases_for.get(mod, [])
        img_dir = out_root / f"imagesTs_{mod}"
        lab_dir = out_root / f"labelsTs_{mod}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for case in cases:
            sub = f"sub-{case}"
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_dseg.nii"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue
            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1
        status = f"{n_ok}/{len(cases)} → {img_dir.name} (+labels)"
        if missing:
            status += f"  [MISSING {len(missing)}: {missing[:5]}]"
        print(f"  {mod:7s} ({suffix:>18s}): {status}")


if __name__ == "__main__":
    main()
