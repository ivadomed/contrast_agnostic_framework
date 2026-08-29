#!/usr/bin/env python3
"""
Build the nnUNet test-input dirs for the LiverHccSeg evaluation set.

4 items -- ce-pre_T1w, ce-art_T1w, ce-pv_T1w, ce-del_T1w (the 4 native phases). Only
the 14/17 patients with a tumour mask are included (matches what we score -- tumour
only, see 00_utils/env.sh) -- the 3 liver-only patients (no tumour annotation) are
skipped entirely for this eval-only test set. atlas-liver-hcc's models are
single-channel, so each volume is fed as channel _0000. Uses rater1 as the active GT
(see 00_utils/00_01_bidsify.py docstring).

Reads:  ../../1_BIDS_liverhccseg/liverhccseg/sub-liverhccsegNNN/anat/  (+ derivatives)
        ../../1_BIDS_liverhccseg/liverhccseg/case_id_map.json (has_tumour flag)
Writes: ../../2_nnUNet_liverhccseg/raw/imagesTs_{item}/{case}_0000.nii.gz
                                       /labelsTs_{item}/{case}.nii.gz
        for item in {ce-pre_T1w, ce-art_T1w, ce-pv_T1w, ce-del_T1w}

    python 05_00_build_test_inputs.py
"""
import gzip
import json
import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_liverhccseg" / "liverhccseg"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_liverhccseg" / "raw"

PHASES = ["ce-pre_T1w", "ce-art_T1w", "ce-pv_T1w", "ce-del_T1w"]
RATER = "rater1"


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".gz":
        shutil.copyfile(src, dst)
    else:
        with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out)


def main() -> None:
    case_id_map = json.loads((BIDS_ROOT / "case_id_map.json").read_text())
    subs = sorted(sub for sub, info in case_id_map.items() if info["has_tumour"])
    if not subs:
        raise SystemExit(f"No tumour-annotated subjects in {BIDS_ROOT}/case_id_map.json — run 00_utils/00_01_bidsify.py first.")

    for item in PHASES:
        img_dir = NNUNET_RAW / f"imagesTs_{item}"
        lab_dir = NNUNET_RAW / f"labelsTs_{item}"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        n_ok, missing = 0, []
        for sub in subs:
            case = sub[len("sub-"):]                       # liverhccseg000
            img = BIDS_ROOT / sub / "anat" / f"{sub}_{item}.nii.gz"
            seg = DERIV_DIR / sub / "anat" / f"{sub}_{item}_tumour-{RATER}_dseg.nii.gz"
            if not img.exists() or not seg.exists():
                missing.append(case)
                continue
            gzip_copy(img, img_dir / f"{case}_0000.nii.gz")
            gzip_copy(seg, lab_dir / f"{case}.nii.gz")
            n_ok += 1

        status = f"{n_ok}/{len(subs)} -> {img_dir.name} (+labels)"
        if missing:
            status += f"  MISSING: {missing}"
        print(status)


if __name__ == "__main__":
    main()
