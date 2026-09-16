#!/usr/bin/env python3
"""
Convert `1_BIDS_totalseg-pelvic/pelvis-totalsegpelvic/` (BIDS-staged by
00_utils/00_01_bidsify.py) into nnU-Net raw datasets for CT and MRI.

Rewritten 2026-09-15: the original version of this script converted straight from the
Zenodo zips to nnU-Net format with no BIDS staging in between (the pattern
autopet/duke-breast-mri use deliberately) -- wrong call for this dataset, per direct user
correction. This version reads ONLY from the BIDS layer -- no Zenodo/network access here
at all, so this step has no login-node-only constraint and could in principle run on a
compute node (though in practice it's fast enough on the login node that it doesn't
matter).

Per case: reads the anat image + the 10 per-structure `label-<name>_seg.nii.gz`
derivative files, merges them into one multiclass integer label map (background=0,
same LABEL_MAP as before), writes nnU-Net imagesTr/labelsTr. A BIDS-staged case is by
construction already usable (bidsify.py only stages non-empty cases) and already
reoriented to canonical -- no further filtering/reorientation needed here, but this
script still verifies non-emptiness + image/label shape match per case as a cheap
sanity check rather than trusting that invariant blindly.

Usage:
  .venv/bin/python 02_00_convert.py --modality ct
  .venv/bin/python 02_00_convert.py --modality mri
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
BIDS_ROOT = DATASET_ROOT / "1_BIDS_totalseg-pelvic" / "pelvis-totalsegpelvic"
DERIV_DIR = BIDS_ROOT / "derivatives" / "labels"
SPLITS_ROOT = DATASET_ROOT / "4_splits_totalseg-pelvic"
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", str(DATASET_ROOT / "2_nnUNet_totalseg-pelvic" / "raw")))

LABEL_MAP = {
    "hip_left": 1,
    "hip_right": 2,
    "sacrum": 3,
    "gluteus_maximus_left": 4,
    "gluteus_maximus_right": 5,
    "gluteus_medius_left": 6,
    "gluteus_medius_right": 7,
    "gluteus_minimus_left": 8,
    "gluteus_minimus_right": 9,
    "iliopsoas_left": 10,
    "iliopsoas_right": 11,
}

# LABEL_MAP's keys are the nnU-Net class names (snake_case is fine there) -- but the BIDS
# derivative filenames' `label-<value>` entity can't contain underscores (that's the
# spec's own key-value separator). Fixed 2026-09-15, matches 00_01_bidsify.py's BIDS_LABEL.
BIDS_LABEL = {
    "hip_left": "hipLeft", "hip_right": "hipRight", "sacrum": "sacrum",
    "gluteus_maximus_left": "gluteusMaximusLeft", "gluteus_maximus_right": "gluteusMaximusRight",
    "gluteus_medius_left": "gluteusMediusLeft", "gluteus_medius_right": "gluteusMediusRight",
    "gluteus_minimus_left": "gluteusMinimusLeft", "gluteus_minimus_right": "gluteusMinimusRight",
    "iliopsoas_left": "iliopsoasLeft", "iliopsoas_right": "iliopsoasRight",
}

MODALITY_SUFFIX = {"ct": "CT", "mri": "MRI"}
MODALITY_OUT = {"ct": "Dataset130_TotalsegPelvic_CT", "mri": "Dataset131_TotalsegPelvic_MRI"}


def _bids_usable_cases(modality: str) -> list[str]:
    """Cases bidsify.py confirmed usable for this modality (source case ids, e.g. 's0004')."""
    p = SPLITS_ROOT / modality / "bids_usable_cases.json"
    return sorted(json.loads(p.read_text())["usable"])


def _sub_id(modality: str, case: str) -> str:
    return f"sub-{modality}{case[1:]}"


def _merge_labels(sub: str, suffix: str):
    import nibabel as nib
    import numpy as np

    combined = None
    ref_affine = None
    ref_header = None
    for name, label_id in LABEL_MAP.items():
        p = DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_label-{BIDS_LABEL[name]}_seg.nii.gz"
        img = nib.load(str(p))
        arr = np.asarray(img.dataobj)
        if combined is None:
            combined = np.zeros(arr.shape, dtype=np.uint8)
            ref_affine = img.affine
            ref_header = img.header
        combined[(arr > 0) & (combined == 0)] = label_id
    return nib.Nifti1Image(combined, ref_affine, ref_header), combined


def _convert_modality(modality: str) -> None:
    import nibabel as nib
    import numpy as np

    suffix = MODALITY_SUFFIX[modality]
    cases = _bids_usable_cases(modality)
    print(f"[convert][{modality}] {len(cases)} BIDS-usable cases to convert")

    out_root = NNUNET_RAW / MODALITY_OUT[modality]
    (out_root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out_root / "labelsTr").mkdir(parents=True, exist_ok=True)

    # Clear stale output before writing -- a real bug found 2026-09-15: this directory
    # can carry leftover case files from earlier conversion attempts (the pre-BIDS
    # direct-from-Zenodo pass, or an earlier top-up round with a different candidate
    # pool) that are no longer in the current usable list. This script previously only
    # ever wrote/overwrote the CURRENT cases and never removed anything else, so 26
    # stale CT cases silently survived a full rewrite and inflated imagesTr/labelsTr to
    # 226 files instead of 200 (caught by a second independent count, not by this
    # script). Delete anything not in `cases` before writing, every run.
    current_names = {f"{c}_0000.nii.gz" for c in cases}
    stale_removed = 0
    for f in (out_root / "imagesTr").glob("*.nii.gz"):
        if f.name not in current_names:
            f.unlink()
            stale_removed += 1
    current_label_names = {f"{c}.nii.gz" for c in cases}
    for f in (out_root / "labelsTr").glob("*.nii.gz"):
        if f.name not in current_label_names:
            f.unlink()
            stale_removed += 1
    if stale_removed:
        print(f"[convert][{modality}] removed {stale_removed} stale file(s) not in the "
              f"current usable list before writing")

    n_ok, n_bad = 0, []
    for i, case in enumerate(cases):
        sub = _sub_id(modality, case)
        img_src = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
        if not img_src.exists():
            n_bad.append((case, "missing BIDS image"))
            continue

        label_img, label_arr = _merge_labels(sub, suffix)
        if not np.any(label_arr):
            n_bad.append((case, "empty after merge (unexpected -- bidsify should have excluded this)"))
            continue

        img = nib.load(str(img_src))
        if img.shape != label_img.shape:
            n_bad.append((case, f"shape mismatch img={img.shape} label={label_img.shape}"))
            continue

        img_dst = out_root / "imagesTr" / f"{case}_0000.nii.gz"
        lbl_dst = out_root / "labelsTr" / f"{case}.nii.gz"
        for dst, obj in ((img_dst, img), (lbl_dst, label_img)):
            # nib.save() infers file type from the filename suffix, so the temp name
            # must still end in .nii.gz (a ".nii.gz.tmp" suffix fails with
            # ImageFileError) -- prefix instead of suffix for the "temp" marker.
            tmp = dst.parent / f".tmp_{dst.name}"
            nib.save(obj, str(tmp))
            os.replace(tmp, dst)
        n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"[convert][{modality}] {i + 1}/{len(cases)} done")

    print(f"[convert][{modality}] converted {n_ok}/{len(cases)} cases "
          f"({len(n_bad)} rejected: {n_bad[:5]}{'...' if len(n_bad) > 5 else ''})")

    final_usable = [c for c in cases if c not in {b[0] for b in n_bad}]
    manifest_path = SPLITS_ROOT / modality / "usable_cases_verified.json"
    manifest_path.write_text(json.dumps({
        "usable": sorted(final_usable),
        "rejected": n_bad,
    }, indent=2))

    dataset_json = {
        "channel_names": {"0": suffix},
        "labels": {"background": 0, **LABEL_MAP},
        "numTraining": len(final_usable),
        "file_ending": ".nii.gz",
    }
    (out_root / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"[convert][{modality}] numTraining={len(final_usable)}, dataset.json written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mri"], required=True)
    args = parser.parse_args()
    _convert_modality(args.modality)


if __name__ == "__main__":
    main()
