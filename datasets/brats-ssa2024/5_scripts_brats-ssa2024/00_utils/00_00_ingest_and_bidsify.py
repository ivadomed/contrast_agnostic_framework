#!/usr/bin/env python3
"""
BIDSify BraTS-SSA 2024 (0_raw_brats-ssa2024/95_Glioma/) -> 1_BIDS_brats-ssa2024/ssa-brain-brats2024/.

Source: kagglehub "kaalmurlidhar/brats2024-africa" (a re-upload of BraTS-Africa/BraTS-SSA;
official distribution is Synapse-gated, see datasets/brats-ssa2024/README.md). The archive
bundles TWO cohorts -- "95_Glioma" (95 cases) and "51_OtherNeoplasms" (51 cases, non-glioma
tumor types). ONLY "95_Glioma" is ingested here -- brats2024-glioma is a glioma-specific
segmenter, so the other cohort is out-of-domain and would confound the comparison.

Each case ships as BraTS-SSA-<id>-000/BraTS-SSA-<id>-000-{t1c,t1n,t2f,t2w,seg}.nii
(uncompressed, orientation RAS -- NOT the project's LPS convention, so
03_preprocess/03_00_reorient_to_lps.py performs a REAL reorientation after this step,
same situation as AMOS/MSLesSeg).

Label note (verified empirically, see module docstring in env.sh): GT here has only 3
of brats2024-glioma's 4 foreground labels {NCR:1, SNFH:2, ET:3} -- RC:4 (resection
cavity) never appears, since these are pre-treatment scans. 06_evaluate scores only
these 3 labels explicitly.

Reads:   0_raw_brats-ssa2024/95_Glioma/BraTS-SSA-<id>-000/BraTS-SSA-<id>-000-*.nii
Writes:  1_BIDS_brats-ssa2024/ssa-brain-brats2024/
           dataset_description.json, participants.tsv
           sub-<id>/anat/sub-<id>_{T1n,T1c,T2w,T2f}.nii.gz (+ .json sidecars)
           derivatives/manual_masks/sub-<id>/anat/sub-<id>_dseg.nii.gz

Images/masks are gzip-compressed on write (source is uncompressed .nii) -- not a
hardlink, since the format changes; orientation is intentionally NOT touched here
(that's 03_preprocess's job).

Usage:  python 00_00_ingest_and_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib

DATASET_ROOT = Path(__file__).resolve().parents[2]                 # datasets/brats-ssa2024
RAW = DATASET_ROOT / "0_raw_brats-ssa2024" / "95_Glioma"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_brats-ssa2024" / "ssa-brain-brats2024"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

# raw filename suffix -> BIDS suffix
CONTRASTS = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}


def _case_id(case_dir_name: str) -> str:
    # "BraTS-SSA-00002-000" -> "00002000"
    return case_dir_name.replace("BraTS-SSA-", "").replace("-", "")


def _write_gz(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    nib.save(nib.load(str(src)), str(dst))


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def bidsify() -> None:
    if not RAW.exists():
        raise SystemExit(f"Raw archive not found at {RAW} -- stage the kaggle 95_Glioma folder there first.")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "BraTS-SSA 2024 (Sub-Saharan Africa glioma cohort) -- 95_Glioma subset",
        "BIDSVersion": "1.9.0",
        "License": "See datasets/brats-ssa2024/README.md -- BraTS DUA via Synapse; "
                    "ingested here from a kagglehub re-upload (kaalmurlidhar/brats2024-africa)",
        "ReferencesAndLinks": ["https://www.synapse.org/brats",
                                "https://www.kaggle.com/datasets/kaalmurlidhar/brats2024-africa"],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "BraTS-SSA 2024 expert-annotated glioma segmentation masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Expert annotation (BraTS-Africa consortium)"}],
    })

    cases = sorted(p for p in RAW.glob("BraTS-SSA-*") if p.is_dir())
    rows = ["participant_id"]
    n = 0
    for case_dir in cases:
        cid = _case_id(case_dir.name)
        sub = f"sub-{cid}"
        anat_dir = BIDS_ROOT / sub / "anat"

        for raw_suffix, bids_suffix in CONTRASTS.items():
            src = case_dir / f"{case_dir.name}-{raw_suffix}.nii"
            if not src.exists():
                raise SystemExit(f"Missing expected file: {src}")
            _write_gz(src, anat_dir / f"{sub}_{bids_suffix}.nii.gz")
            _json(anat_dir / f"{sub}_{bids_suffix}.json",
                  {"Modality": "MR", "Description": f"{raw_suffix} (BraTS-SSA 2024, pre-treatment glioma)"})

        seg_src = case_dir / f"{case_dir.name}-seg.nii"
        if not seg_src.exists():
            raise SystemExit(f"Missing expected mask: {seg_src}")
        _write_gz(seg_src, DERIV_DIR / sub / "anat" / f"{sub}_dseg.nii.gz")

        rows.append(sub)
        n += 1

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"BIDSified {n} glioma cases -> {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
