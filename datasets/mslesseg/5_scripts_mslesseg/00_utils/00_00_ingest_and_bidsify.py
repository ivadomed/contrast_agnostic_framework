#!/usr/bin/env python3
"""
BIDSify MSLesSeg (0_raw_mslesseg/MSLesSeg_Dataset/{train,test}/) -> 1_BIDS_mslesseg/mslesseg-brain/.

Source: MSLesSeg (Guarnera/Rondinella et al., Univ. of Catania), figshare
10.6084/m9.figshare.27919209.v1, CC-BY 4.0. 115 scans / 75 patients, each shipped
already co-registered + skull-stripped + resampled to 1mm-iso MNI152 space (182,218,182),
orientation LAS (verified via nibabel — NOT the project's LPS convention, unlike
open-ms, so 03_preprocess/03_00_check_orientation.py performs a REAL reorientation
here, not just an idempotent check).

Archive layout (kept verbatim under 0_raw, see README):
  train/P<id>/T<k>/P<id>_T<k>_{FLAIR,T1,T2,MASK}.nii.gz   (k in 1..3, up to 3 timepoints)
  test/P<id>/P<id>_{FLAIR,T1,T2,MASK}.nii.gz              (single timepoint)

Per CLAUDE.md / user instruction: MSLesSeg is used ENTIRELY for cross-dataset
evaluation (open-ms models) — the archive's own train/test split is not a train/test
split for US (we never train here); it is collapsed into ONE flat pool of 115 cases.

Case ids (BIDS subject/session, kept traceable to the source):
  train: sub-P<id>_ses-T<k>      e.g. sub-P1_ses-T1
  test:  sub-P<id>                e.g. sub-P54          (no session — single scan)

Reads:   0_raw_mslesseg/MSLesSeg_Dataset/{train,test}/...
Writes:  1_BIDS_mslesseg/mslesseg-brain/
           dataset_description.json, participants.tsv
           sub-<id>[/ses-T<k>]/anat/sub-<id>[_ses-T<k>]_{FLAIR,T1w,T2w}.nii.gz (+ .json)
           derivatives/manual_masks/dataset_description.json
           .../sub-<id>[/ses-T<k>]/anat/sub-<id>[_ses-T<k>]_FLAIR_dseg.nii.gz

Images are HARD-LINKED from 0_raw (lossless, same filesystem); masks are copied as-is
(already uint8 binary {0,1} — verified, no float32-junk gotcha like TRUSTED's kidney
masks). Orientation is intentionally NOT touched here — that's 03_preprocess's job, so
a hard-link tree stays traceable back to 0_raw until the reorient step breaks it.

Usage:  python 00_00_ingest_and_bidsify.py   (light — pure filesystem ops, login-node OK,
                                               but dispatch via .sh/run_job for consistency)
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]                 # datasets/mslesseg
RAW = DATASET_ROOT / "0_raw_mslesseg" / "MSLesSeg_Dataset"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_mslesseg" / "mslesseg-brain"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

# raw filename suffix -> BIDS suffix
CONTRASTS = {"FLAIR": "FLAIR", "T1": "T1w", "T2": "T2w"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:                      # cross-device -> copy
        import shutil
        shutil.copyfile(src, dst)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _cases():
    """Yield (bids_id, src_dir, file_prefix) for every scan in train/ + test/."""
    test_dir = RAW / "test"
    for pdir in sorted(test_dir.glob("P*")):
        yield pdir.name, pdir, pdir.name                     # e.g. P54, .../test/P54, "P54"

    train_dir = RAW / "train"
    for pdir in sorted(train_dir.glob("P*")):
        for tdir in sorted(pdir.glob("T*")):
            pid, tp = pdir.name, tdir.name                    # P1, T1
            bids_id = f"{pid}_ses-{tp}"                       # P1_ses-T1
            yield bids_id, tdir, f"{pid}_{tp}"                # file prefix P1_T1_*


def bidsify() -> None:
    if not RAW.exists():
        raise SystemExit(f"Raw archive not found at {RAW} — extract MSLesSeg Dataset.zip there first.")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "MSLesSeg — Multiple Sclerosis Lesion Segmentation (Univ. of Catania)",
        "BIDSVersion": "1.9.0",
        "License": "CC-BY-4.0",
        "Authors": ["Francesco Guarnera", "Alessia Rondinella", "Elena Crispino",
                    "Giulia Russo", "Clara Di Lorenzo", "Davide Maimone",
                    "Francesco Pappalardo", "Sabastiano Battiato"],
        "ReferencesAndLinks": ["https://doi.org/10.6084/m9.figshare.27919209.v1"],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "MSLesSeg expert-validated lesion masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Expert annotation (Guarnera et al.)"}],
    })

    rows = ["participant_id\tsource_split\tlesion_voxels"]
    n = 0
    for bids_id, src_dir, prefix in _cases():
        sub_parts = bids_id.split("_ses-")
        if len(sub_parts) == 2:
            sub, ses = f"sub-{sub_parts[0]}", f"ses-{sub_parts[1]}"
            anat_rel = Path(sub) / ses / "anat"
            fname_stub = f"{sub}_{ses}"
            source_split = "train"
        else:
            sub = f"sub-{bids_id}"
            anat_rel = Path(sub) / "anat"
            fname_stub = sub
            source_split = "test"

        for raw_suffix, bids_suffix in CONTRASTS.items():
            src = src_dir / f"{prefix}_{raw_suffix}.nii.gz"
            if not src.exists():
                raise SystemExit(f"Missing expected file: {src}")
            _link(src, BIDS_ROOT / anat_rel / f"{fname_stub}_{bids_suffix}.nii.gz")
            _json(BIDS_ROOT / anat_rel / f"{fname_stub}_{bids_suffix}.json",
                  {"Modality": "MR", "Space": "FLAIR",
                   "Description": f"{raw_suffix}, co-registered to FLAIR, 1mm iso MNI152, "
                                   f"skull-stripped (source archive '{source_split}' split, "
                                   "collapsed here into one eval-only pool)"})

        mask_src = src_dir / f"{prefix}_MASK.nii.gz"
        if not mask_src.exists():
            raise SystemExit(f"Missing expected mask: {mask_src}")
        _link(mask_src, DERIV_DIR / anat_rel / f"{fname_stub}_FLAIR_dseg.nii.gz")

        import nibabel as nib
        import numpy as np
        lv = int((np.asanyarray(nib.load(str(mask_src)).dataobj) > 0).sum())
        rows.append(f"{fname_stub}\t{source_split}\t{lv}")
        n += 1

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    print(f"BIDSified {n} scans (train+test archive splits collapsed) → {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
