#!/usr/bin/env python3
"""
Ingest the TRUSTED kidney dataset (IRCAD France) and BIDSify it.

TRUSTED is EVALUATION-ONLY here (see datasets/trusted/README.md): a CT↔US kidney
registration dataset whose labeled volumes become OUR test set for measuring
MR→{CT,US} domain-randomization generalization of chaos-trained models.

What we ingest (the rest of the archive — meshes, landmarks, registration
transforms — is irrelevant to segmentation eval and is NOT extracted):
  * CT_DATA/CT_images/<id>_imgCT.nii.gz              48 volumes, BOTH kidneys
  * CT_DATA/CT_masks/{Annotator1,Annotator2,GT_estimated_masksCT}/<id>_maskCT.nii.gz
  * US_DATA/US_images/<id><L|R>_imgUS.nii.gz         59 volumes, ONE kidney each
  * US_DATA/US_masks/{Annotator1,Annotator2,GT_estimated_masksUS}/<id><L|R>_maskUS.nii.gz
  * README.txt, data_use_agreement.txt

Two gotchas this script fixes (see datasets/trusted onboarding notes):
  1. GT masks are stored as float32 with floating-point junk values
     (1.6e-27, 2.8e-14, …) instead of clean integers → we threshold (>0.5) and
     cast to uint8 {0,1} when writing the BIDS derivative dseg.
  2. Each TRUSTED mask is a single binary kidney label (CT merges both kidneys;
     US has one). chaos predicts right_kidney(2)+left_kidney(3) separately, so the
     evaluator (06_00_evaluate_trusted.py) merges {2,3}→kidney at score time.

Orientation: TRUSTED volumes are already ('L','P','S') — the chaos/sliver07
canonical convention — so NO reorientation is needed (03_preprocess/03_00_check_
orientation.py verifies this idempotently).

Raw layout (0_raw_trusted/) is left pristine: extracted .nii.gz are byte-identical
to the archive. BIDS IMAGES are hardlinks into 0_raw (same filesystem, lossless,
saves ~18 GB); BIDS MASKS are freshly written, binarized uint8.

BIDS output (1_BIDS_trusted/trusted-kidney/):
  sub-<id>/anat/sub-<id>_CT.nii.gz            (+ .json)   <- CT image (both kidneys)
  sub-<id>/anat/sub-<id>_acq-<R|L>_US.nii.gz  (+ .json)   <- US image (one kidney)
  derivatives/manual_masks/sub-<id>/anat/sub-<id>_CT_dseg.nii.gz
  derivatives/manual_masks/sub-<id>/anat/sub-<id>_acq-<R|L>_US_dseg.nii.gz
  dataset_description.json  participants.tsv

Usage (dispatch via 00_00_ingest_and_bidsify.sh — runs on a compute node):
  python 00_00_ingest_and_bidsify.py [--zip PATH] [--skip-extract] [--skip-bids]
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATASET_ROOT = Path(__file__).resolve().parents[2]          # …/datasets/trusted/
RAW_ROOT     = DATASET_ROOT / "0_raw_trusted"
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_trusted" / "trusted-kidney"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
SPLITS_DIR   = DATASET_ROOT / "4_splits_trusted"

ARCHIVE_TOP  = "TRUSTED_dataset_for_nsd"
DEFAULT_ZIP  = Path("/scratch/paulh/us_download/TRUSTED_dataset_for_nsd.zip")

# Single binary foreground label. The kidney↔chaos merge {2,3}→kidney happens in the
# evaluator, not here (GT stays the dataset's own single label = 1).
LABELS = {"background": 0, "kidney": 1}

# 5-fold cross-validation splits, verbatim from the archive README.txt. CT folds are
# per-patient; US folds are per-patient-kidney. Kept for reference / reproducibility
# (TRUSTED is eval-only, so these are not training folds — see 4_splits_trusted/).
CT_CV = {
    1: ["263", "794", "592", "206", "579", "915", "250", "905", "249"],
    2: ["561", "418", "636", "258", "283", "737", "610", "517", "801"],
    3: ["443", "506", "641", "398", "711", "532", "371", "721", "735", "680"],
    4: ["755", "510", "701", "948", "329", "239", "284", "656", "704", "399"],
    5: ["861", "200", "220", "716", "348", "466", "738", "314", "832", "406"],
}
US_CV = {
    1: ["263R", "263L", "794R", "794L", "592R", "206R", "206L", "579R", "579L", "915L", "250R", "250L"],
    2: ["561R", "418R", "418L", "636R", "258R", "258L", "283L", "610L", "517R", "517L"],
    3: ["443R", "506R", "506L", "641R", "641L", "398R", "711L", "532R", "532L", "371R", "721L", "735R", "680L"],
    4: ["755R", "755L", "510R", "510L", "701R", "701L", "948R", "948L", "329R", "329L", "704L", "399R"],
    5: ["861R", "861L", "200R", "220R", "716R", "348R", "348L", "466R", "466L", "738R", "314R", "314L"],
}
# Patients with one or more renal lesions (README) — still scored as kidney.
LESION_KIDNEYS = {"250R", "283L", "371R", "915L", "314L", "314R"}

# Which archive members to extract (images + all 3 mask variants + docs). Meshes,
# landmarks and registration transforms are intentionally skipped.
_EXTRACT_DIRS = [
    "CT_DATA/CT_images/",
    "CT_DATA/CT_masks/Annotator1/", "CT_DATA/CT_masks/Annotator2/",
    "CT_DATA/CT_masks/GT_estimated_masksCT/",
    "US_DATA/US_images/",
    "US_DATA/US_masks/Annotator1/", "US_DATA/US_masks/Annotator2/",
    "US_DATA/US_masks/GT_estimated_masksUS/",
]
_EXTRACT_FILES = ["README.txt", "data_use_agreement.txt"]

CT_GT_DIR = RAW_ROOT / "CT_DATA" / "CT_masks" / "GT_estimated_masksCT"
US_GT_DIR = RAW_ROOT / "US_DATA" / "US_masks" / "GT_estimated_masksUS"


# ── extract ───────────────────────────────────────────────────────────────────

def _wanted(member: str) -> bool:
    rel = member[len(ARCHIVE_TOP) + 1:] if member.startswith(ARCHIVE_TOP + "/") else member
    if rel in _EXTRACT_FILES:
        return True
    return any(rel.startswith(d) and not rel.endswith("/") for d in _EXTRACT_DIRS)


def extract(zip_path: Path) -> None:
    if not zip_path.exists():
        sys.exit(f"ERROR: archive not found: {zip_path}")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if _wanted(m)]
        print(f"  extracting {len(members)} members (images + masks + docs) → {RAW_ROOT}")
        for m in members:
            rel = m[len(ARCHIVE_TOP) + 1:]
            dst = RAW_ROOT / rel
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, open(dst, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            n += 1
            if n % 25 == 0:
                print(f"    … {n} files")
    print(f"  extract done: {n} new files (pristine) → {RAW_ROOT}")


# ── helpers ─────────────────────────────────────────────────────────────────

def hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)                      # same filesystem → lossless, no copy
    except OSError:
        sitk.WriteImage(sitk.ReadImage(str(src)), str(dst))   # fallback: copy


def write_binarized_mask(src_mask: Path, dst: Path) -> int:
    """Read a float32 GT mask (with float junk), threshold >0.5, cast uint8 {0,1}.
    Returns the foreground voxel count. Geometry is preserved from the source mask."""
    m = sitk.ReadImage(str(src_mask))
    arr = (sitk.GetArrayFromImage(m) > 0.5).astype(np.uint8) * LABELS["kidney"]
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dst))
    return int(arr.sum())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _fold_of(key: str, cv: dict) -> str:
    for k, members in cv.items():
        if key in members:
            return str(k)
    return ""


# ── BIDSify ──────────────────────────────────────────────────────────────────

def _ct_cases() -> list[str]:
    return sorted(re.match(r"(\d+)_imgCT", p.name).group(1)
                  for p in (RAW_ROOT / "CT_DATA" / "CT_images").glob("*_imgCT.nii.gz"))


def _us_cases() -> list[tuple[str, str]]:
    out = []
    for p in (RAW_ROOT / "US_DATA" / "US_images").glob("*_imgUS.nii.gz"):
        mo = re.match(r"(\d+)([RL])_imgUS", p.name)
        out.append((mo.group(1), mo.group(2)))
    return sorted(out)


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    ct_cases = _ct_cases()
    us_cases = _us_cases()
    if not ct_cases and not us_cases:
        sys.exit(f"ERROR: no images found under {RAW_ROOT} — run extract first.")

    # subject record: pid → dict(ct=bool, us=set(sides))
    subjects: dict[str, dict] = {}

    print(f"  BIDSifying {len(ct_cases)} CT + {len(us_cases)} US volumes …")
    for pid in ct_cases:
        sub, anat, deriv = _sub_dirs(pid)
        img = RAW_ROOT / "CT_DATA" / "CT_images" / f"{pid}_imgCT.nii.gz"
        seg = CT_GT_DIR / f"{pid}_maskCT.nii.gz"
        if not seg.exists():
            print(f"    ! CT {pid}: GT mask missing — skipping", file=sys.stderr); continue
        hardlink(img, anat / f"{sub}_CT.nii.gz")
        write_json(anat / f"{sub}_CT.json", _sidecar("CT", "TRUSTED contrast-enhanced abdominal CT (both kidneys)"))
        fg = write_binarized_mask(seg, deriv / f"{sub}_CT_dseg.nii.gz")
        subjects.setdefault(pid, {"ct": False, "us": set()})["ct"] = True
        print(f"    CT  {pid}: kidney voxels={fg}")

    for pid, side in us_cases:
        sub, anat, deriv = _sub_dirs(pid)
        img = RAW_ROOT / "US_DATA" / "US_images" / f"{pid}{side}_imgUS.nii.gz"
        seg = US_GT_DIR / f"{pid}{side}_maskUS.nii.gz"
        if not seg.exists():
            print(f"    ! US {pid}{side}: GT mask missing — skipping", file=sys.stderr); continue
        hardlink(img, anat / f"{sub}_acq-{side}_US.nii.gz")
        write_json(anat / f"{sub}_acq-{side}_US.json",
                   _sidecar("US", f"TRUSTED 3D ultrasound, {'right' if side == 'R' else 'left'} kidney"))
        fg = write_binarized_mask(seg, deriv / f"{sub}_acq-{side}_US_dseg.nii.gz")
        subjects.setdefault(pid, {"ct": False, "us": set()})["us"].add(side)
        print(f"    US  {pid}{side}: kidney voxels={fg}")

    _write_participants_tsv(subjects)
    _write_splits(ct_cases, us_cases)
    print(f"  BIDSify done → {BIDS_ROOT}")


def _sub_dirs(pid: str):
    sub = f"sub-{pid}"
    return sub, BIDS_ROOT / sub / "anat", DERIV_DIR / sub / "anat"


def _sidecar(modality: str, descr: str) -> dict:
    return {"Modality": modality, "SeriesDescription": descr, "TRUSTEDSource": True}


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "TRUSTED — kidney CT + 3D US (evaluation set)",
        "BIDSVersion": "1.9.0",
        "License": "IRCAD France TRUSTED Data Use Agreement (see 0_raw_trusted/data_use_agreement.txt) — NOT redistributable",
        "ReferencesAndLinks": ["https://springernature.figshare.com/ (file 51079133)"],
        "DatasetType": "raw",
        "Usage": "EVALUATION ONLY — test set for chaos-trained models (MR→{CT,US} generalization). Kidney only.",
    })


def _write_derivatives_description() -> None:
    write_json(DERIV_DIR / "dataset_description.json", {
        "Name": "TRUSTED Manual Kidney Masks (estimated ground-truth, binarized)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "TRUSTED GT_estimated masks, binarized float32>0.5 → uint8"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(subjects: dict) -> None:
    lines = ["participant_id\thas_ct\thas_us_R\thas_us_L\tct_cv_fold\tlesion_sides\tsplit"]
    for pid in sorted(subjects):
        rec = subjects[pid]
        les = ",".join(sorted(s for s in ("R", "L") if f"{pid}{s}" in LESION_KIDNEYS)) or "n/a"
        lines.append("\t".join([
            f"sub-{pid}",
            "1" if rec["ct"] else "0",
            "1" if "R" in rec["us"] else "0",
            "1" if "L" in rec["us"] else "0",
            _fold_of(pid, CT_CV) or "n/a",
            les,
            "test",                              # TRUSTED is only ever a test set here
        ]))
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def _write_splits(ct_cases, us_cases) -> None:
    """Persist the README 5-fold CV (reference only — eval-only dataset)."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "TRUSTED is evaluation-only; these are the original README 5-fold CV "
                "splits, kept for reference/reproducibility, not used for training.",
        "ct_folds": {str(k): v for k, v in CT_CV.items()},
        "us_folds": {str(k): v for k, v in US_CV.items()},
    }
    (SPLITS_DIR / "trusted_cv_folds.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP,
                    help=f"path to TRUSTED archive (default: {DEFAULT_ZIP})")
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip zip extraction (0_raw already populated)")
    ap.add_argument("--skip-bids", action="store_true", help="skip BIDS conversion")
    args = ap.parse_args()

    print("=" * 64)
    print("TRUSTED — Ingest (CT + US, kidney) + BIDSify")
    print("=" * 64)

    if not args.skip_extract:
        print(f"\n[1/2] Extracting from {args.zip} → {RAW_ROOT}")
        extract(args.zip)
    else:
        print("\n[1/2] Skipping extract (--skip-extract)")

    if not args.skip_bids:
        print(f"\n[2/2] BIDSifying → {BIDS_ROOT}")
        bidsify()
    else:
        print("\n[2/2] Skipping BIDS conversion (--skip-bids)")


if __name__ == "__main__":
    main()
