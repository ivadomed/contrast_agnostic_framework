#!/usr/bin/env python3
"""
Download CHAOS (Combined CT-MR Healthy Abdominal Organ Segmentation) from Kaggle,
convert DICOM → NIfTI, and BIDSify.

CHAOS has two DISJOINT patient databases (different patients, not registered):
  - CT : liver only (binary mask), portal-venous contrast.
  - MR : T1-DUAL (in-phase + out-phase) and T2-SPIR, 4 organ labels.

Only the public training half has ground truth; the sealed test half (GT never
released) is SKIPPED entirely — every split we use comes from the training half.

Raw layout (kagglehub):
  CHAOS_Train_Sets/Train_Sets/
    CT/<id>/DICOM_anon/*.dcm            CT/<id>/Ground/liver_GT_###.png   (1-bit)
    MR/<id>/T1DUAL/DICOM_anon/InPhase/*.dcm   + .../OutPhase/*.dcm
            T1DUAL/Ground/*.png   (shared mask, named by InPhase, applies to both phases)
            T2SPIR/DICOM_anon/*.dcm   + T2SPIR/Ground/*.png   (8-bit grayscale)

MR mask greyscale encoding (verified): 63=liver 126=R-kidney 189=L-kidney 252=spleen.
CT mask: binary → liver.

BIDS output (1_BIDS_chaos/chaos-abdominal/):
  sub-MR##/anat/  sub-MR##_acq-inphase_T1w.nii  (+ .json)   <- T1DUAL InPhase
                  sub-MR##_acq-outphase_T1w.nii (+ .json)   <- T1DUAL OutPhase
                  sub-MR##_T2w.nii              (+ .json)   <- T2SPIR
  sub-CT##/anat/  sub-CT##_CT.nii               (+ .json)
  derivatives/manual_masks/sub-MR##/anat/sub-MR##_<entities>_dseg.nii
                           sub-CT##/anat/sub-CT##_CT_dseg.nii
  dataset_description.json   participants.tsv

In/out-phase are treated as two DISTINCT modalities (domain generalisation).
Unified label map: 0=bg 1=liver 2=right_kidney 3=left_kidney 4=spleen (CT → liver only).

Usage:
  .venv/bin/python 00_00_download_and_bidsify.py [--skip-copy] [--skip-bids]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import kagglehub
from PIL import Image

DATASET_ROOT    = Path(__file__).resolve().parents[2]   # …/datasets/chaos/
RAW_ROOT        = DATASET_ROOT / "0_raw_chaos"
BIDS_ROOT       = DATASET_ROOT / "1_BIDS_chaos" / "chaos-abdominal"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

KAGGLE_DATASET = "omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ"

# Unified label map (also written into nnUNet dataset.json downstream).
LABELS = {"background": 0, "liver": 1, "right_kidney": 2, "left_kidney": 3, "spleen": 4}

# MR ground PNG greyscale anchors → unified labels.
MR_ANCHORS      = np.array([0, 63, 126, 189, 252])
MR_ANCHOR_LABEL = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
# midpoints between consecutive anchors → bin edges for nearest-anchor assignment
MR_BIN_EDGES    = (MR_ANCHORS[:-1] + MR_ANCHORS[1:]) / 2.0


# ── DICOM / mask helpers ─────────────────────────────────────────────────────

def read_dicom_series(folder: Path) -> tuple[sitk.Image, list[str]]:
    """Read a single-series DICOM folder → (image, geometrically-ordered file list)."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(folder))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {folder}")
    # If a folder holds >1 series, pick the one with the most slices.
    best = max(series_ids,
               key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(folder), sid)))
    files = list(reader.GetGDCMSeriesFileNames(str(folder), best))  # sorted by position
    reader.SetFileNames(files)
    return reader.Execute(), files


def decode_mr_mask(arr: np.ndarray) -> np.ndarray:
    """8-bit greyscale → unified label ids via nearest CHAOS anchor."""
    idx = np.digitize(arr.astype(np.float32), MR_BIN_EDGES)
    return MR_ANCHOR_LABEL[idx]


def decode_ct_mask(arr: np.ndarray) -> np.ndarray:
    """Binary CT liver mask → label 1 (liver)."""
    return (np.asarray(arr) > 0).astype(np.uint8) * LABELS["liver"]


def build_label_volume(ref_img: sitk.Image,
                       ordered_files: list[str],
                       mask_for_file,
                       decode) -> sitk.Image:
    """Stack per-slice PNG masks in the SAME order as `ordered_files`, copy geometry.

    `mask_for_file(dcm_path) -> Path | None` resolves each slice's mask PNG.
    Missing masks (None / absent) leave that slice as background.
    """
    size_x, size_y, n = ref_img.GetSize()
    vol = np.zeros((n, size_y, size_x), dtype=np.uint8)
    for i, dcm in enumerate(ordered_files):
        mpath = mask_for_file(Path(dcm))
        if mpath is None or not mpath.exists():
            continue
        m = np.array(Image.open(mpath))
        if m.shape != (size_y, size_x):
            raise ValueError(f"Mask {mpath} shape {m.shape} != image ({size_y},{size_x})")
        vol[i] = decode(m)
    lab = sitk.GetImageFromArray(vol)
    lab.CopyInformation(ref_img)
    return lab


def write_nii(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


# ── raw copy (skip sealed test set) ──────────────────────────────────────────

def copy_raw(kaggle_path: Path, raw_root: Path) -> None:
    train = kaggle_path / "CHAOS_Train_Sets" / "Train_Sets"
    if not train.exists():
        print(f"ERROR: expected {train} not found", file=sys.stderr)
        sys.exit(1)
    raw_root.mkdir(parents=True, exist_ok=True)
    for modality in ("CT", "MR"):
        src = train / modality
        dst = raw_root / modality
        if dst.exists():
            print(f"  {modality}: already copied → {dst}")
            continue
        shutil.copytree(src, dst)
        n = len(list(dst.iterdir()))
        print(f"  {modality}: copied {n} patients → {dst}")
    print(f"  Raw copy done (sealed test set skipped) → {raw_root}")


# ── BIDSify ──────────────────────────────────────────────────────────────────

def bidsify_mr(chaos_id: str, mr_dir: Path, participants: list[dict]) -> None:
    label = f"MR{int(chaos_id):02d}"
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"

    # --- T1DUAL: in-phase (own mask) + out-phase (shares in-phase mask) ---
    t1 = mr_dir / "T1DUAL"
    ground = t1 / "Ground"

    in_img, in_files = read_dicom_series(t1 / "DICOM_anon" / "InPhase")
    in_lab = build_label_volume(
        in_img, in_files,
        mask_for_file=lambda p: ground / f"{p.stem}.png",
        decode=decode_mr_mask)
    write_nii(in_img, anat / f"{sub}_acq-inphase_T1w.nii")
    write_json(anat / f"{sub}_acq-inphase_T1w.json", _sidecar("MR", "T1DUAL in-phase"))
    write_nii(in_lab, deriv / f"{sub}_acq-inphase_T1w_dseg.nii")

    out_img, _ = read_dicom_series(t1 / "DICOM_anon" / "OutPhase")
    # In/out are co-registered with identical slice count & z-order → reuse mask array.
    in_arr = sitk.GetArrayFromImage(in_lab)
    if sitk.GetArrayFromImage(out_img).shape != in_arr.shape:
        raise ValueError(f"{sub}: in/out-phase shape mismatch — cannot share mask")
    out_lab = sitk.GetImageFromArray(in_arr)
    out_lab.CopyInformation(out_img)
    write_nii(out_img, anat / f"{sub}_acq-outphase_T1w.nii")
    write_json(anat / f"{sub}_acq-outphase_T1w.json", _sidecar("MR", "T1DUAL out-phase"))
    write_nii(out_lab, deriv / f"{sub}_acq-outphase_T1w_dseg.nii")

    # --- T2SPIR ---
    t2 = mr_dir / "T2SPIR"
    t2_ground = t2 / "Ground"
    t2_img, t2_files = read_dicom_series(t2 / "DICOM_anon")
    t2_lab = build_label_volume(
        t2_img, t2_files,
        mask_for_file=lambda p: t2_ground / f"{p.stem}.png",
        decode=decode_mr_mask)
    write_nii(t2_img, anat / f"{sub}_T2w.nii")
    write_json(anat / f"{sub}_T2w.json", _sidecar("MR", "T2-SPIR"))
    write_nii(t2_lab, deriv / f"{sub}_T2w_dseg.nii")

    participants.append({"label": label, "modality": "MR", "chaos_id": chaos_id})


def bidsify_ct(chaos_id: str, ct_dir: Path, participants: list[dict]) -> None:
    label = f"CT{int(chaos_id):02d}"
    sub = f"sub-{label}"
    anat = BIDS_ROOT / sub / "anat"
    deriv = DERIVATIVES_DIR / sub / "anat"

    ground = ct_dir / "Ground"
    img, files = read_dicom_series(ct_dir / "DICOM_anon")
    # CT masks are named liver_GT_### by DICOM filename order, not slice position.
    # build_label_volume places each slice at its position index, so resolving the
    # mask by the file's filename-sorted rank keeps mask ↔ volume aligned.
    name_rank = {f: i for i, f in enumerate(sorted(files))}

    def mask_for(p: Path):
        return ground / f"liver_GT_{name_rank[str(p)]:03d}.png"

    lab = build_label_volume(img, files, mask_for_file=mask_for, decode=decode_ct_mask)
    write_nii(img, anat / f"{sub}_CT.nii")
    write_json(anat / f"{sub}_CT.json", _sidecar("CT", "portal-venous abdominal CT"))
    write_nii(lab, deriv / f"{sub}_CT_dseg.nii")

    participants.append({"label": label, "modality": "CT", "chaos_id": chaos_id})


def _sidecar(modality: str, descr: str) -> dict:
    return {"Modality": modality, "SeriesDescription": descr, "CHAOSSource": True}


def bidsify(raw_root: Path) -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    participants: list[dict] = []

    mr_ids = sorted((raw_root / "MR").iterdir(), key=lambda p: int(p.name)) \
        if (raw_root / "MR").exists() else []
    ct_ids = sorted((raw_root / "CT").iterdir(), key=lambda p: int(p.name)) \
        if (raw_root / "CT").exists() else []

    print(f"  BIDSifying {len(mr_ids)} MR + {len(ct_ids)} CT patients …")
    for d in mr_ids:
        bidsify_mr(d.name, d, participants)
        print(f"    MR {d.name} done")
    for d in ct_ids:
        bidsify_ct(d.name, d, participants)
        print(f"    CT {d.name} done")

    _write_participants_tsv(participants)
    print(f"  BIDSify done → {BIDS_ROOT}")


def _write_dataset_description() -> None:
    write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "CHAOS — Combined (CT-MR) Healthy Abdominal Organ Segmentation",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-NC-SA 4.0",
        "Authors": ["CHAOS Challenge Organizers (ISBI 2019)"],
        "ReferencesAndLinks": [
            "https://chaos.grand-challenge.org/",
            "https://www.kaggle.com/datasets/omarxadel/chaos-combined-ct-mr-healthy-abdominal-organ",
        ],
        "DatasetType": "raw",
    })


def _write_derivatives_description() -> None:
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "CHAOS Manual Organ Masks",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "CHAOS Challenge"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list[dict]) -> None:
    lines = ["participant_id\tmodality\tchaos_id\tsplit"]
    for p in sorted(participants, key=lambda x: x["label"]):
        lines.append(f"sub-{p['label']}\t{p['modality']}\t{p['chaos_id']}\ttrain")
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-copy", action="store_true",
                    help="skip copying raw data (0_raw already populated)")
    ap.add_argument("--skip-bids", action="store_true", help="skip BIDS conversion")
    args = ap.parse_args()

    print("=" * 64)
    print("CHAOS — Download + Copy Raw (train only) + DICOM→NIfTI BIDSify")
    print("=" * 64)

    print("\n[1/3] Downloading from Kaggle Hub …")
    kaggle_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"  Kaggle cache: {kaggle_path}")

    if not args.skip_copy:
        print(f"\n[2/3] Copying raw data → {RAW_ROOT}")
        copy_raw(kaggle_path, RAW_ROOT)
    else:
        print("\n[2/3] Skipping raw copy (--skip-copy)")

    if not args.skip_bids:
        print(f"\n[3/3] DICOM→NIfTI BIDSifying → {BIDS_ROOT}")
        bidsify(RAW_ROOT)
    else:
        print("\n[3/3] Skipping BIDS conversion (--skip-bids)")

    print("\nAll done.")
    print(f"  Raw : {RAW_ROOT}")
    print(f"  BIDS: {BIDS_ROOT}")


if __name__ == "__main__":
    main()
