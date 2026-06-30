#!/usr/bin/env python3
"""
BIDSify AMOS22 (0_raw_amos/amos22/) → 1_BIDS_amos/amos-abdominal/.

AMOS raw data is already in NIfTI format (nnUNet-style layout). This step
reorganises it into a BIDS-compliant directory tree and writes metadata
files (dataset_description.json, participants.tsv, per-volume .json sidecars).
Files are hard-linked (no data duplication on the same filesystem).

NOTE: this is a faithful reorganisation — it does NOT touch orientation. AMOS is
stored LAS (CT) / RAS (MRI), whereas CHAOS/SLIVER07 are LPS, so the AMOS voxel
arrays are flipped relative to them. Run 03_preprocess/03_00_reorient_to_lps.py
after this step (and after 05_predict/05_00_build_test_inputs.py) to normalise.

All 360 labeled cases are BIDSified:
  imagesTr / labelsTr  200 CT + 40 MRI  → split=train (reserved for future training)
  imagesVa / labelsVa  100 CT + 20 MRI  → split=val   (used as test set now)
The official test set (imagesTs, no GT) is skipped.

Subject IDs:   sub-AM{id:04d}  (ID encodes modality: CT < 500, MRI ≥ 500)
BIDS suffixes: CT  → _CT.nii.gz
               MRI → _T2w.nii.gz

Reads:
  0_raw_amos/amos22/{imagesTr,imagesVa}/amos_{id:04d}.nii.gz
  0_raw_amos/amos22/{labelsTr,labelsVa}/amos_{id:04d}.nii.gz

Writes:
  1_BIDS_amos/amos-abdominal/
    dataset_description.json
    participants.tsv
    sub-AM{id:04d}/anat/sub-AM{id:04d}_{CT|T2w}.nii.gz       (hard-link)
    sub-AM{id:04d}/anat/sub-AM{id:04d}_{CT|T2w}.json
    derivatives/manual_masks/
      dataset_description.json
      sub-AM{id:04d}/anat/sub-AM{id:04d}_{CT|T2w}_dseg.nii.gz (hard-link)

Usage:
  python 00_01_bidsify.py
  python 00_01_bidsify.py --skip-bids   # dry-run / already done
"""
import argparse
import json
from pathlib import Path

DATASET_ROOT    = Path(__file__).resolve().parents[2]
AMOS22          = DATASET_ROOT / "0_raw_amos" / "amos22"
BIDS_ROOT       = DATASET_ROOT / "1_BIDS_amos" / "amos-abdominal"
DERIVATIVES_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

# Full 15-organ AMOS GT label map (stored in the derivatives description).
LABELS = {
    "background":          0,
    "spleen":              1,
    "right_kidney":        2,
    "left_kidney":         3,
    "gallbladder":         4,
    "esophagus":           5,
    "liver":               6,
    "stomach":             7,
    "aorta":               8,
    "postcava":            9,
    "pancreas":            10,
    "right_adrenal_gland": 11,
    "left_adrenal_gland":  12,
    "duodenum":            13,
    "bladder":             14,
    "prostate_uterus":     15,
}

# (img_dir, lab_dir, split_label)
SPLITS = [
    ("imagesTr", "labelsTr", "train"),
    ("imagesVa", "labelsVa", "val"),
]


def _is_ct(case_id: int) -> bool:
    return case_id < 500


def _bids_suffix(case_id: int) -> str:
    return "CT" if _is_ct(case_id) else "T2w"


def _link(src: Path, dst: Path) -> None:
    """Hard-link src → dst; fall back to copy if cross-device. Skip if dst exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_dataset_description() -> None:
    _write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "AMOS — A Large-Scale Abdominal Multi-Organ Segmentation Benchmark",
        "BIDSVersion": "1.9.0",
        "License": "CC BY-SA 4.0",
        "Authors": [
            "Yuanfeng Ji", "Haotian Bai", "Chongjian GE", "Jie Yang",
            "Ye Zhu", "Ruimao Zhang", "Zhen Li", "Lingyan Zhang",
            "Wanling Ma", "Xiang Wan", "Ping Luo",
        ],
        "ReferencesAndLinks": [
            "https://amos22.grand-challenge.org/",
            "https://zenodo.org/records/7262581",
            "Ji et al., NeurIPS 2022 — https://arxiv.org/abs/2206.08023",
        ],
        "DatasetType": "raw",
        "Usage": (
            "test-only (chaos-trained models evaluated here for cross-dataset "
            "generalization); native AMOS training pipeline pending."
        ),
    })


def _write_derivatives_description() -> None:
    DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(DERIVATIVES_DIR / "dataset_description.json", {
        "Name": "AMOS22 Manual Organ Masks (15 organs)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "AMOS22 Challenge"}],
        "LabelMap": LABELS,
    })


def _write_participants_tsv(participants: list[dict]) -> None:
    lines = ["participant_id\tmodality\tamos22_id\tsplit"]
    for p in sorted(participants, key=lambda x: x["amos22_id"]):
        lines.append(
            f"{p['participant_id']}\t{p['modality']}\t{p['amos22_id']}\t{p['split']}"
        )
    (BIDS_ROOT / "participants.tsv").write_text("\n".join(lines) + "\n")


def bidsify() -> None:
    BIDS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataset_description()
    _write_derivatives_description()

    participants: list[dict] = []

    for img_dir_name, lab_dir_name, split in SPLITS:
        img_dir = AMOS22 / img_dir_name
        lab_dir = AMOS22 / lab_dir_name
        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found — skipping {split} split")
            continue

        cases = sorted(img_dir.glob("amos_*.nii.gz"))
        print(f"  {split:5s}: {len(cases)} cases ({img_dir_name}/) …", flush=True)

        for img_path in cases:
            case_id = int(img_path.name.replace(".nii.gz", "").replace("amos_", ""))
            lab_path = lab_dir / img_path.name
            if not lab_path.exists():
                print(f"    WARN: no label for {img_path.name} — skipping")
                continue

            sub    = f"sub-AM{case_id:04d}"
            suffix = _bids_suffix(case_id)
            mod    = "CT" if _is_ct(case_id) else "MRI"

            anat  = BIDS_ROOT / sub / "anat"
            deriv = DERIVATIVES_DIR / sub / "anat"

            _link(img_path, anat  / f"{sub}_{suffix}.nii.gz")
            _link(lab_path, deriv / f"{sub}_{suffix}_dseg.nii.gz")
            _write_json(anat / f"{sub}_{suffix}.json", {
                "Modality":  mod,
                "AMOS22_ID": f"amos_{case_id:04d}",
                "Split":     split,
            })

            participants.append({
                "participant_id": sub,
                "modality":       mod,
                "amos22_id":      f"amos_{case_id:04d}",
                "split":          split,
            })

    _write_participants_tsv(participants)
    print(f"\n  BIDSify done → {BIDS_ROOT}")
    print(f"  {len(participants)} cases total")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-bids", action="store_true",
                    help="skip BIDS conversion (1_BIDS already populated)")
    args = ap.parse_args()

    print("=" * 64)
    print("AMOS22 — NIfTI → BIDS reorganisation")
    print("=" * 64)

    if not AMOS22.exists():
        raise SystemExit(
            f"ERROR: {AMOS22} not found.\n"
            f"Run 00_00_download_and_extract.py first."
        )

    if args.skip_bids:
        print("\n[1/1] Skipping BIDSify (--skip-bids)")
    else:
        print(f"\n[1/1] BIDSifying → {BIDS_ROOT}")
        bidsify()

    # Sanity report
    if BIDS_ROOT.exists():
        n_subs = sum(1 for p in BIDS_ROOT.iterdir()
                     if p.is_dir() and p.name.startswith("sub-"))
        n_nii  = sum(1 for _ in BIDS_ROOT.rglob("*.nii.gz"))
        tsv    = BIDS_ROOT / "participants.tsv"
        n_tsv  = len(tsv.read_text().strip().splitlines()) - 1 if tsv.exists() else 0
        print(f"\n  Subjects : {n_subs}")
        print(f"  NIfTI    : {n_nii}  (images + labels)")
        print(f"  TSV rows : {n_tsv}")


if __name__ == "__main__":
    main()
