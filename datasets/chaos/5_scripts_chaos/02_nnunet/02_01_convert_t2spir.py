#!/usr/bin/env python3
"""
Convert CHAOS BIDS → nnUNet raw format — MR T2-SPIR (T2w) only.

Produces Dataset061_CHAOS_MR_T2spir: single channel (T2w), 4 organ labels.
Mirrors 02_00_convert.py (T1in) exactly — same train/val pool, same holdout
exclusion, same splits_final.json. Only the BIDS source file differs:
  images: sub-MR{id}_T2w.nii
  masks:  sub-MR{id}_T2w_dseg.nii

Usage:
  python 02_01_convert_t2spir.py [--dataset-id 61] [--jobs N]
"""
import argparse
import gzip
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
BIDS_ROOT    = DATASET_ROOT / "1_BIDS_chaos" / "chaos-abdominal"
NNUNET_RAW   = DATASET_ROOT / "2_nnUNet_chaos" / "raw"
DERIV_DIR    = BIDS_ROOT / "derivatives" / "manual_masks"
TEST_CASES   = DATASET_ROOT / "4_splits_chaos" / "test_cases.json"

LABELS = {"background": 0, "liver": 1, "right_kidney": 2, "left_kidney": 3, "spleen": 4}


def gzip_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out)


def convert_case(case_id: str, images_tr: Path, labels_tr: Path) -> str:
    sub = f"sub-{case_id}"
    src = BIDS_ROOT / sub / "anat" / f"{sub}_T2w.nii"
    if not src.exists():
        raise FileNotFoundError(f"Missing T2w: {src}")
    gzip_copy(src, images_tr / f"{case_id}_0000.nii.gz")

    seg = DERIV_DIR / sub / "anat" / f"{sub}_T2w_dseg.nii"
    if not seg.exists():
        raise FileNotFoundError(f"Missing seg: {seg}")
    gzip_copy(seg, labels_tr / f"{case_id}.nii.gz")
    return case_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, default=61)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    meta = json.loads(TEST_CASES.read_text())
    holdout = set(meta["mr_internal_test"])
    # MR CV cases = all MR subjects minus internal-test holdout (same partition as T1in).
    all_mr = sorted(p.name.removeprefix("sub-") for p in BIDS_ROOT.glob("sub-MR*"))
    cv_cases = [c for c in all_mr if c not in holdout]

    ds_name   = f"Dataset{args.dataset_id:03d}_CHAOS_MR_T2spir"
    out_dir   = NNUNET_RAW / ds_name
    images_tr = out_dir / "imagesTr"
    labels_tr = out_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(cv_cases)} MR CV cases (T2w/T2spir) → {ds_name}")
    print(f"  excluded (test-only): {len(holdout)} MR holdout + all CT")

    case_ids, failed = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(convert_case, c, images_tr, labels_tr): c for c in cv_cases}
        for fut in as_completed(futs):
            try:
                case_ids.append(fut.result())
            except Exception as e:
                failed.append((futs[fut], str(e)))

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for c, err in failed:
            print(f"  {c}: {err}")
        raise SystemExit(1)

    dataset_json = {
        "name": "CHAOS_MR_T2spir",
        "description": "CHAOS MR T2-SPIR — train/val pool for cross-modality generalization",
        "reference": "https://chaos.grand-challenge.org/",
        "licence": "CC BY-NC-SA 4.0",
        "release": "1.0",
        "channel_names": {"0": "T2w"},
        "labels": LABELS,
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    (out_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"\nDone. {len(case_ids)} cases → {out_dir}")


if __name__ == "__main__":
    main()
