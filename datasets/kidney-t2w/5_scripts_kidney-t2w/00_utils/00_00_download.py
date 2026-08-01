#!/usr/bin/env python3
"""
Download the "T2-weighted Kidney MRI Segmentation" dataset (Zenodo record 5153568)
-- 100 T2W abdominal MRI scans (50 healthy controls + 50 chronic kidney disease
patients) with manually-defined binary kidney masks.

KIDNEY-T2W is EVALUATION-ONLY here (see datasets/kidney-t2w/README.md): all 100
scans become OUR test set for MRI->MRI generalization of chaos-trained models on
the kidney label.

Source: https://zenodo.org/records/5153568 -- public, access_right=open,
CC-BY-4.0 (even more permissive than CirrMRI600+'s CC-BY-NC -- no NC restriction).
No account/approval needed, unlike the Duke Spleen/Liver MRI datasets (Zenodo,
human-approval email gate) which were considered and rejected for that reason.

Usage:
  python 00_00_download.py                 # download (if needed) + extract
  python 00_00_download.py --skip-download  # 0_raw already populated
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = DATASET_ROOT / "0_raw_kidney-t2w"

FILES = {
    "Healthy_Control.zip": "https://zenodo.org/api/records/5153568/files/Healthy_Control.zip/content",
    "CKD.zip": "https://zenodo.org/api/records/5153568/files/CKD.zip/content",
    "subject_information.csv": "https://zenodo.org/api/records/5153568/files/subject_information.csv/content",
}
EXPECTED_SIZE = {"Healthy_Control.zip": 76955226, "CKD.zip": 83338533}


def download(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dst = raw_root / fname
        want = EXPECTED_SIZE.get(fname)
        if dst.exists() and dst.stat().st_size > 0 and (want is None or dst.stat().st_size == want):
            print(f"  {fname}: already present -> {dst}")
            continue
        print(f"  downloading {fname} …")
        urllib.request.urlretrieve(url, dst)
        got = dst.stat().st_size
        print(f"    -> {dst} ({got / 1e6:.1f} MB)")
        if want and got != want:
            raise SystemExit(f"ERROR: {fname} size mismatch: got {got}, expected {want} "
                              f"(truncated download? re-run)")

    for fname in ("Healthy_Control.zip", "CKD.zip"):
        zp = raw_root / fname
        extract_dir = raw_root / fname.replace(".zip", "")
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  {fname}: already extracted -> {extract_dir}")
            continue
        print(f"  extracting {fname} …")
        with zipfile.ZipFile(zp) as z:
            z.extractall(extract_dir)
    print(f"  Raw download done -> {raw_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print("T2-weighted Kidney MRI Segmentation — Download + extract")
    print("=" * 64)

    if not args.skip_download:
        print(f"\nDownloading -> {RAW_ROOT}")
        download(RAW_ROOT)
    else:
        print("\nSkipping download (--skip-download)")


if __name__ == "__main__":
    main()
