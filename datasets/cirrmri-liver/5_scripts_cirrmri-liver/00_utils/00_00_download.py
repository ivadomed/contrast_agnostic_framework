#!/usr/bin/env python3
"""
Download CirrMRI600+ (Large Scale MRI Collection and Segmentation of Cirrhotic
Liver, Jha et al., Scientific Data 2025) from its public OSF project.

CirrMRI600+ is EVALUATION-ONLY here (see datasets/cirrmri-liver/README.md): CT-only
sliver07 covers liver-CT generalization for chaos; this dataset covers liver-MRI
generalization instead, using both T1w and T2w 3D volumes (paralleling chaos's own
T1in/T2spir training contrasts).

Source: https://osf.io/cuk24/ — public OSF project, access_requests_enabled=false,
CC BY-NC 4.0 (permits redistribution/adaptation, non-commercial only — see
0_raw_cirrmri-liver/LICENSE.txt). No account/approval needed, unlike the Duke
Liver/Spleen MRI datasets (Zenodo, human-approval email gate) which were
considered and rejected for that reason.

Only the two 3D zips are downloaded (Cirrhosis_T1_3D.zip, Cirrhosis_T2_3D.zip) —
Cirrhosis_T2_2D.zip (2D slices, not usable by a 3d_fullres nnU-Net model) and
Healthy_subjects.zip (no cirrhosis-specific value for a segmentation-generalization
probe; the label is liver either way) are intentionally skipped.

Usage:
  python 00_00_download.py                 # download (if needed) + extract
  python 00_00_download.py --skip-download  # 0_raw already populated
"""
import argparse
import urllib.request
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]   # …/datasets/cirrmri-liver/
RAW_ROOT = DATASET_ROOT / "0_raw_cirrmri-liver"

# OSF direct-download links (osf.io/download/<id>/), resolved via the OSF v2 API
# against project cuk24's osfstorage file listing (2026-07-31).
FILES = {
    "Cirrhosis_T1_3D.zip": "https://osf.io/download/47rxy/",
    "Cirrhosis_T2_3D.zip": "https://osf.io/download/72df5/",
    "LICENSE.txt": "https://osf.io/download/wmy85/",
    "Metadata.zip": "https://osf.io/download/vdxas/",
}


def download(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for fname, url in FILES.items():
        dst = raw_root / fname
        if dst.exists() and dst.stat().st_size > 0:
            print(f"  {fname}: already present -> {dst}")
            continue
        print(f"  downloading {fname} …")
        urllib.request.urlretrieve(url, dst)
        print(f"    -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")

    for fname in ("Cirrhosis_T1_3D.zip", "Cirrhosis_T2_3D.zip", "Metadata.zip"):
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
    ap.add_argument("--skip-download", action="store_true",
                    help="skip OSF download/extract (0_raw already populated)")
    args = ap.parse_args()

    print("=" * 64)
    print("CirrMRI600+ — Download (T1_3D + T2_3D) + extract")
    print("=" * 64)

    if not args.skip_download:
        print(f"\nDownloading -> {RAW_ROOT}")
        download(RAW_ROOT)
    else:
        print("\nSkipping download (--skip-download)")


if __name__ == "__main__":
    main()
