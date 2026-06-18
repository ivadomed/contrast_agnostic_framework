#!/usr/bin/env python3
"""
Download AMOS22 labeled data from Zenodo and extract to 0_raw_amos/.

AMOS ships as NIfTI (nnUNet-style layout) — no DICOM conversion needed.
We download only the labeled split (amos22.zip, 24 GB) and the metadata CSV.
The unlabeled CT/MRI splits (~5 TB total) are NOT downloaded.

Zenodo record: https://zenodo.org/records/7262581

Raw layout after extraction (0_raw_amos/amos22/):
  imagesTr/  amos_0001.nii.gz … amos_0399.nii.gz
  labelsTr/  amos_0001.nii.gz … amos_0399.nii.gz
  imagesVa/  amos_0400.nii.gz … amos_0499.nii.gz + MRI 0500–0599
  labelsVa/  …
  imagesTs/  (test images, no GT — optionally skipped during evaluation)
  dataset.json

ID convention: CT 0001–0499, MRI 0500–0599.
15 labels: 0=bg, 1=spleen, 2=R_kidney, 3=L_kidney, 4=gallbladder, 5=esophagus,
           6=liver, 7=stomach, 8=aorta, 9=ivc, 10=pancreas, 11=R_adrenal,
           12=L_adrenal, 13=duodenum, 14=R_bladder, 15=prostate_uterus.

Usage:
  python 00_00_download_and_extract.py                  # download + extract
  python 00_00_download_and_extract.py --skip-download  # 0_raw already populated
"""
import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT     = DATASET_ROOT / "0_raw_amos"

ZENODO_RECORD = "7262581"
ZENODO_BASE   = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
# Only the labeled zip + metadata. Unlabeled parts (900CT/1100CT/1200MRI) are skipped.
ZENODO_FILES  = [
    ("labeled_data_meta_0000_0599.csv", "labeled_data_meta.csv"),
    ("amos22.zip",                       "amos22.zip"),
]


class ProgressBar:
    def __init__(self, name: str):
        self.name = name
        self._last = -1

    def __call__(self, count: int, block: int, total: int) -> None:
        pct = min(int(count * block * 100 / total), 100) if total > 0 else 0
        if pct != self._last:
            print(f"\r  {self.name}: {pct:3d}%", end="", flush=True)
            self._last = pct
        if pct == 100:
            print()


def download(raw_root: Path) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in ZENODO_FILES:
        dst = raw_root / dst_name
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  {dst_name}: already present ({dst.stat().st_size / 1e9:.2f} GB)")
            continue
        url = f"{ZENODO_BASE}/{src_name}/content"
        print(f"  downloading {dst_name} …")
        urllib.request.urlretrieve(url, dst, reporthook=ProgressBar(dst_name))
        print(f"    → {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


def extract(raw_root: Path) -> None:
    zp = raw_root / "amos22.zip"
    if not zp.exists():
        print(f"ERROR: {zp} not found — run without --skip-download first", file=sys.stderr)
        sys.exit(1)
    if (raw_root / "amos22").exists():
        n = len(list((raw_root / "amos22").rglob("*.nii.gz")))
        print(f"  amos22/: already extracted ({n} .nii.gz files)")
        return
    print(f"  extracting amos22.zip ({zp.stat().st_size / 1e9:.1f} GB) …")
    with zipfile.ZipFile(zp) as z:
        z.extractall(raw_root)
    n = len(list((raw_root / "amos22").rglob("*.nii.gz")))
    print(f"  extracted → {raw_root / 'amos22'}  ({n} .nii.gz files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true",
                    help="skip Zenodo download (0_raw already populated)")
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip zip extraction (amos22/ already present)")
    args = ap.parse_args()

    print("=" * 64)
    print("AMOS22 — Download labeled data from Zenodo + extract")
    print("=" * 64)

    if not args.skip_download:
        print(f"\n[1/2] Downloading from Zenodo → {RAW_ROOT}")
        download(RAW_ROOT)
    else:
        print("\n[1/2] Skipping download (--skip-download)")

    if not args.skip_extract:
        print(f"\n[2/2] Extracting amos22.zip → {RAW_ROOT}/amos22/")
        extract(RAW_ROOT)
    else:
        print("\n[2/2] Skipping extraction (--skip-extract)")

    # Sanity report
    amos22 = RAW_ROOT / "amos22"
    if amos22.exists():
        for d in ("imagesTr", "labelsTr", "imagesVa", "labelsVa", "imagesTs"):
            p = amos22 / d
            n = len(list(p.glob("*.nii.gz"))) if p.exists() else 0
            print(f"  {d:12s}: {n:4d} .nii.gz")
        dsj = amos22 / "dataset.json"
        print(f"  dataset.json: {'present' if dsj.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
