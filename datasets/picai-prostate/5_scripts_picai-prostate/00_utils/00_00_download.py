#!/usr/bin/env python3
"""Download the PI-CAI public training/development data into 0_raw_picai-prostate/.

Two sources, both fully open:
  1. Images  — Zenodo record 6624726 ("The PI-CAI Challenge: Public Training and
     Development Dataset", CC-BY-NC-4.0): 5 zips (`picai_public_images_fold{0..4}.zip`,
     ~27 GB total) holding 1500 bpMRI studies as MHA (`*_t2w.mha`, `*_adc.mha`,
     `*_hbv.mha`, plus unused sagittal/coronal series).
  2. Labels  — GitHub `DIAGNijmegen/picai_labels` (~35 MB): the human-expert csPCa
     lesion delineations (`csPCa_lesion_delineations/human_expert/resampled/`, already
     resampled into each study's T2W frame) and the AI whole-gland prostate masks
     (`anatomical_delineations/whole_gland/AI/Bosma22b/`), which 00_01_bidsify.py uses
     only to CENTRE the common resampling grid on the prostate.

LOGIN-NODE ONLY — this is deliberate and is why there is no `run_job` wrapper around the
download itself (contrast: amos/cirrmri-liver, whose downloads do go through run_job on
clusters where compute nodes can reach the internet). On tamia the compute nodes have no
outbound network at all, so a submitted job cannot fetch anything. The work here is pure
network I/O (no CPU, <1 GB RAM), which is inside the login-node exception in CLAUDE.md.
Downloads are sequential (one stream) to stay polite to both Zenodo and the login node.

Usage:
    bash 00_00_download.sh                 # images + labels
    bash 00_00_download.sh --labels-only   # refresh just picai_labels
    bash 00_00_download.sh --skip-extract  # download zips but do not unzip
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ZENODO_RECORD = "6624726"
IMAGE_ZIPS = [f"picai_public_images_fold{k}.zip" for k in range(5)]
LABELS_REPO = "https://github.com/DIAGNijmegen/picai_labels.git"


def sh(cmd: list[str], **kw) -> None:
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_record() -> dict:
    import json
    import urllib.request

    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download_images(raw_root: Path, skip_extract: bool) -> None:
    rec = fetch_record()
    files = {f["key"]: f for f in rec["files"]}
    zips_dir = raw_root / "_zips"
    zips_dir.mkdir(parents=True, exist_ok=True)

    for name in IMAGE_ZIPS:
        if name not in files:
            sys.exit(f"ERROR: {name} not present in Zenodo record {ZENODO_RECORD}")
        meta = files[name]
        want_md5 = meta["checksum"].split(":")[-1]
        dest = zips_dir / name
        if dest.exists() and md5(dest) == want_md5:
            print(f"[images] {name}: already downloaded and checksum-verified", flush=True)
        else:
            print(f"[images] {name}: downloading {meta['size'] / 1e9:.2f} GB", flush=True)
            # -C - resumes a partial file; --retry rides out transient Zenodo 5xx.
            sh(["curl", "-L", "--fail", "--retry", "8", "--retry-delay", "15",
                "-C", "-", "-o", str(dest), meta["links"]["self"]])
            got = md5(dest)
            if got != want_md5:
                sys.exit(f"ERROR: md5 mismatch for {name}: got {got}, want {want_md5}")
            print(f"[images] {name}: md5 OK", flush=True)

        if skip_extract:
            continue
        marker = raw_root / f".extracted_{name}"
        if marker.exists():
            print(f"[images] {name}: already extracted", flush=True)
            continue
        print(f"[images] {name}: extracting", flush=True)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(raw_root / "images")
        marker.touch()


def download_labels(raw_root: Path) -> None:
    dest = raw_root / "picai_labels"
    if (dest / ".git").is_dir():
        print("[labels] picai_labels present — pulling", flush=True)
        sh(["git", "-C", str(dest), "pull", "--ff-only"])
    else:
        print("[labels] cloning picai_labels", flush=True)
        sh(["git", "clone", "--depth", "1", LABELS_REPO, str(dest)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    raw_root = Path(os.environ["RAW_ROOT"]).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"RAW_ROOT = {raw_root}", flush=True)

    download_labels(raw_root)
    if not args.labels_only:
        download_images(raw_root, args.skip_extract)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
