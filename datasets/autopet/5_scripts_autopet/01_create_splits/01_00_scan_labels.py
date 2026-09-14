#!/usr/bin/env python3
"""
Scan every FDG and PSMA label volume directly out of the archive zip and record, per
case: positive-voxel count (the real usable-N — do NOT trust the "diagnosis" metadata
column, see below), orientation (nib.aff2axcodes — verified EVERY case, not a sample,
per this project's own "verify orientation on onboarding" lesson), shape, and spacing.

Why not use fdg_metadata.csv's "diagnosis" column for positivity: verified directly
(2026-09-13) that 30 FDG patients have MULTIPLE studies with DIFFERING diagnosis values
(e.g. one study "LYMPHOMA", another study of the SAME patient "NEGATIVE") — this is a
real clinical fact (a later surveillance scan can be lesion-free even for a patient with
that underlying disease), not a data error. "diagnosis" is patient-level disease
category, not per-study lesion presence. psma_metadata.csv doesn't even have a
diagnosis/lesion column at all. The only correct way to determine per-STUDY positivity
is to check that study's own label file for a nonzero voxel — which is what this script
does, matching this project's pre-flight checklist item #2 (claimed-N vs directly-
counted usable-N).

Output: 4_splits_autopet/label_scan_manifest.json — one entry per case:
  {case_id, tracer, n_positive_voxels, shape, zooms, axcodes}
01_01_create_splits.py consumes this instead of re-scanning.

This is real compute (~1611 cases, each needs its label decompressed + loaded), not a
login-node-light check — run via run_job, not directly.

Usage: .venv/bin/python 01_00_scan_labels.py
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", str(DATASET_ROOT / "4_splits_autopet")))
ARCHIVE_ZIP = Path(os.environ.get(
    "AUTOPET_ARCHIVE_ZIP", "/scratch/paulh/pet_task_staging/autopet/psma-fdg-pet-ct-lesions_v2.zip"))


def _case_id_from_label_path(name: str) -> str:
    # e.g. ".../labelsTr/fdg_01140d52d8_08-13-2005-NA-PET-CT ....nii.gz" -> case id
    # (the archive's own case-id convention: everything after "labelsTr/", minus ".nii.gz")
    base = name.split("labelsTr/")[-1]
    return base[:-len(".nii.gz")] if base.endswith(".nii.gz") else base


def _scan_one(zf: zipfile.ZipFile, member: str) -> dict:
    with zf.open(member) as src:
        data = src.read()
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        arr = img.get_fdata()
        return {
            "n_positive_voxels": int((arr > 0).sum()),
            "shape": list(img.shape),
            "zooms": [float(z) for z in img.header.get_zooms()],
            "axcodes": list(nib.aff2axcodes(img.affine)),
        }
    finally:
        os.unlink(tmp_path)


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ARCHIVE_ZIP)
    names = zf.namelist()

    manifest = []
    axcode_counts: dict[tuple, int] = {}
    for tracer in ("fdg", "psma"):
        labels = sorted([n for n in names if f"labelsTr/{tracer}_" in n])
        print(f"[scan] {tracer}: {len(labels)} label files")
        for i, lbl in enumerate(labels):
            case_id = _case_id_from_label_path(lbl)
            info = _scan_one(zf, lbl)
            info["case_id"] = case_id
            info["tracer"] = tracer
            manifest.append(info)
            ax = tuple(info["axcodes"])
            axcode_counts[ax] = axcode_counts.get(ax, 0) + 1
            if (i + 1) % 100 == 0:
                print(f"  [{tracer}] {i+1}/{len(labels)} done")

    n_pos = {t: sum(1 for m in manifest if m["tracer"] == t and m["n_positive_voxels"] > 0)
             for t in ("fdg", "psma")}
    n_total = {t: sum(1 for m in manifest if m["tracer"] == t) for t in ("fdg", "psma")}
    print(f"[scan] FDG: {n_pos['fdg']}/{n_total['fdg']} studies positive")
    print(f"[scan] PSMA: {n_pos['psma']}/{n_total['psma']} studies positive")
    print(f"[scan] axcode distribution: {axcode_counts}")

    out = SPLITS_DIR / "label_scan_manifest.json"
    out.write_text(json.dumps({
        "manifest": manifest,
        "n_positive": n_pos,
        "n_total": n_total,
        "axcode_counts": {"_".join(k): v for k, v in axcode_counts.items()},
    }, indent=2))
    print(f"[scan] wrote {out}")


if __name__ == "__main__":
    main()
