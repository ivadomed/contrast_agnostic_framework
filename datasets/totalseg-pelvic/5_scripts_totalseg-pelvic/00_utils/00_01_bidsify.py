#!/usr/bin/env python3
"""
BIDSify TotalSegmentator CT + MRI (pelvic/hip subset) -> 1_BIDS_totalseg-pelvic/pelvis-totalsegpelvic/.

Corrects the original onboarding, which converted straight from the Zenodo zips to
nnU-Net format with no BIDS staging in between (the pattern autopet/duke-breast-mri use
deliberately -- wrong call here, per direct user correction 2026-09-15). This project's
standard pattern is raw -> BIDS -> nnUNet; this script is the raw -> BIDS half.

CT and MRI are UNPAIRED, disjoint cohorts (different patients, like chaos's CT/MR) --
same pattern chaos uses (sub-CT##/sub-MR## disjoint id spaces). Here: sub-ct<id>/sub-mri<id>.

Per case: writes the anat image (already correctly reoriented in most cases, see below)
plus the 10 individual pelvic/hip structure masks UNMERGED (one file per structure, BIDS
"label-<name>_seg.nii.gz" derivative convention -- same shape as liverhccseg's
00_01_bidsify.py) -- merging into one multiclass nnU-Net label happens downstream in
02_nnunet/02_00_convert.py, not here.

Source of the 10 masks: MUST be (re)fetched from the Zenodo archives -- the original
conversion pass merged them in-memory and never wrote the unmerged per-structure files to
disk, so there is no local copy to reuse for the derivative layer even for cases whose
merged nnUNet output already exists. The anat image itself IS reused from the existing
nnUNet imagesTr output where present (already fetched + reoriented correctly there), to
avoid re-fetching image bytes unnecessarily.

Usable-case determination is NOT redone here (same source bytes as before) -- read from
each modality's usable_cases.json where a case's usability is already known from an
earlier candidate that finished the (image+label) fetch; for brand-new CT top-up
candidates never touched before, usability is (re)determined here since this fetch IS the
first exposure to that case's mask content.

Usage:
  .venv/bin/python 00_01_bidsify.py --modality mri
  .venv/bin/python 00_01_bidsify.py --modality ct --target-usable 200
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenodo_zip import CT_ZENODO_URL, MRI_ZENODO_URL, open_zip  # noqa: E402

DATASET_ROOT = Path(os.environ["DATASET_ROOT"]) if "DATASET_ROOT" in os.environ else \
    Path(__file__).resolve().parents[2]
SPLITS_ROOT = DATASET_ROOT / "4_splits_totalseg-pelvic"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_totalseg-pelvic" / "pelvis-totalsegpelvic"
DERIV_DIR = BIDS_ROOT / "derivatives" / "labels"
OLD_NNUNET_RAW = DATASET_ROOT / "2_nnUNet_totalseg-pelvic" / "raw"

CT_ZIP = Path(os.environ.get("TOTALSEG_CT_ZIP", "/scratch/paulh/totalseg_preflight/ct_full.zip"))
MRI_ZIP = Path(os.environ.get("TOTALSEG_MRI_ZIP", "/scratch/paulh/totalseg_preflight/mri.zip"))
MODALITY_URLS = {"ct": CT_ZENODO_URL, "mri": MRI_ZENODO_URL}
MODALITY_ZIPS = {"ct": CT_ZIP, "mri": MRI_ZIP}
MODALITY_SUFFIX = {"ct": "CT", "mri": "MRI"}  # non-standard BIDS suffix, same pattern chaos uses for _CT
MODALITY_IMG_MEMBER = {"ct": "ct.nii.gz", "mri": "mri.nii.gz"}
OLD_DATASET_DIR = {"ct": "Dataset130_TotalsegPelvic_CT", "mri": "Dataset131_TotalsegPelvic_MRI"}

STRUCTURES = [
    "hip_left", "hip_right", "sacrum",
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right",
    "iliopsoas_left", "iliopsoas_right",
]

# BIDS entity values can't contain underscores (that's the key-value separator the spec
# itself uses to split filenames) -- STRUCTURES above stays snake_case because it also
# doubles as the Zenodo zip member name (case}/segmentations/{name}.nii.gz), but every
# `label-<value>` BIDS filename must use this camelCase form instead. Fixed 2026-09-15
# (audit found 3830 on-disk derivative files violating this -- renamed in place).
BIDS_LABEL = {
    "hip_left": "hipLeft", "hip_right": "hipRight", "sacrum": "sacrum",
    "gluteus_maximus_left": "gluteusMaximusLeft", "gluteus_maximus_right": "gluteusMaximusRight",
    "gluteus_medius_left": "gluteusMediusLeft", "gluteus_medius_right": "gluteusMediusRight",
    "gluteus_minimus_left": "gluteusMinimusLeft", "gluteus_minimus_right": "gluteusMinimusRight",
    "iliopsoas_left": "iliopsoasLeft", "iliopsoas_right": "iliopsoasRight",
}


def _sub_id(modality: str, case: str) -> str:
    # case like "s0004" -> "sub-ct0004" / "sub-mri0004"
    return f"sub-{modality}{case[1:]}"


def _candidates(modality: str) -> list[str]:
    return sorted(json.loads((SPLITS_ROOT / modality / "candidate_cases.json").read_text())["candidates"])


def _load_nifti_bytes(raw_bytes: bytes):
    import nibabel as nib
    return nib.Nifti1Image.from_bytes(gzip.decompress(raw_bytes))


def _to_gzip_bytes(img) -> bytes:
    buf = io.BytesIO()
    file_map = img.make_file_map()
    file_map["image"].fileobj = buf
    img.to_file_map(file_map)
    return gzip.compress(buf.getvalue())


def _reorient_canonical(img):
    import nibabel as nib
    return nib.as_closest_canonical(img)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_dataset_descriptions() -> None:
    _write_json(BIDS_ROOT / "dataset_description.json", {
        "Name": "TotalSegmentator pelvic/hip subset (CT + MRI, unpaired)",
        "BIDSVersion": "1.9.0",
        "License": "See LICENSE.md (CT: CC BY 4.0; MRI: CC BY-NC-SA 2.0 -- different "
                   "licenses per modality, do not blend into one statement)",
        "Authors": ["Wasserthal, Jakob", "Akinci D'Antonoli, Tugba", "et al."],
        "ReferencesAndLinks": [
            "https://doi.org/10.5281/zenodo.10047292",
            "https://doi.org/10.5281/zenodo.11367005",
            "https://doi.org/10.1148/ryai.230024",
        ],
        "DatasetType": "raw",
    })
    _write_json(DERIV_DIR / "dataset_description.json", {
        "Name": "TotalSegmentator pelvic/hip structure masks (10 classes, unmerged "
                "per-structure binary masks; multiclass merge happens downstream at "
                "nnU-Net conversion time)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "TotalSegmentator model + iterative manual "
                                  "correction (source project's own pipeline)"}],
    })


def _existing_image_bytes(modality: str, case: str) -> bytes | None:
    """Reuse the already-fetched+reoriented nnUNet image if this case was already
    converted in the original (pre-BIDS) pass -- avoids re-fetching image bytes."""
    p = OLD_NNUNET_RAW / OLD_DATASET_DIR[modality] / "imagesTr" / f"{case}_0000.nii.gz"
    return p.read_bytes() if p.exists() else None


def _bidsify_case(zf: zipfile.ZipFile, modality: str, case: str) -> str:
    """Returns 'usable', 'empty', or 'missing'. On 'usable', writes the BIDS anat image
    + 10 per-structure derivative masks; on 'empty'/'missing' writes nothing (and cleans
    up any partial output from an interrupted prior attempt)."""
    import numpy as np

    sub = _sub_id(modality, case)
    suffix = MODALITY_SUFFIX[modality]
    img_dst = BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz"
    lbl_dsts = {name: DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_label-{BIDS_LABEL[name]}_seg.nii.gz"
                for name in STRUCTURES}

    member_prefix = f"{case}/"
    names_in_case = {n for n in zf.namelist() if n.startswith(member_prefix)}
    img_member = f"{case}/{MODALITY_IMG_MEMBER[modality]}"
    if img_member not in names_in_case:
        return "missing"

    # Fetch + merge (in memory only, to test emptiness) the 10 structure masks.
    label_imgs = {}
    ref_affine = None
    any_nonzero = False
    for name in STRUCTURES:
        member = f"{case}/segmentations/{name}.nii.gz"
        with zf.open(member) as fh:
            raw = fh.read()
        img = _load_nifti_bytes(raw)
        if ref_affine is None:
            ref_affine = img.affine
        arr = np.asarray(img.dataobj)
        if np.any(arr):
            any_nonzero = True
        label_imgs[name] = img

    if not any_nonzero:
        for f in [img_dst, *lbl_dsts.values()]:
            f.unlink(missing_ok=True)
        return "empty"

    # Image: reuse existing reoriented bytes if this case was already converted before,
    # else fetch fresh from the zip.
    existing = _existing_image_bytes(modality, case)
    if existing is not None:
        # Already-fetched nnUNet image bytes are gzip-compressed nifti, same as what
        # _load_nifti_bytes expects.
        image_img = _load_nifti_bytes(existing)
    else:
        with zf.open(img_member) as src:
            image_img = _load_nifti_bytes(src.read())
        image_img = _reorient_canonical(image_img)

    for name in STRUCTURES:
        label_imgs[name] = _reorient_canonical(label_imgs[name])

    img_dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = img_dst.with_suffix(img_dst.suffix + ".tmp")
    tmp.write_bytes(_to_gzip_bytes(image_img))
    os.replace(tmp, img_dst)
    _write_json(img_dst.with_suffix("").with_suffix(".json"),
                {"Modality": suffix, "Description": f"TotalSegmentator source case {case}"})

    for name, dst in lbl_dsts.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(_to_gzip_bytes(label_imgs[name]))
        os.replace(tmp, dst)

    return "usable"


def bidsify_modality(modality: str, target_usable: int | None) -> None:
    _write_dataset_descriptions()
    candidates = _candidates(modality)
    url = MODALITY_URLS[modality]
    zip_path = MODALITY_ZIPS[modality]
    print(f"[bidsify][{modality}] source: "
          f"{'local ' + str(zip_path) if zip_path.exists() else 'remote (HTTP range-read) ' + url}")
    print(f"[bidsify][{modality}] {len(candidates)} total candidates available")

    usable, empty, missing = [], [], []
    participants_rows = ["participant_id\tsource_case_id\tmodality"]

    with open_zip(zip_path, url) as zf:
        for i, case in enumerate(candidates):
            if target_usable is not None and len(usable) >= target_usable:
                print(f"[bidsify][{modality}] reached target_usable={target_usable}, stopping "
                      f"({i}/{len(candidates)} candidates tried)")
                break
            sub = _sub_id(modality, case)
            img_dst = BIDS_ROOT / sub / "anat" / f"{sub}_{MODALITY_SUFFIX[modality]}.nii.gz"
            if img_dst.exists():
                usable.append(case)
                participants_rows.append(f"{sub}\t{case}\t{modality}")
                continue

            print(f"[bidsify][{modality}] ({i + 1}/{len(candidates)}) case '{case}'")
            result = _bidsify_case(zf, modality, case)
            if result == "usable":
                usable.append(case)
                participants_rows.append(f"{sub}\t{case}\t{modality}")
            elif result == "empty":
                empty.append(case)
            else:
                missing.append(case)

    print(f"[bidsify][{modality}] usable={len(usable)} empty={len(empty)} missing={len(missing)}")

    manifest_path = SPLITS_ROOT / modality / "bids_usable_cases.json"
    manifest_path.write_text(json.dumps({
        "usable": sorted(usable), "empty": sorted(empty), "missing": sorted(missing),
    }, indent=2))
    print(f"[bidsify][{modality}] wrote {manifest_path}")

    # participants.tsv is cumulative across modalities -- merge rather than overwrite.
    ptsv = BIDS_ROOT / "participants.tsv"
    existing_rows = ptsv.read_text().splitlines() if ptsv.exists() else [participants_rows[0]]
    existing_subs = {line.split("\t")[0] for line in existing_rows[1:]}
    new_rows = [r for r in participants_rows[1:] if r.split("\t")[0] not in existing_subs]
    ptsv.write_text("\n".join(existing_rows + new_rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["ct", "mri"], required=True)
    parser.add_argument("--target-usable", type=int, default=None,
                         help="stop once this many usable cases are found (default: try all candidates)")
    args = parser.parse_args()
    bidsify_modality(args.modality, args.target_usable)


if __name__ == "__main__":
    main()
