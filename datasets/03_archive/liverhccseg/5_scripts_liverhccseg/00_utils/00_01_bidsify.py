#!/usr/bin/env python3
"""
BIDSify the LiverHccSeg archive (0_raw_liverhccseg/nifti_and_segms/) ->
1_BIDS_liverhccseg/liverhccseg/.

17 TCGA-LIHC (The Cancer Genome Atlas Liver Hepatocellular Carcinoma) patients, 4
native multiphasic CE-T1w phases each (pre-contrast, arterial, portal-venous,
delayed) -- genuinely HCC-only (same disease as atlas-liver-hcc, unlike lld-mmri-hcc's
broader liver-lesion pool), independent cohort/scanner/protocol from ATLAS's own Dijon
cohort, cross-eval-only test set for atlas-liver-hcc. Same CE-T1w contrast FAMILY as
ATLAS's training data (not a cross-contrast test the way lld-mmri-hcc's T2WI/DWI are),
but a genuinely independent site/scanner/protocol -- added as an extra concordant
stratum in the pooled significance test, not to replace the cross-contrast axis.

Two independent board-certified radiologist raters annotated each case (real expert
masks, unlike lld-mmri-hcc's MedSAM2 semi-automated ones) -- this pipeline uses
**rater1 only** as the active ground truth (arbitrary but consistent and documented;
the dataset's own contribution is studying INTER-rater variability, so it deliberately
provides no single blessed consensus). rater2 is hard-linked into derivatives too for
future reference but not read by 05_00_build_test_inputs.py.

Only 14/17 patients have tumour annotations (3 have liver-only: TCGA-BC-4073,
TCGA-BC-A216, TCGA-DD-A4NB) -- those 3 are BIDSified (liver mask only) but will be
excluded from the nnUNet test-input build (05_00_build_test_inputs.py), since we score
`tumour` only, matching lld-mmri-hcc. One patient (TCGA-BC-A10Y) has 3 separate tumour
instances (rater{1,2}_tumor{1,2,3}.nii.gz) -- merged into ONE binary lesion mask here
(logical OR), since atlas-liver-hcc's tumour class doesn't distinguish instances.

Subject ids: sequential liverhccseg000..016 (TCGA barcodes aren't BIDS-legal as-is --
contain hyphens). Mapping preserved in participants.tsv / case_id_map.json.

Reads:   0_raw_liverhccseg/nifti_and_segms/<TCGA-ID>/<date>/{pre,art,pv,del}.nii.gz
                                                            /rater{1,2}_liver.nii.gz
                                                            /rater{1,2}_tumor{1,2,3}.nii.gz
Writes:  1_BIDS_liverhccseg/liverhccseg/
           dataset_description.json, participants.tsv, case_id_map.json
           sub-liverhccsegNNN/anat/sub-liverhccsegNNN_{phase}.nii.gz (+ .json sidecars)
           derivatives/manual_masks/sub-liverhccsegNNN/anat/
             sub-liverhccsegNNN_{phase}_liver-rater1_dseg.nii.gz
             sub-liverhccsegNNN_{phase}_liver-rater2_dseg.nii.gz
             sub-liverhccsegNNN_{phase}_tumour-rater1_dseg.nii.gz  (only the 14 with tumour)
             sub-liverhccsegNNN_{phase}_tumour-rater2_dseg.nii.gz

Usage:  python 00_01_bidsify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]                    # datasets/liverhccseg
RAW = DATASET_ROOT / "0_raw_liverhccseg" / "nifti_and_segms"
BIDS_ROOT = DATASET_ROOT / "1_BIDS_liverhccseg" / "liverhccseg"
DERIV_DIR = BIDS_ROOT / "derivatives" / "manual_masks"

PHASES = {"pre": "ce-pre_T1w", "art": "ce-art_T1w", "pv": "ce-pv_T1w", "del": "ce-del_T1w"}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil
        shutil.copyfile(src, dst)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _merge_tumor_instances(case_dir: Path, rater: str) -> Path | None:
    """Logical-OR all rater{N}_tumor{1,2,3...}.nii.gz for this case into one binary
    mask, written to a temp file next to the source. Returns None if no tumour files
    exist for this rater (the 3 liver-only patients)."""
    tumor_files = sorted(case_dir.glob(f"{rater}_tumor*.nii.gz"))
    if not tumor_files:
        return None
    if len(tumor_files) == 1:
        return tumor_files[0]
    ref = nib.load(str(tumor_files[0]))
    merged = np.zeros(ref.shape, dtype=np.uint8)
    for f in tumor_files:
        merged |= (np.asanyarray(nib.load(str(f)).dataobj) > 0).astype(np.uint8)
    out = case_dir / f"{rater}_tumor_merged.nii.gz"
    nib.save(nib.Nifti1Image(merged, ref.affine, ref.header), str(out))
    return out


def bidsify() -> None:
    cases = sorted(p for p in RAW.iterdir() if p.is_dir() and p.name.startswith("TCGA-"))
    if len(cases) != 17:
        raise SystemExit(f"expected 17 patients, found {len(cases)}")

    _json(BIDS_ROOT / "dataset_description.json", {
        "Name": "LiverHccSeg -- TCGA-LIHC HCC liver+tumour segmentation, multiphasic "
                "CE-T1w, cross-eval-only set for atlas-liver-hcc",
        "BIDSVersion": "1.9.0",
        "License": "CC BY 4.0",
        "Authors": ["Sundar Iyer, Andres Chomicki, Sasan Partovi, Rondell Graham, et al."],
        "ReferencesAndLinks": [
            "https://doi.org/10.5281/zenodo.7957515",
            "https://doi.org/10.1016/j.dib.2023.109680",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=6885436",
        ],
        "DatasetType": "raw",
    })
    _json(DERIV_DIR / "dataset_description.json", {
        "Name": "LiverHccSeg liver + HCC tumour segmentation masks (2 independent "
                "board-certified radiologist raters; rater1 used as active GT)",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "Manual expert annotation, 2 independent raters"}],
    })

    case_id_map = {}
    rows = ["participant_id\tsource_tcga_id\tstudy_date\thas_tumour"]
    for i, case_dir in enumerate(cases):
        date_dirs = [d for d in case_dir.iterdir() if d.is_dir()]
        if len(date_dirs) != 1:
            raise SystemExit(f"{case_dir}: expected exactly 1 study-date dir, found {len(date_dirs)}")
        src_dir = date_dirs[0]
        sub = f"sub-liverhccseg{i:03d}"
        has_tumour = _merge_tumor_instances(src_dir, "rater1") is not None
        case_id_map[sub] = {"source_tcga_id": case_dir.name, "study_date": src_dir.name,
                             "has_tumour": has_tumour}

        for raw_phase, suffix in PHASES.items():
            img_src = src_dir / f"{raw_phase}.nii.gz"
            if not img_src.exists():
                raise SystemExit(f"{src_dir}: missing {raw_phase}.nii.gz")
            _link(img_src, BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.nii.gz")
            _json(BIDS_ROOT / sub / "anat" / f"{sub}_{suffix}.json",
                  {"Modality": "MR", "Description": f"{raw_phase} phase, HCC (TCGA-LIHC)"})

            for rater in ("rater1", "rater2"):
                liver_src = src_dir / f"{rater}_liver.nii.gz"
                _link(liver_src, DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_liver-{rater}_dseg.nii.gz")
                tumor_src = _merge_tumor_instances(src_dir, rater)
                if tumor_src is not None:
                    _link(tumor_src, DERIV_DIR / sub / "anat" / f"{sub}_{suffix}_tumour-{rater}_dseg.nii.gz")

        rows.append(f"{sub}\t{case_dir.name}\t{src_dir.name}\t{has_tumour}")

    (BIDS_ROOT / "participants.tsv").write_text("\n".join(rows) + "\n")
    _json(BIDS_ROOT / "case_id_map.json", case_id_map)
    n_tumour = sum(1 for v in case_id_map.values() if v["has_tumour"])
    print(f"BIDSified {len(cases)} patients ({n_tumour} with tumour) -> {BIDS_ROOT}")


if __name__ == "__main__":
    bidsify()
