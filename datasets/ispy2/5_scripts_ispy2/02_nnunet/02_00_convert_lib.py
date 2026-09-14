#!/usr/bin/env python3
"""
Shared BIDS -> nnU-Net conversion library for ispy2 (I-SPY2), the project's
PRIMARY breast-cancer TRAINING dataset as of 2026-09-04 (it replaces ambl, whose
99-case cohort overfit; ambl is now the malignant-only external test set).

Two training modalities, one nnU-Net Dataset id each (same shape as ambl
t1wce/t2w, chaos t1in/t2spir, brats2024-glioma t1n/t2w):

    t1wce -> Dataset100_ISPY2T1wce      (early post-contrast T1w DCE phase)
    t2w   -> Dataset101_ISPY2T2w

Ids are the ones already declared in 00_utils/env.sh (DATASET_ID_T1WCE=100,
DATASET_ID_T2W=101) -- do not renumber here without fixing that file.

WHAT DRIVES THE CASE LIST (never re-derived here)
-------------------------------------------------
  4_splits_ispy2/partition.json   -> cases[<modality>][train_pool|test], the
                                     authoritative case ids
  4_splits_ispy2/fov_variants.json-> per patient, per "<modality>/<fov>" variant,
                                     the exact BIDS-relative image AND label
                                     path (native or script-derived)

Case ids are `ispy2_<PID>_<bil|uni>`: I-SPY2 carries BOTH a bilateral (both
breasts) and a unilateral (single-breast crop) FOV variant of the same patient,
and FOV is a first-class axis of this task, so each (patient, FOV) pair is its
OWN nnU-Net case. The splits in 4_splits_ispy2/ are patient-level and
FOV-stratified, so a patient's two variants never straddle train/val.

Expected sizes (verified against partition.json before writing):
  t1wce : 580 train_pool + 102 test   (bilateral DCE exists for only the 122
          natively-bilateral patients -- TCIA has no bilateral DCE series for the
          438 natively-unilateral ones, so t1wce/bil is legitimately absent
          there. This asymmetry is expected and correct.)
  t2w   : 952 train_pool + 168 test   (both FOVs obtainable for all 560 patients)

LABELS: {0: background, 1: tumour} -- single binary lesion mask, and every label
file loaded is the one registered to the image space actually being written
(the t2w bilateral/unilateral masks are `desc-sharedframe` re-derivations; see
01_00_audit_t2w_label_registration.py / 01_01_rederive_t2w_labels.py). A mask
from one modality is NEVER reused for the other -- the same correctness rule
ambl's converters had to learn the hard way.

CROSS-CONTRAST TEST SETS: each Dataset dir carries BOTH modalities' held-out
test cases (imagesTs_t1wce + imagesTs_t2w), so the cross-contrast generalization
axis lives inside the Dataset dir, exactly as in ambl's Dataset090/091.

Output under 2_nnUNet_ispy2/raw/Dataset10X_ISPY2.../:
  imagesTr/<case>_0000.nii.gz     train pool, this Dataset's own modality
  labelsTr/<case>.nii.gz          tumour mask uint8 {0,1}, that image's space
  imagesTs_t1wce/<case>_0000.nii.gz + labelsTs_t1wce/<case>.nii.gz
  imagesTs_t2w/<case>_0000.nii.gz   + labelsTs_t2w/<case>.nii.gz
  dataset.json

LEGACY DIRS: 2_nnUNet_ispy2/raw/ previously held FLAT imagesTs_*/labelsTs_*
dirs (122 bilateral-only cases, old case ids `ispy2<PID>`) built by
05_predict/05_00_build_test_inputs.py back when ispy2 was an eval-only test set
for ambl-trained models. Those are stale under the current design and were moved
to 2_nnUNet_ispy2/raw/_legacy_ambl_crosseval/ (see the README written there) so
they can never be mixed with the new Dataset-scoped test sets. They remain
regenerable by re-running 05_00_build_test_inputs.py.

ORIENTATION: every written image and label is checked to be project-canonical
LPS via nib.aff2axcodes -- a hard requirement in this project (CLAUDE.md,
"Verify orientation on onboarding"). A non-LPS file aborts the conversion.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[2]              # datasets/ispy2
BIDS_ROOT = DATASET_ROOT / "1_BIDS_ispy2" / "breast-ispy2"
DERIV_ROOT = BIDS_ROOT / "derivatives" / "labels"
SPLITS = DATASET_ROOT / "4_splits_ispy2"
RAW = DATASET_ROOT / "2_nnUNet_ispy2" / "raw"

EXPECTED_AXCODES = ("L", "P", "S")
MODALITIES = ("t1wce", "t2w")

DATASET_DIRNAME = {
    "t1wce": "Dataset100_ISPY2T1wce",
    "t2w": "Dataset101_ISPY2T2w",
}
CHANNEL_NAME = {"t1wce": "T1WCE", "t2w": "T2w"}

REFERENCE = (
    "I-SPY2 Trial (ISPY2) — The Cancer Imaging Archive. "
    "https://www.cancerimagingarchive.net/collection/ispy2/ "
    "(DOI: https://doi.org/10.7937/TCIA.D8Z0-9T85)"
)
LICENCE = "CC BY 4.0"


def load_inputs():
    partition = json.loads((SPLITS / "partition.json").read_text())
    variants = json.loads((SPLITS / "fov_variants.json").read_text())["patients"]
    return partition, variants


def variant_paths(variants: dict, case_id: str, modality: str) -> tuple[Path, Path]:
    """`ispy2_<PID>_<fov>` + modality -> (image path, label path), straight out of
    fov_variants.json. Raises KeyError if the case/variant is not listed -- we
    never guess a filename."""
    _, pid, fov = case_id.split("_")
    rec = variants[f"sub-ispy2{pid}"]["variants"][f"{modality}/{fov}"]
    return BIDS_ROOT / rec["image"], DERIV_ROOT / rec["label"]


def _check_axcodes(path: Path) -> None:
    ax = nib.aff2axcodes(nib.load(str(path)).affine)
    if tuple(ax) != EXPECTED_AXCODES:
        raise RuntimeError(f"non-LPS orientation {ax} in {path}")


def copy_image(src: Path, dst: Path) -> None:
    _check_axcodes(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_mask(src: Path, dst: Path) -> int:
    """Load a lesion mask, enforce uint8 {0,1}, preserve affine/header geometry.
    Returns the foreground voxel count so callers can assert no empty masks."""
    img = nib.load(str(src))
    if tuple(nib.aff2axcodes(img.affine)) != EXPECTED_AXCODES:
        raise RuntimeError(f"non-LPS orientation in label {src}")
    arr = np.asanyarray(img.dataobj)
    uniq = np.unique(arr)
    if not set(np.asarray(uniq).ravel().tolist()) <= {0, 1}:
        raise RuntimeError(f"label values {uniq.tolist()} not in {{0,1}}: {src}")
    arr = arr.astype(np.uint8)
    out = nib.Nifti1Image(arr, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return int(arr.sum())


def convert(modality: str) -> None:
    assert modality in MODALITIES
    partition, variants = load_inputs()
    out = RAW / DATASET_DIRNAME[modality]
    train_pool = partition["cases"][modality]["train_pool"]

    empty: list[str] = []

    # --- training pool: this Dataset's own modality ---------------------------
    for case_id in train_pool:
        img, lab = variant_paths(variants, case_id, modality)
        copy_image(img, out / "imagesTr" / f"{case_id}_0000.nii.gz")
        if write_mask(lab, out / "labelsTr" / f"{case_id}.nii.gz") == 0:
            empty.append(f"Tr/{case_id}")

    # --- held-out test, BOTH modalities (cross-contrast axis) -----------------
    for test_mod in MODALITIES:
        for case_id in partition["cases"][test_mod]["test"]:
            img, lab = variant_paths(variants, case_id, test_mod)
            copy_image(img, out / f"imagesTs_{test_mod}" / f"{case_id}_0000.nii.gz")
            n = write_mask(lab, out / f"labelsTs_{test_mod}" / f"{case_id}.nii.gz")
            if n == 0:
                empty.append(f"Ts_{test_mod}/{case_id}")

    if empty:
        raise RuntimeError(f"{len(empty)} EMPTY masks: {empty[:20]}")

    dataset_json = {
        "name": DATASET_DIRNAME[modality].split("_", 1)[1],
        "description": (
            "I-SPY2 breast tumour segmentation on "
            + ("early post-contrast T1w DCE-MRI (T1WCE)" if modality == "t1wce" else "T2w MRI")
            + ". Primary breast-cancer training task; two training modalities "
            "(t1wce=Dataset100, t2w=Dataset101), each case being one "
            "(patient, FOV) pair -- bilateral and unilateral FOV variants are "
            "separate cases, patient-level FOV-stratified 3-fold splits."
        ),
        "reference": REFERENCE,
        "licence": LICENCE,
        "release": "1.0",
        "channel_names": {"0": CHANNEL_NAME[modality]},
        "labels": {"background": 0, "tumour": 1},
        "numTraining": len(train_pool),
        "file_ending": ".nii.gz",
    }
    (out / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    def n(sub: str) -> int:
        return len(list((out / sub).glob("*.nii.gz")))

    print(f"{DATASET_DIRNAME[modality]} written -> {out}")
    print(f"  imagesTr: {n('imagesTr')}  labelsTr: {n('labelsTr')}  (expected {len(train_pool)})")
    for m in MODALITIES:
        exp = len(partition["cases"][m]["test"])
        print(f"  imagesTs_{m}: {n(f'imagesTs_{m}')}  labelsTs_{m}: {n(f'labelsTs_{m}')}  (expected {exp})")
