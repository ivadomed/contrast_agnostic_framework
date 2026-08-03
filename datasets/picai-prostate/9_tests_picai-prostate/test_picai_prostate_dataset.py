#!/usr/bin/env python3
"""
picai-prostate dataset invariants.

These guard the two things that are easy to break silently and expensive to notice late:
the PATIENT-level split (a leak inflates every score) and the COMMON-GRID property (if the
three contrasts stop sharing one geometry, the single lesion mask silently stops being
valid for the cross-contrast test dirs, and Dice degrades for a reason that looks like a
modelling result).

Skips cleanly when the data isn't materialised on this machine (the bulk tree lives on
$SCRATCH on tamia and is not present on vulcan).

Run:  .venv/bin/python -m pytest datasets/picai-prostate/9_tests_picai-prostate/ -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

DATASET_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = Path(os.environ.get("SPLITS_DIR", DATASET_ROOT / "4_splits_picai-prostate"))
BIDS_ROOT = Path(os.environ.get(
    "BIDS_ROOT", DATASET_ROOT / "1_BIDS_picai-prostate" / "picai-prostate-bpmri"))
NNUNET_RAW = Path(os.environ.get(
    "nnUNet_raw", DATASET_ROOT / "2_nnUNet_picai-prostate" / "raw"))

CONTRASTS = ("t2w", "adc", "hbv")
N_FOLDS_FILE = 4          # folds 0/1/2 are trained; the file carries 4 (CLAUDE.md FOLD POLICY)


def _patient(case_id: str) -> str:
    return case_id.split("_", 1)[0]


@pytest.fixture(scope="module")
def partition() -> dict:
    p = SPLITS_DIR / "partition.json"
    if not p.is_file():
        pytest.skip(f"{p} not present — run 01_create_splits first")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def splits() -> list:
    p = SPLITS_DIR / "splits_final.json"
    if not p.is_file():
        pytest.skip(f"{p} not present — run 01_create_splits first")
    return json.loads(p.read_text())


def test_test_set_disjoint_from_train_pool(partition):
    assert not set(partition["test"]) & set(partition["train_pool"])


def test_split_is_patient_level(partition):
    """No PATIENT may appear on both sides — a few PI-CAI patients have two studies, and
    case-level splitting would leak the same prostate between train and test."""
    test_pats = {_patient(c) for c in partition["test"]}
    pool_pats = {_patient(c) for c in partition["train_pool"]}
    assert not test_pats & pool_pats


def test_folds_are_patient_level_and_cover_the_pool(splits, partition):
    assert len(splits) == N_FOLDS_FILE
    pool = set(partition["train_pool"])
    seen_val: set[str] = set()
    for k, s in enumerate(splits):
        assert not set(s["train"]) & set(s["val"]), f"fold {k}: train/val overlap"
        assert not {_patient(c) for c in s["train"]} & {_patient(c) for c in s["val"]}, \
            f"fold {k}: same patient in train and val"
        assert set(s["train"]) | set(s["val"]) == pool, f"fold {k}: does not cover the pool"
        assert not seen_val & set(s["val"]), f"fold {k}: val cases reused by another fold"
        seen_val |= set(s["val"])
    assert seen_val == pool, "the 4 val folds must partition the train pool exactly"


def test_no_held_out_test_case_in_any_fold(splits, partition):
    test = set(partition["test"])
    for k, s in enumerate(splits):
        assert not test & (set(s["train"]) | set(s["val"])), f"fold {k}: test contamination"


@pytest.mark.parametrize("ds_name,channel", [("Dataset080_PICAI_T2W", "T2W"),
                                             ("Dataset081_PICAI_ADC", "ADC")])
def test_nnunet_dataset_is_complete(ds_name, channel, partition):
    ds = NNUNET_RAW / ds_name
    if not ds.is_dir():
        pytest.skip(f"{ds} not present — run 02_nnunet convert first")
    meta = json.loads((ds / "dataset.json").read_text())
    assert meta["channel_names"] == {"0": channel}
    assert meta["labels"] == {"background": 0, "lesion": 1}
    assert meta["numTraining"] == len(partition["train_pool"])

    assert len(list((ds / "imagesTr").glob("*.nii.gz"))) == len(partition["train_pool"])
    assert len(list((ds / "labelsTr").glob("*.nii.gz"))) == len(partition["train_pool"])
    # Every contrast must carry the FULL held-out test set — cross-contrast evaluation is
    # the headline result, so a short imagesTs_<c> would quietly shrink one column.
    for c in CONTRASTS:
        assert len(list((ds / f"imagesTs_{c}").glob("*.nii.gz"))) == len(partition["test"]), c
        assert len(list((ds / f"labelsTs_{c}").glob("*.nii.gz"))) == len(partition["test"]), c


def test_all_contrasts_share_one_grid():
    """The whole cross-contrast protocol rests on this: one mask, valid for every contrast."""
    nib = pytest.importorskip("nibabel")
    numpy = pytest.importorskip("numpy")
    cases = sorted(BIDS_ROOT.glob("sub-*"))
    if not cases:
        pytest.skip(f"{BIDS_ROOT} not populated — run 00_01_bidsify first")

    for sub in cases[:10]:
        anat = next(sub.glob("ses-*/anat"), None)
        assert anat is not None, f"{sub}: no ses-*/anat"
        ref = None
        for suffix in ("T2w", "ADC", "HBV"):
            f = next(anat.glob(f"*_{suffix}.nii.gz"), None)
            assert f is not None, f"{sub}: missing {suffix}"
            img = nib.load(str(f))
            if ref is None:
                ref = img.affine
                assert img.shape == (256, 256, 24), f"{sub}: unexpected shape {img.shape}"
            else:
                assert numpy.allclose(img.affine, ref), f"{sub}: {suffix} affine differs"

        ses = anat.parent.name
        mask_f = (BIDS_ROOT / "derivatives" / "manual_masks" / sub.name / ses / "anat"
                  / f"{sub.name}_{ses}_dseg.nii.gz")
        assert mask_f.is_file(), f"{sub}: missing lesion mask"
        mask = nib.load(str(mask_f))
        assert numpy.allclose(mask.affine, ref), f"{sub}: mask affine differs from images"
        arr = numpy.asanyarray(mask.dataobj)
        assert set(numpy.unique(arr)) <= {0, 1}, f"{sub}: mask not binarised"
        assert arr.sum() > 0, f"{sub}: empty mask — only csPCa-positive cases are kept"
