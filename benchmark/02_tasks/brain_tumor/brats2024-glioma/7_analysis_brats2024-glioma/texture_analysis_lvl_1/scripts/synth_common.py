#!/usr/bin/env python
"""
Shared library for the cross-contrast FEATURE-LEVEL synthesis experiment (containment
hypothesis, Paul 2026-09-24): for a train->eval ladder cell, R2(train | eval) asks whether a CNN
can recover the TRAINING contrast's local content from the EVAL contrast's real image alone,
inside a given anatomical region. High R2(train|eval) predicts the training contrast's learned
texture transfers well when tested on that eval contrast (real-fill ablation rung4->5 "HELPS");
low R2(train|eval) predicts "HURTS"/n.s. Voxelwise measures (NGF, correlation-ratio eta^2) in
this same directory already tried and only partially explained the ladder; this is the
feature-level (learned-synthesis) follow-up.

Reuses compute_cross_contrast_ngf.py in this same scripts/ dir for BIDS loading, label loading
and the 7 region masks (NCR/SNFH/ET/RC/tumor_core/whole_tumor/healthy) -- do NOT re-derive those
independently, they must match the ladder's own region definitions exactly.

Cache ($SCRATCH/brats_synth_cache/<patient>.npz), built ONCE by build_cache.py, never rebuilt by
train_synth.py/infer_synth.py (parallel jobs would race on it): per patient, cropped-to-brain
bounding box (+8vox pad, >=96 per axis), z-scored per contrast over that contrast's OWN
brain-mask voxels, background zeroed, float16 4-contrast stack + region masks (native to the
crop) + crop bbox + original shape (for uncropping QC NIfTIs back to the native 240x240x155 grid
with the original affine).
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import compute_cross_contrast_ngf as ngf  # noqa: E402  (reuse BIDS loading / region masks)

PROJECT_ROOT = ngf.PROJECT_ROOT
DS_ROOT = ngf.DS_ROOT
BIDS_ROOT = ngf.BIDS_ROOT
LABELS_DIR = ngf.LABELS_DIR
LVL1_DIR = THIS_DIR.parent
OUT_DIR = LVL1_DIR / "outputs"

CONTRASTS = ("t1n", "t1c", "t2w", "t2f")
REGIONS = ("NCR", "SNFH", "ET", "RC", "tumor_core", "whole_tumor", "healthy")
MIN_REGION_VOX = 200

SCRATCH = Path(os.environ.get("SCRATCH") or "/scratch/paulh")
CACHE_DIR = SCRATCH / "brats_synth_cache"
CKPT_DIR = SCRATCH / "brats_synth_checkpoints"
QC_DIR = SCRATCH / "brats_synth_qc"

MIN_PATCH = 96
BBOX_PAD = 8

SEED = 0
N_TRAIN = 100
N_VAL = 20


def targets_for(source: str) -> tuple[str, ...]:
    """Fixed output-channel order for a given source model: the other 3 contrasts, in the
    CONTRASTS order. Saved into every checkpoint so inference can never mis-align channels."""
    return tuple(c for c in CONTRASTS if c != source)


def eval_patient_ids() -> list[str]:
    """The 70 held-out patients used by the fill-swap ablation -- read from its own per-patient
    delta CSV so this can never drift from the ladder's actual eval set."""
    csv_path = OUT_DIR / "data" / "patient_region_deltas.csv"
    ids = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ids.add(row["case"])
    return sorted(ids)


def patient_prefix(pid: str) -> str:
    """'BraTSGLI00005101' -> 'BraTSGLI00005' (drops the 3-digit timepoint suffix). Two BraTS
    scans of the same patient at different timepoints must never be split across train/eval."""
    assert pid.startswith("BraTSGLI") and len(pid) == 16, pid
    return pid[:13]


def all_bids_patient_ids() -> list[str]:
    ids = []
    for d in sorted(BIDS_ROOT.glob("sub-BraTSGLI*")):
        pid = d.name[len("sub-"):]
        anat = d / "anat"
        if not anat.is_dir():
            continue
        if not all((anat / f"{d.name}_{suf}").exists() for suf in ngf.CONTRAST_SUFFIX.values()):
            continue
        if not (LABELS_DIR / f"{pid}.nii.gz").exists():
            continue
        ids.append(pid)
    return ids


def train_val_split():
    """100 train / 20 val patients (seed 0), excluding every patient whose 5-digit BraTS ID
    (any timepoint) appears among the 70 fill-swap eval patients -- prevents leakage through a
    different timepoint of an eval patient's own brain into the training set."""
    eval_ids = eval_patient_ids()
    eval_prefixes = {patient_prefix(p) for p in eval_ids}
    candidates = [p for p in all_bids_patient_ids() if patient_prefix(p) not in eval_prefixes]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(candidates))
    chosen = [candidates[i] for i in idx[: N_TRAIN + N_VAL]]
    train, val = chosen[:N_TRAIN], chosen[N_TRAIN:]
    assert len(train) == N_TRAIN and len(val) == N_VAL, (len(candidates), len(train), len(val))
    return train, val


def cache_path(pid: str) -> Path:
    return CACHE_DIR / f"{pid}.npz"


def build_one_cache(pid: str):
    out = cache_path(pid)
    if out.exists():
        return
    vols, label = ngf.load_patient(pid, "cpu")
    masks = ngf.region_masks(label, vols["t1n"])
    brain = torch.zeros_like(label, dtype=torch.bool)
    for c in CONTRASTS:
        brain |= vols[c] > 0

    nz = torch.nonzero(brain)
    lo = nz.min(dim=0).values.numpy()
    hi = (nz.max(dim=0).values + 1).numpy()
    shape = np.array(brain.shape)
    lo = np.maximum(lo - BBOX_PAD, 0)
    hi = np.minimum(hi + BBOX_PAD, shape)
    for d in range(3):
        if hi[d] - lo[d] < MIN_PATCH:
            extra = MIN_PATCH - (hi[d] - lo[d])
            lo[d] = max(0, lo[d] - extra // 2 - 1)
            hi[d] = min(shape[d], lo[d] + MIN_PATCH)
            lo[d] = max(0, hi[d] - MIN_PATCH)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    brain_c = brain[sl].numpy()
    data = {
        "bbox_lo": lo.astype(np.int32), "bbox_hi": hi.astype(np.int32),
        "orig_shape": shape.astype(np.int32), "brain": brain_c,
    }
    for c in CONTRASTS:
        v = vols[c][sl].numpy().astype(np.float32)
        m = brain_c
        mu, sd = float(v[m].mean()), float(v[m].std() + 1e-6)
        v = (v - mu) / sd
        v[~m] = 0.0
        data[c] = v.astype(np.float16)
        data[f"{c}_mu"] = np.float32(mu)
        data[f"{c}_sd"] = np.float32(sd)
    label_c = label[sl].numpy().astype(np.uint8)
    data["label"] = label_c
    for r in REGIONS:
        data[f"mask_{r}"] = masks[r][sl].numpy()

    # np.savez_compressed silently appends ".npz" to a string path that doesn't already end in
    # ".npz" -- pass an open binary file object instead so the actual written path is `tmp`.
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **data)
    os.replace(tmp, out)


def load_cache(pid: str) -> dict:
    z = np.load(cache_path(pid))
    return {k: z[k] for k in z.files}
