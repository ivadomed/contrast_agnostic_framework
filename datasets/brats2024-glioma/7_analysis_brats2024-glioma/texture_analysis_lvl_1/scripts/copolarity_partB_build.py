#!/usr/bin/env python
"""
PRE-REGISTERED Part B: co-polarity intervention, testing whether restoring "core and edema have
the same polarity" (as Part A's training-data analysis finds is the STATISTICAL norm for T2w,
even though the augmentation code has no explicit rule forcing it) rescues a bright edema step on
T1n. See intervention_step_build.py (bright edema step on T1n hurts real-fill; dark step helps
it) and copolarity_partA_build.py (K-means/label-remap co-polarity analysis) for the two prior
results this combines.

MASKS: core = GT NCR (label==1) UNION GT ET (label==3) -- the solid/enhancing tumor, distinct
from the edema mask E (SNFH, label==2) used throughout the visibility-step round. w_core =
feathered core (gaussian sigma=1.5 of core, zeroed outside brain) as in the step round.
ring(mask) = voxels at Euclidean distance 1-5 from `mask`, in brain, excluding ALL tumor labels
(reused unchanged from intervention_step_build.ring_mask).

INTERVENTIONS (fold 0, same 30 patients, t2w-trained noise-/real-fill models, T1n input):
  C-FLIP(img): inside core (feathered), reflect intensities about the core's own ring mean:
    img_new = img*(1-w_core) + (2*mean(img[ring(core)]) - img)*w_core
    On T1n, the tumor core is typically DARK relative to surrounding brain; this reflection makes
    it BRIGHT relative to its own ring -- restoring "core polarity = bright", matching what Part A
    finds is the training-data norm relative to edema on T2w.
  S-ADD(edema, k=+1): unchanged from the visibility-step round (reused directly, no rebuild --
    same file used there, `v1_sadd_t1n_kp100` and its sham `sham_v1_sadd_t1n_kp100`) -- the
    BRIGHT edema step that broke real-fill (DiD dice -0.137, p=5.7e-7 vs noise-fill).
  COMBINED = C-FLIP(img) then S-ADD(., k=+1) on top (ring statistics for S-ADD's edema mask are
    unaffected by C-FLIP, since that ring excludes all tumor labels, same as the core's own ring).
  SHAM: core and edema are BOTH mirrored using the EXACT SAME flip+shift already found valid for
    edema's sham mask in the very first intervention round (outputs/data/intervention_manifest.csv,
    `sham_status`, parsed for its shift amount) -- this keeps the sham core and sham edema in the
    same relative (mirrored) geometry as the real tumor, rather than deriving a second,
    independently-shifted sham core that could drift apart from the sham edema. Only patients
    where BOTH the sham core and this shift are usable are kept (checked: mirrored core doesn't
    overlap real tumor, and has enough brain coverage).

PRE-REGISTERED PREDICTION: DiD(real-noise) for [C-FLIP + S-ADD(+1)] MINUS DiD(real-noise) for
[S-ADD(+1) alone] > 0 -- i.e. restoring core co-polarity should make the bright edema step LESS
damaging (or actively helpful) to real-fill specifically, relative to how damaging the bare bright
step was on its own. C-FLIP ALONE (no edema step) is reported descriptively, no direction
pre-registered (the training-data analysis is about edema's relationship to core, not a claim
that flipping the core by itself should move edema Dice). SHAM: no true-edema Dice/recall change
expected in either condition; report new false-positive SNFH voxels (noise vs real) as before.

Run as a CPU job (--gpus 0); reuses intervention_build.py's/intervention_step_build.py's helpers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from intervention_build import RAW051, OUT_DATA, PATIENT_CSV, SNFH, N_PATIENTS, load_case, load_label, brain_mask  # noqa: E402
from intervention_step_build import feather_mask, ring_mask, s_add  # noqa: E402

INPUTS_ROOT = Path("/scratch/paulh/brats_intervention/inputs")
NCR, ET = 1, 3


def parse_shift(status: str):
    if status == "mirror_ok":
        return 0
    if status == "failed" or not isinstance(status, str):
        return None
    m = re.match(r"mirror_shift([+-]\d+)", status)
    return int(m.group(1)) if m else None


def apply_mirror_shift(mask: np.ndarray, shift: int) -> np.ndarray:
    mirrored = mask[::-1, :, :]
    return np.roll(mirrored, shift, axis=0) if shift != 0 else mirrored


def c_flip(img: np.ndarray, core: np.ndarray, whole_tumor: np.ndarray, brain: np.ndarray) -> np.ndarray:
    if core.sum() == 0:
        return img.copy()
    ring = ring_mask(core, whole_tumor, brain)
    mean_ring = float(img[ring].mean()) if ring.sum() > 0 else float(img[core].mean())
    w = feather_mask(core, brain)
    reflected = 2 * mean_ring - img
    return img * (1 - w) + reflected * w


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    all_cases = sorted(df["case"].unique())
    cases = all_cases[:N_PATIENTS]

    orig_manifest = pd.read_csv(OUT_DATA / "intervention_manifest.csv")
    shift_by_case = {r["case"]: parse_shift(r["sham_status"]) for _, r in orig_manifest.iterrows()}

    sets = ["corefeather_t1n", "corefeather_sadd_t1n", "sham_corefeather_t1n", "sham_corefeather_sadd_t1n"]
    for s in sets:
        (INPUTS_ROOT / s).mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        label = load_label(case)
        E = label == SNFH
        core = (label == NCR) | (label == ET)
        whole_tumor = label > 0
        n_core = int(core.sum())
        n_snfh = int(E.sum())
        if n_core < 50 or n_snfh < 50:
            rows.append(dict(case=case, ok=False))
            continue

        t1n, t1n_img = load_case(case, "t1n", RAW051)
        brain = brain_mask(t1n)
        w_E = feather_mask(E, brain)
        ring_E = ring_mask(E, whole_tumor, brain)

        flipped = c_flip(t1n, core, whole_tumor, brain)
        combined = s_add(flipped, w_E, ring_E, 1.0)

        def write(arr, ref_img, set_name):
            out_f = INPUTS_ROOT / set_name / f"{case}_0000.nii.gz"
            if out_f.exists():
                return
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header), str(out_f))

        write(flipped, t1n_img, "corefeather_t1n")
        write(combined, t1n_img, "corefeather_sadd_t1n")

        shift = shift_by_case.get(case)
        sham_ok = False
        if shift is not None:
            sham_core = apply_mirror_shift(core, shift) & brain
            sham_E = apply_mirror_shift(E, shift) & brain
            if (int(sham_core.sum()) >= 50 and int(sham_E.sum()) >= 50
                    and not (sham_core & whole_tumor).any() and not (sham_E & whole_tumor).any()):
                sham_ring_E = ring_mask(sham_E, whole_tumor, brain)
                sham_w_E = feather_mask(sham_E, brain)
                sham_flipped = c_flip(t1n, sham_core, whole_tumor, brain)
                sham_combined = s_add(sham_flipped, sham_w_E, sham_ring_E, 1.0)
                write(sham_flipped, t1n_img, "sham_corefeather_t1n")
                write(sham_combined, t1n_img, "sham_corefeather_sadd_t1n")
                sham_ok = True

        rows.append(dict(case=case, ok=True, n_core=n_core, n_snfh=n_snfh, shift=shift, sham_ok=sham_ok))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DATA / "copolarity_partB_manifest.csv", index=False)
    print(f"n_ok={int(manifest['ok'].sum())}/{len(manifest)}, n_sham_ok={int(manifest.get('sham_ok', pd.Series(dtype=bool)).sum())}")
    for s in sets:
        n = len(list((INPUTS_ROOT / s).glob("*.nii.gz")))
        print(f"  {s}: {n} files")


if __name__ == "__main__":
    main()
