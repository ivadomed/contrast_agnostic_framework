#!/usr/bin/env python
"""
PRE-REGISTERED texture-transplant test (see intervention_leakage_build.py's own docstring for
the F-ERODE/F-SMOOTH leakage-control results this follows on from). Written and committed to
disk BEFORE any inference was run on these new input sets.

BACKGROUND: the leakage control found the models are sensitive to the fine spatial TEXTURE of
the edema interior specifically (F-ERODE, mean+iid-noise replacement of the depth>3 core, helps
noise-fill / hurts real-fill; F-SMOOTH, removing only the low-frequency trend, does ~nothing).
Open question this test targets: does the t2w-trained real-fill model under-call edema on t1n
because t1n's real edema texture is not t2w-like (i.e. does real-fill generalize when the
INTERIOR TEXTURE is swapped for texture the model was actually trained on, even though the
image is nominally the "wrong" contrast)?

INTERVENTION — T-SWAP(target, donor), restricted to the depth>3 eroded core (same core as
F-ERODE; the 0-3 voxel border shell and everything outside E untouched):
  x_new = sign * (x_donor - mean(donor_core)) / std(donor_core) * std(target_core) + mean(target_core)
  donor_core / target_core are the SAME (co-registered) core voxels read from the donor / target
  contrast. `sign` in {+1, -1} is chosen per patient to MAXIMIZE the voxelwise Pearson
  correlation of x_new with the ORIGINAL target core (equivalently: sign = +1 if
  corr(donor_z, target_orig) >= 0 else -1) — this preserves as much of the donor's texture
  polarity as agrees with the target's own local structure, rather than assuming which way a
  contrast inversion should go. The chosen sign is logged per patient.

EXPERIMENTS (t2w-trained noise-/real-fill models unless noted; fold 0, first 30 of 70 eval
patients, edema=SNFH; DiD reported as delta_real - delta_noise, per the coordinator's framing
this round):
  X1 (primary): T-SWAP(t1n, donor=t2w) -> predict on the resulting t1n.
    PREDICTION: real-fill edema recall/Dice on t1n INCREASES more than noise-fill's
    (DiD = delta_real - delta_noise > 0) — i.e. giving t1n's edema interior t2w-like texture
    should specifically help the model trained to expect t2w's own real texture.
  X2 (in-domain check): T-SWAP(t2w, donor=t1n) -> predict on the resulting t2w.
    PREDICTION: real-fill decreases more than noise-fill (DiD < 0) — replacing t2w's edema
    texture with t1n's should specifically hurt the model trained on t2w's real texture.
  X3 (control, "any other T1 texture?"): T-SWAP(t1n, donor=t1c) -> predict on the resulting t1n.
    NO DIRECTIONAL PREDICTION pre-registered — reported descriptively. If real-fill is
    unaffected (unlike X1), that argues the X1 effect is t2w-texture-specific, not "any texture
    replacement helps."
  X4 (control, texture-destroyed baseline): T-SWAP degenerates to a same-contrast, same-patient
    SPATIAL PERMUTATION of the core voxels (t1n core values shuffled among themselves) — destroys
    spatial autocorrelation while keeping the exact intensity histogram (unlike F-ERODE's
    Gaussian-noise replacement, which changes the histogram shape too). NO DIRECTIONAL
    PREDICTION pre-registered — reported descriptively, as a second point of reference besides
    F-ERODE for "what happens when spatial texture is destroyed but distribution shape is not."
  SYMMETRY CHECK (t1n-trained noise-/real-fill models): T-SWAP(t2f, donor=t1n) -> predict on the
    resulting t2f (the t1n-trained analogue of X1: does giving t2f's edema interior the
    training contrast's own real texture specifically help t1n-trained real-fill?). NO
    DIRECTIONAL PREDICTION pre-registered ("report whatever happens") — included because if the
    mechanism is symmetric, real-fill should improve relative to noise-fill here too.
  SHAM (X1, X2 only): T-SWAP applied to the same mirrored sham cores used in the leakage control
    (17/30 patients with a valid sham core) — specificity check, no directional prediction
    (expect ~0 DiD).

Run as a CPU job (--gpus 0), after intervention_leakage_build.py (reuses its erode_core /
MIN_CORE_VOX / ERODE_DEPTH and intervention_build.py's label/brain-mask/sham-mask helpers so
cores and sham masks here are identical to the earlier runs').
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from intervention_build import (  # noqa: E402
    RAW051, OUT_DATA, PATIENT_CSV, SNFH, N_PATIENTS, RNG_SEED,
    load_case, load_label, brain_mask, make_sham_mask,
)
from intervention_leakage_build import erode_core, MIN_CORE_VOX  # noqa: E402

INPUTS_ROOT = Path(os.environ["SCRATCH"]) / "brats_intervention/inputs"


def t_swap(target: np.ndarray, donor: np.ndarray, core: np.ndarray) -> tuple[np.ndarray, int]:
    out = target.copy()
    if core.sum() < 2:
        return out, 0
    t = target[core]
    d = donor[core]
    t_mean, t_std = float(t.mean()), float(t.std()) + 1e-8
    d_mean, d_std = float(d.mean()), float(d.std()) + 1e-8
    z = (d - d_mean) / d_std
    corr = float(np.corrcoef(z, t)[0, 1]) if len(z) > 1 else 0.0
    sign = 1 if (np.isnan(corr) or corr >= 0) else -1
    out[core] = sign * z * t_std + t_mean
    return out, sign


def permute_core(target: np.ndarray, core: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = target.copy()
    idx = np.flatnonzero(core.reshape(-1))
    flat = out.reshape(-1)
    vals = flat[idx].copy()
    rng.shuffle(vals)
    flat[idx] = vals
    return out


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    all_cases = sorted(df["case"].unique())
    assert len(all_cases) == 70
    cases = all_cases[:N_PATIENTS]

    sets = ["x1_tswap_t1n_donor_t2w", "x2_tswap_t2w_donor_t1n", "x3_tswap_t1n_donor_t1c",
            "x4_permute_t1n", "symmetry_tswap_t2f_donor_t1n",
            "sham_tswap_t1n_donor_t2w", "sham_tswap_t2w_donor_t1n"]
    for s in sets:
        (INPUTS_ROOT / s).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for case in cases:
        label = load_label(case)
        E = label == SNFH
        n_snfh = int(E.sum())
        whole_tumor = label > 0
        if n_snfh < 50:
            rows.append(dict(case=case, core_ok=False))
            continue

        t1n, t1n_img = load_case(case, "t1n", RAW051)
        t2w, t2w_img = load_case(case, "t2w", RAW051)
        t2f, t2f_img = load_case(case, "t2f", RAW051)
        t1c, _ = load_case(case, "t1c", RAW051)
        brain = brain_mask(t1n)

        core = erode_core(E)
        core_ok = int(core.sum()) >= MIN_CORE_VOX

        sham_mask, sham_status = make_sham_mask(E, whole_tumor, brain)
        sham_core = None
        if sham_mask is not None:
            sc = erode_core(sham_mask)
            if int(sc.sum()) >= MIN_CORE_VOX:
                sham_core = sc

        def write(arr: np.ndarray, ref_img: nib.Nifti1Image, set_name: str):
            out_f = INPUTS_ROOT / set_name / f"{case}_0000.nii.gz"
            if out_f.exists():
                return
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header), str(out_f))

        row = dict(case=case, core_ok=core_ok, sham_status=sham_status)
        if core_ok:
            x1, s1 = t_swap(t1n, t2w, core)
            x2, s2 = t_swap(t2w, t1n, core)
            x3, s3 = t_swap(t1n, t1c, core)
            x4 = permute_core(t1n, core, rng)
            xsym, ssym = t_swap(t2f, t1n, core)
            write(x1, t1n_img, "x1_tswap_t1n_donor_t2w")
            write(x2, t2w_img, "x2_tswap_t2w_donor_t1n")
            write(x3, t1n_img, "x3_tswap_t1n_donor_t1c")
            write(x4, t1n_img, "x4_permute_t1n")
            write(xsym, t2f_img, "symmetry_tswap_t2f_donor_t1n")
            row.update(x1_sign=s1, x2_sign=s2, x3_sign=s3, symmetry_sign=ssym)

        if sham_core is not None:
            sh1, ssh1 = t_swap(t1n, t2w, sham_core)
            sh2, ssh2 = t_swap(t2w, t1n, sham_core)
            write(sh1, t1n_img, "sham_tswap_t1n_donor_t2w")
            write(sh2, t2w_img, "sham_tswap_t2w_donor_t1n")
            row.update(sham_x1_sign=ssh1, sham_x2_sign=ssh2, n_sham_core=int(sham_core.sum()))
        else:
            row.update(n_sham_core=0)

        rows.append(row)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DATA / "intervention_transplant_manifest.csv", index=False)
    print(f"n_core_ok={int(manifest['core_ok'].sum())}/{len(manifest)}, "
          f"n_sham_core_ok={(manifest.get('n_sham_core', 0) >= MIN_CORE_VOX).sum()}")
    for s in sets:
        n = len(list((INPUTS_ROOT / s).glob("*.nii.gz")))
        print(f"  {s}: {n} files")
    if "x1_sign" in manifest:
        print("x1 sign counts:", manifest["x1_sign"].value_counts(dropna=True).to_dict())
        print("x2 sign counts:", manifest["x2_sign"].value_counts(dropna=True).to_dict())


if __name__ == "__main__":
    main()
