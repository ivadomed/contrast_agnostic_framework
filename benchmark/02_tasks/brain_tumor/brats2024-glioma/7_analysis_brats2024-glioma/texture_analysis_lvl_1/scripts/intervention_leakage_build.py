#!/usr/bin/env python
"""
PRE-REGISTERED label-leakage control for the FLATTEN intervention (see intervention_build.py's
own docstring for the original E1/E2/E3 pre-registration and results). Written and committed
to disk BEFORE any inference was run on these new input sets.

MOTIVATION: full FLATTEN(img) replaces ALL of the GT edema mask E with a homogeneous
mean+noise fill, so the flattened region's border is EXACTLY the GT border — a model could in
principle detect "there is a homogeneous patch here" and use its EXACT boundary as a shortcut
to the GT segmentation ("boundary leakage") rather than genuinely finding no internal structure.
The controls below preserve the real GT border and its real transition, touching only voxels
well inside the region, so no new hard edge coincides with (or even exists near) the GT border.

INTERVENTIONS (all restricted to the eroded core = voxels with
distance_transform_edt(E) > 3, i.e. depth > 3 inside E; the 0-3 voxel shell next to the true
border, and everything outside E, is left completely untouched):
  F-ERODE:  identical recipe to FLATTEN, but computed and applied ONLY on the eroded core —
            mean and residual std (same highpass, sigma=2) are taken from the CORE's own
            voxels, not all of E. No new boundary is introduced at the true GT border; a new
            (much smaller, and non-tumor-shaped) boundary exists only at the core/shell
            interface, 3+ voxels inside real tumor tissue on both sides.
  F-SMOOTH: does not replace intensities — removes only the smooth low-frequency trend within
            the region (img - (G_sigma3(img*E)/G_sigma3(E) - mean(E))), computed using the
            WHOLE region E for context (so the trend estimate isn't itself edge-biased), but
            the correction is only APPLIED on the eroded core. Preserves fine texture/noise
            and the true border's real transition entirely; only removes low-freq ramp
            structure from the region's interior.
  SHAM-ERODE: F-ERODE applied to the same mirrored sham masks used in the original run (same
            deterministic construction, reproduced here via make_sham_mask/erode).

PRE-REGISTERED PREDICTIONS (fold 0, first 30 of 70 eval patients, edema=SNFH; E1 = t1n-trained
noise/real on t2f; E2-cross = t2w-trained on t2f; E2-indomain = t2w-trained on t2w):
  For BOTH F-ERODE and F-SMOOTH, in E1 and E2 (cross + in-domain):
    noise-fill edema Dice/recall still increases more than real-fill's (difference-in-differences
    noise-delta minus real-delta > 0), same DIRECTION as full FLATTEN, but SMALLER in magnitude
    than full FLATTEN's (since the true border is untouched, giving both models a correct
    boundary to key off regardless of interior fill).
  If the noise-vs-real difference-in-differences collapses to ~0 / loses significance under
  F-ERODE specifically, the original full-FLATTEN E1/E2 effect was (at least partly) boundary
  leakage rather than genuine interior-structure sensitivity — to be reported plainly, not
  explained away.
  SHAM-ERODE (both metrics, all experiments): ~0 change vs sham's own original prediction —
  same specificity check as the original sham, at the smaller eroded scale.

Run as a CPU job (--gpus 0), immediately after intervention_build.py (reuses its label loading,
brain mask, flatten(), highpass-based residual, and sham-mask construction so the sham masks
here are byte-identical to the ones used in the original run).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_region_surround_texture import highpass  # noqa: E402
from intervention_build import (  # noqa: E402
    RAW051, LABELS_DIR, OUT_DATA, PATIENT_CSV, SNFH, N_PATIENTS, HP_SIGMA, RNG_SEED,
    load_case, load_label, brain_mask, flatten, make_sham_mask,
)

INPUTS_ROOT = Path("/scratch/paulh/brats_intervention/inputs")

ERODE_DEPTH = 3
MIN_CORE_VOX = 50
SMOOTH_SIGMA = 3.0


def erode_core(mask: np.ndarray, depth: int = ERODE_DEPTH) -> np.ndarray:
    edt = distance_transform_edt(mask)
    return mask & (edt > depth)


def flatten_core(img: np.ndarray, core: np.ndarray, brain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """F-ERODE: same recipe as flatten(), but mean/std taken from `core`'s own voxels and
    applied only there."""
    out = img.copy()
    if core.sum() == 0:
        return out
    hp = highpass(img, brain)
    resid_std = float(hp[core].std())
    mean_val = float(img[core].mean())
    out[core] = mean_val + rng.normal(0.0, resid_std, size=int(core.sum()))
    return out


def smooth_trend_remove(img: np.ndarray, mask: np.ndarray, core: np.ndarray, sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """F-SMOOTH: mask-normalized low-pass trend of img within `mask` (context = full region),
    correction applied only on `core`."""
    out = img.copy()
    if core.sum() == 0:
        return out
    m = mask.astype(np.float64)
    smooth = gaussian_filter(img * m, sigma) / np.maximum(gaussian_filter(m, sigma), 1e-6)
    mean_val = float(img[mask].mean())
    trend_correction = smooth - mean_val
    out[core] = img[core] - trend_correction[core]
    return out


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    all_cases = sorted(df["case"].unique())
    assert len(all_cases) == 70
    cases = all_cases[:N_PATIENTS]

    sets = ["f_erode_t2f", "f_erode_t2w", "f_smooth_t2f", "f_smooth_t2w",
            "sham_erode_t2f", "sham_erode_t2w"]
    for s in sets:
        (INPUTS_ROOT / s).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)  # same seed/order as intervention_build.py
    rows = []
    for case in cases:
        label = load_label(case)
        E = label == SNFH
        n_snfh = int(E.sum())
        whole_tumor = label > 0
        if n_snfh < 50:
            rows.append(dict(case=case, n_snfh=n_snfh, n_core=0, core_ok=False, sham_status="skipped"))
            continue

        t1n, _ = load_case(case, "t1n", RAW051)
        t2w, t2w_img = load_case(case, "t2w", RAW051)
        t2f, t2f_img = load_case(case, "t2f", RAW051)
        brain = brain_mask(t1n)

        core = erode_core(E)
        n_core = int(core.sum())
        core_ok = n_core >= MIN_CORE_VOX

        # IMPORTANT: draw the sham mask exactly as intervention_build.py did (same rng calls
        # happen inside flatten(), not make_sham_mask, and make_sham_mask itself is
        # deterministic) so sham masks here are identical to the original run's.
        sham_mask, sham_status = make_sham_mask(E, whole_tumor, brain)
        sham_core = None
        if sham_mask is not None:
            sham_core = erode_core(sham_mask)
            if int(sham_core.sum()) < MIN_CORE_VOX:
                sham_core = None

        def write(arr: np.ndarray, ref_img: nib.Nifti1Image, set_name: str):
            out_f = INPUTS_ROOT / set_name / f"{case}_0000.nii.gz"
            if out_f.exists():
                return
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header), str(out_f))

        if core_ok:
            write(flatten_core(t2f, core, brain, rng), t2f_img, "f_erode_t2f")
            write(flatten_core(t2w, core, brain, rng), t2w_img, "f_erode_t2w")
            write(smooth_trend_remove(t2f, E, core), t2f_img, "f_smooth_t2f")
            write(smooth_trend_remove(t2w, E, core), t2w_img, "f_smooth_t2w")

        if sham_core is not None:
            write(flatten_core(t2f, sham_core, brain, rng), t2f_img, "sham_erode_t2f")
            write(flatten_core(t2w, sham_core, brain, rng), t2w_img, "sham_erode_t2w")

        rows.append(dict(case=case, n_snfh=n_snfh, n_core=n_core, core_ok=core_ok,
                          sham_status=sham_status,
                          n_sham_core=int(sham_core.sum()) if sham_core is not None else 0))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DATA / "intervention_leakage_manifest.csv", index=False)
    print(f"n_core_ok={int(manifest['core_ok'].sum())}/{len(manifest)}, "
          f"n_sham_core_ok={(manifest.get('n_sham_core', 0) >= MIN_CORE_VOX).sum() if 'n_sham_core' in manifest else 0}")
    for s in sets:
        n = len(list((INPUTS_ROOT / s).glob("*.nii.gz")))
        print(f"  {s}: {n} files")


if __name__ == "__main__":
    main()
