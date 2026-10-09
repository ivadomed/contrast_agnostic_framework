#!/usr/bin/env python
"""
PRE-REGISTERED visibility-step intervention (see intervention_transplant_build.py's docstring
for the transplant-test results this follows on from: swapping the edema core's texture for a
different real contrast's texture did ~nothing (X1/X2 near-zero, not significant), while
destroying spatial order (X4, permutation) hurt real-fill sharply. This narrows the question
from "which contrast's texture" to "is there a macroscopic step at all"). Written and committed
to disk BEFORE any inference was run on these new input sets.

HYPOTHESIS: real-fill is more EVIDENCE(step)-DRIVEN — it keys more on a genuine mean-intensity
step between the region and its surroundings — while noise-fill is more CONTEXT-DRIVEN (relies
on where edema-shaped homogeneous patches tend to sit, less on the step itself). On t1n, edema
has ~zero mean step against its surroundings (the correlational finding motivating this test) —
predicting that this specifically starves real-fill of the evidence it depends on.

MASKS (per patient):
  E = GT SNFH (label==2). brain = t1n > 0 (BraTS is skull-stripped).
  w = feathered mask = gaussian_filter(E.astype(float), sigma=1.5), then zeroed outside brain —
      a smooth field ~1 well inside E, tapering to 0 over a few voxels past the true border (no
      hard new edge is introduced; texture is untouched since only a smooth constant-ish FIELD is
      added, not a new pattern).
  ring = voxels at Euclidean distance 1-5 from E, inside brain, EXCLUDING any tumor label
      (label==0) — same "non-tumor immediate surroundings" as compute_region_surround_texture.py's
      ring, narrowed to d_out<=5 (only need it for a local mean/std reference, not the wider
      d_out in {3,5,8} sweep used there).

INTERVENTIONS:
  S-ADD(img, k):    img + k * std(img[ring]) * w  — adds a soft step of magnitude
                     k standard-deviations-of-the-local-surroundings, same sign convention
                     regardless of the image's own polarity (k>0 raises the region, k<0 lowers
                     it) — k in {+0.5, +1.0, -0.5, -1.0} tests both directions since e.g. T2w
                     edema is hyperintense but T1c edema-adjacent effects can go either way.
  S-REMOVE(img):    img - (mean(img[E]) - mean(img[ring])) * w — sets the region's mean toward
                     the ring's own mean (removes the real step) while leaving the region's own
                     internal texture/variance completely intact (a smooth, region-shaped mean
                     shift, not a texture edit).
  SHAM: the identical S-ADD/S-REMOVE recipe computed and applied on the mirrored sham mask
        (same 17/30 patients with a valid sham mask as every earlier round), using the sham
        mask's own w and its own ring (recomputed around the sham location, not E's).

EXPERIMENTS (fold 0, first 30 of 70 eval patients, edema=SNFH; DiD = delta_real - delta_noise,
matching the transplant round's convention):
  V1: t2w-trained noise-/real-fill models on S-ADD(t1n, k) for all 4 k.
      PREDICTION: real-fill edema recall/Dice increases more than noise-fill's (DiD > 0),
      growing in magnitude with |k| (both signs) — a genuine dose-response.
  V2: t2w-trained models on S-REMOVE(t2w) [in-domain] and S-REMOVE(t2f) [cross-contrast].
      PREDICTION: real-fill decreases more than noise-fill (DiD < 0) — removing the real
      contrast's own step should specifically hurt the model trained on that contrast's real
      structure.
  V3: t1n-trained noise-/real-fill models on S-REMOVE(t2f). NO DIRECTIONAL PREDICTION
      pre-registered (symmetry check, reported as-is).
  SHAM (V1 all k + V2 both targets): applied to the mirrored sham region. NO real-edema Dice/
      recall change expected; additionally reports NEW predicted-SNFH voxels created inside the
      sham region (hallucination count) for noise vs real — a purely step-driven model should
      hallucinate edema more readily when an artificial step of the right sign/magnitude is
      planted on otherwise-healthy tissue.

Run as a CPU job (--gpus 0); reuses intervention_build.py's label/brain-mask/sham-mask helpers.
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
from intervention_build import (  # noqa: E402
    RAW051, OUT_DATA, PATIENT_CSV, SNFH, N_PATIENTS,
    load_case, load_label, brain_mask, make_sham_mask,
)

INPUTS_ROOT = Path("/scratch/paulh/brats_intervention/inputs")

FEATHER_SIGMA = 1.5
RING_LO, RING_HI = 1, 5
K_VALUES = [0.5, 1.0, -0.5, -1.0]


def feather_mask(mask: np.ndarray, brain: np.ndarray) -> np.ndarray:
    w = gaussian_filter(mask.astype(np.float64), FEATHER_SIGMA)
    return w * brain


def ring_mask(mask: np.ndarray, whole_tumor: np.ndarray, brain: np.ndarray) -> np.ndarray:
    dist_out = distance_transform_edt(~mask)
    return (dist_out >= RING_LO) & (dist_out <= RING_HI) & brain & ~whole_tumor


def s_add(img: np.ndarray, w: np.ndarray, ring: np.ndarray, k: float) -> np.ndarray:
    sigma_ring = float(img[ring].std()) if ring.sum() > 0 else float(img.std())
    return img + k * sigma_ring * w


def s_remove(img: np.ndarray, mask: np.ndarray, w: np.ndarray, ring: np.ndarray) -> np.ndarray:
    mean_e = float(img[mask].mean())
    mean_ring = float(img[ring].mean()) if ring.sum() > 0 else mean_e
    return img - (mean_e - mean_ring) * w


def k_tag(k: float) -> str:
    sign = "p" if k >= 0 else "m"
    return f"{sign}{abs(k):.2f}".replace(".", "")


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    all_cases = sorted(df["case"].unique())
    assert len(all_cases) == 70
    cases = all_cases[:N_PATIENTS]

    sets = [f"v1_sadd_t1n_k{k_tag(k)}" for k in K_VALUES]
    sets += ["v2_sremove_t2w", "v2_sremove_t2f"]
    sets += [f"sham_v1_sadd_t1n_k{k_tag(k)}" for k in K_VALUES]
    sets += ["sham_v2_sremove_t2w", "sham_v2_sremove_t2f"]
    for s in sets:
        (INPUTS_ROOT / s).mkdir(parents=True, exist_ok=True)

    rows = []
    for case in cases:
        label = load_label(case)
        E = label == SNFH
        n_snfh = int(E.sum())
        whole_tumor = label > 0
        if n_snfh < 50:
            rows.append(dict(case=case, ok=False, sham_status="skipped"))
            continue

        t1n, t1n_img = load_case(case, "t1n", RAW051)
        t2w, t2w_img = load_case(case, "t2w", RAW051)
        t2f, t2f_img = load_case(case, "t2f", RAW051)
        brain = brain_mask(t1n)

        w_E = feather_mask(E, brain)
        ring_E = ring_mask(E, whole_tumor, brain)

        sham_mask, sham_status = make_sham_mask(E, whole_tumor, brain)
        w_sham = ring_sham = None
        if sham_mask is not None:
            w_sham = feather_mask(sham_mask, brain)
            ring_sham = ring_mask(sham_mask, whole_tumor, brain)

        def write(arr, ref_img, set_name):
            out_f = INPUTS_ROOT / set_name / f"{case}_0000.nii.gz"
            if out_f.exists():
                return
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header), str(out_f))

        for k in K_VALUES:
            write(s_add(t1n, w_E, ring_E, k), t1n_img, f"v1_sadd_t1n_k{k_tag(k)}")
            if sham_mask is not None:
                write(s_add(t1n, w_sham, ring_sham, k), t1n_img, f"sham_v1_sadd_t1n_k{k_tag(k)}")

        write(s_remove(t2w, E, w_E, ring_E), t2w_img, "v2_sremove_t2w")
        write(s_remove(t2f, E, w_E, ring_E), t2f_img, "v2_sremove_t2f")
        if sham_mask is not None:
            write(s_remove(t2w, sham_mask, w_sham, ring_sham), t2w_img, "sham_v2_sremove_t2w")
            write(s_remove(t2f, sham_mask, w_sham, ring_sham), t2f_img, "sham_v2_sremove_t2f")

        rows.append(dict(case=case, ok=True, n_snfh=n_snfh,
                          mean_step_t1n=float(t1n[E].mean() - t1n[ring_E].mean()) if ring_E.sum() else np.nan,
                          mean_step_t2w=float(t2w[E].mean() - t2w[ring_E].mean()) if ring_E.sum() else np.nan,
                          mean_step_t2f=float(t2f[E].mean() - t2f[ring_E].mean()) if ring_E.sum() else np.nan,
                          sham_status=sham_status, n_sham_vox=int(sham_mask.sum()) if sham_mask is not None else 0))

    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DATA / "intervention_step_manifest.csv", index=False)
    print(f"n_ok={int(manifest['ok'].sum())}/{len(manifest)}, "
          f"n_sham_ok={(manifest['sham_status']!='failed').sum() if 'sham_status' in manifest else 0}")
    for s in sets:
        n = len(list((INPUTS_ROOT / s).glob("*.nii.gz")))
        print(f"  {s}: {n} files")
    print("\nmean step (region - ring) by contrast, median over patients:")
    for c in ("t1n", "t2w", "t2f"):
        print(f"  {c}: {manifest[f'mean_step_{c}'].median():.2f}")


if __name__ == "__main__":
    main()
