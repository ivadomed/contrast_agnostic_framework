#!/usr/bin/env python
"""
PRE-REGISTERED intervention builder — causal test of the "each model expects target interiors
like its training distribution" mechanism (noise-fill: flat; real-fill: training contrast's real
structure). Written and committed to disk BEFORE any inference was run (see run log timestamps).

PRE-REGISTERED PREDICTIONS (fold 0 only, first 30 of the 70 eval patients in
outputs/data/patient_region_deltas.csv, sorted by case id; edema = SNFH, label id 2):

  E1: t1n-trained noise-fill and real-fill models, predicted on FLATTEN(t2f).
      Predict: noise-fill edema recall/Dice INCREASES vs original t2f; real-fill changes less
      (difference-in-differences > 0, i.e. delta_noise - delta_real > 0).

  E2: t2w-trained noise-fill and real-fill models, predicted on FLATTEN(t2f) [cross-contrast]
      and FLATTEN(t2w) [in-domain].
      Predict: real-fill edema Dice DECREASES more than noise-fill's (delta_real < delta_noise,
      i.e. difference-in-differences < 0), in both conditions but especially cross-contrast.

  E3: t2w-trained noise-fill and real-fill models, predicted on RAMP(t1n, donor=t2w).
      Predict: real-fill edema recall INCREASES vs original t1n more than noise-fill's
      (difference-in-differences > 0).

  Controls (sham): every FLATTEN/RAMP operation is also applied to a SHAM mask (E mirrored
  across the mid-sagittal plane onto healthy tissue, translated along axis 0 if the mirror
  clips the tumor or leaves the brain) to confirm the effect is specific to the edema interior,
  not an artifact of touching image intensities.

Interventions (per patient, using GT edema mask E from labelsTr; background/other voxels of the
image are left untouched):
  FLATTEN(img, mask): inside `mask`, replace intensities by mean(img[mask]) + iid Gaussian noise
    with the SAME std as the high-pass residual of img within `mask` (img minus a mask-normalized
    Gaussian(sigma=2) smooth, computed over the whole brain — see highpass() reused from
    compute_region_surround_texture.py). Removes the region's low-frequency internal structure
    (the depth ramp) while preserving the mean level and the fine noise floor.
  RAMP(img, donor, mask): inside `mask`, add a depth ramp derived from `donor`'s own per-depth-bin
    mean intensity inside `mask` (depth = voxels from the mask border, via distance_transform_edt),
    z-scored across bins then mapped back to voxels by their depth bin, scaled so the added ramp's
    std (over mask voxels) equals donor's own (ramp_std / donor_std) fraction — i.e. donor's own
    depth-explained-variance fraction — applied to img's own within-mask std. The ramp's mean is
    subtracted so mean(img[mask]) is unchanged.

Intervenes directly on the nnU-Net *test-time* input files (2_nnUNet_.../imagesTs_<contrast>/
<case>_0000.nii.gz) rather than the raw BIDS files, because those are exactly what
05_00_build_test_inputs.py fed to nnUNetv2_predict for the stored headline predictions (content
verified identical, byte-for-byte array equality, across the Dataset051/052 imagesTs_<contrast>
copies — see session notes). GT comes from Dataset051's labelsTr (content-identical to Dataset052's,
verified by np.array_equal).

Outputs (this dataset's own analysis dir, not $SCRATCH):
  outputs/data/intervention_manifest.csv  — per-patient bookkeeping (region sizes, sham status).
Outputs ($SCRATCH, ephemeral per project convention, job I/O never on $HOME):
  $SCRATCH/brats_intervention/inputs/<set_name>/<case>_0000.nii.gz

Run inside a CPU run_job (--gpus 0) — no GPU needed, this is pure numpy/nibabel.
  .venv/bin/python intervention_build.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from compute_region_surround_texture import highpass  # noqa: E402  (HP_SIGMA=2.0 baked in)

PROJECT = Path(__file__).resolve().parents[7]
DS = PROJECT / "benchmark" / "02_tasks" / "brain_tumor" / "brats2024-glioma"
RAW051 = DS / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset051_BraTS2024GliomaT1n"
RAW052 = DS / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset052_BraTS2024GliomaT2w"
LABELS_DIR = RAW051 / "labelsTr"  # content-identical to RAW052's labelsTr (verified)

OUT_DATA = DS / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1" / "outputs" / "data"
PATIENT_CSV = OUT_DATA / "patient_region_deltas.csv"

SCRATCH_ROOT = Path(os.environ["SCRATCH"]) / "brats_intervention"
INPUTS_ROOT = SCRATCH_ROOT / "inputs"

SNFH = 2
N_PATIENTS = 30
MIN_REGION_VOX = 50
HP_SIGMA = 2.0
MAX_DEPTH_BIN = 8
MIN_BIN_VOX = 10
RNG_SEED = 0


def load_case(case: str, contrast: str, family_raw: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    f = family_raw / f"imagesTs_{contrast}" / f"{case}_0000.nii.gz"
    img = nib.load(str(f))
    return np.asarray(img.dataobj, dtype=np.float32), img


def load_label(case: str) -> np.ndarray:
    f = LABELS_DIR / f"{case}.nii.gz"
    img = nib.load(str(f))
    return np.asarray(img.dataobj).round().astype(np.int16)


def brain_mask(t1n: np.ndarray) -> np.ndarray:
    # BraTS is skull-stripped: background is exactly 0. Simple, robust foreground def.
    return t1n > 0


def flatten(img: np.ndarray, mask: np.ndarray, brain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if mask.sum() == 0:
        return img.copy()
    hp = highpass(img, brain)
    resid_std = float(hp[mask].std())
    mean_val = float(img[mask].mean())
    out = img.copy()
    out[mask] = mean_val + rng.normal(0.0, resid_std, size=int(mask.sum()))
    return out


def depth_bins(mask: np.ndarray) -> np.ndarray:
    edt = distance_transform_edt(mask)
    return np.clip(np.floor(edt).astype(np.int32), 0, MAX_DEPTH_BIN)


def ramp(img: np.ndarray, donor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return img.copy()
    bins = depth_bins(mask)
    mvox_bins = bins[mask]
    donor_vals = donor[mask]
    img_vals = img[mask]

    uniq, counts = np.unique(mvox_bins, return_counts=True)
    valid_bins = uniq[counts >= MIN_BIN_VOX]
    if len(valid_bins) < 2:
        return img.copy()  # not enough depth structure to define a ramp

    prof_raw = np.array([donor_vals[mvox_bins == b].mean() for b in valid_bins])
    prof_z = (prof_raw - prof_raw.mean()) / (prof_raw.std() + 1e-8)

    bin_to_idx = {b: i for i, b in enumerate(valid_bins)}
    keep = np.isin(mvox_bins, valid_bins)
    idxs = np.array([bin_to_idx[b] for b in mvox_bins[keep]])
    mapped_z = prof_z[idxs]
    mapped_raw = prof_raw[idxs]  # donor's own raw (non-z-scored) depth-ramp signal

    donor_std = float(donor_vals[keep].std()) + 1e-8
    rel_std_donor = float(mapped_raw.std()) / donor_std  # donor's depth-explained fraction of its own std

    img_std_here = float(img_vals[keep].std()) + 1e-8
    mapped_z_std = float(mapped_z.std()) + 1e-8
    a = rel_std_donor * img_std_here / mapped_z_std

    new_ramp = a * mapped_z
    new_ramp = new_ramp - new_ramp.mean()  # keep mean(img[mask]) unchanged

    out = img.copy()
    idx_full = np.flatnonzero(mask)[keep]
    flat_out = out.reshape(-1)
    flat_out[idx_full] = img_vals[keep] + new_ramp
    return out


def make_sham_mask(true_mask: np.ndarray, whole_tumor: np.ndarray, brain: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Mirror true_mask across the mid-sagittal plane (axis 0). If it overlaps the tumor or
    falls mostly outside the brain, translate along axis 0 in +/-5 voxel steps until it clears,
    up to +/-40 voxels. Returns (mask_or_None, status)."""
    mirror = true_mask[::-1, :, :].copy()
    for shift in [0, 5, -5, 10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 35, -35, 40, -40]:
        cand = np.roll(mirror, shift, axis=0) if shift != 0 else mirror
        overlap = int((cand & whole_tumor).sum())
        n = int(cand.sum())
        if n == 0:
            continue
        brain_frac = float((cand & brain).sum()) / n
        if overlap == 0 and brain_frac > 0.95:
            status = "mirror_ok" if shift == 0 else f"mirror_shift{shift:+d}"
            # restrict to brain to avoid any residual out-of-brain voxels
            return (cand & brain), status
    return None, "failed"


def main() -> None:
    df = pd.read_csv(PATIENT_CSV)
    all_cases = sorted(df["case"].unique())
    assert len(all_cases) == 70, f"expected 70 eval cases, got {len(all_cases)}"
    cases = all_cases[:N_PATIENTS]
    print(f"{len(all_cases)} total eval cases; using first {N_PATIENTS}: {cases[0]} .. {cases[-1]}")

    sets = ["orig_t2f", "orig_t2w", "orig_t1n",
            "e1_flatten_t2f", "e2_flatten_t2w", "e3_ramp_t1n_donor_t2w",
            "sham_flatten_t2f", "sham_flatten_t2w", "sham_ramp_t1n"]
    for s in sets:
        (INPUTS_ROOT / s).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    n_ok, n_excluded, n_sham_failed = 0, 0, 0

    for case in cases:
        label = load_label(case)
        E = label == SNFH
        n_snfh = int(E.sum())
        whole_tumor = label > 0

        if n_snfh < MIN_REGION_VOX:
            n_excluded += 1
            rows.append(dict(case=case, n_snfh=n_snfh, excluded=True, sham_status="skipped"))
            continue
        n_ok += 1

        t1n, t1n_img = load_case(case, "t1n", RAW051)
        t2w, t2w_img = load_case(case, "t2w", RAW051)
        t2f, t2f_img = load_case(case, "t2f", RAW051)
        brain = brain_mask(t1n)

        sham_mask, sham_status = make_sham_mask(E, whole_tumor, brain)
        if sham_mask is None:
            n_sham_failed += 1

        # --- originals (symlink-equivalent: write identical copy so nnUNet sees a clean subset dir)
        def write(arr: np.ndarray, ref_img: nib.Nifti1Image, set_name: str):
            out_f = INPUTS_ROOT / set_name / f"{case}_0000.nii.gz"
            if out_f.exists():
                return
            nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header), str(out_f))

        write(t2f, t2f_img, "orig_t2f")
        write(t2w, t2w_img, "orig_t2w")
        write(t1n, t1n_img, "orig_t1n")

        write(flatten(t2f, E, brain, rng), t2f_img, "e1_flatten_t2f")
        write(flatten(t2w, E, brain, rng), t2w_img, "e2_flatten_t2w")
        write(ramp(t1n, donor=t2w, mask=E), t1n_img, "e3_ramp_t1n_donor_t2w")

        if sham_mask is not None:
            write(flatten(t2f, sham_mask, brain, rng), t2f_img, "sham_flatten_t2f")
            write(flatten(t2w, sham_mask, brain, rng), t2w_img, "sham_flatten_t2w")
            write(ramp(t1n, donor=t2w, mask=sham_mask), t1n_img, "sham_ramp_t1n")
            n_sham_vox = int(sham_mask.sum())
        else:
            n_sham_vox = 0

        rows.append(dict(case=case, n_snfh=n_snfh, excluded=False,
                          sham_status=sham_status, n_sham_vox=n_sham_vox))

    manifest = pd.DataFrame(rows)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_DATA / "intervention_manifest.csv", index=False)
    print(f"\nn_ok={n_ok} n_excluded(<{MIN_REGION_VOX}vox)={n_excluded} n_sham_failed={n_sham_failed}")
    print(f"manifest -> {OUT_DATA / 'intervention_manifest.csv'}")
    print("input sets written under", INPUTS_ROOT)
    for s in sets:
        n = len(list((INPUTS_ROOT / s).glob("*.nii.gz")))
        print(f"  {s}: {n} files")


if __name__ == "__main__":
    main()
