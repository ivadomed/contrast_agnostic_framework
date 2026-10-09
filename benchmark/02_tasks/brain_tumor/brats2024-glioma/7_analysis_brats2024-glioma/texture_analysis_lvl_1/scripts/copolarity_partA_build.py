#!/usr/bin/env python
"""
PRE-REGISTERED Part A: does the T2w real-fill training augmentation give edema and tumor
core the SAME polarity (sign) more often than chance, explaining why real-fill is
polarity-specific on T1n (dark-edema-step rescues it, bright-edema-step breaks it, both found
in the visibility-step round; see intervention_step_build.py's docstring)?

TRANSFORM CODE READ FIRST (sub-workspaces/auglab_workspace/AugLab/auglab/transforms/gpu/fromSeg.py,
class RandomV26_6_2ContrastGPU, lines ~333-540), and the config actually used to train the t2w
real-fill model (v26_6_2_t2w_train050_val100 -> nnUNetTrainerBraTS2024GliomaV26_6_2, which per its
own docstring gets its train synth config from
transform_params_gpu_v26_6_2_synth_spatialDA_train050.json). Exact structure, restated precisely:

  STEP 1 (whole-image V26_6 K-means+Voronoi remap; NOT anatomical-label-aware):
    - Image is min-max normalised to [0,1]; foreground = pixels > dark_threshold (0.01).
    - With probability skip_parcellation_prob (0.10), OR if <4 foreground voxels: a single
      GLOBAL remap is used (one (mu, alpha) for the WHOLE image) -- edema and core trivially get
      the SAME sign in this branch, since there is only one region.
    - Otherwise: draw C_k UNIFORMLY from c_choices=[2,3,4,5,6]; fit 1-D K-means (10 Lloyd
      iterations, deterministic init = linspace(min,max,C)) on up to n_kmeans_subsample=10000
      foreground values (themselves drawn from a random subsample of up to 40000 voxels);
      each foreground voxel gets a K-means cluster id via nearest-centroid bucketize.
    - Each of the C_k clusters is THEN independently spatially sub-divided by Voronoi seeding:
      per cluster, with probability skip_sub_parc_prob (0.40) [or if <2 fg voxels in the
      cluster] no sub-split (S=1, whole cluster is one region); otherwise S is drawn UNIFORMLY
      from s_choices=[2..10] Voronoi seeds are placed at S random foreground voxels of that
      cluster and every voxel in the cluster joins its nearest seed's sub-region.
    - EVERY final (cluster x Voronoi-sub-region) gets its OWN independent affine remap:
      mu ~ U(0,1), sign ~ Bernoulli(0.5) in {-1,+1}, mag ~ U(0.5, 2.0), alpha = sign*mag.
      So voxels sharing a final region share exactly the same sign; voxels in different final
      regions have INDEPENDENT 50/50 signs, uncorrelated with each other in the code itself --
      any co-polarity between edema and core at this step can ONLY come from edema and core
      voxels landing in the SAME final region (same K-means cluster AND same Voronoi sub-region),
      not from any explicit code path that correlates their signs.
  STEP 2 (per-anatomical-label remap, the "_2" in V26_6_2 -- THIS is anatomical-label-aware):
    - For each foreground label class present (NCR=1, SNFH/edema=2, ET=3, RC=4), INDEPENDENTLY,
      with probability label_remap_prob (0.5) AND if the label has >= min_label_voxels (4)
      voxels: draw a FRESH (mu_c, alpha_c) -- again sign ~ Bernoulli(0.5), mag ~ U(0.5,2.0) --
      and OVERWRITE every voxel of that label with mu_c + alpha_c*(current_value - label_mean).
      This is drawn PER LABEL INDEPENDENTLY -- there is no code path linking NCR's coin flip to
      edema's coin flip. If a label's remap does NOT trigger (50% of the time), that label's
      voxels keep whatever step-1 per-region values/signs they already had (which may differ
      voxel-to-voxel if the label spans multiple step-1 regions).
  Net effect: co-polarity between edema and core is NOT built into the code as a rule -- it can
  only emerge STATISTICALLY, via the two labels' voxels tending to fall into the same intensity
  (K-means) cluster before either coin flip happens, since same-cluster membership survives the
  case where NEITHER label's step-2 remap triggers (25% of the time: (1-0.5)*(1-0.5)).

PRE-REGISTERED PREDICTION: edema-core same-sign fraction, simulated end-to-end (including the
label-remap coin flips), is SUBSTANTIALLY > 0.5 -- because on T2w training images, edema (SNFH)
and the solid/enhancing tumor core (NCR/ET) both sit toward the BRIGHT end of the intensity
range relative to normal brain, so they are expected to fall in the same (or adjacent, same-sign)
K-means clusters more often than chance, and this correlation survives whenever step 2 doesn't
independently re-randomize both labels.

METHOD: ~40 T2w TRAINING cases (Dataset052 imagesTr/labelsTr), the FIRST 40 case ids (sorted)
that are NOT in the 70-case held-out eval test set (4_splits_brats2024-glioma/test_cases.json;
verified this exclusion is necessary -- imagesTr/labelsTr for this dataset actually contains all
770 cases, all 70 test cases included, unlike a standard nnU-Net imagesTr/imagesTs split).
The real AugLab CPU-portable functions (_kmeans_1d, _voronoi_region_ids, imported directly from
auglab.transforms.gpu.fromSeg -- pure torch tensor ops, run fine on CPU) are called with the
EXACT config values from transform_params_gpu_v26_6_2_synth_spatialDA_train050.json. Since the
GPU transform operates on nnU-Net TRAINING PATCHES (this dataset's 3d_fullres patch_size =
[128,160,112], not the full ~182x218x182 volume), each training case is cropped to its tumor
bounding box padded to roughly patch scale (clipped to volume bounds) before simulation, matching
what an oversampled foreground-centered training patch would actually contain.
"Normal-WM" is NOT independently segmented (no SynthSeg run for this analysis) -- it is
approximated as "normal (non-tumor) brain tissue inside the crop" (label==0, foreground), and
reported under that honest name, not claimed to be true white matter.

REPORTS (per training case, then pooled):
  (1) K-means-cluster co-membership sweep over c in {2,...,6} (30 repeats/case/c, subsampling
      randomness only -- K-means init itself is deterministic given the sample): share of edema
      voxels in the SAME final K-means cluster as the cluster holding the MAJORITY of NCR / ET /
      normal-tissue voxels.
  (2) End-to-end random-draw simulation (200 draws/case) of the FULL step-1 + step-2 pipeline
      (C_k drawn from c_choices, Voronoi sub-split, then the two independent label-remap coin
      flips for NCR and edema, and separately ET and edema): fraction of draws in which the
      MAJORITY sign of edema voxels equals the majority sign of NCR voxels (and separately ET).

Run as a CPU job (--gpus 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[7]
AUGLAB = PROJECT / "sub-workspaces/auglab_workspace/AugLab"
sys.path.insert(0, str(AUGLAB))
from auglab.transforms.gpu.fromSeg import _kmeans_1d, _voronoi_region_ids  # noqa: E402

DS = PROJECT / "benchmark" / "02_tasks" / "brain_tumor" / "brats2024-glioma"
RAW052 = DS / "2_nnUNet_brats2024-glioma" / "raw" / "Dataset052_BraTS2024GliomaT2w"
TEST_CASES = DS / "4_splits_brats2024-glioma" / "test_cases.json"

ANALYSIS_DIR = DS / "7_analysis_brats2024-glioma" / "texture_analysis_lvl_1"
OUT_DATA = ANALYSIS_DIR / "outputs" / "data"

N_TRAIN_CASES = 40
CROP_MARGIN = 24  # ~half the gap to nominal patch_size [128,160,112] beyond the tumor bbox

# exact config values from transform_params_gpu_v26_6_2_synth_spatialDA_train050.json
C_CHOICES = [2, 3, 4, 5, 6]
S_CHOICES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
DARK_THRESHOLD = 0.01
N_KMEANS_SUBSAMPLE = 10_000
SKIP_PARCELLATION_PROB = 0.10
SKIP_SUB_PARC_PROB = 0.40
ALPHA_LO, ALPHA_HI = 0.5, 2.0
LABEL_REMAP_PROB = 0.5
MIN_LABEL_VOXELS = 4

N_CLUSTER_REPEATS = 30
N_DRAWS = 200
SEED = 0

NCR, SNFH, ET = 1, 2, 3


def get_train_cases() -> list[str]:
    import json
    test_cases = set(json.loads(TEST_CASES.read_text()))
    all_files = sorted((RAW052 / "labelsTr").glob("*.nii.gz"))
    train_cases = [f.name.replace(".nii.gz", "") for f in all_files
                   if f.name.replace(".nii.gz", "") not in test_cases]
    return train_cases[:N_TRAIN_CASES]


def crop_case(case: str):
    img = nib.load(str(RAW052 / "imagesTr" / f"{case}_0000.nii.gz"))
    lab = nib.load(str(RAW052 / "labelsTr" / f"{case}.nii.gz"))
    x = np.asarray(img.dataobj, dtype=np.float32)
    y = np.asarray(lab.dataobj).round().astype(np.int64)
    fg = y > 0
    if fg.sum() == 0:
        return None
    idx = np.argwhere(fg)
    lo = np.maximum(idx.min(0) - CROP_MARGIN, 0)
    hi = np.minimum(idx.max(0) + CROP_MARGIN, np.array(x.shape) - 1)
    sl = tuple(slice(lo[i], hi[i] + 1) for i in range(3))
    return x[sl], y[sl]


def kmeans_cluster_ids(flat_01: torch.Tensor, flat_m: torch.Tensor, C: int, rng: torch.Generator):
    N = flat_01.shape[0]
    idx = torch.randint(0, N, (min(N, 40_000),), generator=rng)
    samp = flat_01[idx]
    sub_fg = samp[samp > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
    if sub_fg.numel() < 4:
        sub_fg = samp[:N_KMEANS_SUBSAMPLE]
    centroids = _kmeans_1d(sub_fg, C)
    sorted_c, sort_idx = torch.sort(centroids)
    boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
    lbl_s = torch.bucketize(flat_01, boundaries)
    lbl_l = sort_idx[lbl_s].long()
    return lbl_l


def majority_cluster(cluster_ids: torch.Tensor, mask_idx: torch.Tensor) -> int | None:
    if mask_idx.numel() == 0:
        return None
    vals, counts = torch.unique(cluster_ids[mask_idx], return_counts=True)
    return int(vals[torch.argmax(counts)].item())


def part1_cluster_membership(case: str, x: np.ndarray, y: np.ndarray, rng: torch.Generator) -> list[dict]:
    flat_all = torch.from_numpy(x.reshape(-1))
    vmin, vmax = flat_all.min(), flat_all.max()
    flat_01 = ((flat_all - vmin) / (vmax - vmin + 1e-7)).clamp(0, 1)
    flat_m = (flat_01 > DARK_THRESHOLD).float()
    lbl = torch.from_numpy(y.reshape(-1))

    edema_idx = torch.nonzero(lbl == SNFH).squeeze(1)
    ncr_idx = torch.nonzero(lbl == NCR).squeeze(1)
    et_idx = torch.nonzero(lbl == ET).squeeze(1)
    normal_idx = torch.nonzero((lbl == 0) & (flat_m > 0)).squeeze(1)
    if normal_idx.numel() > 20000:
        perm = torch.randperm(normal_idx.numel(), generator=rng)[:20000]
        normal_idx = normal_idx[perm]

    rows = []
    if edema_idx.numel() < MIN_LABEL_VOXELS:
        return rows
    for C in C_CHOICES:
        same_ncr, same_et, same_normal, n_rep = 0.0, 0.0, 0.0, 0
        for _ in range(N_CLUSTER_REPEATS):
            cluster_ids = kmeans_cluster_ids(flat_01, flat_m, C, rng)
            edema_clusters = cluster_ids[edema_idx]
            maj_ncr = majority_cluster(cluster_ids, ncr_idx)
            maj_et = majority_cluster(cluster_ids, et_idx)
            maj_normal = majority_cluster(cluster_ids, normal_idx)
            if maj_ncr is not None:
                same_ncr += float((edema_clusters == maj_ncr).float().mean())
            if maj_et is not None:
                same_et += float((edema_clusters == maj_et).float().mean())
            if maj_normal is not None:
                same_normal += float((edema_clusters == maj_normal).float().mean())
            n_rep += 1
        rows.append(dict(case=case, c=C, n_rep=n_rep,
                          frac_edema_same_cluster_as_ncr=same_ncr / n_rep if n_rep else np.nan,
                          frac_edema_same_cluster_as_et=same_et / n_rep if n_rep else np.nan,
                          frac_edema_same_cluster_as_normal=same_normal / n_rep if n_rep else np.nan))
    return rows


def simulate_final_sign(flat_01: torch.Tensor, flat_m: torch.Tensor, lbl: torch.Tensor,
                         coords: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """One full draw of step 1 (K-means+Voronoi) + step 2 (per-label remap), returning the
    final per-voxel SIGN (+1/-1), matching RandomV26_6_2ContrastGPU exactly for B=1."""
    N = flat_01.shape[0]
    n_fg = flat_m.sum()
    device = flat_01.device

    if n_fg < 4 or torch.rand(1, generator=rng).item() < SKIP_PARCELLATION_PROB:
        sign = (torch.rand(1, generator=rng) > 0.5).float() * 2 - 1
        sign_field = torch.full((N,), float(sign.item()))
    else:
        C_k = C_CHOICES[int(torch.rand(1, generator=rng).item() * len(C_CHOICES))]
        cluster_ids = kmeans_cluster_ids(flat_01, flat_m, C_k, rng)
        rid, R = _voronoi_region_ids(coords, cluster_ids, flat_m, C_k, device,
                                      S_CHOICES, SKIP_SUB_PARC_PROB)
        sign_c = (torch.rand(R, generator=rng) > 0.5).float() * 2 - 1
        sign_field = sign_c[rid]

    for c_val in (NCR, SNFH, ET):
        c_mask = lbl == c_val
        c_cnt = int(c_mask.sum().item())
        if c_cnt < MIN_LABEL_VOXELS:
            continue
        if torch.rand(1, generator=rng).item() < LABEL_REMAP_PROB:
            sign_c = (torch.rand(1, generator=rng) > 0.5).float() * 2 - 1
            sign_field = torch.where(c_mask, torch.full_like(sign_field, float(sign_c.item())), sign_field)
    return sign_field


def part2_sign_simulation(case: str, x: np.ndarray, y: np.ndarray, rng: torch.Generator) -> dict | None:
    D, H, W = x.shape
    N = D * H * W
    flat_all = torch.from_numpy(x.reshape(-1))
    vmin, vmax = flat_all.min(), flat_all.max()
    flat_01 = ((flat_all - vmin) / (vmax - vmin + 1e-7)).clamp(0, 1)
    flat_m = (flat_01 > DARK_THRESHOLD).float()
    lbl = torch.from_numpy(y.reshape(-1))
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, dtype=torch.float32), torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32), indexing="ij"), dim=-1).reshape(N, 3)

    edema_idx = torch.nonzero(lbl == SNFH).squeeze(1)
    ncr_idx = torch.nonzero(lbl == NCR).squeeze(1)
    et_idx = torch.nonzero(lbl == ET).squeeze(1)
    if edema_idx.numel() < MIN_LABEL_VOXELS:
        return None

    same_ncr, same_et, n_ok = 0, 0, 0
    for _ in range(N_DRAWS):
        sign_field = simulate_final_sign(flat_01, flat_m, lbl, coords, rng)
        edema_maj = float(sign_field[edema_idx].mean().sign().item())  # majority via mean sign
        ok = True
        if ncr_idx.numel() >= MIN_LABEL_VOXELS:
            ncr_maj = float(sign_field[ncr_idx].mean().sign().item())
            same_ncr += int(edema_maj == ncr_maj)
        else:
            ok = False
        if et_idx.numel() >= MIN_LABEL_VOXELS:
            et_maj = float(sign_field[et_idx].mean().sign().item())
            same_et += int(edema_maj == et_maj)
        n_ok += 1
    return dict(case=case, n_draws=n_ok,
                frac_same_sign_edema_ncr=same_ncr / n_ok if n_ok else np.nan,
                frac_same_sign_edema_et=same_et / n_ok if (n_ok and et_idx.numel() >= MIN_LABEL_VOXELS) else np.nan,
                n_ncr=int(ncr_idx.numel()), n_et=int(et_idx.numel()), n_edema=int(edema_idx.numel()))


def main() -> None:
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    cases = get_train_cases()
    print(f"{len(cases)} training cases (excluding the 70 held-out eval cases)")

    cluster_rows, sign_rows = [], []
    for i, case in enumerate(cases):
        cropped = crop_case(case)
        if cropped is None:
            continue
        x, y = cropped
        cluster_rows.extend(part1_cluster_membership(case, x, y, rng))
        s = part2_sign_simulation(case, x, y, rng)
        if s is not None:
            sign_rows.append(s)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(cases)} done")

    cluster_df = pd.DataFrame(cluster_rows)
    sign_df = pd.DataFrame(sign_rows)
    cluster_df.to_csv(OUT_DATA / "copolarity_cluster_membership.csv", index=False)
    sign_df.to_csv(OUT_DATA / "copolarity_sign_simulation.csv", index=False)

    print("\n=== Part A summary: K-means cluster co-membership (mean over cases) ===")
    print(cluster_df.groupby("c")[["frac_edema_same_cluster_as_ncr", "frac_edema_same_cluster_as_et",
                                    "frac_edema_same_cluster_as_normal"]].mean())
    print("\n=== Part A summary: end-to-end same-sign fraction (mean over cases) ===")
    print(f"edema-NCR: {sign_df['frac_same_sign_edema_ncr'].mean():.3f} "
          f"(n_cases={sign_df['frac_same_sign_edema_ncr'].notna().sum()})")
    print(f"edema-ET:  {sign_df['frac_same_sign_edema_et'].mean():.3f} "
          f"(n_cases={sign_df['frac_same_sign_edema_et'].notna().sum()})")


if __name__ == "__main__":
    main()
