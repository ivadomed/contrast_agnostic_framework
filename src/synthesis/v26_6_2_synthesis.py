"""
V26_6_2 synthesis — V26_6 whole-image synthesis + per-label affine remap.

Adds a label-wise stochastic decoupling step (from v19_c) on top of V26_6:

  1. V26_6 whole-image K-means synthesis  →  synth_01  (blurred, [0,1])
  2. Per-label affine remap (50% probability per label per sample):
       new_i = (mu_c + alpha_c * (synth_i - mean_c)).clamp(0, 1)
     where mean_c is the label's mean intensity in synth_01.
  3. Z-score within foreground  →  synth_z  (network input).

The label-wise step is fully vectorised over the batch dimension.
Labels with fewer than 4 voxels in a sample are skipped for that sample.
Ignore-label voxels (-1) are treated as background and never remapped.
"""
from __future__ import annotations

import random

import torch

from src.synthesis.v26_6_synthesis import (
    BLUR_SIGMAS,
    C_CHOICES,
    DARK_THRESHOLD,
    N_KMEANS_SUBSAMPLE,
    _gaussian_blur_3d,
    _kmeans_1d,
    _signed_alpha,
    _voronoi_region_ids,
    synthesize_batch_fast,
)


@torch.no_grad()
def synthesize_batch_fast_v2(
    images_01: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    V26_6_2 batched synthesis with label-guided intensity remap.

    Parameters
    ----------
    images_01 : (B, 1, D, H, W)  pre-normalised to [0, 1], on GPU.
    labels    : (B, 1, D, H, W)  integer segmentation labels; -1 = ignore → treated as 0.

    Returns
    -------
    synth_z  : (B, 1, D, H, W)  z-scored within foreground (network input).
    synth_01 : (B, 1, D, H, W)  [0, 1] synthesised image (for visualisation).
    """
    B, _, D, H, W = images_01.shape
    N      = D * H * W
    device = images_01.device
    eps    = 1e-7

    # ── Step 1: V26_6 whole-image synthesis ───────────────────────────────────
    _, synth_01 = synthesize_batch_fast(images_01)   # (B, 1, D, H, W), [0,1], already blurred

    # ── Step 2: per-label affine remap ────────────────────────────────────────
    synth = synth_01[:, 0].reshape(B, N)             # (B, N) working buffer
    lbl   = labels[:, 0].reshape(B, N).long()        # (B, N) — ignore label (-1) → treat as 0
    lbl   = lbl.clamp(min=0)

    unique_classes = lbl.unique()
    unique_classes = unique_classes[unique_classes > 0]  # skip background

    for c in unique_classes:
        c_val  = int(c.item())
        c_mask = (lbl == c_val).float()                     # (B, N) in {0,1}
        c_cnt  = c_mask.sum(dim=1, keepdim=True)            # (B, 1)

        # 50% probability per sample; skip samples where label has < 4 voxels
        apply  = ((torch.rand(B, 1, device=device) > 0.5) & (c_cnt >= 4)).float()

        if apply.sum() == 0:
            continue

        # Mean intensity of this label in current synthesis
        c_mean = (synth * c_mask).sum(dim=1, keepdim=True) / c_cnt.clamp(min=1)  # (B, 1)

        # Independent (mu, alpha) per sample
        mu_c   = torch.rand(B, 1, device=device)
        mag_c  = torch.rand(B, 1, device=device) * 1.5 + 0.5
        sign_c = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
        alp_c  = mag_c * sign_c                                          # (B, 1)

        new_vals     = (mu_c + alp_c * (synth - c_mean)).clamp(0, 1)    # (B, N)
        write_mask   = c_mask * apply                                    # (B, N) in {0,1}
        synth        = synth * (1.0 - write_mask) + new_vals * write_mask

    # ── Step 3: optional blur, then z-score within foreground ─────────────────
    synth_01 = synth.reshape(B, 1, D, H, W)

    sigma = random.choice(BLUR_SIGMAS)
    if sigma > 0.0:
        synth_01 = _gaussian_blur_3d(synth_01, sigma)
        synth    = synth_01.reshape(B, N)

    flat_m = (images_01[:, 0].reshape(B, N) > DARK_THRESHOLD).float()
    b_sum  = (synth * flat_m).sum(dim=1, keepdim=True)
    b_cnt  = flat_m.sum(dim=1, keepdim=True).clamp(min=1)
    b_mean = b_sum / b_cnt
    b_sq   = ((synth - b_mean) * flat_m).pow(2).sum(dim=1, keepdim=True)
    b_std  = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth - b_mean) / b_std * flat_m).reshape(B, 1, D, H, W)

    return synth_z, synth_01


@torch.no_grad()
def synthesize_debug_v2(
    image_01: torch.Tensor,
    labels: torch.Tensor,
    C: int = 6,
) -> dict[str, torch.Tensor]:
    """
    Single-sample synthesis exposing every intermediate stage — for visualisation only.

    Unlike the training path, this ALWAYS parcellates (never the 10 % global-remap
    branch) and ALWAYS applies the per-label remap to every label, so the panel
    reliably shows both effects.  Do not use in train/val.

    Parameters
    ----------
    image_01 : (1, 1, D, H, W)  [0, 1] on GPU.
    labels   : (1, 1, D, H, W)  integer anatomical labels; -1 = ignore → treated as 0.
    C        : number of K-means clusters for the parcellation.

    Returns
    -------
    dict with:
      parcellation  : (D, H, W) float — Voronoi sub-region id per voxel, -1 = background.
      synth01_whole : (1, 1, D, H, W) — V26_6 whole-image synth (K-means + Voronoi), [0,1].
      synth01_label : (1, 1, D, H, W) — after per-label affine remap (step 2), [0,1].
      synth_z       : (1, 1, D, H, W) — final z-scored network input.
    """
    eps    = 1e-7
    device = image_01.device
    _, _, D, H, W = image_01.shape
    N      = D * H * W

    flat   = image_01[0, 0].float().reshape(-1)          # (N,)
    flat_m = (flat > DARK_THRESHOLD).float()             # (N,)

    # ── K-means parcellation (forced, for viz) ────────────────────────────────
    idx    = torch.randint(0, N, (min(N, 40_000),), device=device)
    samp   = flat[idx]
    sub_fg = samp[samp > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
    if sub_fg.numel() < 4:
        sub_fg = samp[:N_KMEANS_SUBSAMPLE]

    centroids          = _kmeans_1d(sub_fg, C)
    sorted_c, sort_idx = torch.sort(centroids)
    boundaries         = (sorted_c[:-1] + sorted_c[1:]) / 2.0
    lbl_s              = torch.bucketize(flat, boundaries)        # (N,) ∈ [0, C)
    lbl_l              = sort_idx[lbl_s].long()                  # (N,) original cluster

    # ── Voronoi spatial sub-parcellation (forced on every cluster, for viz) ────
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"), dim=-1).reshape(N, 3)
    rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C, device,
                                 force_split=True)

    # parcellation map shows the Voronoi sub-regions (mod 10 for a discrete cmap)
    parcel = torch.where(flat_m > 0, (rid % 10).float(),
                         torch.full_like(rid.float(), -1.0)).reshape(D, H, W)

    # ── Step 1: whole-image per-region signed-alpha remap ──────────────────────
    s_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
    n_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
    mean_c = s_c / n_c.clamp(min=eps)

    mu_c   = torch.rand(R, device=device)
    mag_c  = torch.rand(R, device=device) * 1.5 + 0.5
    sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
    alp_c  = mag_c * sign_c

    synth         = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m
    synth01_whole = synth.clone().reshape(1, 1, D, H, W)

    # ── Step 2: per-anatomical-label affine remap (forced on every label) ──────
    lbl_anat       = labels[0, 0].reshape(-1).long().clamp(min=0)
    unique_classes = lbl_anat.unique()
    unique_classes = unique_classes[unique_classes > 0]
    for c in unique_classes:
        c_val  = int(c.item())
        c_mask = (lbl_anat == c_val).float()
        c_cnt  = c_mask.sum()
        if c_cnt < 4:
            continue
        c_mean   = (synth * c_mask).sum() / c_cnt.clamp(min=1)
        mu       = torch.rand(1, device=device)
        mag      = torch.rand(1, device=device) * 1.5 + 0.5
        sign     = (torch.rand(1, device=device) > 0.5).float() * 2 - 1
        alp      = mag * sign
        new_vals = (mu + alp * (synth - c_mean)).clamp(0, 1)
        synth    = synth * (1.0 - c_mask) + new_vals * c_mask
    synth01_label = synth.clone().reshape(1, 1, D, H, W)

    # ── Step 3: z-score within foreground ──────────────────────────────────────
    b_cnt   = flat_m.sum().clamp(min=1)
    b_mean  = (synth * flat_m).sum() / b_cnt
    b_sq    = ((synth - b_mean) * flat_m).pow(2).sum()
    b_std   = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth - b_mean) / b_std * flat_m).reshape(1, 1, D, H, W)

    return {
        "parcellation":  parcel,
        "synth01_whole": synth01_whole,
        "synth01_label": synth01_label,
        "synth_z":       synth_z,
    }
