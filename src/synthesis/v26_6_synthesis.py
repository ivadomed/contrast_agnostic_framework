"""
V26_6 synthesis — vectorised on-the-fly MRI contrast randomisation.

Approach
--------
Given a T1w volume normalised to [0, 1]:

  1. K-means parcellation of foreground voxels into C ∈ {4, 6, 8} clusters
     (cluster boundaries computed via 1-D bucketize — no boolean indexing).
  2. Per-cluster signed-alpha affine remap: y = μ + α · (x − x̄_c)
     • μ ~ U(0, 1)  — random target mean
     • α = sign · magnitude,  sign ~ Bernoulli(0.5),  magnitude ~ U(0.5, 2.0)
     Negative α inverts the local intensity ordering within a cluster, enabling
     contrast inversions (T1w-like → T2w-like) without explicit contrast targets.
  3. Optional Gaussian blur: σ ∈ {0, 0, 0, 0.3, 0.5, 0.8}  (mostly no blur).
  4. Zero non-brain (dark) voxels: image < 0.01 → 0.
  5. Z-score within foreground region.

Key property: NO boolean-indexing anywhere — only masked arithmetic and
scatter_add into small tables (C ≤ 8 bins).  This makes the synthesis fast
enough to run on-the-fly every training step without a pre-synthesis cache.

Public API
----------
  synthesize_volume_fast(image_01)             → (synth_z, synth_01)
      Full-volume GPU synthesis.  ~50 ms on A6000 for a 192×224×160 brain.
      Use in both train_step and validation_step for zero distribution mismatch.

  synthesize_patch_fast(image_01, seg, centroids) → (synth_z, synth_01)
      Patch-level CPU synthesis with pre-computed centroids.  ~150 ms per
      168×210×147 patch on a single CPU core.  Use with OnHarmonyBatchPool.

  compute_kmeans_centroids(image_01)            → centroids (C,)
      Pre-compute full-volume K-means once per training case (~10 ms CPU).
      Pass centroids to synthesize_patch_fast for consistent cluster boundaries
      between train and val synthesis.

  gpu_spatial_augment(image_01, seg)            → (aug_image, aug_seg)
      GPU random affine augment (rotation + scaling) applied BEFORE synthesis
      so that synthesis sees the augmented anatomy.  ~30 ms on A6000.
      Replaces the 1-3 s CPU SpatialTransform when rotation fires.

Design constraints
------------------
- No labels, no atlas, no anatomical templates anywhere in the synthesis path.
  Synthesis must work on pathological brains where such priors fail.
- No targeted contrast: all contrast diversity comes from random (μ, α) draws.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────

C_CHOICES: list[int] = [2, 3, 4, 5, 6]  # number of K-means intensity clusters
S_CHOICES: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Voronoi sub-regions per cluster
BLUR_SIGMAS: list[float] = [0.0, 0.0, 0.0, 0.3, 0.5, 0.8]  # mostly no blur
DARK_THRESHOLD: float = 0.01           # voxels below this are treated as non-brain
N_KMEANS_SUBSAMPLE: int = 10_000       # max foreground voxels used for K-means fit
SKIP_PARCELLATION_PROB: float = 0.10   # probability of single global remap (no clusters)
SKIP_SUB_PARC_PROB: float = 0.40       # per-cluster probability of NO Voronoi sub-split


# ── Internal helpers ──────────────────────────────────────────────────────────

def _signed_alpha(device: torch.device) -> float:
    """Draw one signed alpha: sign ~ Bernoulli(0.5), magnitude ~ U(0.5, 2.0)."""
    sign = 1.0 if random.random() > 0.5 else -1.0
    return sign * (random.random() * 1.5 + 0.5)


def _gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur.  x must be (1, 1, D, H, W)."""
    k_r = max(1, int(3.0 * sigma + 0.5))
    k1d = torch.arange(-k_r, k_r + 1, dtype=x.dtype, device=x.device)
    k1d = torch.exp(-0.5 * (k1d / sigma) ** 2)
    k1d = k1d / k1d.sum()
    pad = len(k1d) // 2
    y = x
    y = F.conv3d(y, k1d.view(1, 1, -1, 1, 1), padding=(pad, 0, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, -1, 1), padding=(0, pad, 0))
    y = F.conv3d(y, k1d.view(1, 1, 1, 1, -1), padding=(0, 0, pad))
    return y.clamp(0, 1)


def _kmeans_1d(values: torch.Tensor, C: int, n_iter: int = 10) -> torch.Tensor:
    """
    1-D K-means on a 1-D tensor of foreground values.  Returns (C,) centroids.

    Uses scatter_add — no loops over N elements, O(n_iter × N) total.
    """
    centroids = torch.linspace(values.min().item(), values.max().item(), C,
                               device=values.device)
    for _ in range(n_iter):
        d   = torch.abs(values.unsqueeze(1) - centroids.unsqueeze(0))  # (N, C)
        lbl = torch.argmin(d, dim=1)                                   # (N,)
        s   = torch.zeros(C, device=values.device).scatter_add_(0, lbl, values)
        n   = torch.zeros(C, device=values.device).scatter_add_(
            0, lbl, torch.ones_like(values))
        new_c = torch.where(n > 0, s / n, centroids)
        if torch.allclose(centroids, new_c):
            break
        centroids = new_c
    return centroids


def _voronoi_region_ids(
    coords:      torch.Tensor,   # (N, 3) float voxel coordinates
    lbl_l:       torch.Tensor,   # (N,)   K-means cluster id ∈ [0, C)
    fg:          torch.Tensor,   # (N,)   foreground mask in {0,1}
    C:           int,
    device:      torch.device,
    force_split: bool = False,   # if True, never skip sub-parcellation (viz only)
) -> tuple[torch.Tensor, int]:
    """
    Spatially subdivide each K-means cluster into S Voronoi sub-regions.

    For each cluster c (sequential, C ≤ 6): with prob SKIP_SUB_PARC_PROB keep it
    whole (S=1); otherwise pick S random foreground voxels as Voronoi seeds and
    assign every voxel of the cluster to its nearest seed (squared-euclidean).

    Returns
    -------
    rid : (N,) long — combined (cluster × sub-region) id ∈ [0, R).
    R   : int        — total number of regions.

    Cost: one cdist per *subdivided* cluster over N voxels × S seeds (S ≤ 10).
    Clusters that skip (or have < 2 fg voxels) add no cdist.
    """
    N   = lbl_l.shape[0]
    rid = torch.zeros(N, dtype=torch.long, device=device)
    offset = 0
    for c in range(C):
        c_mask    = lbl_l == c                       # (N,) all voxels of cluster c
        c_fg_mask = c_mask & (fg > 0)                # foreground voxels only
        n_fg      = int(c_fg_mask.sum().item())
        if n_fg == 0:
            continue
        if n_fg < 2 or (not force_split and torch.rand(1, device=device).item() < SKIP_SUB_PARC_PROB):
            S = 1
        else:
            s_idx = int(torch.rand(1, device=device).item() * len(S_CHOICES))
            S = min(S_CHOICES[s_idx], n_fg)
        if S <= 1:
            rid = torch.where(c_mask, torch.full_like(rid, offset), rid)
            offset += 1
            continue
        # S random foreground seeds (multinomial = gather, no boolean indexing)
        seed_idx  = torch.multinomial(c_fg_mask.float(), S, replacement=False)  # (S,)
        centroids = coords.index_select(0, seed_idx)                            # (S, 3)
        d         = torch.cdist(coords, centroids)                              # (N, S)
        sub       = torch.argmin(d, dim=1)                                      # (N,) ∈ [0, S)
        rid       = torch.where(c_mask, offset + sub, rid)
        offset   += S
    return rid, offset


# ── Public API ────────────────────────────────────────────────────────────────

@torch.no_grad()
def synthesize_volume_fast(
    image_01: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full-volume V26_6 synthesis.  ~50 ms on GPU (A6000) for a typical brain.

    Parameters
    ----------
    image_01 : (1, 1, D, H, W)  pre-normalised to [0, 1], any device.

    Returns
    -------
    synth_z  : (1, 1, D, H, W)  z-scored within foreground (network input).
    synth_01 : (1, 1, D, H, W)  [0, 1] synthesised image (for visualisation).
    """
    eps    = 1e-7
    device = image_01.device
    _, _, D, H, W = image_01.shape
    N      = D * H * W

    img    = image_01[0, 0].float()   # (D, H, W)
    flat   = img.reshape(-1)          # (N,)
    flat_m = (flat > DARK_THRESHOLD).float()  # foreground mask as float

    if flat_m.sum() < 4 or torch.rand(1, device=device).item() < SKIP_PARCELLATION_PROB:
        # Global single-region remap
        b_mean = (flat * flat_m).sum() / flat_m.sum().clamp(min=1)
        mu     = torch.rand(1, device=device).item()
        alpha  = _signed_alpha(device)
        synth  = (mu + alpha * (flat - b_mean)).clamp(0, 1) * flat_m
    else:
        C   = C_CHOICES[int(torch.rand(1, device=device).item() * len(C_CHOICES))]
        idx = torch.randint(0, N, (min(N, 40_000),), device=device)
        samp   = flat[idx]
        sub_fg = samp[samp > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
        if sub_fg.numel() < 4:
            sub_fg = samp[:N_KMEANS_SUBSAMPLE]

        centroids = _kmeans_1d(sub_fg, C)

        sorted_c, sort_idx = torch.sort(centroids)
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0   # (C-1,)
        lbl_s = torch.bucketize(flat, boundaries)            # (N,) ∈ [0, C)
        lbl_l = sort_idx[lbl_s].long()                       # (N,) original cluster

        # Voronoi spatial sub-parcellation of each intensity cluster
        coords = torch.stack(torch.meshgrid(
            torch.arange(D, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"), dim=-1).reshape(N, 3)
        rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C, device)

        s_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
        n_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
        mean_c = s_c / n_c.clamp(min=eps)                   # (R,) per-region mean

        mu_c   = torch.rand(R, device=device)
        mag_c  = torch.rand(R, device=device) * 1.5 + 0.5
        sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
        alp_c  = mag_c * sign_c

        synth = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m

    synth_01 = synth.reshape(1, 1, D, H, W)

    sigma = random.choice(BLUR_SIGMAS)
    if sigma > 0.0:
        synth_01 = _gaussian_blur_3d(synth_01, sigma)
        synth    = synth_01.reshape(-1)

    # Z-score within foreground
    b_sum   = (synth * flat_m).sum()
    b_cnt   = flat_m.sum().clamp(min=1)
    b_mean  = b_sum / b_cnt
    b_sq    = ((synth - b_mean) * flat_m).pow(2).sum()
    b_std   = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth - b_mean) / b_std * flat_m).reshape(1, 1, D, H, W)

    return synth_z, synth_01


@torch.no_grad()
def synthesize_batch_fast(
    images_01: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched V26_6 synthesis.  images_01: (B, 1, D, H, W)

    K-means is sequential per sample (unavoidable — different foreground each
    volume).  Gaussian blur and z-score are vectorised over the full batch in a
    single GPU pass, saving one conv3d launch per extra sample.

    Returns
    -------
    synth_z  : (B, 1, D, H, W)  z-scored within foreground.
    synth_01 : (B, 1, D, H, W)  [0, 1] synthesised images.
    """
    B, _, D, H, W = images_01.shape
    N      = D * H * W
    device = images_01.device
    eps    = 1e-7

    flat_all  = images_01[:, 0].float().reshape(B, N)    # (B, N)
    flat_m_all = (flat_all > DARK_THRESHOLD).float()      # (B, N)

    # Voxel coordinates (shared across the batch — same D,H,W) for Voronoi seeding
    coords = torch.stack(torch.meshgrid(
        torch.arange(D, device=device, dtype=torch.float32),
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"), dim=-1).reshape(N, 3)            # (N, 3)

    # Per-sample K-means → Voronoi sub-parcellation → affine remap
    synth_list = []
    for i in range(B):
        flat   = flat_all[i]
        flat_m = flat_m_all[i]
        n_fg   = flat_m.sum()

        if n_fg < 4 or torch.rand(1, device=device).item() < SKIP_PARCELLATION_PROB:
            b_mean_i = (flat * flat_m).sum() / n_fg.clamp(min=1)
            mu       = torch.rand(1, device=device).item()
            alpha    = _signed_alpha(device)
            synth_i  = (mu + alpha * (flat - b_mean_i)).clamp(0, 1) * flat_m
        else:
            C   = C_CHOICES[int(torch.rand(1, device=device).item() * len(C_CHOICES))]
            idx = torch.randint(0, N, (min(N, 40_000),), device=device)
            samp   = flat[idx]
            sub_fg = samp[samp > DARK_THRESHOLD][:N_KMEANS_SUBSAMPLE]
            if sub_fg.numel() < 4:
                sub_fg = samp[:N_KMEANS_SUBSAMPLE]

            centroids = _kmeans_1d(sub_fg, C)
            sorted_c, sort_idx = torch.sort(centroids)
            boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
            lbl_s = torch.bucketize(flat, boundaries)
            lbl_l = sort_idx[lbl_s].long()

            # Voronoi spatial sub-parcellation of each intensity cluster
            rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C, device)

            s_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat * flat_m)
            n_c    = torch.zeros(R, device=device).scatter_add_(0, rid, flat_m)
            mean_c = s_c / n_c.clamp(min=eps)

            mu_c   = torch.rand(R, device=device)
            mag_c  = torch.rand(R, device=device) * 1.5 + 0.5
            sign_c = (torch.rand(R, device=device) > 0.5).float() * 2 - 1
            alp_c  = mag_c * sign_c

            synth_i = (mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])).clamp(0, 1) * flat_m

        synth_list.append(synth_i)

    synth    = torch.stack(synth_list)           # (B, N)
    synth_01 = synth.reshape(B, 1, D, H, W)

    # Batched Gaussian blur — one conv3d call on (B, 1, D, H, W)
    sigma = random.choice(BLUR_SIGMAS)
    if sigma > 0.0:
        synth_01 = _gaussian_blur_3d(synth_01, sigma)
        synth    = synth_01.reshape(B, N)

    # Batched z-score within foreground — vectorised over (B, N)
    b_sum  = (synth * flat_m_all).sum(dim=1, keepdim=True)            # (B, 1)
    b_cnt  = flat_m_all.sum(dim=1, keepdim=True).clamp(min=1)         # (B, 1)
    b_mean = b_sum / b_cnt
    b_sq   = ((synth - b_mean) * flat_m_all).pow(2).sum(dim=1, keepdim=True)
    b_std  = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth - b_mean) / b_std * flat_m_all).reshape(B, 1, D, H, W)

    return synth_z, synth_01


def compute_kmeans_centroids(
    image_01: torch.Tensor,
    n_sample: int = N_KMEANS_SUBSAMPLE,
) -> torch.Tensor:
    """
    Pre-compute full-volume K-means centroids once per training case.  ~10 ms CPU.

    Use with synthesize_patch_fast when the same cluster boundaries should be
    shared across all patches of one volume (consistent train/val synthesis).

    Parameters
    ----------
    image_01 : (1, 1, D, H, W)  [0, 1] float32 CPU tensor.

    Returns
    -------
    centroids : (C,) float tensor on CPU.
    """
    flat = image_01[0, 0].float().reshape(-1)
    N    = flat.shape[0]

    # Over-sample then filter to foreground (~75% of voxels) in one step
    idx    = torch.randint(0, N, (n_sample * 4,))
    sampled = flat[idx]
    sub_fg  = sampled[sampled > DARK_THRESHOLD][:n_sample]

    if sub_fg.numel() < 4:
        return torch.zeros(C_CHOICES[0])

    C = C_CHOICES[int(torch.rand(1).item() * len(C_CHOICES))]
    return _kmeans_1d(sub_fg, C)


@torch.no_grad()
def synthesize_patch_fast(
    image_01:  torch.Tensor,
    seg:       torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patch-level V26_6 synthesis using pre-computed full-volume centroids.

    ~150 ms for a 168×210×147 patch on a single CPU core.  Designed for use
    in OnHarmonyBatchPool where multiple CPU workers synthesize in parallel.

    Parameters
    ----------
    image_01  : (1, 1, D, H, W)  [0, 1] from full-volume min-max.
    seg       : (1, 1, D, H, W)  integer labels (>0 = brain foreground).
    centroids : (C,)              full-volume K-means centroids.

    Returns
    -------
    synth_z  : (1, 1, D, H, W)  z-scored within segmentation mask.
    synth_01 : (1, 1, D, H, W)  [0, 1] synthesis.
    """
    eps = 1e-7
    _, _, D, H, W = image_01.shape
    N   = D * H * W
    C   = centroids.shape[0]
    dev = image_01.device

    img    = image_01[0, 0].float()
    flat   = img.reshape(-1)
    flat_m = (seg[0, 0].reshape(-1) > 0).float()   # seg-based brain mask

    if torch.rand(1, device=dev).item() < SKIP_PARCELLATION_PROB:
        brain_mean = (flat * flat_m).sum() / flat_m.sum().clamp(min=1)
        mu    = torch.rand(1, device=dev).item()
        alpha = _signed_alpha(dev)
        synth_flat = mu + alpha * (flat - brain_mean)
    else:
        sorted_c, sort_idx = torch.sort(centroids.to(dev))
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        lbl_sorted = torch.bucketize(flat, boundaries)     # (N,) ∈ [0, C)
        flat_lbl   = sort_idx[lbl_sorted]                  # (N,) original cluster
        lbl_l      = flat_lbl.long()

        # Voronoi spatial sub-parcellation of each intensity cluster
        coords = torch.stack(torch.meshgrid(
            torch.arange(D, device=dev, dtype=torch.float32),
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32),
            indexing="ij"), dim=-1).reshape(N, 3)
        rid, R = _voronoi_region_ids(coords, lbl_l, flat_m, C, dev)

        s_c    = torch.zeros(R, device=dev).scatter_add_(0, rid, flat * flat_m)
        n_c    = torch.zeros(R, device=dev).scatter_add_(0, rid, flat_m)
        mean_c = s_c / n_c.clamp(min=eps)

        mu_c   = torch.rand(R, device=dev)
        mag_c  = torch.rand(R, device=dev) * 1.5 + 0.5
        sign_c = (torch.rand(R, device=dev) > 0.5).float() * 2 - 1
        alp_c  = mag_c * sign_c

        synth_flat = mu_c[rid] + alp_c[rid] * (flat - mean_c[rid])

    synth_flat = synth_flat.clamp(0, 1) * flat_m
    synth_01   = synth_flat.reshape(1, 1, D, H, W)

    sigma = random.choice(BLUR_SIGMAS)
    if sigma > 0.0:
        synth_01   = _gaussian_blur_3d(synth_01, sigma)
        synth_flat = synth_01.reshape(-1)

    # Zero dark voxels
    dark_mask  = (flat < DARK_THRESHOLD).float()
    synth_flat = synth_flat * (1 - dark_mask)
    synth_01   = synth_flat.reshape(1, 1, D, H, W)

    # Z-score within seg mask
    b_sum   = (synth_flat * flat_m).sum()
    b_cnt   = flat_m.sum().clamp(min=1)
    b_mean  = b_sum / b_cnt
    b_sq    = ((synth_flat - b_mean) * flat_m).pow(2).sum()
    b_std   = (b_sq / b_cnt + eps).sqrt()
    synth_z = ((synth_flat - b_mean) / b_std * flat_m).reshape(1, 1, D, H, W)

    return synth_z, synth_01


@torch.no_grad()
def gpu_spatial_augment(
    image_01:         torch.Tensor,
    seg:              torch.Tensor,
    p_rotation:       float = 0.2,
    p_scaling:        float = 0.2,
    max_rotation_rad: float = 0.35,   # ≈ 20 degrees
    scale_range:      tuple = (0.7, 1.4),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Random 3D affine augmentation on GPU.  ~30 ms on A6000.

    Replaces batchgeneratorsv2 SpatialTransform on CPU (1–3 s when rotation
    fires).  Applied BEFORE synthesis so the generator sees augmented anatomy
    and the labels correctly correspond to the augmented image.

    Parameters
    ----------
    image_01 : (1, 1, D, H, W)  [0, 1] on GPU.
    seg      : (1, 1, D, H, W)  integer labels on GPU.

    Returns
    -------
    Augmented (image_01, seg) with the same shapes as inputs.
    """
    device   = image_01.device
    do_rot   = random.random() < p_rotation
    do_scale = random.random() < p_scaling

    if not do_rot and not do_scale:
        return image_01, seg

    def _rot_x(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]], dtype=torch.float32, device=device)

    def _rot_y(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]], dtype=torch.float32, device=device)

    def _rot_z(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]], dtype=torch.float32, device=device)

    R = torch.eye(3, device=device)
    if do_rot:
        ax = (random.random() * 2 - 1) * max_rotation_rad
        ay = (random.random() * 2 - 1) * max_rotation_rad
        az = (random.random() * 2 - 1) * max_rotation_rad
        R  = _rot_z(az) @ _rot_y(ay) @ _rot_x(ax)
    if do_scale:
        s  = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
        R  = R * s

    B            = image_01.shape[0]
    theta        = torch.zeros(B, 3, 4, device=device)
    theta[:, :3, :3] = R
    grid         = F.affine_grid(theta, image_01.shape, align_corners=False)
    aug_img      = F.grid_sample(image_01, grid, mode='bilinear',
                                 align_corners=False, padding_mode='zeros').clamp(0, 1)
    aug_seg      = F.grid_sample(seg.float(), grid, mode='nearest',
                                 align_corners=False, padding_mode='zeros').to(seg.dtype)
    return aug_img, aug_seg
