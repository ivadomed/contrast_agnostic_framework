"""
Verify the sync-free synthesize_volume_fast:
  (1) augmentation output distribution is statistically unchanged vs the original
      (per-voxel mean/std/percentiles over many random draws on a real volume), and
  (2) it no longer blocks the CUDA pipeline (time many calls without manual sync).

Run: set_slot 0 .venv/bin/python scripts/experiments/verify_synth_syncfree.py
"""
from __future__ import annotations
import sys, time, math, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREP = ROOT / "datasets/on-harmony/2_nnUNet_on-harmony/preprocessed/Dataset030_OnHarmonyT1w"

import src.synthesis.v26_6_synthesis as S


# ---- reference: the ORIGINAL (sync-ful) implementation, inlined ----
def _kmeans_orig(values, C, n_iter=10):
    centroids = torch.linspace(values.min().item(), values.max().item(), C, device=values.device)
    for _ in range(n_iter):
        d = torch.abs(values.unsqueeze(1) - centroids.unsqueeze(0))
        lbl = torch.argmin(d, dim=1)
        s = torch.zeros(C, device=values.device).scatter_add_(0, lbl, values)
        n = torch.zeros(C, device=values.device).scatter_add_(0, lbl, torch.ones_like(values))
        new_c = torch.where(n > 0, s / n, centroids)
        if torch.allclose(centroids, new_c):
            break
        centroids = new_c
    return centroids


@torch.no_grad()
def synth_orig(image_01):
    eps = 1e-7; device = image_01.device
    _, _, D, H, W = image_01.shape; N = D * H * W
    img = image_01[0, 0].float(); flat = img.reshape(-1)
    flat_m = (flat > S.DARK_THRESHOLD).float()
    if flat_m.sum() < 4 or torch.rand(1, device=device).item() < S.SKIP_PARCELLATION_PROB:
        b_mean = (flat * flat_m).sum() / flat_m.sum().clamp(min=1)
        mu = torch.rand(1, device=device).item(); alpha = S._signed_alpha(device)
        synth = (mu + alpha * (flat - b_mean)).clamp(0, 1) * flat_m
    else:
        C = S.C_CHOICES[int(torch.rand(1, device=device).item() * len(S.C_CHOICES))]
        idx = torch.randint(0, N, (min(N, 40_000),), device=device)
        samp = flat[idx]; sub_fg = samp[samp > S.DARK_THRESHOLD][:S.N_KMEANS_SUBSAMPLE]
        if sub_fg.numel() < 4: sub_fg = samp[:S.N_KMEANS_SUBSAMPLE]
        centroids = _kmeans_orig(sub_fg, C)
        sorted_c, sort_idx = torch.sort(centroids)
        boundaries = (sorted_c[:-1] + sorted_c[1:]) / 2.0
        lbl_s = torch.bucketize(flat, boundaries); lbl_l = sort_idx[lbl_s].long()
        s_c = torch.zeros(C, device=device).scatter_add_(0, lbl_l, flat * flat_m)
        n_c = torch.zeros(C, device=device).scatter_add_(0, lbl_l, flat_m)
        mean_c = s_c / n_c.clamp(min=eps)
        mu_c = torch.rand(C, device=device); mag_c = torch.rand(C, device=device) * 1.5 + 0.5
        sign_c = (torch.rand(C, device=device) > 0.5).float() * 2 - 1; alp_c = mag_c * sign_c
        synth = (mu_c[lbl_l] + alp_c[lbl_l] * (flat - mean_c[lbl_l])).clamp(0, 1) * flat_m
    synth_01 = synth.reshape(1, 1, D, H, W)
    sigma = random.choice(S.BLUR_SIGMAS)
    if sigma > 0.0:
        synth_01 = S._gaussian_blur_3d(synth_01, sigma); synth = synth_01.reshape(-1)
    b_mean = (synth * flat_m).sum() / flat_m.sum().clamp(min=1)
    b_std = (((synth - b_mean) * flat_m).pow(2).sum() / flat_m.sum().clamp(min=1) + eps).sqrt()
    return ((synth - b_mean) / b_std * flat_m).reshape(1, 1, D, H, W)


def load_vol():
    import blosc2
    f = sorted((PREP / "nnUNetPlans_3d_fullres").glob("*_T1w.b2nd"))[0]
    data = np.asarray(blosc2.open(urlpath=str(f), mode="r"))
    d = torch.from_numpy(np.ascontiguousarray(data[0:1][None])).float().cuda()
    vmin, vmax = d.min(), d.max()
    return ((d - vmin) / (vmax - vmin + 1e-7)).clamp(0, 1)


def dist_stats(fn, img, n=200, seed0=0):
    # accumulate per-draw foreground mean/std and a global intensity histogram
    means, stds = [], []
    hist = torch.zeros(50, device=img.device)
    fg = (img.reshape(-1) > S.DARK_THRESHOLD)
    for i in range(n):
        random.seed(seed0 + i); torch.manual_seed(seed0 + i)
        z = fn(img).reshape(-1)[fg]
        means.append(z.mean().item()); stds.append(z.std().item())
        hist += torch.histc(z, bins=50, min=-5, max=5)
    hist /= hist.sum()
    return np.array(means), np.array(stds), hist.cpu().numpy()


def main():
    img = load_vol()
    print("verifying distribution equivalence (200 draws each)...")
    mo, so, ho = dist_stats(synth_orig, img, seed0=1000)
    mn, sn, hn = dist_stats(S.synthesize_volume_fast_zonly if hasattr(S, "synthesize_volume_fast_zonly") else (lambda x: S.synthesize_volume_fast(x)[0]), img, seed0=1000)
    print(f"  per-draw fg-mean : orig {mo.mean():+.4f}±{mo.std():.4f}   new {mn.mean():+.4f}±{mn.std():.4f}")
    print(f"  per-draw fg-std  : orig {so.mean():.4f}±{so.std():.4f}   new {sn.mean():.4f}±{sn.std():.4f}")
    l1 = np.abs(ho - hn).sum()
    print(f"  histogram L1 distance (0=identical, 2=disjoint): {l1:.4f}")
    print(f"  => {'PASS: distributions match' if l1 < 0.05 else 'CHECK: histograms differ'}")

    # timing without manual sync between calls — sync-free version should not stall
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(100):
        random.seed(i); torch.manual_seed(i)
        _ = S.synthesize_volume_fast(img)
    torch.cuda.synchronize()
    print(f"\n  new synth: {(time.perf_counter()-t0)/100*1000:.2f} ms/call (queued, 1 sync at end)")


if __name__ == "__main__":
    main()


def noise_floor():
    """L1 between two independent orig sample-sets = the sampling-noise floor."""
    img = load_vol()
    _, _, ho1 = dist_stats(synth_orig, img, n=200, seed0=1000)
    _, _, ho2 = dist_stats(synth_orig, img, n=200, seed0=5000)
    _, _, hn = dist_stats(lambda x: S.synthesize_volume_fast(x)[0], img, n=200, seed0=9000)
    print(f"orig-vs-orig L1 (noise floor): {np.abs(ho1-ho2).sum():.4f}")
    print(f"orig-vs-new  L1             : {np.abs(ho1-hn).sum():.4f}")


if __name__ != "__main__":
    pass
