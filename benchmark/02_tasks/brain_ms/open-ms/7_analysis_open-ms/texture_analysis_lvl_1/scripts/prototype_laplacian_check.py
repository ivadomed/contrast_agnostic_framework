#!/usr/bin/env python
"""
Prototype-and-validate check: Laplacian (second-derivative) zero-crossing sign agreement as a
SECOND, mechanistically-distinct corroborator for NGF (which is first-derivative/gradient-
direction based). Marr & Hildreth (1980), "Theory of edge detection," Proc. R. Soc. Lond. B —
edges occur at Laplacian zero-crossings; this is the classic alternative to gradient-magnitude
edge detection, still standard (e.g. LoG/DoG edge detectors).

Metric: fraction of ROI voxels where sign(Laplacian(source)) == sign(Laplacian(generated)).
Independent-field floor should be ~0.5 (coin flip) -- verify, don't assume.

Caveat to check empirically (not assumed): under an AFFINE remap g(x)=a*x+b, the chain rule gives
Laplacian(g(x)) = a*Laplacian(x) exactly, so sign is preserved (a>0) or flipped (a<0) UNIFORMLY --
clean, like NGF. Under a genuinely NONLINEAR monotonic remap (gamma, log), there is an extra
g''(x)*|grad(x)|^2 curvature term that can perturb the sign locally -- unlike NGF's gradient
DIRECTION, which chain-rule-survives ANY monotonic remap exactly. This is a real, expected
limitation (weaker invariance than NGF) -- test whether it is small enough in practice to still be
useful as a second opinion, not a replacement.
"""
import torch
import torch.nn.functional as F

LAPLACIAN_KERNEL_3D = torch.tensor([
    [[[0,0,0],[0,1,0],[0,0,0]],
     [[0,1,0],[1,-6,1],[0,1,0]],
     [[0,0,0],[0,1,0],[0,0,0]]],
], dtype=torch.float32).reshape(1,1,3,3,3)   # 6-connected discrete Laplacian


def laplacian3d(x, device):
    k = LAPLACIAN_KERNEL_3D.to(device)
    xp = F.pad(x[None,None], (1,1,1,1,1,1), mode="replicate")
    return F.conv3d(xp, k)[0,0]


def sign_agreement(src, gen, mask, device, dead_zone=1e-6):
    ls, lg = laplacian3d(src, device), laplacian3d(gen, device)
    # ignore near-zero Laplacian voxels (flat regions, sign is noise) on the SOURCE side
    valid = mask & (ls.abs() > dead_zone)
    agree = (torch.sign(ls) == torch.sign(lg))[valid]
    return float(agree.float().mean()), int(valid.sum())


def run_check(device):
    torch.manual_seed(0)
    D = 44
    def boxf(x, w):
        return F.avg_pool3d(x[None,None], w, 1, w//2, count_include_pad=False)[0,0]
    source = boxf(torch.randn(D,D,D, device=device), 5)
    source = (source - source.min())/(source.max()-source.min()+1e-7)
    mask = torch.ones(D,D,D, dtype=torch.bool, device=device)
    mask[:3]=mask[-3:]=mask[:,:3]=mask[:,-3:]=mask[:,:,:3]=mask[:,:,-3:]=False

    noise_a = torch.rand(D,D,D, device=device)
    noise_b = torch.rand(D,D,D, device=device)

    cases = {
        "identity": source.clone(),
        "gamma (nonlinear monotone)": source.clamp_min(1e-4)**2.0,
        "log (nonlinear monotone)": torch.log1p(9*source.clamp_min(0)),
        "inverted (affine, a<0)": 1.0 - source,
        "noise (vs source)": noise_a,
    }
    print(f"{'case':28s} {'agree':>8s} {'n_valid':>10s}")
    for name, gen in cases.items():
        a, n = sign_agreement(source, gen, mask, device)
        print(f"{name:28s} {a:8.3f} {n:10d}")

    a, n = sign_agreement(noise_a, noise_b, mask, device)
    print(f"{'noise_A vs noise_B':28s} {a:8.3f} {n:10d}   <- should be ~0.5 (independent) if valid floor")


if __name__ == "__main__":
    run_check(torch.device("cpu"))
