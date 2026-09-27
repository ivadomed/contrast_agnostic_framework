#!/usr/bin/env python
"""
Prototype-and-reject check: does MIND (Heinrich et al. 2012, Medical Image Analysis — Modality
Independent Neighbourhood Descriptor) work as a SECOND, independent-mechanism corroborating
metric alongside NGF for THIS problem (distinguishing genuine derived texture from i.i.d.-noise
fill)?

MIND(I,x,r) = exp(-D_p(I,x,x+r) / V(I,x)) for r in a small fixed offset set (here: 6-connected
3-D neighbours), D_p = Gaussian-weighted patch SSD (approximated here via blur((I-shift(I,r))^2)),
V(I,x) = mean of D_p over the same offsets (local noise/variance estimate). Compare two images via
mean squared difference of their per-voxel MIND descriptor VECTORS; similarity = 1 - that.

Hypothesis to test (before committing engineering time to a full real-data run): for i.i.d.
isotropic noise, D_p(I,x,x+r) is expected to be roughly EQUAL across all 6 offsets (noise has no
preferred direction) -> MIND(I,x,r) ~ exp(-1) for every r, REGARDLESS of the specific noise
realization -> two INDEPENDENT noise fields could have nearly IDENTICAL (both ~flat) MIND
descriptors -> falsely HIGH similarity for the one case (SynthSeg-floor / baseline_kmeans_label_remap_voronoi) where we
most need a clean floor. Only real data at stake here is a go/no-go on adopting MIND; run the same
--sanity-style phantom battery already used for census and NGF before trusting it further.
"""
import torch
import torch.nn.functional as F

OFFSETS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
EPS = 1e-5


def gauss_blur_3d(x, sigma=1.0, radius=2):
    """Separable Gaussian blur, replicate-padded. x: (D,H,W)."""
    ax = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    k1 = torch.exp(-0.5*(ax/sigma)**2); k1 = k1/k1.sum()
    out = x[None, None]
    for dim in (2, 3, 4):                    # dims of the (1,1,D,H,W) tensor
        kshape = [1, 1, 1, 1, 1]; kshape[dim] = 2*radius+1
        kernel = k1.reshape(kshape)
        padding = [0, 0, 0, 0, 0, 0]
        idx = 4 - dim                          # F.pad order is last-dim-first
        padding[idx*2] = padding[idx*2+1] = radius
        out = F.conv3d(F.pad(out, padding, mode="replicate"), kernel)
    return out[0, 0]


def shift3d(x, offset):
    return torch.roll(x, shifts=offset, dims=(0,1,2))


def mind_descriptor(img, sigma=1.0):
    dists = []
    for off in OFFSETS:
        diff2 = (img - shift3d(img, off))**2
        dists.append(gauss_blur_3d(diff2, sigma=sigma))
    V = torch.stack(dists, 0).mean(0) + EPS
    mind = torch.stack([torch.exp(-d/V) for d in dists], 0)  # (6,D,H,W)
    return mind


def mind_similarity(src, gen, mask, sigma=1.0):
    m_src = mind_descriptor(src, sigma)
    m_gen = mind_descriptor(gen, sigma)
    ssd = ((m_src - m_gen)**2).mean(0)  # (D,H,W)
    sim = 1.0 - ssd
    return float(sim[mask].mean())


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
    noise_b = torch.rand(D,D,D, device=device)   # a SECOND, INDEPENDENT noise field

    cases = {
        "identity": source.clone(),
        "gamma": source.clamp_min(1e-4)**2.0,
        "inverted": 1.0 - source,
        "noise (vs source)": noise_a,
    }
    print(f"{'case':20s} {'mind_sim':>10s}")
    for name, gen in cases.items():
        s = mind_similarity(source, gen, mask)
        print(f"{name:20s} {s:10.3f}")

    # THE key test: two INDEPENDENT noise fields against EACH OTHER (not against source)
    s_noise_vs_noise = mind_similarity(noise_a, noise_b, mask)
    print(f"{'noise_A vs noise_B':20s} {s_noise_vs_noise:10.3f}   <- should be LOW (independent) if MIND is a valid floor")
    s_source_vs_noiseA = mind_similarity(source, noise_a, mask)
    print(f"{'source vs noise_A':20s} {s_source_vs_noiseA:10.3f}   <- the actual floor case we need")


if __name__ == "__main__":
    run_check(torch.device("cpu"))
