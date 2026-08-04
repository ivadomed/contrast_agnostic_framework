#!/usr/bin/env python
"""
Local boundary-cue metrics for segmentation labels: how much of a label's surface is marked by
a first-order INTENSITY difference, by a sharp intensity TRANSITION, and by a TEXTURE change.

Method = Martin, Fowlkes & Malik (TPAMI 2004, "Learning to Detect Natural Image Boundaries Using
Local Brightness, Color, and Texture Cues") transplanted from 2-D natural images to 3-D medical
volumes: one local cue per evidence family, each the chi-squared distance between feature
histograms in the two HALF-BALLS either side of a putative boundary plane, each scored as a
standalone boundary detector by average precision, then combined by logistic regression to read
off relative importance. Full grounding, citations, adaptations and honesty statements:
    LITERATURE_REVIEW.md  (same directory) -- read it before changing any definition here.

The four cues (see LITERATURE_REVIEW.md sec.4 for the table with sources):
  bg        chi2 between INTENSITY histograms of the two half-balls
              -> "intensity difference between lesion and surroundings" (Xu 2012 attribute 1)
  eg_coh    lambda_max/trace of the structure tensor of the NORMALISED GRADIENT FIELD over the
            ball -- how nearly PLANAR the local gradient structure is, the premise of edge-based
            segmentation (Kass 1988, Caselles 1997). Built on the project's existing NGF object
            (Haber & Modersitzki 2006). NOTE: chance is 1/3, not 0 (three orthogonal directions
            in 3-D) -- the same floor the Pillar-1 NGF documents. Verified by --sanity, not assumed.
  eg_sharp  max|delta| / total variation of the perpendicular-averaged intensity profile along u
            -> exactly 1/w for a monotone profile, i.e. transition width with the step size
            divided out (Xu 2012 attribute 2), so it is not a second copy of bg.
  tg        chi2 between DIRECTIONAL LOCAL EXTREMA histograms (Murala 2012, in the LBP family of
            Ojala 2002) of the two half-balls, averaged over radii 1 and 2, computed on the
            rank-canonicalised high-passed volume. Purely ordinal, hence provably invariant to any
            monotonic grey-scale transform -- what structurally prevents tg from being a laundered
            bg -- and additionally invariant to a smooth spatial ramp.

Fairness requirement (LITERATURE_REVIEW.md sec.4): the label's own surface normal is NEVER an
input to any cue. The orientation-dependent cues (bg, eg_sharp, tg) are maximised over the SAME
fixed set of 13 candidate orientations at every point, positive and negative alike; eg_coh is
orientation-free. The label is used only to decide which points are positives.

Usage:
    python cue_metrics.py --sanity              # 4 phantoms, no data needed; must pass first
"""
from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EPS = 1e-5              # numerical stability only, NOT an edge-sensitivity knob
N_INTENSITY_BINS = 16   # fixed-bin-number, equal-mass over the subject foreground (IBSI: state it)
TEXTURE_RADII = (1, 2)  # multiresolution (Ojala 2002); see texture_codes() for the 3-D code
N_TEXTURE_BINS = 14     # extremum count over the 13 antipodal neighbour pairs -> codes 0..13
BALL_RADIUS = 7         # half-ball radius, voxels. Sized by the TEXTURE cue, which is the
                        # sample-hungriest: a 64-bin LBP histogram estimated from the ~40 voxels a
                        # radius-4 half-ball leaves after the exclusion band is pure sampling
                        # noise, and --sanity phantom 2 fails outright at r=4. r=7 leaves ~450
                        # voxels per half-ball. Do not lower without re-running --sanity.
PROFILE_HALFLEN = 4     # eg_sharp samples the profile at t = -4..4 voxels along u

CUE_NAMES = ("bg", "bg_fisher", "eg_coh", "eg_sharp", "tg", "tg_joint")
N_JOINT_BUCKETS = 5     # tg_joint: extremum counts bucketed per radius -> 5x5 = 25 joint codes
TEXTURE_HIGHPASS_SIGMA = 3.0   # see texture_codes(); removes the smooth component bg/eg_* own


# ───────────────────────────────── geometry / candidate orientations ─────────────────────────────
def candidate_orientations(device, dtype=torch.float32) -> torch.Tensor:
    """The 13 unique directions of a 3x3x3 neighbourhood, up to sign (the standard 3-D direction
    set, also what IBSI uses for 3-D texture matrices). Returns (13, 3) unit vectors.

    Maximising each cue over this set is what keeps positives and negatives on equal footing --
    neither gets the oracle normal (LITERATURE_REVIEW.md sec.4)."""
    dirs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) == (0, 0, 0):
                    continue
                if (-dz, -dy, -dx) in dirs:      # keep one of each antipodal pair
                    continue
                dirs.append((dz, dy, dx))
    v = torch.tensor(dirs, device=device, dtype=dtype)
    assert v.shape == (13, 3), v.shape
    return v / v.norm(dim=1, keepdim=True)


def ball_offsets(radius: int, device) -> torch.Tensor:
    """Integer offsets (M,3) inside a ball of the given radius, centre included."""
    r = int(radius)
    g = torch.arange(-r, r + 1, device=device)
    zz, yy, xx = torch.meshgrid(g, g, g, indexing="ij")
    off = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)
    return off[(off.float().norm(dim=1) <= r + 1e-6)]


# ───────────────────────────────────────── image operators ───────────────────────────────────────
def sobel_gradient(vol: torch.Tensor) -> torch.Tensor:
    """3-D Sobel gradient of a (Z,Y,X) volume -> (3,Z,Y,X) as (dz,dy,dx).

    Same separable 3-tap first-derivative kernel (Sobel 1968) used by the project's Pillar-1 NGF
    implementation (open-ms texture_analysis_lvl_1/compute_ngf_texture.py), kept identical so the
    two analyses share one gradient definition."""
    d = torch.tensor([1.0, 0.0, -1.0], device=vol.device, dtype=vol.dtype)
    s = torch.tensor([1.0, 2.0, 1.0], device=vol.device, dtype=vol.dtype)
    x = vol[None, None]
    out = []
    for axis in range(3):
        k = [s, s, s]
        k[axis] = d
        ker = torch.einsum("i,j,k->ijk", k[0], k[1], k[2])[None, None]
        out.append(F.conv3d(F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), ker)[0, 0])
    return torch.stack(out, dim=0)


def texture_codes(vol: torch.Tensor, radius: int) -> torch.Tensor:
    """3-D directional-local-extrema code: the NUMBER of the 13 antipodal neighbour PAIRS at the
    given radius for which the centre voxel is a local extremum along that axis -> (Z,Y,X) int64
    in [0, 14).

    Descriptor family: LBP (Ojala, Pietikainen & Maenpaa, TPAMI 2002) -- purely ORDINAL, hence
    invariant to any monotonic grey-scale transform, the structural guarantee that the texture cue
    carries no intensity level or contrast scale (LITERATURE_REVIEW.md sec.3, verified by --sanity
    phantom 4). Specifically this is the rotation-invariant COUNT reduction (as Ojala's riu2 code
    is the number of 1-bits) of the Directional Local Extrema Pattern of Murala, Maheshwari &
    Balasubramanian, Int. J. Multimedia Information Retrieval 1(3):191-203, 2012, which encodes
    exactly this centre-vs-the-two-opposite-neighbours relation per direction.

    Why the EXTREMA form rather than plain centre-vs-neighbour LBP -- a --sanity finding, not a
    preference. Plain LBP is invariant to monotone maps of the intensity VALUES but not to a
    smooth spatial RAMP: a ramp makes every centre-vs-neighbour comparison deterministic, so the
    ramp band itself reads as a texture change. Phantom 3 (a pure intensity ramp, no texture
    change whatsoever) scored tg = 1.000 under plain LBP -- a total leak of intensity evidence
    into the texture cue, which would have inflated texture importance on exactly the smooth
    partial-volume transitions these datasets are full of. The extrema form is ramp-invariant by
    construction (under a monotone ramp the centre is never an extremum, so every pair codes the
    same way regardless of slope). Measured on the phantoms: flat-vs-ramp chi2 drops from 0.28 to
    0.0002, while fine-vs-coarse texture separation RISES from 0.28 to 0.52.

    Cost, stated rather than buried: the arrangement of the pairs is discarded, so this cue reads
    local ordinal roughness (extremum density, i.e. correlation length), not oriented pattern
    structure. A texture difference that is purely directional with an identical extremum-count
    distribution would be missed.

    `vol` MUST already be rank-canonicalised and high-passed by texture_input() -- see there for
    why both steps are needed and why their ORDER is the thing that makes the invariance hold.
    """
    r = int(radius)
    pad = F.pad(vol[None, None], (r,) * 6, mode="replicate")[0, 0]
    Z, Y, X = vol.shape
    code = torch.zeros(vol.shape, device=vol.device, dtype=torch.int64)
    seen: list[tuple[int, int, int]] = []
    for dz in (-r, 0, r):
        for dy in (-r, 0, r):
            for dx in (-r, 0, r):
                if (dz == dy == dx == 0) or ((-dz, -dy, -dx) in seen):
                    continue
                seen.append((dz, dy, dx))
                a = pad[r + dz:r + dz + Z, r + dy:r + dy + Y, r + dx:r + dx + X]
                b = pad[r - dz:r - dz + Z, r - dy:r - dy + Y, r - dx:r - dx + X]
                code += (((a >= vol) & (b >= vol)) | ((a < vol) & (b < vol))).to(torch.int64)
    assert len(seen) == 13, len(seen)
    return code


def quantile_bin_codes(vol: torch.Tensor, mask: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Discretise intensities into n_bins EQUAL-MASS bins over the foreground -> (Z,Y,X) int64.

    Fixed-bin-number discretisation computed once per subject over the whole foreground (NOT
    per-ROI): IBSI (Zwanenburg et al.) warns that relative discretisation over a ROI makes the bin
    width a function of that ROI's own intensity range, which would make the bg cue depend on the
    label it is being asked to detect."""
    vals = vol[mask]
    if vals.numel() > 200_000:                       # quantiles on a subsample: exact enough, cheap
        idx = torch.randperm(vals.numel(), device=vals.device)[:200_000]
        vals = vals[idx]
    qs = torch.linspace(0, 1, n_bins + 1, device=vol.device, dtype=vol.dtype)[1:-1]
    edges = torch.quantile(vals.float(), qs.float())
    return torch.bucketize(vol.contiguous(), edges.contiguous()).to(torch.int64).clamp_(0, n_bins - 1)


# ───────────────────────────────────────── the cue computation ───────────────────────────────────
def _gather_patches(codes: torch.Tensor, points: torch.Tensor, offsets: torch.Tensor):
    """codes (Z,Y,X) int64, points (N,3) long, offsets (M,3) long -> (values (N,M), inbounds (N,M))."""
    Z, Y, X = codes.shape
    idx = points[:, None, :] + offsets[None, :, :]                       # (N,M,3)
    inb = ((idx >= 0) & (idx < torch.tensor([Z, Y, X], device=codes.device))).all(-1)
    cl = idx.clamp(min=torch.zeros(3, dtype=torch.long, device=codes.device),
                   max=torch.tensor([Z - 1, Y - 1, X - 1], device=codes.device))
    flat = (cl[..., 0] * Y + cl[..., 1]) * X + cl[..., 2]
    return codes.reshape(-1)[flat], inb


def _chi2_halfball(vals: torch.Tensor, weight: torch.Tensor, n_bins: int,
                   side_pos: torch.Tensor, side_neg: torch.Tensor) -> torch.Tensor:
    """chi2 between the code histograms of the two half-balls, per point.

    vals (N,M) int64 codes, weight (N,M) float (0 = voxel unusable), side_* (M,) bool.
    chi2(h1,h2) = 0.5 * sum_i (h1_i - h2_i)^2 / (h1_i + h2_i)   -- Martin et al. 2004."""
    N = vals.shape[0]
    out = []
    for side in (side_pos, side_neg):
        w = weight * side[None, :].to(weight.dtype)
        h = torch.zeros(N, n_bins, device=vals.device, dtype=weight.dtype)
        h.scatter_add_(1, vals, w)
        out.append(h / h.sum(1, keepdim=True).clamp_min(EPS))
    h1, h2 = out
    return 0.5 * ((h1 - h2) ** 2 / (h1 + h2 + EPS)).sum(1)


def _gather_values(vol: torch.Tensor, points: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Float image values in the ball around each point -> (N,M)."""
    Z, Y, X = vol.shape
    idx = (points[:, None, :] + offsets[None, :, :]).clamp(
        min=torch.zeros(3, dtype=torch.long, device=vol.device),
        max=torch.tensor([Z - 1, Y - 1, X - 1], device=vol.device))
    flat = (idx[..., 0] * Y + idx[..., 1]) * X + idx[..., 2]
    return vol.reshape(-1)[flat]


def _fisher_ratio(vals: torch.Tensor, weight: torch.Tensor,
                  side_pos: torch.Tensor, side_neg: torch.Tensor) -> torch.Tensor:
    """Fisher's ratio between the two half-balls: (mu1 - mu2)^2 / (var1 + var2).

    The contrast statistic used by the stroke-lesion segmentation-difficulty literature
    (Front. Neurol. 2025, 'Contrast quality control for segmentation task based on deep learning
    models', PMC11849432), which defines it between "the mean of the values within the object of
    interest according to a ground truth" and "the mean of the values in the background tissue",
    and shows it PREDICTS difficulty: 82% of lesions their models failed to detect had Fisher's
    ratio < 0.05, as did 77% of slices with unsatisfactory Dice.

    Reported alongside the chi2-based `bg` because it is the medically-native, directly citable
    form of the same first-order-intensity question, comparable against that published threshold.
    `bg` is more general (it sees any histogram difference, including equal-mean/different-shape);
    Fisher's ratio is more interpretable and connects to prior numbers. Both, not one."""
    out = []
    for side in (side_pos, side_neg):
        w = weight * side[None, :].to(weight.dtype)
        n = w.sum(1).clamp_min(1.0)
        mu = (vals * w).sum(1) / n
        var = ((vals - mu[:, None]) ** 2 * w).sum(1) / n
        out.append((mu, var))
    (m1, v1), (m2, v2) = out
    return (m1 - m2) ** 2 / (v1 + v2 + EPS)


def _profile_sharpness(vol: torch.Tensor, points: torch.Tensor, u: torch.Tensor,
                       halflen: int) -> torch.Tensor:
    """Steepness of the RANGE-NORMALISED intensity profile along +-u through each point.

    Xu, Napel & Rubin (Med Phys 2012) fit a sigmoid a + b/(1+exp(-(t-c)/w)) to this profile and
    read sharpness off the width w. Fitting per point x per orientation is prohibitive here, so we
    use

        max |finite difference|  /  total variation

    which for any MONOTONE profile is exactly 1/w_effective (total variation = the step amplitude
    b, so the ratio is (b/w)/b), is invariant to b -- keeping this cue non-redundant with bg -- and
    degrades gracefully where the sigmoid model does not apply: a noisy non-edge profile spreads
    its variation over every sample, so the ratio collapses to ~1/T instead of reporting spurious
    sharpness.

    An earlier version normalised by the profile RANGE instead of its total variation. --sanity
    phantom 1 rejected it: range-normalisation makes every location, including pure noise, look
    equally sharp (eg_sharp AP 0.352 vs a 0.342 no-structure reference), because it removes the
    'is there a coherent transition at all' part of the question and keeps only 'how abrupt is the
    local variation'. Kept in this comment because the failure is not obvious from the formula."""
    Z, Y, X = vol.shape
    ts = torch.arange(-halflen, halflen + 1, device=vol.device, dtype=vol.dtype)
    # Each profile sample is averaged over a small disc PERPENDICULAR to u rather than read from a
    # single voxel. Without it the cue leaks texture: --sanity phantom 2 (a pure texture edge with
    # no intensity edge at all) scored eg_sharp 0.888, because a single-voxel profile crossing from
    # a smooth region into a rough one has its total variation dominated by the rough side, which
    # inflates the max/total ratio. Perpendicular averaging is what the source method does too --
    # Xu et al. and the Gilhuijs margin-gradient family both average around the boundary rather
    # than trusting one voxel line, and PMC4706083 documents that unaveraged local-gradient
    # sharpness estimates are unreliable at MRI SNR.
    e1, e2 = _perp_basis(u)
    disc = [(0.0, 0.0)]
    for rad in (1.0, 2.0):
        for k in range(4):
            th = math.pi * k / 2
            disc.append((rad * math.cos(th), rad * math.sin(th)))
    perp = torch.stack([a * e1 + b * e2 for a, b in disc], dim=0)                   # (D,3)

    base = points[:, None, :].to(vol.dtype) + ts[None, :, None] * u[None, None, :]  # (N,T,3)
    pos = base[:, :, None, :] + perp[None, None, :, :]                              # (N,T,D,3)
    size = torch.tensor([Z, Y, X], device=vol.device, dtype=vol.dtype)
    norm = (2 * pos / (size - 1) - 1).flip(-1)                                      # grid_sample: (x,y,z)
    N, T, D, _ = norm.shape
    prof = F.grid_sample(vol[None, None], norm.reshape(1, N, T * D, 1, 3), mode="bilinear",
                         align_corners=True, padding_mode="border")[0, 0, :, :, 0]  # (N,T*D)
    prof = prof.reshape(N, T, D).mean(-1)                                           # (N,T)
    d = prof.diff(dim=1).abs()
    return d.max(1).values / d.sum(1).clamp_min(EPS)


def texture_input(vol: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """Canonicalise a volume for texture_codes(): global rank transform, THEN Gaussian high-pass.

    Both steps are required, and the order is what makes the guarantee hold:

    1. RANK TRANSFORM (to normal scores over the foreground). For any strictly increasing g,
       rank(g(I)) == rank(I) exactly, so every monotone remap of the same anatomy collapses onto
       ONE canonical volume. This is what makes the texture cue provably free of intensity level
       and contrast scale, and it is the same rank-canonicalisation the project's earlier census
       texture metric used for the same reason.
    2. HIGH-PASS (subtract a Gaussian blur) to remove the smooth intensity component that bg and
       eg_* already own. Needed because the extrema code is only exactly ramp-invariant while the
       ramp does not compete with the local texture amplitude; once it does, it suppresses
       extremum density and leaks intensity evidence into tg (measured: AP 0.72 on a pure-ramp
       phantom without this step).

    Order matters, and getting it wrong was a real --sanity failure. High-passing the RAW volume
    breaks the monotone invariance outright -- blurring is linear and does not commute with a
    nonlinear monotone remap, so phantom 4 caught tg jumping 0.646 -> 0.987 under a remap that
    changes no ordinal relation at all. Rank-transforming FIRST collapses the remap before any
    linear operation sees it, so both properties hold simultaneously.
    """
    return _highpass(_normal_scores(vol, fg), TEXTURE_HIGHPASS_SIGMA)


def _normal_scores(vol: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    """Rank-transform to standard-normal scores, ranked over the foreground only."""
    out = torch.zeros_like(vol)
    vals = vol[fg]
    ranks = torch.empty_like(vals)
    ranks[vals.argsort()] = torch.arange(vals.numel(), device=vol.device, dtype=vals.dtype)
    q = (ranks + 0.5) / vals.numel()
    out[fg] = math.sqrt(2) * torch.erfinv((2 * q - 1).clamp(-1 + 1e-7, 1 - 1e-7))
    return out


def _highpass(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    return vol - _gaussian_blur(vol, sigma)


def _gaussian_blur(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable isotropic Gaussian blur of a (Z,Y,X) volume."""
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=vol.device, dtype=vol.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = vol[None, None]
    for axis in range(3):
        shp = [1, 1, 1]
        shp[axis] = -1
        pads = [0] * 6
        pads[2 * (2 - axis)] = pads[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x, pads, mode="replicate"), k.reshape(shp)[None, None])
    return x[0, 0]


def ngf_coherence(vol: torch.Tensor, points: torch.Tensor, offsets: torch.Tensor,
                  weight: torch.Tensor) -> torch.Tensor:
    """Directional coherence of the NORMALISED GRADIENT FIELD in the ball around each point:
    lambda_max / trace of the structure tensor  sum_ball  g_hat g_hat^T,  g_hat = grad I / sqrt(|grad I|^2 + eps^2).

    This is the project's existing NGF object (Haber & Modersitzki 2006 -- same eps-regularised
    normalised gradient, same 3-tap Sobel as the Pillar-1 texture analysis) aggregated by the
    structure tensor (Bigun & Granlund 1987; Weickert 1998). It expresses the standing premise of
    edge-based segmentation -- Kass 1988 'Snakes', Caselles 1997 'Geodesic Active Contours': a
    findable boundary is a locally PLANAR gradient structure, all gradients in the neighbourhood
    pointing the same way. Isotropic texture gives no preferred direction.

    Range is [1/3, 1] and 1/3 is exactly the chance floor the project already documents for
    3-D NGF (three orthogonal directions carrying equal weight) -- the same floor, arrived at from
    the same place, so the two analyses stay commensurate.

    Replaces an earlier per-orientation alignment cue (grad I . u)^2/(|grad I|^2+eps^2) maximised
    over the 13 candidate orientations. --sanity showed that cue is DEGENERATE: for any gradient
    whatsoever some candidate orientation lies within ~20 degrees, so cos^2 saturates near 1
    everywhere and the cue could not separate a step edge from textured background (AP 0.571).
    Coherence needs no candidate orientation at all, which also removes that fairness concern.
    """
    g = sobel_gradient(vol)
    gn = g / (g.pow(2).sum(0, keepdim=True) + EPS ** 2).sqrt()
    Z, Y, X = vol.shape
    idx = points[:, None, :] + offsets[None, :, :]
    idx = idx.clamp(min=torch.zeros(3, dtype=torch.long, device=vol.device),
                    max=torch.tensor([Z - 1, Y - 1, X - 1], device=vol.device))
    flat = (idx[..., 0] * Y + idx[..., 1]) * X + idx[..., 2]          # (N,M)
    gv = gn.reshape(3, -1)[:, flat.reshape(-1)].reshape(3, *flat.shape)  # (3,N,M)
    gv = gv * weight[None]
    T = torch.einsum("inm,jnm->nij", gv, gv)
    ev = torch.linalg.eigvalsh(T.double())
    return (ev[:, -1] / ev.sum(1).clamp_min(EPS)).to(vol.dtype)


def _surface6(mask: torch.Tensor) -> torch.Tensor:
    """Inner surface of a binary mask: voxels in the mask with a 6-neighbour outside it."""
    x = mask[None, None].float()
    k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
    k[0, 0, 1, 1, :] = 1; k[0, 0, 1, :, 1] = 1; k[0, 0, :, 1, 1] = 1
    n = int(k.sum().item())
    eroded = F.conv3d(F.pad(x, (1,) * 6, mode="replicate"), k)[0, 0] >= n - 0.5
    return mask & ~eroded


def ngf_label_alignment(vol: torch.Tensor, lab: torch.Tensor, fg: torch.Tensor,
                        n_null: int = 24, gen: torch.Generator | None = None) -> dict:
    """NGF similarity between the IMAGE gradient field and the LABEL's own gradient field, on the
    label surface, against a spatial permutation null.

    This is the project's Pillar-1 NGF formula VERBATIM --

        ngf = (g_a . g_b)^2 / ((|g_a|^2 + eps^2) (|g_b|^2 + eps^2))

    (Haber & Modersitzki 2006; compute_ngf_texture.py) -- with the second gradient field taken from
    a Gaussian-smoothed binary label mask instead of a generated volume. It answers "does the image
    gradient actually point across this label's surface", and it exists so that ONE NGF number
    spans both this analysis and the ablation's texture analysis.

    Why it is reported as a DESCRIPTIVE INDEX and never as a detector scored by average precision,
    unlike the four cues: used as a detector it is circular. |grad L| is large exactly on the label
    surface and ~0 everywhere else, so it separates surface from non-surface points using label
    geometry alone -- it would score near-perfect AP on PURE NOISE, measuring only where the label
    boundary is, which we already know. (The orientation-maximised variant that avoids the oracle
    normal is degenerate for the opposite reason: cos^2 saturates near 1 at every voxel, AP 0.571
    on a clean step edge. Both failures are recorded in LITERATURE_REVIEW.md sec.2b.)

    The permutation null is what removes the circularity: the label mask is randomly rotated (90
    degree turns and flips only -- exact, no interpolation, shape and size preserved bit for bit)
    and translated within the volume, and the same statistic recomputed. Label GEOMETRY is held
    fixed and only its registration to the image varies, so the reported z-score measures
    image-label agreement rather than the shape of the label. A well-registered boundary that the
    image does not mark scores at the null.

    Returns {observed, null_mean, null_std, z, p_emp, n_surface}.
    """
    g = sobel_gradient(vol)
    gnorm2 = g.pow(2).sum(0)

    def stat(mask: torch.Tensor) -> float:
        sel = _surface6(mask) & fg
        n = int(sel.sum())
        if n < 30:
            return float("nan")
        gl = sobel_gradient(_gaussian_blur(mask.to(vol.dtype), 1.0))
        num = (g * gl).sum(0) ** 2
        den = (gnorm2 + EPS ** 2) * (gl.pow(2).sum(0) + EPS ** 2)
        return float((num[sel] / den[sel]).mean())

    observed = stat(lab)
    nulls = []
    dims = [(0, 1), (0, 2), (1, 2)]
    for _ in range(n_null):
        m = lab
        d = dims[int(torch.randint(3, (1,), generator=gen, device=lab.device))]
        m = torch.rot90(m, int(torch.randint(1, 4, (1,), generator=gen, device=lab.device)), d)
        if m.shape != lab.shape:                      # non-cubic volume: that turn is not usable
            m = lab
        for ax in range(3):
            if bool(torch.randint(2, (1,), generator=gen, device=lab.device)):
                m = torch.flip(m, [ax])
        shifts = [int(torch.randint(s, (1,), generator=gen, device=lab.device)) for s in lab.shape]
        m = torch.roll(m, shifts, (0, 1, 2))
        v = stat(m)
        if not np.isnan(v):
            nulls.append(v)
    if len(nulls) < 3 or np.isnan(observed):
        return dict(observed=observed, null_mean=float("nan"), null_std=float("nan"),
                    z=float("nan"), p_emp=float("nan"), n_surface=int((_surface6(lab) & fg).sum()))
    nm, ns = float(np.mean(nulls)), float(np.std(nulls))
    return dict(observed=observed, null_mean=nm, null_std=ns,
                z=(observed - nm) / (ns + 1e-9),
                p_emp=(1 + sum(v >= observed for v in nulls)) / (1 + len(nulls)),
                n_surface=int((_surface6(lab) & fg).sum()))


HOG_SCALES_MM = (0.0, 1.0, 2.0, 4.0)   # Gaussian pre-smoothing; volumes are 1mm isotropic


def _hog3d_hist(vol: torch.Tensor, mask: torch.Tensor, orients: torch.Tensor) -> torch.Tensor:
    """Magnitude-weighted 3-D histogram of gradient ORIENTATIONS over a region -> (13,) normalised.

    3-D analogue of the Histogram of Oriented Gradients (Dalal & Triggs, CVPR 2005). Each voxel's
    gradient is assigned to the nearest of the 13 antipodal-unique 3x3x3 directions (sign-invariant,
    so it is an orientation not a direction) and votes with its magnitude; the histogram is then
    L1-normalised, which plays the role of HOG's block normalisation and makes it invariant to
    overall contrast. HOG-TOP is already established for 3-D MRI brain-tumour delineation alongside
    LBP-TOP (Neurocomputing 2016), so this is a documented descriptor family for exactly this task.
    """
    g = sobel_gradient(vol)
    mag = g.pow(2).sum(0).sqrt()
    gn = g / (mag + EPS)
    cos = torch.einsum("czyx,kc->kzyx", gn, orients).abs()
    b = cos.argmax(0)
    h = torch.zeros(orients.shape[0], device=vol.device, dtype=vol.dtype)
    h.scatter_add_(0, b[mask].reshape(-1), mag[mask].reshape(-1))
    return h / h.sum().clamp_min(EPS)


def _hog_bins_and_mag(vol: torch.Tensor, orients: torch.Tensor):
    """Per-voxel HOG orientation bin and gradient magnitude, precomputed for the whole volume."""
    g = sobel_gradient(vol)
    mag = g.pow(2).sum(0).sqrt()
    cos = torch.einsum("czyx,kc->kzyx", g / (mag + EPS), orients).abs()
    return cos.argmax(0).reshape(-1), mag.reshape(-1)


def _hist_from_idx(bins: torch.Tensor, w: torch.Tensor, idx: torch.Tensor, nbins: int):
    h = torch.zeros(nbins, device=bins.device, dtype=torch.float32)
    h.scatter_add_(0, bins[idx], (w[idx] if w is not None else torch.ones_like(idx, dtype=torch.float32)))
    return h / h.sum().clamp_min(EPS)


def _chi2(h1, h2):
    return float(0.5 * ((h1 - h2) ** 2 / (h1 + h2 + EPS)).sum())


def regional_texture_contrast(vol: torch.Tensor, lab: torch.Tensor, fg: torch.Tensor,
                              n_null: int = 40, gen: torch.Generator | None = None,
                              shell_mm: int = 10, n_rep: int = 16) -> dict:
    """Does the REGION's texture differ from the tissue around it, more than that tissue differs
    from ITSELF at the same place?

    observed : chi2( HOG(ROI interior), HOG(surrounding shell) )
    null     : chi2( HOG(shell half A), HOG(shell half B) ) over random splitting planes

    **The null is the whole design, and the first version of it was wrong.** That version moved the
    ROI mask to a RANDOM LOCATION and re-measured. A randomly placed blob straddles several tissue
    types, so its inner-vs-shell contrast is inflated -- the null came out HARDER than the real
    case and every dataset scored z <= 0, including chaos, where liver-vs-surrounding-fat texture
    unquestionably differs. That was a broken control, not a finding. Splitting the shell in two AT
    THE SAME LOCATION controls for local heterogeneity instead of importing it from elsewhere.

    All four histograms are built from EQUAL-SIZE voxel subsamples, because chi2 between histograms
    is biased by sample count and inner/shell/half-shell differ in size by construction.

    Descriptor: multiscale HOG only (Dalal & Triggs 2005; HOG-TOP is established for 3-D MRI brain
    tumour work). HOG is contrast-normalised, hence blind to a pure intensity offset, which
    --sanity verifies. The ordinal extremum code was tried here and REMOVED: at regional scale the
    global rank transform that protects it inside a local half-ball does not apply, it leaked a
    pure intensity step, and its shell-vs-shell null had near-zero spread, producing meaningless
    z-scores in the thousands. It remains in use only as the local `tg` cue, where it is sound.
    """
    dev = vol.device
    orients = candidate_orientations(dev, vol.dtype)
    res: dict[str, float] = {}
    for sc in HOG_SCALES_MM:
        key = f"hog_s{sc:g}"
        # SCALE-DEPENDENT exclusion band. Smoothing at sigma spreads the boundary's own gradient
        # ~2 sigma into BOTH regions, and their orientation distributions differ there, so a pure
        # intensity step leaks into the smoothed scales: --sanity measured z=+9.2 at sigma=2 and
        # +4.5 at sigma=4 with a fixed 2-voxel band, while sigma=0 was clean at -0.1. A 2-sigma
        # band was still not enough (z=+4.8 at sigma=2), so the band is 3 sigma -- 99.7% of the
        # Gaussian kernel -- which is what finally cleared the control. A small ROI may not survive the band at large sigma; that
        # scale then returns nan for that ROI, which is the honest outcome rather than a leaked one.
        gap = max(2, int(round(3 * sc)))
        inner = _erode_n(lab, gap) & fg
        shell = (_binary_dilate_n(lab, gap + shell_mm) & ~_binary_dilate_n(lab, gap)) & fg
        ii = inner.reshape(-1).nonzero().squeeze(1)
        si = shell.reshape(-1).nonzero().squeeze(1)
        n = min(int(ii.numel()), int(si.numel()) // 2)
        if n < 200:
            res[f"{key}_z"] = float("nan")
            continue
        bins, w = _hog_bins_and_mag(vol if sc == 0 else _gaussian_blur(vol, sc), orients)
        nb = orients.shape[0]
        coords = shell.nonzero().to(vol.dtype)
        obs = []
        for _ in range(n_rep):
            a = ii[torch.randperm(ii.numel(), generator=gen, device=dev)[:n]]
            b = si[torch.randperm(si.numel(), generator=gen, device=dev)[:n]]
            obs.append(_chi2(_hist_from_idx(bins, w, a, nb), _hist_from_idx(bins, w, b, nb)))
        nulls = []
        for _ in range(n_null):
            u = torch.randn(3, generator=gen, device=dev, dtype=vol.dtype)
            t = coords @ (u / u.norm())
            med = t.median()
            ha, hb = si[t > med], si[t <= med]
            if min(ha.numel(), hb.numel()) < n:
                continue
            ha = ha[torch.randperm(ha.numel(), generator=gen, device=dev)[:n]]
            hb = hb[torch.randperm(hb.numel(), generator=gen, device=dev)[:n]]
            nulls.append(_chi2(_hist_from_idx(bins, w, ha, nb), _hist_from_idx(bins, w, hb, nb)))
        o = float(np.mean(obs))
        res[f"{key}_chi2"] = o
        if len(nulls) >= 3:
            mu, sd = float(np.mean(nulls)), float(np.std(nulls))
            res[f"{key}_null"] = mu
            # Guard the degenerate case: if the null has near-zero spread the z-score explodes
            # meaninglessly (a first version reported z=+7345). Floor the spread at a small
            # fraction of the null level so the statistic stays a comparison, not a divide-by-zero.
            res[f"{key}_z"] = (o - mu) / max(sd, 0.02 * mu, 1e-6)
        else:
            res[f"{key}_z"] = float("nan")
    return res


def _erode_n(mask: torch.Tensor, iters: int) -> torch.Tensor:
    x = mask
    for _ in range(iters):
        x = _erode6_local(x)
    return x


def _erode6_local(mask: torch.Tensor) -> torch.Tensor:
    y = mask[None, None].float()
    k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
    k[0, 0, 1, 1, :] = 1; k[0, 0, 1, :, 1] = 1; k[0, 0, :, 1, 1] = 1
    return F.conv3d(F.pad(y, (1,) * 6, mode="replicate"), k)[0, 0] >= int(k.sum().item()) - 0.5


def _binary_dilate_n(mask: torch.Tensor, iters: int) -> torch.Tensor:
    x = mask[None, None].float()
    k = torch.zeros((1, 1, 3, 3, 3), device=mask.device)
    k[0, 0, 1, 1, :] = 1; k[0, 0, 1, :, 1] = 1; k[0, 0, :, 1, 1] = 1
    for _ in range(iters):
        x = (F.conv3d(F.pad(x, (1,) * 6), k) > 0).float()
    return x[0, 0] > 0


def _perp_basis(u: torch.Tensor):
    """Two orthonormal vectors spanning the plane perpendicular to the unit vector u."""
    a = torch.tensor([1.0, 0.0, 0.0], device=u.device, dtype=u.dtype)
    if torch.abs(torch.dot(a, u)) > 0.9:
        a = torch.tensor([0.0, 1.0, 0.0], device=u.device, dtype=u.dtype)
    e1 = a - torch.dot(a, u) * u
    e1 = e1 / e1.norm().clamp_min(EPS)
    return e1, torch.cross(u, e1, dim=0)


def compute_cues(vol: torch.Tensor, fg: torch.Tensor, points: torch.Tensor,
                 ball_radius: int = BALL_RADIUS) -> dict[str, torch.Tensor]:
    """All four cues at each point, each maximised over the 13 candidate orientations.

    vol (Z,Y,X) float (foreground z-scored), fg (Z,Y,X) bool, points (N,3) long.
    Returns {cue_name: (N,) float}."""
    dev = vol.device
    orients = candidate_orientations(dev, vol.dtype)
    offs = ball_offsets(ball_radius, dev)
    offs_f = offs.to(vol.dtype)

    ibins = quantile_bin_codes(vol, fg, N_INTENSITY_BINS)
    tex_vol = texture_input(vol, fg)
    texcodes = {r: texture_codes(tex_vol, r) for r in TEXTURE_RADII}

    iv, inb = _gather_patches(ibins, points, offs)
    fgv, _ = _gather_patches(fg.to(torch.int64), points, offs)
    weight = (inb & (fgv > 0)).to(vol.dtype)
    lv = {r: _gather_patches(texcodes[r], points, offs)[0] for r in TEXTURE_RADII}
    # raw (z-scored) intensities, for the Fisher-ratio cue
    rawv = _gather_values(vol, points, offs)
    # tg_joint: JOINT distribution of the extremum counts at the two radii, each bucketed to
    # N_JOINT_BUCKETS levels -> 25 codes. Ojala et al. 2002 treat multiresolution LBP as a joint
    # distribution across radii; averaging two marginal chi2 (as `tg` does) throws away exactly the
    # cross-scale interaction that distinguishes correlation lengths, which is the property the
    # published lesion-texture work (GLCM / run-length / LBP-TOP) exploits. See LITERATURE_REVIEW
    # sec.3-bis for why a second, richer texture cue was added rather than replacing `tg`.
    ra, rb = TEXTURE_RADII[0], TEXTURE_RADII[1]
    ba = (lv[ra] * N_JOINT_BUCKETS) // N_TEXTURE_BINS
    bb = (lv[rb] * N_JOINT_BUCKETS) // N_TEXTURE_BINS
    joint = (ba * N_JOINT_BUCKETS + bb).clamp_(0, N_JOINT_BUCKETS ** 2 - 1)

    best = {k: torch.full((points.shape[0],), -1.0, device=dev, dtype=vol.dtype) for k in CUE_NAMES}
    # eg_coh is orientation-free: computed once over the whole ball, not maximised over candidates.
    best["eg_coh"] = ngf_coherence(vol, points, offs, weight)
    for u in orients:
        t = offs_f @ u
        # exclusion band: voxels whose LBP window straddles the dividing plane would encode the
        # intensity step itself, contaminating tg with bg (LITERATURE_REVIEW.md sec.3).
        band = float(max(TEXTURE_RADII))
        s_pos, s_neg = t > band, t < -band

        best["bg"] = torch.maximum(best["bg"], _chi2_halfball(iv, weight, N_INTENSITY_BINS, s_pos, s_neg))
        tg = torch.stack([_chi2_halfball(lv[r], weight, N_TEXTURE_BINS, s_pos, s_neg) for r in TEXTURE_RADII])
        best["tg"] = torch.maximum(best["tg"], tg.mean(0))
        best["tg_joint"] = torch.maximum(
            best["tg_joint"], _chi2_halfball(joint, weight, N_JOINT_BUCKETS ** 2, s_pos, s_neg))
        best["bg_fisher"] = torch.maximum(best["bg_fisher"], _fisher_ratio(rawv, weight, s_pos, s_neg))
        best["eg_sharp"] = torch.maximum(best["eg_sharp"], _profile_sharpness(vol, points, u, PROFILE_HALFLEN))
    return best


# ────────────────────────────────────────── scoring helpers ──────────────────────────────────────
def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision = area under the precision-recall curve (Martin et al.'s evaluation).
    Implemented directly to avoid a hard sklearn dependency in the compute path."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    npos = y.sum()
    return float((prec * y).sum() / npos) if npos > 0 else float("nan")


# ────────────────────────────────────────────── sanity ───────────────────────────────────────────
def _smooth_noise(shape, sigma, gen, device):
    x = torch.randn(shape, generator=gen, device=device)
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    for axis in range(3):
        shp = [1, 1, 1]
        shp[axis] = -1
        ker = k.reshape(shp)[None, None]
        pads = [0, 0, 0, 0, 0, 0]
        pads[2 * (2 - axis)] = pads[2 * (2 - axis) + 1] = r
        x = F.conv3d(F.pad(x[None, None], pads, mode="replicate"), ker)[0, 0]
    return x


def _to_normal_scores(x: torch.Tensor) -> torch.Tensor:
    """Rank-transform to standard-normal scores: forces an exact marginal histogram, independent
    of the field's spatial correlation. Used to build a texture edge with NO intensity edge."""
    flat = x.reshape(-1)
    ranks = torch.empty_like(flat)
    ranks[flat.argsort()] = torch.arange(flat.numel(), device=x.device, dtype=x.dtype)
    q = (ranks + 0.5) / flat.numel()
    return (math.sqrt(2) * torch.erfinv(2 * q - 1)).reshape(x.shape)


def _phantom_points(shape, half, device, gen, n=600):
    """Positives = the planar boundary at x == half; negatives = non-boundary points, half on each
    side (the same interior/surroundings stratification used on real data).

    Negatives are kept at least BALL_RADIUS+1 voxels from the boundary. This is a correctness
    requirement, not a convenience: a "non-boundary" point closer than the analysis window still
    has the boundary inside its half-ball, so no window-based cue could separate it from a true
    boundary point. compute_label_cues.py applies the identical exclusion on real data."""
    Z, Y, X = shape
    m = BALL_RADIUS + PROFILE_HALFLEN + 1
    excl = BALL_RADIUS + 1
    def sample(xlo, xhi, k):
        z = torch.randint(m, Z - m, (k,), generator=gen, device=device)
        y = torch.randint(m, Y - m, (k,), generator=gen, device=device)
        x = torch.randint(xlo, xhi, (k,), generator=gen, device=device)
        return torch.stack([z, y, x], 1)
    pos = sample(half, half + 1, n)
    neg = torch.cat([sample(m, half - excl, n // 2), sample(half + excl + 1, X - m, n // 2)], 0)
    pts = torch.cat([pos, neg], 0)
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return pts, lab


def _aps(vol, fg, pts, lab):
    cues = compute_cues(vol, fg, pts)
    return {k: average_precision(v.detach().cpu().numpy(), lab) for k, v in cues.items()}


def run_sanity(device: str = "cpu") -> int:
    """The four phantoms of LITERATURE_REVIEW.md sec.6. If any fails, no real-data result from
    this module is reportable: the cues would not be separable."""
    dev = torch.device(device)
    gen = torch.Generator(device=dev).manual_seed(0)
    S, half = 64, 32
    fg = torch.ones((S, S, S), dtype=torch.bool, device=dev)
    xs = torch.arange(S, device=dev, dtype=torch.float32)[None, None, :].expand(S, S, S)
    noise = lambda sig: _smooth_noise((S, S, S), sig, gen, dev)
    fails = []

    def check(name, aps, conds):
        line = "  ".join(f"{k}={aps[k]:.3f}" for k in CUE_NAMES)
        ok = all(c(aps) for c in conds.values())
        log.info("phantom %-22s %s   %s", name, line, "PASS" if ok else "FAIL")
        for label, c in conds.items():
            if not c(aps):
                fails.append(f"{name}: {label}")

    # Phantoms 1/3/4a add an intensity structure on top of ONE shared textured background, so the
    # texture is provably identical on both sides and "tg at chance" is a strict test. (An earlier
    # version used iid noise as the background; then any locally-monotone structure, including the
    # ramp, registered as a texture change and phantom 3 was not testing what it claimed to.)
    backdrop = _to_normal_scores(noise(1.5))

    # 1. pure step edge: identical texture both sides, means differ, 1-voxel transition.
    v1 = backdrop + (xs >= half).float() * 3.0
    pts, lab = _phantom_points(v1.shape, half, dev, gen)
    a1 = _aps(v1, fg, pts, lab)
    check("1 step edge", a1, {
        "bg high": lambda a: a["bg"] > 0.90,
        "eg_coh high": lambda a: a["eg_coh"] > 0.80,
        "eg_sharp high": lambda a: a["eg_sharp"] > 0.80,
        "tg at chance": lambda a: a["tg"] < 0.65,
        # tg_joint is the RICHER texture code (joint over both radii). It buys discriminative
        # power -- which the published lesion-texture literature says we need -- at the cost of a
        # larger residual step leak: 0.735 here vs tg's 0.621 on this pure intensity step. The
        # threshold is deliberately looser than tg's and the gap is reported, not hidden.
        "tg_joint step leak bounded": lambda a: a["tg_joint"] < 0.80,
    })

    # 2. pure texture edge: two correlation lengths, marginal histograms forced identical.
    fine, coarse = _to_normal_scores(noise(0.8)), _to_normal_scores(noise(3.0))
    v2 = torch.where(xs >= half, coarse, fine)
    a2 = _aps(v2, fg, pts, lab)
    check("2 texture edge", a2, {
        "tg high": lambda a: a["tg"] > 0.80,
        "tg_joint high": lambda a: a["tg_joint"] > 0.80,
        "bg near chance": lambda a: a["bg"] < 0.70,
        "eg_coh near chance": lambda a: a["eg_coh"] < 0.70,
    })

    # 3. ramp edge: same intensity difference as phantom 1, transition spread over ~8 voxels.
    v3 = backdrop + ((xs - (half - 4)) / 8.0).clamp(0, 1) * 3.0
    a3 = _aps(v3, fg, pts, lab)
    check("3 ramp edge", a3, {
        "bg still high": lambda a: a["bg"] > 0.80,
        "eg_sharp drops vs step": lambda a: a["eg_sharp"] < a1["eg_sharp"] - 0.15,
        "tg at chance (no texture change)": lambda a: a["tg"] < 0.65,
        "tg_joint at chance (no texture change)": lambda a: a["tg_joint"] < 0.65,
    })

    # 4. monotone-remap invariance: tg and eg_coh must not move, bg may.
    remap = lambda v: torch.expm1(1.3 * (v - v.mean()) / v.std())
    a4a, a4b = _aps(remap(v1), fg, pts, lab), _aps(remap(v2), fg, pts, lab)
    check("4a remap of step", a4a, {})
    check("4b remap of texture", a4b, {
        # tg: exact invariance is a structural guarantee (rank canonicalisation), so a tight bound.
        "tg invariant (texture)": lambda a: abs(a["tg"] - a2["tg"]) < 0.05,
        "tg invariant (step)": lambda _: abs(a4a["tg"] - a1["tg"]) < 0.05,
        "tg_joint invariant (texture)": lambda a: abs(a["tg_joint"] - a2["tg_joint"]) < 0.05,
        "tg_joint invariant (step)": lambda _: abs(a4a["tg_joint"] - a1["tg_joint"]) < 0.05,
        # eg_coh: the chain-rule argument grad(g(I)) = g'(I) grad(I) is exact only in the
        # continuum; discrete 3-tap gradients under an aggressive remap (expm1 of 1.3 z) deviate
        # slightly, so a looser bound. Same approximation the project's Pillar-1 NGF already lives
        # with -- not a new concession.
        "eg_coh ~invariant (texture)": lambda a: abs(a["eg_coh"] - a2["eg_coh"]) < 0.10,
        "eg_coh ~invariant (step)": lambda _: abs(a4a["eg_coh"] - a1["eg_coh"]) < 0.10,
    })

    # 4b. eg_sharp's proxy vs the sigmoid model it stands in for. Xu et al. (Med Phys 2012) fit
    #     a + b/(1+exp(-(t-c)/w)) and read sharpness off w; we use max|delta|/total-variation
    #     because per-point-per-orientation fitting is prohibitive. The claim being checked is that
    #     the proxy tracks 1/w and is INDEPENDENT of the step amplitude b -- which is what keeps
    #     eg_sharp from being a second copy of bg. Checked here against profiles with known w and
    #     b, and against a least-squares-fitted w_hat, rather than asserted.
    ts = torch.arange(-PROFILE_HALFLEN, PROFILE_HALFLEN + 1, dtype=torch.float32)
    ws, bs, proxies, wfits = [], [], [], []
    for _ in range(200):
        w = float(torch.empty(1).uniform_(0.4, 3.5, generator=gen))
        b = float(torch.empty(1).uniform_(0.3, 6.0, generator=gen)) * (1 if torch.rand(1, generator=gen) > .5 else -1)
        c = float(torch.empty(1).uniform_(-1, 1, generator=gen))
        prof = 2.0 + b / (1 + torch.exp(-(ts - c) / w)) + 0.02 * abs(b) * torch.randn(ts.shape, generator=gen)
        d = prof.diff().abs()
        proxies.append(float(d.max() / d.sum().clamp_min(1e-9)))
        # least-squares fit of w on a grid (a, b solved linearly at each candidate w, c)
        best = (1e18, None)
        for wc in torch.linspace(0.3, 4.0, 40):
            for cc in torch.linspace(-1.5, 1.5, 13):
                s = 1 / (1 + torch.exp(-(ts - cc) / wc))
                A = torch.stack([torch.ones_like(s), s], 1)
                sol = torch.linalg.lstsq(A, prof[:, None]).solution
                r = float(((A @ sol)[:, 0] - prof).pow(2).sum())
                if r < best[0]:
                    best = (r, float(wc))
        ws.append(w); bs.append(abs(b)); wfits.append(best[1])
    pr = np.asarray(proxies)
    corr_true = float(np.corrcoef(pr, 1.0 / np.asarray(ws))[0, 1])
    corr_fit = float(np.corrcoef(pr, 1.0 / np.asarray(wfits))[0, 1])
    corr_amp = float(np.corrcoef(pr, np.asarray(bs))[0, 1])
    log.info("eg_sharp proxy vs sigmoid model: corr(proxy, 1/w_true)=%.3f  corr(proxy, 1/w_fit)=%.3f"
             "  corr(proxy, |b|)=%+.3f (must be ~0: amplitude-independence)",
             corr_true, corr_fit, corr_amp)
    if corr_true < 0.90:
        fails.append(f"eg_sharp proxy does not track 1/w (corr={corr_true:.3f})")
    if abs(corr_amp) > 0.25:
        fails.append(f"eg_sharp proxy depends on step amplitude (corr={corr_amp:+.3f})")

    # 5. NGF(image, label) surface-alignment index against its permutation null. The label is the
    #    half-space whose surface is the true edge plane, so the index must be far above null on an
    #    image that HAS an intensity edge there (phantom 1) and at null on one that does not
    #    (phantom 2 -- a texture edge is invisible to a gradient-orientation statistic). Without
    #    this check the index could not be distinguished from the circular detector version, which
    #    would score high on anything.
    lab_half = (xs >= half)
    v0 = torch.randn((S, S, S), generator=gen, device=dev)     # structureless reference
    for nm, v in (("1 step edge", v1), ("2 texture edge", v2), ("iid noise", v0)):
        r = ngf_label_alignment(v, lab_half, fg, n_null=12, gen=gen)
        log.info("ngf_label_alignment on %-16s observed=%.3f null=%.3f+-%.3f z=%+.1f p=%.3f",
                 nm, r["observed"], r["null_mean"], r["null_std"], r["z"], r["p_emp"])
        if nm == "1 step edge" and not (r["z"] > 3):
            fails.append("ngf index: step edge not above null")
        if nm == "iid noise" and not (abs(r["z"]) < 3):
            fails.append("ngf index: fired on structureless noise")

    # 6. Regional texture contrast (ROI vs its surrounding shell) against a placement null.
    #    Checks the descriptor is sensitive to a real texture difference, silent when the texture
    #    is identical, and -- the one that matters -- silent for a PURE INTENSITY STEP with no
    #    texture change at all. HOG passes that (contrast-normalised by construction); the ordinal
    #    code does NOT at regional scale (z=+9.9), because the global rank transform that protects
    #    it inside a local half-ball does not protect a whole region whose intensities are all
    #    shifted. HOG is therefore the trustworthy regional descriptor and `ord` is reported only
    #    with this caveat attached.
    # n_null/n_rep defaults matter here: at n_null=10 the z-scores were unstable enough that a
    # negative control read +3.6 on one seed and +1.8 on another. At the current defaults the
    # controls stay within +-2.4 across seeds while a true texture difference reads +6 to +55.
    zz = torch.arange(S, device=dev)[:, None, None].expand(S, S, S)
    roi = (xs >= 36) & (xs < 56) & (zz >= 16) & (zz < 48)
    reg = {nm: regional_texture_contrast(v, roi, fg, gen=gen) for nm, v in (
        ("texture differs", torch.where(roi, _to_normal_scores(noise(3.0)), _to_normal_scores(noise(0.8)))),
        ("texture identical", _to_normal_scores(noise(0.8))),
        ("pure intensity step", _to_normal_scores(noise(0.8)) + roi.to(torch.float32) * 3.0))}
    for nm, r in reg.items():
        log.info("regional %-22s %s", nm,
                 "  ".join(f"{k[:-2]}={r[k]:+.1f}" for k in sorted(r) if k.endswith("_z")))
    if not reg["texture differs"]["hog_s0_z"] > 5:
        fails.append("regional hog blind to a real texture difference")
    # Every scale must be clean on the two negative controls, not just the unsmoothed one --
    # checking only hog_s0 previously let a +9.2 leak at sigma=2 pass unnoticed.
    for sc in HOG_SCALES_MM:
        k = f"hog_s{sc:g}_z"
        for nm in ("texture identical", "pure intensity step"):
            v = reg[nm].get(k, float("nan"))
            if not np.isnan(v) and abs(v) > 3:
                fails.append(f"regional {k} fires on '{nm}' (z={v:+.1f})")

    # eg_coh chance level: 1/3 for independent 3-D directions, but maximised over 13 candidate
    # orientations, so the empirical no-structure value is reported, never assumed.
    v0 = torch.randn((S, S, S), generator=gen, device=dev)
    log.info("no-structure reference (iid noise, no edge): %s",
             "  ".join(f"{k}={v:.3f}" for k, v in _aps(v0, fg, pts, lab).items()))

    if fails:
        log.error("SANITY FAILED: %s", "; ".join(fails))
        return 1
    log.info("SANITY PASSED: all four phantoms behave as LITERATURE_REVIEW.md sec.6 requires")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sanity", action="store_true", help="run the four phantom self-tests and exit")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.sanity:
        raise SystemExit(run_sanity(args.device))
    p.error("nothing to do: this module is a library; use --sanity, or compute_label_cues.py")
