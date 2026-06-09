#!/usr/bin/env python
"""
Analyze contrast diversity of v19_c synthetic data.

Part 2: Empirical contrast diversity of histogram randomization.
Part 3: Theoretical chunk remapping simulation.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─── PART 2: Empirical contrast diversity ─────────────────────────────────────

print("=" * 70)
print("PART 2: Empirical contrast diversity of v19_c synthetic data")
print("=" * 70)

V19_C_ROOT = PROJECT_ROOT / "data" / "ON-Harmony" / "derivatives" / "synthetic_v19_c"

# Pick 5 subjects, first session for each
all_subjects = sorted(V19_C_ROOT.iterdir())
subjects_5 = all_subjects[:5]

print(f"\nUsing subjects: {[s.name for s in subjects_5]}")
print(f"Dataset root: {V19_C_ROOT}\n")


def compute_inversion_score(nii_path: Path):
    """Load NIfTI, reorient to RAS, compute inversion score and histogram."""
    img = nib.load(str(nii_path))
    # Reorient to RAS
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.dataobj, dtype=np.float32)

    # Foreground voxels (>0)
    fg = data[data > 0].ravel()
    if len(fg) == 0:
        return np.nan, None

    # Split into 3 tertiles
    t33 = np.percentile(fg, 33.33)
    t66 = np.percentile(fg, 66.67)

    dark = fg[fg <= t33]
    mid = fg[(fg > t33) & (fg <= t66)]
    bright = fg[fg > t66]

    dark_mean = dark.mean() if len(dark) > 0 else 0.0
    mid_mean = mid.mean() if len(mid) > 0 else 0.0
    bright_mean = bright.mean() if len(bright) > 0 else 0.0

    inversion_score = float(bright_mean - dark_mean)

    # 20-bin histogram of foreground
    hist, bin_edges = np.histogram(fg, bins=20, range=(0, 1))
    hist_norm = hist / hist.sum()

    return inversion_score, hist_norm, dark_mean, mid_mean, bright_mean


print(f"{'Subject':<12} {'Session':<20} {'Run':<6} {'InvScore':>10} {'DarkMean':>10} {'MidMean':>10} {'BrightMean':>12} {'T2w?':>6}")
print("-" * 92)

all_results = []

for subj in subjects_5:
    # Pick first session only for brevity
    sessions = sorted(subj.iterdir())
    ses = sessions[0]

    # Get all 10 runs
    nii_files = sorted(ses.glob("*.nii.gz"))

    subject_scores = []
    for nii_path in nii_files:
        run_tag = nii_path.stem.split("_run-")[1].split("_")[0] if "_run-" in nii_path.stem else "?"
        inv_score, hist_norm, dark_mean, mid_mean, bright_mean = compute_inversion_score(nii_path)
        is_t2w = "YES" if inv_score < 0 else "no"
        print(f"{subj.name:<12} {ses.name:<20} {run_tag:<6} {inv_score:>10.4f} {dark_mean:>10.4f} {mid_mean:>10.4f} {bright_mean:>12.4f} {is_t2w:>6}")
        subject_scores.append(inv_score)
        all_results.append({
            "subject": subj.name, "session": ses.name, "run": run_tag,
            "inv_score": inv_score, "dark_mean": dark_mean,
            "mid_mean": mid_mean, "bright_mean": bright_mean,
        })

    print(f"  --> {subj.name}/{ses.name} | inv_score range: [{min(subject_scores):.4f}, {max(subject_scores):.4f}] | any T2w-like (neg)? {'YES' if any(s < 0 for s in subject_scores) else 'NO'}")
    print()

all_scores = [r["inv_score"] for r in all_results]
n_t2w = sum(1 for s in all_scores if s < 0)
print(f"\nSUMMARY (Part 2):")
print(f"  Total images analyzed: {len(all_scores)}")
print(f"  Inversion score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")
print(f"  Mean inversion score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")
print(f"  Images with T2w-like inversion (score < 0): {n_t2w} / {len(all_scores)}")
print(f"  => Ever goes T2w-like? {'YES' if n_t2w > 0 else 'NO'}")

# ─── PART 3: Theoretical chunk remapping simulation ───────────────────────────

print()
print("=" * 70)
print("PART 3: Theoretical chunk remapping simulation")
print("=" * 70)

# Find any NIfTI in synthetic_v19_c for the simulation
sample_nii = None
for subj in sorted(V19_C_ROOT.iterdir())[:1]:
    for ses in sorted(subj.iterdir())[:1]:
        files = sorted(ses.glob("*.nii.gz"))
        if files:
            sample_nii = files[0]
            break

if sample_nii is None:
    print("ERROR: No NIfTI found for simulation!")
    sys.exit(1)

print(f"\nUsing sample NIfTI for simulation: {sample_nii}")

# Load and extract foreground histogram
img = nib.load(str(sample_nii))
img = nib.as_closest_canonical(img)
data = np.asarray(img.dataobj, dtype=np.float32)
fg = data[data > 0].ravel()

print(f"Foreground voxels: {len(fg):,}")
print(f"Intensity range: [{fg.min():.4f}, {fg.max():.4f}]")

# Compute reference T1w inversion score
t33_ref = np.percentile(fg, 33.33)
t66_ref = np.percentile(fg, 66.67)
dark_ref = fg[fg <= t33_ref].mean()
bright_ref = fg[fg > t66_ref].mean()
ref_inv_score = float(bright_ref - dark_ref)
print(f"Reference (T1w) inversion score: {ref_inv_score:.4f}")

# ─── Chunk remapping simulation ───────────────────────────────────────────────
# V19 parameters: 8 chunks, mu~U(0,1), alpha~U(0.5, 2)
# The chunk remapping maps each input intensity bin to a target via piecewise
# linear segments between chunk boundary values drawn from U(0,1).
# We simulate this:
#   1. Draw 8 boundary x-coords uniformly in [0,1] -> sorted -> define chunk edges
#   2. Draw 8 mu values U(0,1) -> sorted -> define output values at chunk edges
#   3. Interpolate: for each fg voxel, map its intensity via piecewise linear
#   4. Compute inversion score on the remapped values

N_SIM = 1000
N_CHUNKS = 8
RNG = np.random.default_rng(42)

print(f"\nSimulating {N_SIM} random chunk remappings (N_CHUNKS={N_CHUNKS}, params as V19)...")
print("  mu ~ U(0, 1) (sorted), alpha ~ U(0.5, 2.0)")
print()

inv_scores_sim = []

# For speed, work on a subsample of fg voxels
MAX_VOXELS = 200_000
if len(fg) > MAX_VOXELS:
    idx = RNG.choice(len(fg), size=MAX_VOXELS, replace=False)
    fg_sim = fg[idx]
    print(f"  (Subsampled to {MAX_VOXELS:,} voxels for speed)")
else:
    fg_sim = fg.copy()

for i in range(N_SIM):
    # Draw chunk x-positions (sorted), include 0 and 1 as anchors
    inner_x = np.sort(RNG.uniform(0.0, 1.0, N_CHUNKS - 1))
    chunk_x = np.concatenate([[0.0], inner_x, [1.0]])  # N_CHUNKS+1 boundaries

    # Draw mu (sorted output values at boundaries)
    inner_mu = np.sort(RNG.uniform(0.0, 1.0, N_CHUNKS - 1))
    chunk_mu = np.concatenate([[0.0], inner_mu, [1.0]])  # N_CHUNKS+1 boundary mu values

    # Apply random non-linear scaling within each chunk via alpha
    # alpha ~ U(0.5, 2) — controls local contrast
    # We apply: within each chunk segment, warp with power alpha
    # piecewise linear between chunk_x -> chunk_mu, then apply alpha warp
    alphas = RNG.uniform(0.5, 2.0, N_CHUNKS)

    # Map fg_sim intensities via piecewise linear then alpha warp
    remapped = np.zeros_like(fg_sim)

    for c in range(N_CHUNKS):
        x0, x1 = chunk_x[c], chunk_x[c + 1]
        y0, y1 = chunk_mu[c], chunk_mu[c + 1]
        a = alphas[c]

        mask = (fg_sim >= x0) & (fg_sim < x1) if c < N_CHUNKS - 1 else (fg_sim >= x0) & (fg_sim <= x1)

        if not mask.any():
            continue

        # Normalize to [0,1] within chunk
        if x1 > x0:
            t = (fg_sim[mask] - x0) / (x1 - x0)
        else:
            t = np.zeros(mask.sum())

        # Apply alpha warp: t^alpha within [0,1]
        t_warped = np.power(np.clip(t, 0, 1), a)

        # Map back to output range [y0, y1]
        remapped[mask] = y0 + t_warped * (y1 - y0)

    # Compute inversion score on remapped values
    remapped = np.clip(remapped, 0, 1)
    if len(remapped) == 0:
        continue
    t33 = np.percentile(remapped, 33.33)
    t66 = np.percentile(remapped, 66.67)
    d_mean = remapped[remapped <= t33].mean() if (remapped <= t33).any() else 0.0
    b_mean = remapped[remapped > t66].mean() if (remapped > t66).any() else 0.0
    inv_scores_sim.append(float(b_mean - d_mean))

inv_scores_sim = np.array(inv_scores_sim)
n_t2w_sim = (inv_scores_sim < 0).sum()
frac_t2w = n_t2w_sim / len(inv_scores_sim)

print(f"SIMULATION RESULTS ({N_SIM} remappings):")
print(f"  Inversion score range: [{inv_scores_sim.min():.4f}, {inv_scores_sim.max():.4f}]")
print(f"  Mean: {inv_scores_sim.mean():.4f} ± {inv_scores_sim.std():.4f}")
print(f"  Median: {np.median(inv_scores_sim):.4f}")
print(f"  Fraction T2w-like (score < 0): {n_t2w_sim}/{N_SIM} = {frac_t2w:.4f} ({frac_t2w*100:.2f}%)")
print()

# Distribution deciles
deciles = np.percentile(inv_scores_sim, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("Distribution of inversion scores (deciles):")
for p, v in zip([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], deciles):
    marker = " <-- NEGATIVE (T2w-like zone)" if v < 0 else ""
    print(f"  P{p:3d}: {v:8.4f}{marker}")

print()
# ASCII histogram
bins = np.linspace(inv_scores_sim.min(), inv_scores_sim.max(), 21)
hist_vals, bin_edges = np.histogram(inv_scores_sim, bins=bins)
print("ASCII histogram of inversion scores (simulation):")
print(f"  x-axis: [{inv_scores_sim.min():.4f}, {inv_scores_sim.max():.4f}]")
print(f"  (negative = T2w-like, positive = T1w-like)")
print()
max_bar = 40
for j in range(len(hist_vals)):
    bar_len = int(hist_vals[j] / hist_vals.max() * max_bar)
    is_neg = bin_edges[j + 1] < 0
    marker = "*" if is_neg else " "
    print(f"  {bin_edges[j]:7.4f}–{bin_edges[j+1]:7.4f} |{'#'*bar_len}{marker}")

print()
print(f"CONCLUSION: Chunk remapping CAN produce T2w-like images in {frac_t2w*100:.1f}% of random draws.")
print(f"  (Reference T1w score was {ref_inv_score:.4f}; negative score = T2w-like contrast ordering)")
print()
print("Done.")
