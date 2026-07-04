#!/usr/bin/env python3
"""Mechanism illustration: take ONE real BraTS T1n slice + its GT, and show what each
augmentation method's synthetic training image looks like:
  - SynthSeg: each label region -> per-region mean + white Gaussian noise (EM subdivides
    background into a few flat-noise clusters). TEXTURE DESTROYED.
  - v26: affine remap y = mu + a*(x-mean_region) per region of the REAL intensities.
    TEXTURE PRESERVED.
Saves a PNG (base64-embeddable in the artifact)."""
from pathlib import Path
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
DS = ROOT/"datasets/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n"
case = "BraTSGLI00009100"; Z = 96
img = sitk.GetArrayFromImage(sitk.ReadImage(str(DS/"imagesTr"/f"{case}_0000.nii.gz"))).astype(float)
gt  = sitk.GetArrayFromImage(sitk.ReadImage(str(DS/"labelsTr"/f"{case}.nii.gz")))
im = img[Z]; seg = gt[Z]
# normalize brain to [0,1]
brain = im>0
lo,hi = np.percentile(im[brain],1), np.percentile(im[brain],99)
im01 = np.clip((im-lo)/(hi-lo+1e-6),0,1)*brain
rng = np.random.default_rng(0)

# --- SynthSeg-style: regions = labels + EM-style background subdivision; fill mean+noise ---
ss = np.zeros_like(im01)
# subdivide background (brain & label0) by intensity into 5 clusters (EM-ish via quantile)
bgmask = brain & (seg==0)
qs = np.quantile(im01[bgmask], np.linspace(0,1,6))
bg_clusters = np.digitize(im01, qs[1:-1])  # 0..4
regions = []
for c in range(5):
    regions.append(bgmask & (bg_clusters==c))
for l in [1,2,3,4]:
    regions.append(seg==l)
for r in regions:
    if r.sum()<5: continue
    mu = rng.uniform(0.1,0.9); sd = rng.uniform(0.03,0.12)
    ss[r] = np.clip(mu + sd*rng.standard_normal(r.sum()),0,1)

# --- v26-style: affine remap per region of REAL intensities (texture preserved) ---
v26 = np.zeros_like(im01)
for r in regions:
    if r.sum()<5: continue
    x = im01[r]; m = x.mean()
    mu = rng.uniform(0.1,0.9); a = rng.choice([-1,1])*rng.uniform(0.5,2.0)
    v26[r] = np.clip(mu + a*(x-m),0,1)

# overlay GT
def overlay(base, seg):
    rgb = np.stack([base]*3,-1)
    colors={1:(0.9,0.1,0.1),2:(1,0.6,0),3:(0.1,0.8,0.1),4:(0.2,0.4,1)}
    for l,c in colors.items():
        m=seg==l
        for k in range(3): rgb[...,k][m]=0.55*rgb[...,k][m]+0.45*c[k]
    return rgb

fig,ax=plt.subplots(1,4,figsize=(16,4.4))
for a in ax: a.axis("off")
ax[0].imshow(im01,cmap="gray"); ax[0].set_title("Real T1n (input)",fontsize=13)
ax[1].imshow(overlay(im01,seg)); ax[1].set_title("Ground truth\n(red=NCR, orange=SNFH/edema,\ngreen=ET, blue=RC)",fontsize=10)
ax[2].imshow(ss,cmap="gray"); ax[2].set_title("SynthSeg synthetic\n= per-region mean + white noise\n(texture & gradients DESTROYED)",fontsize=10)
ax[3].imshow(v26,cmap="gray"); ax[3].set_title("v26 synthetic\n= affine remap of real intensities\n(texture & gradients PRESERVED)",fontsize=10)
plt.suptitle("Why the methods differ — same slice, one synthetic draw each",fontsize=14,y=1.02)
plt.tight_layout()
out=Path("/project/aip-jcohen/paulh/mri_synthesis_project/datasets/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1n/exp_texture_advantage/mechanism_illustration.png")
plt.savefig(out,dpi=110,bbox_inches="tight"); print("saved",out, out.stat().st_size,"bytes")
