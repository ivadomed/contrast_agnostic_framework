#!/usr/bin/env python3
"""
Verify the ladder's rung 4.5 (FLAT fill) config BEFORE any training (2026-10-08).

Rung 4.5 = the real-fill rung 5 (PALETTE alone) with the within-region texture term removed: every region of PALETTE's
partition (k-means classes x Voronoi cells) is filled with its target mean mu_c, then PALETTE's own label step
(mu_l + alpha_l * (synth - mean_l), an affine re-remap of the CURRENT synthetic image) runs unchanged. Implemented with
the existing noise-fill class at sigma = 0 and label_fill_noise = false (config only, no AugLab code change):
    transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json

Checks (any failure -> exit 1):
  T0  pipeline: AugTransformsGPU built from the flat / rung-4 / rung-5 JSONs has the same transform list and
      probabilities except the synthesis step; the flat instance really has sigma range [0,0], label_fill_noise False,
      label_voronoi False, p 0.5, and the same partition parameters as rung 5.
  T1  invariant on real patches (BraTS T1c 4 labels, MS FLAIR 1 label, CHAOS T1in 4 labels), blur off, synthesis forced:
      the output is CONSTANT on every (step-1 region x label) cell -- step-1 regions recorded by wrapping
      _voronoi_region_ids. Discrimination: the same test must FAIL for rung 4 (noise) and rung 5 (real texture).
      Also counts how often the label step fired (label voxels differ from the rest of their region) and how often the
      whole foreground came out constant (skip-parcellation branch with no label step: z-score -> all zero).
  T2  pipeline forward (blur off): fraction of iterations whose foreground is piecewise constant ~ 0.5 (= p).
  T3  figure: one BraTS slice through rungs 4 / 4.5 / 5 (blur as configured), PNG for Paul.

Run through run_job (CPU is enough):  bash scripts/cluster/rung45_flat/verify_flatfill.sh
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
AUG = REPO / "sub-workspaces/auglab_workspace/AugLab"
sys.path.insert(0, str(AUG))
CFG = AUG / "auglab/configs"
FLAT = CFG / "transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json"
R4 = CFG / "transform_params_gpu_baseline_kmeans_label_remap_voronoi_lblvor_spatialDA_train050.json"
R5 = CFG / "transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
T = REPO / "benchmark/02_tasks"
CASES = [
    ("brats_t1c", T / "brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset054_BraTS2024GliomaT1c", "BraTSGLI02520101"),
    ("openms_flair", T / "brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR", "patient01"),
    ("chaos_t1in", T / "abdomen_healthy/chaos/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in", "MR01"),
]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "scripts/cluster/rung45_flat/_verify_out"
N_SEEDS = 40
PATCH = 96

import auglab.transforms.gpu.fromSeg as fs          # noqa: E402
import auglab.transforms.gpu.palette_noisefill as pn  # noqa: E402
from auglab.transforms.gpu.transforms import AugTransformsGPU  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def synth_of(pipe):
    s = [t for t in pipe.modules() if isinstance(t, (fs.RandomV26_6_2ContrastGPU,))]
    assert len(s) == 1, f"expected exactly one PALETTE-family transform, got {len(s)}"
    return s[0]


def describe(pipe):
    """(class name, p) of every leaf transform of the pipeline, in order."""
    out = []
    for t in pipe.children():
        p = getattr(t, "p", None)
        out.append((type(t).__name__, None if p is None else float(p)))
    return out


def load_patch(ds: Path, case: str):
    img = nib.load(str(ds / "imagesTr" / f"{case}_0000.nii.gz")).get_fdata().astype(np.float32)
    lbl = np.asarray(nib.load(str(ds / "labelsTr" / f"{case}.nii.gz")).dataobj).astype(np.int64)
    c = np.array(np.nonzero(lbl)).mean(1).round().astype(int)
    sl = []
    for ax, n in enumerate(img.shape):
        h = min(PATCH, n) // 2
        lo = int(np.clip(c[ax] - h, 0, n - 2 * h))
        sl.append(slice(lo, lo + 2 * h))
    img, lbl = img[tuple(sl)], lbl[tuple(sl)]
    return torch.from_numpy(img)[None, None], torch.from_numpy(lbl)[None, None]


class Recorder:
    """Wrap _voronoi_region_ids in a module namespace and keep the FIRST call (= step-1 partition for B=1)."""

    def __init__(self, mod):
        self.mod, self.orig, self.rid = mod, mod._voronoi_region_ids, None

    def __enter__(self):
        def wrapped(*a, **k):
            r = self.orig(*a, **k)
            if self.rid is None:
                self.rid = r[0].clone()
            return r
        self.mod._voronoi_region_ids = wrapped
        return self

    def __exit__(self, *exc):
        self.mod._voronoi_region_ids = self.orig


def cell_test(tr, mod, img, lbl, seed):
    """Run tr once (synthesis forced, blur off); return (max within-cell spread, label_fired, all_constant)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    with Recorder(mod) as rec:
        out = tr.apply_transform(img.clone(), {"seg": lbl}, {})
    o = out[0, 0].reshape(-1).double()
    flat = img[0, 0].reshape(-1).float()
    v01 = (flat - flat.min()) / (flat.max() - flat.min() + 1e-7)
    fg = v01 > tr.dark_threshold
    rid = rec.rid if rec.rid is not None else torch.zeros_like(flat, dtype=torch.long)   # skip-parcellation: one region
    l = lbl[0, 0].reshape(-1)
    key = (rid.long() * 64 + l)[fg]
    vals = o[fg]
    uk, inv = torch.unique(key, return_inverse=True)
    mx = torch.full((len(uk),), -1e30, dtype=torch.float64).scatter_reduce(0, inv, vals, "amax")
    mn = torch.full((len(uk),), 1e30, dtype=torch.float64).scatter_reduce(0, inv, vals, "amin")
    spread = float((mx - mn).max())
    # label step fired: some label voxel's value differs from the non-label voxels of its own step-1 region
    fired = False
    r_fg, l_fg = rid[fg].long(), l[fg]
    for r in torch.unique(r_fg[l_fg > 0])[:200]:
        inr = r_fg == r
        a, b = vals[inr & (l_fg > 0)], vals[inr & (l_fg == 0)]
        if len(a) and len(b) and abs(float(a.mean() - b.mean())) > 1e-6:
            fired = True
            break
    all_const = float(vals.max() - vals.min()) < 1e-9
    return spread, fired, all_const


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("== T0 pipeline construction")
    pipes = {k: AugTransformsGPU(json_path=str(p)) for k, p in (("flat", FLAT), ("r4", R4), ("r5", R5))}
    d = {k: describe(v) for k, v in pipes.items()}
    for k in d:
        print(f"  {k}: {d[k]}")
    strip = lambda L: [x for x in L if not x[0].startswith("RandomV26_6_2")]
    check(strip(d["flat"]) == strip(d["r5"]) == strip(d["r4"]), "non-synthesis transforms + probabilities identical across rungs 4 / 4.5 / 5")
    check([i for i, x in enumerate(d["flat"]) if x[0].startswith("RandomV26_6_2")] ==
          [i for i, x in enumerate(d["r5"]) if x[0].startswith("RandomV26_6_2")], "synthesis step at the same position as rung 5")
    sf, s4, s5 = (synth_of(pipes[k]) for k in ("flat", "r4", "r5"))
    check(type(sf) is pn.RandomV26_6_2NoiseFillContrastGPU, f"flat synthesis class = {type(sf).__name__}")
    check(list(sf.noise_std_range) == [0.0, 0.0], f"flat noise_std_range = {sf.noise_std_range}")
    check(sf.label_fill_noise is False and sf.label_voronoi is False, "flat label_fill_noise=False, label_voronoi=False (PALETTE label step)")
    for a in ("c_choices", "s_choices", "blur_sigmas", "dark_threshold", "n_kmeans_subsample", "skip_parcellation_prob",
              "skip_sub_parc_prob", "alpha_magnitude_range", "label_remap_prob", "min_label_voxels", "label_classes"):
        check(getattr(sf, a) == getattr(s5, a), f"flat.{a} == rung5.{a} ({getattr(sf, a)})")
    check(float(sf.p) == float(s5.p) == 0.5, f"synthesis probability flat {float(sf.p)} rung5 {float(s5.p)}")
    check(getattr(s5, "pv_prob", 0.0) == 0.0, "rung 5 has no PV (flat class has none either)")

    print("== T1 cell invariant on real patches (blur off, synthesis forced)")
    for t in (sf, s4, s5):
        t.blur_sigmas = [0.0]
    summary = {}
    for name, ds, case in CASES:
        img, lbl = load_patch(ds, case)
        print(f"  {name}: patch {tuple(img.shape[2:])}, labels {sorted(int(x) for x in torch.unique(lbl))}")
        res = {}
        for tag, t, mod in (("flat", sf, pn), ("r4", s4, pn), ("r5", s5, fs)):
            sp, fi, ac = zip(*(cell_test(t, mod, img, lbl, s) for s in range(N_SEEDS)))
            res[tag] = dict(max_spread=max(sp), n_nonconst=sum(x > 1e-6 for x in sp), label_fired=sum(fi), all_const=sum(ac))
            print(f"    {tag:5s} max within-cell spread {max(sp):.3g}; non-constant in {res[tag]['n_nonconst']}/{N_SEEDS}; "
                  f"label step fired {sum(fi)}/{N_SEEDS}; whole fg constant {sum(ac)}/{N_SEEDS}")
        check(res["flat"]["max_spread"] < 1e-6, f"{name}: flat output constant on every (region x label) cell in all {N_SEEDS} draws")
        check(res["flat"]["label_fired"] > 0, f"{name}: flat label step fired in {res['flat']['label_fired']}/{N_SEEDS} draws")
        check(res["r4"]["n_nonconst"] >= N_SEEDS * 0.8, f"{name}: discrimination -- rung 4 (noise) fails the invariant")
        check(res["r5"]["n_nonconst"] >= N_SEEDS * 0.8, f"{name}: discrimination -- rung 5 (real fill) fails the invariant")
        summary[name] = res

    print("== T2 pipeline forward (blur off): fraction of synthesized iterations")
    pipe = pipes["flat"]
    img, lbl = load_patch(CASES[0][1], CASES[0][2])
    n_syn, n_it = 0, 200
    for s in range(n_it):
        torch.manual_seed(1000 + s); random.seed(1000 + s)
        o, _ = pipe(img.clone(), lbl.clone().float())
        # not synthesized -> output is the input up to the flip (same multiset of values)
        same = torch.equal(torch.sort(o.reshape(-1)).values, torch.sort(img.reshape(-1)).values)
        n_syn += int(not same)
    frac = n_syn / n_it
    print(f"  synthesized outputs {n_syn}/{n_it} = {frac:.2f} (expected ~0.5)")
    check(0.38 < frac < 0.62, "pipeline synthesizes on ~half of the iterations")
    summary["pipeline_frac"] = frac

    print("== T3 figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for t in (sf, s4, s5):
        t.blur_sigmas = [0.0, 0.0, 0.0, 0.3, 0.5, 0.8]
    for name, ds, case in CASES:
        img, lbl = load_patch(ds, case)
        zs = lbl[0, 0].sum((0, 1)).numpy()
        zc = int(np.argmax(zs)); nz = np.nonzero(zs)[0]
        zo = int(nz[len(nz) // 4]) if len(nz) > 4 else zc
        rows = [(3, zc), (7, zc), (11, zo), (19, zo)]
        fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
        for r, (s, z) in enumerate(rows):
            ax[r, 0].imshow(img[0, 0, :, :, z].T, cmap="gray", origin="lower")
            ax[r, 0].contour(lbl[0, 0, :, :, z].T.numpy(), levels=sorted({0.5, 1.5, 2.5, 3.5}), colors="r", linewidths=0.5, origin="lower")
            ax[r, 0].set_title(f"{name} slice {z} (labels red)", fontsize=8)
            for c, (tag, t) in enumerate((("rung 4: noise fill", s4), ("rung 4.5: flat fill", sf), ("rung 5: real fill", s5)), 1):
                torch.manual_seed(s); random.seed(s)
                o = t.apply_transform(img.clone(), {"seg": lbl}, {})
                ax[r, c].imshow(o[0, 0, :, :, z].T, cmap="gray", origin="lower"); ax[r, c].set_title(f"{tag} (seed {s})", fontsize=8)
            for a in ax[r]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"rung4_45_5_{name}.png", dpi=100)
        plt.close(fig)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {OUT}")
    print("RESULT:", "ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
