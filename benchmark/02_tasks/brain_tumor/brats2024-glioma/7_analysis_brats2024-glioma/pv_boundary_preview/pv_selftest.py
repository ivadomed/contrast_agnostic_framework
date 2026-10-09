"""Self-tests for PALETTE boundary-PV before any training. Prints SELFTEST PASSED/FAILED.

1. pv_prob=0 is bit-identical to the committed (git HEAD) AugLab transform -- on CPU, since CUDA
   scatter_add_ is non-deterministic (also reported: HEAD-vs-HEAD on GPU, to show that floor).
2. pv_prob=1 with only the [0,0,0] level == pv off (blur_sigmas=[0] so python RNG drift can't matter).
3. PV on: finite, background stays exactly 0, partition unchanged (|diff| small vs full-image change).
4. K-means intensity PV makes the remap continuous: largest jump between intensity-adjacent voxels
   of the same Voronoi cell shrinks.
5. Timing on a BraTS-sized batch (2 x 128x160x112) on GPU: legacy vs every PV level.
"""
import importlib.util, random, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np, nibabel as nib, torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
AUG = REPO / "sub-workspaces/auglab_workspace/AugLab"
sys.path.insert(0, str(AUG))
from auglab.transforms.gpu.fromSeg import RandomV26_6_2ContrastGPU as New  # noqa: E402

src = subprocess.run(["git", "-C", str(AUG), "show", "HEAD:auglab/transforms/gpu/fromSeg.py"],
                     capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / "fromSeg_head.py"; tmp.write_text(src)
spec = importlib.util.spec_from_file_location("auglab.transforms.gpu.fromSeg_head", tmp)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Old = mod.RandomV26_6_2ContrastGPU

gpu = "cuda" if torch.cuda.is_available() else "cpu"
RAW = REPO / "benchmark/02_tasks/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n"
PS = (128, 160, 112)
xs, ss = [], []
for case in ["BraTSGLI00005100", "BraTSGLI00006100"]:
    img = nib.load(RAW / "imagesTr" / f"{case}_0000.nii.gz").get_fdata().astype(np.float32)
    seg = nib.load(RAW / "labelsTr" / f"{case}.nii.gz").get_fdata().astype(np.int64)
    c = np.array(np.nonzero(seg)).mean(1).astype(int)
    sl = tuple(slice(max(0, ci - p // 2), max(0, ci - p // 2) + p) for ci, p in zip(c, PS))
    a, b = img[sl], seg[sl]
    pad = [(0, p - n) for p, n in zip(PS, a.shape)]
    xs.append(np.pad(a, pad)); ss.append(np.pad(b, pad))
X = torch.from_numpy(np.stack(xs))[:, None]
S = torch.from_numpy(np.stack(ss))[:, None]
x, s, dev = X, S, "cpu"
fg = x > 0

def on(d):
    global x, s, dev, fg
    dev = d; x, s = X.to(d), S.to(d); fg = x > 0

def run(T, seed, **kw):
    random.seed(seed); torch.manual_seed(seed)
    out = T(p=1.0, **kw).apply_transform(x, {"seg": s}, {})
    if dev == "cuda": torch.cuda.synchronize()
    return out

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")

if gpu == "cuda":
    on("cuda")
    nd = sum(not torch.equal(run(Old, k), run(Old, k)) for k in range(3))
    print(f"[info] GPU HEAD-vs-HEAD non-identical in {nd}/3 seeds (CUDA non-determinism floor)")
    on("cpu")
for seed in range(5):
    a, b = run(Old, seed), run(New, seed)
    check(f"1. pv off == HEAD (seed {seed})", torch.equal(a, b), f"max|d|={(a-b).abs().max().item():.2e}")
for seed in range(3):
    a = run(New, seed, blur_sigmas=[0.0])
    b = run(New, seed, blur_sigmas=[0.0], pv_prob=1.0, pv_levels=[[0, 0, 0]])
    check(f"2. level [0,0,0] == off (seed {seed})", torch.equal(a, b))
for lvl in [[0.7, 0.4, 0.02], [1.0, 0.7, 0.04], [1.5, 1.0, 0.08]]:
    for seed in range(3):
        a = run(New, seed, blur_sigmas=[0.0])
        b = run(New, seed, blur_sigmas=[0.0], pv_prob=1.0, pv_levels=[lvl])
        d = (a - b)[fg].abs()
        check(f"3. level {lvl} seed {seed}: finite, bg==0", torch.isfinite(b).all() and (b[~fg] == 0).all(),
              f"median|dz|={d.median().item():.3f} p99={d.quantile(0.99).item() if d.numel() < 2**24 else float('nan'):.3f}")

# 4. continuity: K-means-only PV (no spatial bands), single sample, compare remap curve jumps
x1, s1 = x[:1], torch.zeros_like(s[:1])
v = x1[0, 0]; v01 = (v - v.min()) / (v.max() - v.min() + 1e-7)
m4 = v01 > 0.02  # inside PALETTE's own fg mask (dark_threshold 0.01), away from that hard cutoff  # no labels -> no label remap; s_choices=[1]-like via skip_sub_parc_prob=1
def curve(**kw):
    random.seed(3); torch.manual_seed(3)
    T = New(p=1.0, blur_sigmas=[0.0], skip_parcellation_prob=0.0, skip_sub_parc_prob=1.0, c_choices=[4], **kw)
    y = T.apply_transform(x1, {"seg": s1}, {})[0, 0][m4]
    xi = x1[0, 0][m4]
    o = torch.argsort(xi); return xi[o], y[o]
xo, y0 = curve()
_, y1 = curve(pv_prob=1.0, pv_levels=[[0, 0, 0.06]])
# A discontinuity = large dy between intensity-adjacent voxels with a TINY dx (in [0,1] units);
# a large dy across a big dx gap (sparse bright tail) is just slope, not a jump.
x01 = (xo - v.min()) / (v.max() - v.min() + 1e-7)
dx = x01.diff()
def jumps(y):
    dy = y.diff().abs(); tiny = dx < 1e-3
    return dy[tiny].max().item(), int((dy[tiny] > 0.1).sum())
(j0, n0), (j1, n1) = jumps(y0), jumps(y1)
for tag, y in [("off", y0), ("on", y1)]:
    k = torch.topk(y.diff().abs(), 3)
    print(f"[info] {tag}: top dy " + ", ".join(f"{a:.3f}@dx={dx[i].item():.2e},x={x01[i].item():.3f}" for a, i in zip(k.values.tolist(), k.indices.tolist())))
check("4. K-means intensity PV removes remap discontinuities", n1 == 0 and n0 > 0,
      f"max dy at dx<1e-3: off={j0:.3f} on={j1:.3f}; #(dy>0.1z) off={n0} on={n1}")

# 5. timing
on(gpu)
def t(**kw):
    run(New, 0, **kw); ts = []
    for k in range(5):
        t0 = time.time(); run(New, k, **kw); ts.append(time.time() - t0)
    return np.median(ts) * 1e3
base = t()
print(f"timing ({dev}, batch 2 x {PS}): legacy {base:.0f} ms")
for lvl in [[0.7, 0.4, 0.02], [1.0, 0.7, 0.04], [1.5, 1.0, 0.08]]:
    print(f"  level {lvl}: {t(pv_prob=1.0, pv_levels=[lvl]):.0f} ms")

# 6. the real rung-6 JSON, built through AugLab's own config parser (the path training uses)
from auglab.transforms.gpu.transforms import AugTransformsGPU  # noqa: E402
cfg = AUG / "auglab/configs/transform_params_gpu_v26_6_2_pv_synth_spatialDA_train050.json"
pipe = AugTransformsGPU(str(cfg))
mods = [m for m in pipe.modules() if isinstance(m, New)]
check("6. rung-6 JSON -> PALETTE carries pv_prob=1 and 5 levels",
      len(mods) == 1 and mods[0].pv_prob == 1.0 and len(mods[0].pv_levels) == 5,
      f"found {len(mods)} PALETTE module(s)")
cfg5 = AUG / "auglab/configs/transform_params_gpu_v26_6_2_synth_spatialDA_train050.json"
m5 = [m for m in AugTransformsGPU(str(cfg5)).modules() if isinstance(m, New)]
check("6b. rung-5 JSON -> PV off", len(m5) == 1 and m5[0].pv_prob == 0.0)
print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
