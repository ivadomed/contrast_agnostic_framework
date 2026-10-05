#!/usr/bin/env python3
"""Learn each scan's TRUE in-plane orientation and validate it (the source image headers are unreliable for many MCF scans).
Self-supervised: scans of the reference centers (NYU + AHN: orientation verified by eye and by T1<->T2 pair tests) are taken as canonical (BIDS image reoriented to
LPS by its header); a small CNN is trained to recognise which of 4 in-plane variants (0 as stored, 1 flip A-P, 2 flip L-R, 3 both = 180 deg rotation) was applied
to a body-centred 360 mm axial patch (7 slices around the mask-centroid slice, random shifts, gamma jitter). NYU subjects are split 80/20 BY SUBJECT: the held-out
accuracy is reported. The trained net is then applied to EVERY scan (softmax averaged over the 7 slices): predicted variant v = the flips the stored array
needs to look canonical (flips are involutions). Validation of the corrections on the unseen centers:
  (a) T1<->T2 same-shape pairs: in-plane distance of the two mask centroids (fraction of extent) before vs after applying the predicted flips to both scans;
  (b) mask centroid offset from the body centre in the corrected frame vs the NYU prototype (does the mask follow its image?).
Writes 9_tests_pansegdata/orientation_classifier.tsv (case, site, contrast, pred, conf, p_orig) and prints the validation. CPU only."""
import csv, json, collections
from pathlib import Path
import numpy as np, nibabel as nib, torch, torch.nn as nn
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform, apply_orientation
from scipy import ndimage as ndi

torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
DS = Path(__file__).resolve().parents[2]
BIDS = DS / "1_BIDS_pansegdata" / "onc-pancreas-pansegdata"
TESTS = DS / "9_tests_pansegdata"
part = json.loads((DS / "4_splits_pansegdata" / "partition.json").read_text())
cases = part["train_pool"] + part["test"]; site = part["site"]
N, WIN, NS = 64, 360.0, 7
FLIPS = {0: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}   # (flip A-P axis1, flip L-R axis0)


def lps(arr, affine):
    return apply_orientation(arr, ornt_transform(io_orientation(affine), axcodes2ornt(("L", "P", "S"))))


def otsu(v):
    h, e = np.histogram(v, bins=128); c = (e[:-1] + e[1:]) / 2; w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def flip2d(p, v):
    ap, lr = FLIPS[v]
    q = p[::-1, :] if lr else p
    return q[:, ::-1] if ap else q


def load(cid, item):
    sub = "sub-" + cid.split("pansegdata_")[1]
    stem, suf = (f"{sub}_acq-venous", "T1w") if item == "t1wce" else (sub, "T2w")
    img = nib.load(str(BIDS / sub / "anat" / f"{stem}_{suf}.nii.gz"))
    x = lps(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    m = lps(np.asanyarray(nib.load(str(BIDS / "derivatives/labels" / sub / "anat" / f"{stem}_label-pancreas_seg.nii.gz")).dataobj) > 0, img.affine)
    z = np.array(img.header.get_zooms()[:3])[[int(a) for a in io_orientation(img.affine)[:, 0]]]
    return x, m, z


def patches(x, m, z):
    zc = int(round(np.argwhere(m)[:, 2].mean()))
    sl0 = x[:, :, max(0, zc - 1):zc + 2].mean(2)
    g0 = ndi.gaussian_filter(sl0, 2)
    body = ndi.binary_fill_holes(g0 > otsu(g0.ravel())); lab, n = ndi.label(body)
    if n > 1: body = lab == (np.argmax(ndi.sum(body, lab, range(1, n + 1))) + 1)
    bx = np.argwhere(body); cen = (bx.min(0) + bx.max(0)) / 2.0
    g = (np.arange(N) - (N - 1) / 2) * (WIN / N)
    ii, jj = np.meshgrid(cen[0] + g / z[0], cen[1] + g / z[1], indexing="ij")
    out = []
    for dz in range(-(NS // 2), NS // 2 + 1):
        s = x[:, :, int(np.clip(zc + dz, 0, x.shape[2] - 1))]
        lo, hi = np.percentile(s, [1, 99.5]); s = np.clip(s, lo, hi)
        p = ndi.map_coordinates(s, [ii, jj], order=1, cval=float(s.min())); out.append((p - p.mean()) / (p.std() + 1e-6))
    mc = np.argwhere(m).mean(0)
    return np.stack(out).astype(np.float32), np.array([(mc[0] - cen[0]) * z[0], (mc[1] - cen[1]) * z[1]]), x.shape, mc / np.array(x.shape)


D = {}
for c in cases:
    for it in ("t1wce", "t2w"):
        D[(c, it)] = patches(*load(c, it))
print("patches built", len(D), flush=True)

ref = [c for c in cases if site[c] in ("NYU", "AHN")]
nyu = sorted(c for c in ref if site[c] == "NYU"); rng.shuffle(nyu)
val_sub = set(nyu[: len(nyu) // 5]); train_ids = [c for c in ref if c not in val_sub]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 64), blk(64, 96), nn.Flatten(), nn.Dropout(0.3), nn.Linear(96 * 16, 64), nn.ReLU(), nn.Linear(64, 4))
    def forward(self, x): return self.f(x)


def sample(ids, n, train=True):
    X, Y = [], []
    for _ in range(n):
        c = ids[rng.integers(len(ids))]; it = ("t1wce", "t2w")[rng.integers(2)]
        p = D[(c, it)][0][rng.integers(NS)]; v = int(rng.integers(4))
        p = flip2d(p, v)
        if train:
            p = np.roll(p, tuple(rng.integers(-4, 5, 2)), (0, 1)); p = np.sign(p) * np.abs(p) ** float(rng.uniform(0.8, 1.25))
        X.append(np.stack([p, np.full_like(p, 1.0 if it == "t1wce" else 0.0)])); Y.append(v)
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(Y)


net = Net(); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4); lossf = nn.CrossEntropyLoss()
Xv, Yv = sample(sorted(val_sub), 1200, train=False)
for ep in range(60):
    net.train()
    for _ in range(12):
        X, Y = sample(train_ids, 128); opt.zero_grad(); l = lossf(net(X), Y); l.backward(); opt.step()
    if ep % 10 == 9 or ep == 59:
        net.eval()
        with torch.no_grad(): acc = (net(Xv).argmax(1) == Yv).float().mean().item()
        print(f"epoch {ep + 1} loss {l.item():.3f}  held-out NYU subjects, 4-way variant accuracy {acc:.3f}", flush=True)
net.eval()
with torch.no_grad():
    pv = net(Xv).argmax(1); cm = np.zeros((4, 4), int)
    for a, b in zip(Yv.numpy(), pv.numpy()): cm[a, b] += 1
print("held-out confusion (rows true variant, cols predicted):\n", cm)

rows = []
for (c, it), (P, off, shp, mfr) in D.items():
    X = torch.tensor(np.stack([np.stack([P[k], np.full_like(P[k], 1.0 if it == "t1wce" else 0.0)]) for k in range(NS)]), dtype=torch.float32)
    with torch.no_grad(): pr = torch.softmax(net(X), 1).mean(0).numpy()
    rows.append(dict(case=c, site=site[c], contrast=it, pred=int(pr.argmax()), conf=round(float(pr.max()), 3), p_orig=round(float(pr[0]), 3),
                     p_apflip=round(float(pr[1] + pr[3]), 3), p0=round(float(pr[0]), 3), p1=round(float(pr[1]), 3), p2=round(float(pr[2]), 3), p3=round(float(pr[3]), 3),
                     in_ref=bool(site[c] in ("NYU", "AHN") and c not in val_sub), off_L=float(off[0]), off_P=float(off[1])))
with open(TESTS / "orientation_classifier.tsv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(rows)
g = collections.defaultdict(list)
for r in rows: g[(r["site"], r["contrast"])].append(r)
print("center contrast n | predicted correction variant counts (0 none,1 A-P,2 L-R,3 both) | median conf | frac conf>0.8")
for k in sorted(g):
    v = g[k]; print(f"{k[0]:4s} {k[1]:5s} {len(v):4d} | {dict(sorted(collections.Counter(r['pred'] for r in v).items()))} | {np.median([r['conf'] for r in v]):.2f} | {np.mean([r['conf'] > 0.8 for r in v]):.2f}")
# (a) pair agreement before / after correction on same-shape pairs
pred = {(r["case"], r["contrast"]): r["pred"] for r in rows}
def cen_after(c, it, v):
    mfr = D[(c, it)][3]; ap, lr = FLIPS[v]
    return np.array([1 - mfr[0] if lr else mfr[0], 1 - mfr[1] if ap else mfr[1]])
print("same-shape T1<->T2 pairs: fraction with in-plane mask-centroid distance < 0.06 (fraction of extent), before -> after correction")
for s in ("NYU", "AHN", "MCF"):
    ids = [c for c in cases if site[c] == s and D[(c, "t1wce")][2] == D[(c, "t2w")][2]]
    b = [np.linalg.norm(cen_after(c, "t1wce", 0) - cen_after(c, "t2w", 0)) < 0.06 for c in ids]
    a = [np.linalg.norm(cen_after(c, "t1wce", pred[(c, "t1wce")]) - cen_after(c, "t2w", pred[(c, "t2w")])) < 0.06 for c in ids]
    print(f"  {s}: n={len(ids)}  before {np.mean(b):.2f} -> after {np.mean(a):.2f}")
# (b) mask-vs-image consistency after correction
prot = {it: np.median([np.array([r["off_L"], r["off_P"]]) for r in rows if r["site"] == "NYU" and r["contrast"] == it], 0) for it in ("t1wce", "t2w")}
print("mask centroid offset (mm from body centre) NYU prototype:", {k: np.round(v, 1).tolist() for k, v in prot.items()})
print("fraction of masks consistent with the corrected image (closer to the NYU prototype than to its 180-degree mirror):")
for k in sorted(g):
    ok = []
    for r in g[k]:
        ap, lr = FLIPS[r["pred"]]; o = np.array([r["off_L"] * (-1 if lr else 1), r["off_P"] * (-1 if ap else 1)])
        ok.append(np.linalg.norm(o - prot[k[1]]) < np.linalg.norm(o + prot[k[1]]))
    print(f"  {k[0]:4s} {k[1]:5s} {np.mean(ok):.2f}")
