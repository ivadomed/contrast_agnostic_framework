#!/usr/bin/env python
"""
Anatomy-vs-pathology texture test (Paul's hypothesis, 2026-10-01).

Hypothesis: within one contrast, a region's texture reflects the underlying ANATOMY it sits in
(WM/GM/CSF-adjacent) rather than the PATHOLOGY, so e.g. edema texture varies with tumour location
across patients (inconsistent cue), and real-fill (keeps real texture) can be misled.

PRE-REGISTERED (written before any number was computed). Input: region_surround_texture_shard*.csv
(12-dim acf fingerprints). PRIMARY config: variant=highpass, part=ring d_out=5, all 3 axes (12 feats).
ROBUSTNESS: raw; in-plane axes 0-1 only (8 feats); ring_healthy (d_out 5).
Per contrast c, region R: fingerprints z-scored per feature across all rows (region + ring rows of R)
of that contrast; patients with any NaN in region or ring fingerprint dropped. Euclidean distance d.
 1. PC(c,R) = mean_p d(F_pR, mean of OTHER patients' R)          (lower = consistent pathology texture)
 2. AC(c,R) = mean_p d(F_pR, F_p,ring)                           (anatomy coupling)
 3. SI(c,R) = frac of patients whose region fp is closer to other-patients' REGION centroid than to
    other-patients' RING centroid (LOO nearest centroid). ~0.5 = region indistinguishable from
    surroundings (anatomy-driven); ~1 = pathology-specific. Also own-ring pull = frac of patients whose
    region fp is closer to their OWN ring than to other-patients' region centroid.
 4. Mantel-style: Spearman between pairwise region-fp distance and pairwise ring-fp distance across
    patients; permutation p (2000 perms of patient labels, one-sided positive), seed 0.
Predictions:
 P1 SNFH (edema) SI lower in t1n/t1c than in t2w/t2f (mean of pair comparison).
 P2 Mantel rho > 0 for SNFH in all four contrasts.
 P3 over the 36 OOD cells (train in t1n,t2w,t2f; eval != train; ranking_within_train_cells.csv) the
    delta (real-noise Dice) is higher when SI in TRAINING contrast is high: Spearman over cells, and
    within-(train,region) Kendall pooled (kendall_S, pooled_p from ranking_within_train.py) -- NB SI_train
    is constant within a (train,region) group so the within test is degenerate for it (S=0) and is
    reported for SI_eval and min(SI_train,SI_eval) instead; Spearman reported for all three.
Run: run_job ... .venv/bin/python this_script
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from ranking_within_train import kendall_S, pooled_p  # noqa: E402
OUT = THIS.parent / "outputs"
DATA = OUT / "data"
CONTRASTS = ["t1n", "t1c", "t2w", "t2f"]
REGIONS = ["SNFH", "RC", "ET", "NCR"]
rng = np.random.default_rng(0)

df = pd.concat([pd.read_csv(p) for p in sorted(DATA.glob("region_surround_texture_shard*.csv"))], ignore_index=True)
df = df.drop_duplicates(["patient", "contrast", "region", "part", "d_out", "variant"])
ALL = [c for c in df.columns if c.startswith("acf_")]
INPLANE = [c for c in ALL if "ax2" not in c]


def fp_table(variant, ring_part, feats, d_out=5):
    a = df[(df.variant == variant)]
    reg = a[a.part == "region"]
    ring = a[(a.part == ring_part) & (a.d_out == d_out)]
    return reg, ring, feats


def metrics(variant="highpass", ring_part="ring", feats=ALL, perms=2000):
    reg, ring, feats = fp_table(variant, ring_part, feats)
    rows = []
    for c in CONTRASTS:
        for R in REGIONS:
            r = reg[(reg.contrast == c) & (reg.region == R)].set_index("patient")[feats]
            g = ring[(ring.contrast == c) & (ring.region == R)].set_index("patient")[feats]
            both = r.dropna().index.intersection(g.dropna().index)
            r, g = r.loc[both], g.loc[both]
            n = len(both)
            if n < 6:
                rows.append(dict(contrast=c, region=R, n=n)); continue
            allv = np.vstack([r.values, g.values])
            mu, sd = allv.mean(0), allv.std(0); sd[sd == 0] = 1
            X, Y = (r.values - mu) / sd, (g.values - mu) / sd
            sx, sy = X.sum(0), Y.sum(0)
            pc, ac, si, pull = [], [], [], []
            for i in range(n):
                cR = (sx - X[i]) / (n - 1); cG = (sy - Y[i]) / (n - 1)
                dR = np.linalg.norm(X[i] - cR); dG = np.linalg.norm(X[i] - cG)
                dOwn = np.linalg.norm(X[i] - Y[i])
                pc.append(dR); ac.append(dOwn); si.append(dR < dG); pull.append(dOwn < dR)
            DX, DY = pdist(X), pdist(Y)
            rho = spearmanr(DX, DY)[0]
            Dy = squareform(DY); cnt = 0
            for _ in range(perms):
                p = rng.permutation(n)
                if spearmanr(DX, squareform(Dy[np.ix_(p, p)]))[0] >= rho: cnt += 1
            rows.append(dict(contrast=c, region=R, n=n, PC=np.mean(pc), AC=np.mean(ac), SI=np.mean(si),
                             pull=np.mean(pull), mantel_rho=rho, mantel_p=(cnt + 1) / (perms + 1)))
    return pd.DataFrame(rows)


configs = {
    "PRIMARY highpass/ring5/3ax": dict(variant="highpass", ring_part="ring", feats=ALL),
    "raw/ring5/3ax": dict(variant="raw", ring_part="ring", feats=ALL),
    "highpass/ring5/axes0-1": dict(variant="highpass", ring_part="ring", feats=INPLANE),
    "highpass/ring_healthy5/3ax": dict(variant="highpass", ring_part="ring_healthy", feats=ALL),
}
res = {k: metrics(**v) for k, v in configs.items()}


def predictions(m):
    s = m.pivot(index="region", columns="contrast", values="SI")
    p1_lo = s.loc["SNFH", ["t1n", "t1c"]].mean(); p1_hi = s.loc["SNFH", ["t2w", "t2f"]].mean()
    mm = m[m.region == "SNFH"].set_index("contrast")
    cells = pd.read_csv(DATA / "ranking_within_train_cells.csv")
    sig = pd.read_csv(DATA / "region_fill_swap_significance.csv")[["train", "eval", "region", "direction"]]
    cells = cells.merge(sig, on=["train", "eval", "region"], how="left")
    cells["SI_train"] = [s.loc[r, t] for r, t in zip(cells["region"], cells["train"])]
    cells["SI_eval"] = [s.loc[r, e] for r, e in zip(cells["region"], cells["eval"])]
    cells["SI_min"] = cells[["SI_train", "SI_eval"]].min(axis=1)
    out = dict(P1_low=p1_lo, P1_high=p1_hi, P1_ok=p1_lo < p1_hi,
               P2_rho=mm["mantel_rho"].to_dict(), P2_p=mm["mantel_p"].to_dict(),
               P2_ok=bool((mm["mantel_rho"] > 0).all()), cells=cells)
    for k in ("SI_train", "SI_eval", "SI_min"):
        rho, p = spearmanr(cells[k], cells["delta"])
        S, nr = 0, 0
        for _, g in cells.groupby(["train", "region"]):
            if len(g) == 3:
                S += kendall_S(g[k].values, g["delta"].values); nr += 1
        out[k] = (rho, p, S, nr, pooled_p(S, nr))
    return out


preds = {k: predictions(v) for k, v in res.items()}

# ---------- markdown
L = ["# Anatomy vs pathology texture (pre-registered; see script docstring)\n"]
for k, m in res.items():
    L.append(f"\n## {k}\n\n| contrast | region | n | PC | AC | SI | own-ring pull | Mantel rho | Mantel p |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in m.iterrows():
        if "SI" not in r or pd.isna(r.get("SI")): L.append(f"| {r.contrast} | {r.region} | {r.n} | - | - | - | - | - | - |"); continue
        L.append(f"| {r.contrast} | {r.region} | {r.n:.0f} | {r.PC:.2f} | {r.AC:.2f} | {r.SI:.2f} | {r.pull:.2f} | {r.mantel_rho:+.2f} | {r.mantel_p:.3f} |")
    P = preds[k]
    L.append(f"\n**P1** SNFH SI t1n/t1c mean {P['P1_low']:.2f} vs t2w/t2f {P['P1_high']:.2f} -> {'supported' if P['P1_ok'] else 'NOT supported'}")
    L.append("\n**P2** SNFH Mantel rho: " + ", ".join(f"{c} {P['P2_rho'][c]:+.2f} (p={P['P2_p'][c]:.3f})" for c in CONTRASTS) + f" -> {'all positive' if P['P2_ok'] else 'NOT all positive'}")
    L.append("\n**P3** delta vs SI (36 OOD cells):\n\n| SI used | Spearman rho | p | pooled Kendall S (12 rows) | pooled p |\n|---|---|---|---|---|")
    for kk in ("SI_train", "SI_eval", "SI_min"):
        rho, p, S, nr, pp = P[kk]
        L.append(f"| {kk} | {rho:+.2f} | {p:.3f} | {S} ({nr} rows) | {pp:.3f} |" + (" (degenerate within-group)" if kk == "SI_train" else ""))
(OUT / "tables" / "anatomy_vs_pathology_texture.md").write_text("\n".join(L) + "\n")

# ---------- plot
prim = res["PRIMARY highpass/ring5/3ax"]; P = preds["PRIMARY highpass/ring5/3ax"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
H = prim.pivot(index="region", columns="contrast", values="SI").loc[REGIONS, CONTRASTS]
im = ax[0].imshow(H.values, vmin=0.4, vmax=1, cmap="viridis")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(CONTRASTS); ax[0].set_yticks(range(4)); ax[0].set_yticklabels(REGIONS)
for i in range(4):
    for j in range(4): ax[0].text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center", color="w")
ax[0].set_title("Self-identity index SI (0.5 = anatomy-like)"); plt.colorbar(im, ax=ax[0])
c = P["cells"]
col = c.direction.map(lambda d: "#2f7d6b" if d == "HELPS" else ("#c0392b" if d == "HURTS" else "grey")).fillna("grey")
ax[1].scatter(c.SI_train, c.delta * 100, c=col, s=40)
ax[1].axhline(0, color="k", lw=.5)
rho, p = P["SI_train"][:2]
ax[1].set_xlabel("SI in training contrast"); ax[1].set_ylabel("delta real - noise (Dice pts)")
ax[1].set_title(f"36 OOD cells, Spearman {rho:+.2f} (p={p:.2f})\ngreen helps / red hurts / grey n.s.")
plt.tight_layout(); plt.savefig(OUT / "plots" / "anatomy_vs_pathology_texture.png", dpi=130)
print("\n".join(L))
print(c.direction.value_counts())
