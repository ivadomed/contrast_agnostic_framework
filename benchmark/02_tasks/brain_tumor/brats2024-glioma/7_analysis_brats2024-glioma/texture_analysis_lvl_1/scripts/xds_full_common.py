"""
xds_full_common.py -- shared helpers for the "full-feature" cross-dataset fill-swap model (xds_full_*).

Structural blind-test guard: this module only builds DEV cell specs (brats, breast family, on-harmony,
toothfairy2) and never reads chaos / open-ms ladder JSON or eval_all.csv.  Test-cell specs live in xds_full_test.py.
(Image FEATURES of chaos / open-ms may be extracted before the freeze -- features are not outcomes.)

Unit of analysis: ROW = (cell, case).  cell = (train key, eval key[, BraTS region]).  Target = per-case
delta Dice (points) = rung "real fill" minus the rung before it, from the ladder's own run_keys, folds 0-2
(ladder_ood_common.load_case_means: mean over labels and folds), i.e. labels pooled per case exactly as in the ladder.
Feature "units": (feature key, case); BraTS key = "brats:<contrast>:<region>", case = patient.
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
REPO = Path("/project/aip-jcohen/paulh/mri_synthesis_project")
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import ladder_ood_common as loc  # noqa: E402

W = THIS.parent
DATA = W / "outputs" / "data"
TABLES = W / "outputs" / "tables"
PLOTS = W / "outputs" / "plots"
LOGS = W / "outputs" / "logs"
TS = REPO / "benchmark/02_tasks"
CAP = 20  # cases per non-BraTS key (matches the existing radiomics_xds extraction)
BREG = ("NCR", "SNFH", "ET", "RC")


def jload(p):
    return json.loads(Path(p).read_text())


# ----------------------------------------------------------------------------- manifests (feature images)
def manifest():
    import cross_dataset_fillswap_model as xd  # import only; dev_cells()/openms_cells() are never called here
    return xd.MAN


def _name_to_root(json_path):
    p = Path(json_path).parent
    while p.name != "ablations" and p != p.parent:
        p = p.parent
    return p.parent


def _ir(d):
    return next(i for i, l in enumerate(d["labels"]) if "real fill" in l)


# ----------------------------------------------------------------------------- DEV cell specs
def _ladder_specs(rel, train_key, fam, evmap, item_of=None):
    d = jload(TS / rel)
    pcs = d.get("fill_swap_significance", {}).get("dice", {}).get("per_contrast", {})
    out = []
    for ek, fk in evmap.items():
        if ek not in d["per_contrast"]["dice"]:
            continue
        out.append(dict(fam=fam, train_key=train_key, eval_key=fk, json=str(TS / rel), item=ek, region=None,
                        p=pcs.get(ek, np.nan),
                        name=f"{train_key}->{fk}"))
    return out


def dev_specs():
    """Mirrors cross_dataset_fillswap_model.dev_cells() minus chaos (a TEST set here) and minus open-ms."""
    c = []
    oh = "brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model"
    for t in ("T1w", "T2w", "dwi_ap"):
        rel = f"{oh}/{t}/ablations/ladder_series.json"
        c += _ladder_specs(rel, f"onharmony:{t}", "onharmony", {e: f"onharmony:{e}" for e in jload(TS / rel)["ood_contrasts"]})
    rel = "mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct/ablations/ladder_series.json"
    c += _ladder_specs(rel, "toothfairy2:cbct", "toothfairy2",
                       {k: k.replace("/", ":") for k in jload(TS / rel)["fill_swap_significance"]["dice"]["per_contrast"]})
    i2 = "breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model"
    c += _ladder_specs(f"{i2}/t1wce/ablations/ladder_series.json", "ispy2:t1wce", "breast", {"t2w": "ispy2:t2w"})
    sp = "breast_cancer/{c}/8_results_{c}/02_metrics/ispy2_model/{t}/ablations/{s}/ladder_series.json"
    for t, tk in (("t2w", "ispy2:t2w"), ("t1wce", "ispy2:t1wce")):
        for cohort, sub, fk in (("ispy1", "precontrast", "ispy1:precontrast"), ("ispy1", "t1wce", "ispy1:t1wce"),
                                ("duke-breast-mri", "precontrast_uniap", "duke:precontrast_uniap"),
                                ("duke-breast-mri", "t1wce_uniap", "duke:t1wce_uniap"),
                                ("acrin6698", "dwi_uniap", "acrin:dwi_uniap")):
            if t == "t1wce" and sub in ("t1wce", "t1wce_uniap"):
                continue  # same-contrast cross-dataset only: not an OOD-contrast cell
            c += _ladder_specs(sp.format(c=cohort, t=t, s=sub), tk, "breast", {sub: fk})
    rg = pd.read_csv(DATA / "region_fill_swap_significance.csv")
    rg = rg[rg.family == "OOD"]
    for r in rg.itertuples():
        c.append(dict(fam="brats", train_key=f"brats:{r.train}:{r.region}", eval_key=f"brats:{r.eval}:{r.region}", json=None,
                      item=None, region=r.region, p=r.p_raw, name=f"brats:{r.train}->{r.eval}:{r.region}",
                      train=r.train, eval=r.eval, n_csv=int(r.n), delta_csv=float(r.mean_delta_pts)))
    return c


def dev_feature_keys():
    s = dev_specs()
    return sorted({k for x in s for k in (x["train_key"], x["eval_key"]) if not k.startswith("brats:")})


TEST_KEYS = ["chaos:t1in", "chaos:t1out", "chaos:t2spir", "chaos:ct", "open-ms:flair", "open-ms:t1w", "open-ms:t2w"]


# ----------------------------------------------------------------------------- targets (outcomes)
def _find_root_item(spec):
    """metrics root + item name inside load_case_means(...) for a non-BraTS spec."""
    d = jload(spec["json"])
    if "ood_sources" in d:  # cross-dataset ladder (toothfairy2): item "<dataset>/<item>"
        ds, item = spec["item"].split("/")
        root = next(Path(s) for s in d["ood_sources"] if loc._dataset_name(Path(s)) == ds)
        return d, root, item
    return d, _name_to_root(spec["json"]), spec["item"]


def cell_targets(spec):
    """{case_id: delta_pts} (rung real-fill minus previous rung), plus meta dict."""
    d, root, item = _find_root_item(spec)
    ir = _ir(d)
    rk_prev, rk_cur = d["run_keys"][ir - 1], d["run_keys"][ir]
    dp, dc = loc.resolve_run_dir(root, rk_prev), loc.resolve_run_dir(root, rk_cur)
    assert dp.is_dir() and dc.is_dir(), f"missing run dir for {spec['name']}: {dp} {dc}"
    prev = loc.load_case_means(dp, "dice").get(item, {})
    cur = loc.load_case_means(dc, "dice").get(item, {})
    cases = sorted(set(prev) & set(cur))
    delta = {k: 100.0 * (cur[k] - prev[k]) for k in cases}
    a = d["per_contrast"]["dice"][spec["item"].split("/")[-1] if "ood_sources" not in d else spec["item"]]
    jd = a[ir] - a[ir - 1]
    full = float(np.mean(list(delta.values()))) if delta else float("nan")
    return delta, dict(json_delta=float(jd), full_mean=full, n_full=len(cases), root=str(root))


_BR = None


def brats_targets(spec):
    """BraTS per-patient delta for a (train, eval, region) cell; GT-present patients (gt_vox>0)."""
    global _BR
    if _BR is None:
        df = pd.read_csv(DATA / "patient_region_deltas.csv")
        g = pd.read_csv(DATA / "gt_region_presence.csv")
        df = df.merge(g, left_on=["case", "region"], right_on=["case", "region"], how="left")
        _BR = df[df.gt_vox > 0]
    s = _BR[(_BR.train == spec["train"]) & (_BR["eval"] == spec["eval"]) & (_BR.region == spec["region"])]
    delta = {r.case: 100.0 * r.delta_dice for r in s.itertuples()}
    full = float(np.mean(list(delta.values()))) if delta else float("nan")
    return delta, dict(json_delta=spec["delta_csv"], full_mean=full, n_full=len(delta), n_csv=spec["n_csv"])


def targets(spec):
    return brats_targets(spec) if spec["fam"] == "brats" else cell_targets(spec)


# ----------------------------------------------------------------------------- feature tables
def pkey(case, key):
    """physical-patient identity of a manifest/eval case id under feature key 'ds:item'."""
    return loc._patient_key(case, key.split(":")[-1])


def load_hand():
    """DataFrame indexed (key, case) with hand-made base features (H prefix dropped later)."""
    fr = []
    b = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(DATA / "bigfeat_shard*.csv")))], ignore_index=True)
    b["key"] = "brats:" + b["contrast"] + ":" + b["region"]
    b = b.rename(columns={"patient": "case"}).drop(columns=["contrast", "region"])
    fr.append(b)
    for f in sorted(glob.glob(str(DATA / "xds_full_hand_shard*.csv"))):
        fr.append(pd.read_csv(f))
    H = pd.concat(fr, ignore_index=True)
    H["case"] = H["case"].astype(str)
    H = H.drop_duplicates(["key", "case"], keep="last").drop(columns=[c for c in ("n_R", "n_ring") if c in H.columns])
    return H.set_index(["key", "case"]).astype(float)


def load_rad():
    """DataFrame indexed (key, case) of radiomics: R__, ring__, RmR__ (R - ring), S__ shape.  Labels pooled per case,
    weighted by n_R (non-BraTS); BraTS rows are already per (patient, contrast, region)."""
    sh = [s for s in sorted(glob.glob(str(DATA / "radiomics_*_shard*.csv"))) if "_TEST" not in s and "radiomics_A" not in s
          and ("_xds_" in s or "_brats_" in s or "_openms_" in s)]
    parts = []
    for s in sh:
        F = pd.read_csv(s)
        if "patient" in F.columns:
            F["key"] = "brats:" + F["contrast"] + ":" + F["region"]
            F["case"] = F["patient"]
        F["case"] = F["case"].astype(str)
        if "label" not in F.columns:
            F["label"] = 0
        parts.append(F)
    F = pd.concat(parts, ignore_index=True)
    F = F.drop_duplicates(["key", "case", "label"], keep="last")
    R = [c for c in F.columns if c.startswith("R__")]
    G = [c for c in F.columns if c.startswith("ring__")]
    S = [c for c in F.columns if c.startswith("S__")]
    D = pd.DataFrame({"RmR__" + c[3:]: F[c].to_numpy() - F["ring__" + c[3:]].to_numpy() for c in R}, index=F.index)
    cols = R + G + list(D.columns) + S
    F = pd.concat([F[["key", "case", "label", "n_R"]], F[R + G + S], D], axis=1)
    w = F["n_R"].to_numpy(float)
    num = F[cols].mul(w, axis=0).groupby([F["key"], F["case"]]).sum(min_count=1)
    den = pd.Series(w, index=F.index).groupby([F["key"], F["case"]]).sum()
    return num.div(den, axis=0).astype(float)


def family_of(col, kind):
    """(family, subfamily) of a base feature column for grouping importances."""
    if kind == "hand":
        for p, f in (("R_", "target-region"), ("ring_", "ring"), ("brain_", "whole-brain/fg"), ("con_", "R-vs-ring contrast"),
                     ("bnd_", "boundary"), ("shape_", "shape")):
            if col.startswith(p):
                body = col[len(p):]
                sub = ("first-order" if body.split("_")[0] in ("mean", "std", "p5", "p25", "p50", "p75", "p95", "iqr", "skew", "kurt", "ent")
                       else "acf" if body.startswith("acf") else "glcm/lbp" if body.startswith(("glcm", "lbp"))
                       else "gradient/highpass" if body.split("_")[0] in ("hp", "gm", "lap") else "other")
                return f, sub
        return "other", "other"
    pre, rest = col.split("__", 1)
    fam = {"R": "target-region", "ring": "ring", "RmR": "R-minus-ring", "S": "shape"}[pre]
    parts = rest.split("_")
    imtype = "wavelet" if parts[0] == "wavelet" else "LoG" if parts[0] == "log" else "original"
    cls = next((x for x in parts if x in ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape")), "other")
    return fam, f"{imtype}/{cls}"


# ----------------------------------------------------------------------------- row assembly (dev AND test; features only)
_HDR = {}


def _hdr(path):
    if path not in _HDR:
        import nibabel as nib
        h = nib.load(str(path))
        _HDR[path] = (tuple(h.shape[:3]), np.asarray(h.affine))
    return _HDR[path]


def coregistered(p1, p2):
    s1, a1 = _hdr(p1)
    s2, a2 = _hdr(p2)
    return s1 == s2 and np.allclose(a1, a2, atol=1e-3)


def build_rows(specs, tgt_fn, H, R, MAN):
    """Rows = (cell, case).  tgt_fn(spec) -> ({case: delta_pts}, meta).  Feature units come from H (hand) and R (radiomics);
    a row needs its EVAL unit in both.  Train-contrast features: the SAME patient's unit if the two images are
    co-registered (same patient key, same shape and allclose affine; BraTS always), else the mean over that train key's
    extracted cases (dataset-mean for the target).  Returns meta DataFrame and arrays Eh, Th, Er, Tr (rows x base feats)."""
    paths = {}

    def key_paths(k):
        if k not in paths:
            paths[k] = {str(c): ip for c, ip, lp in MAN[k][1]()} if not k.startswith("brats:") else {}
        return paths[k]

    hcols, rcols = list(H.columns), list(R.columns)
    Hk = {k: H.xs(k, level=0) for k in H.index.get_level_values(0).unique()}
    Rk = {k: R.xs(k, level=0) for k in R.index.get_level_values(0).unique()}
    mean_cache = {}

    def kmean(tab, k, tag):
        if (tag, k) not in mean_cache:
            mean_cache[(tag, k)] = tab[k].mean(0).to_numpy(float)
        return mean_cache[(tag, k)]

    meta, Eh, Th, Er, Tr, cells = [], [], [], [], [], []
    for si, s in enumerate(specs):
        d, m = tgt_fn(s)
        ek, tk = s["eval_key"], s["train_key"]
        n_join = n_match = 0
        if ek not in Hk or ek not in Rk or tk not in Hk or tk not in Rk:
            cells.append(dict(cell=si, name=s["name"], fam=s["fam"], n_full=m["n_full"], n_rows=0, n_matched=0, **{k: m[k] for k in ("json_delta", "full_mean")}, p=s["p"]))
            continue
        thc, trc = Hk[tk], Rk[tk]
        tcases = [c for c in thc.index if c in trc.index]
        for case in sorted(d):
            c = str(case)
            if c not in Hk[ek].index or c not in Rk[ek].index:
                continue
            tc = None
            if s["fam"] == "brats":
                tc = c if (c in thc.index and c in trc.index) else None
                if tc is None:
                    continue
            else:
                cand = [x for x in tcases if pkey(x, tk) == pkey(c, ek)]
                cand.sort(key=lambda x: (x != c, x))
                pe = key_paths(ek).get(c)
                for x in cand:
                    px = key_paths(tk).get(x)
                    if pe is not None and px is not None and coregistered(px, pe):
                        tc = x
                        break
            if tc is not None:
                th, tr = thc.loc[tc].to_numpy(float), trc.loc[tc].to_numpy(float)
                n_match += 1
            else:
                th, tr = kmean(Hk, tk, "h"), kmean(Rk, tk, "r")
            Eh.append(Hk[ek].loc[c].to_numpy(float)); Er.append(Rk[ek].loc[c].to_numpy(float))
            Th.append(th); Tr.append(tr)
            meta.append(dict(cell=si, case=c, y=float(d[case]), fam=s["fam"], matched=tc is not None))
            n_join += 1
        cells.append(dict(cell=si, name=s["name"], fam=s["fam"], n_full=m["n_full"], n_rows=n_join, n_matched=n_match,
                          json_delta=m["json_delta"], full_mean=m["full_mean"], p=s["p"]))
    return (pd.DataFrame(meta), pd.DataFrame(cells), np.array(Eh), np.array(Th), np.array(Er), np.array(Tr), hcols, rcols)
