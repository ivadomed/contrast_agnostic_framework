#!/usr/bin/env python
"""
Visibility sign rule, all ladders, tied to the REGION (Paul, 2026-10-07).
  Rule (frozen before any outcome was read against it): real-fill (rung 5) beats noise-fill (rung 4) on a (train contrast,
  eval contrast, label) cell iff the label is at least as visible on the EVAL contrast as on the TRAINING contrast:
      gap = vis_eval - vis_train >= 0  ->  real-fill helps;  gap < 0  ->  real-fill hurts.
  vis = AUC of the label's voxels vs its 1-5 vox ring (max(AUC, 1-AUC)), z-scored foreground, mean over <=40 test cases of
  that dataset/contrast (cross_dataset_fillswap_model.label_feats, unchanged). Pooled cells: voxel-weighted mean over labels.
  Outcomes: per-label Delta rebuilt from each ladder's own rung-4 / rung-5 metrics (run_keys of ladder_series.json, folds
  0-2, mean per case, paired Wilcoxon over cases, Holm within dataset); own-source test items only (external companions
  enter through their own per-source ladders: duke / ispy1 / acrin6698).
Stages:  extract  -> outputs/data/vis_label_<key>.csv (per case x label);  score -> outputs/tables/visibility_sign_rule.md
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, nibabel as nib
from scipy.stats import spearmanr, wilcoxon
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS)); sys.path.insert(0, str(THIS.parents[6] / "benchmark/00_commun_scripts/00_00_utils"))
import cross_dataset_fillswap_model as xd  # noqa: E402  (label_feats, prep, MAN)
import ramp_out_of_sample as ro  # noqa: E402
from stat_tests import holm  # noqa: E402
REPO = THIS.parents[6]; T2 = REPO / "benchmark/02_tasks"; DATA = THIS.parent / "outputs/data"; TAB = THIS.parent / "outputs/tables"
MAXC = 40
RUNG4 = os.environ.get("RUNG4", "")   # "" = the ladder json's rung 4; "lblvor" = the 2026-10-07 retrained rung 4
TAG = "_lblvor" if RUNG4 == "lblvor" else ""


def _nn(base, i, l):
    base = Path(base)
    return ro._list_cases(base / i, base / l) if (base / i).is_dir() and (base / l).is_dir() else []


def manifest():
    m = {k: v for k, v in xd.MAN.items() if k not in ("duke:t1wce_uni", "duke:precontrast_uni")}
    p = T2 / "pancreas_disease"
    m["pansegdata:t1wce"] = (False, lambda: _nn(p / "pansegdata/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE", "imagesTs_t1wce", "labelsTs_t1wce"))
    m["pansegdata:t2w"] = (False, lambda: _nn(p / "pansegdata/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE", "imagesTs_t2w", "labelsTs_t2w"))
    m["totalsegmri-pancreas:t1gre"] = (False, lambda: _nn(p / "totalsegmri-pancreas/2_nnUNet_totalsegmri-pancreas/raw", "imagesTs_t1gre", "labelsTs_t1gre"))
    m["totalsegmri-pancreas:t2like"] = (False, lambda: _nn(p / "totalsegmri-pancreas/2_nnUNet_totalsegmri-pancreas/raw", "imagesTs_t2like", "labelsTs_t2like"))
    m["msd-pancreas:ct"] = (False, lambda: _nn(p / "msd-pancreas/2_nnUNet_msd-pancreas/raw", "imagesTs_ct", "labelsTs_ct"))
    q = T2 / "pelvis_healthy/totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT"
    m["totalseg-pelvic:ct"] = (False, lambda: _nn(q, "imagesTs_ct", "labelsTs_ct"))
    m["totalseg-pelvic:mri"] = (False, lambda: _nn(q, "imagesTs_mri", "labelsTs_mri"))
    return m


def case_task(task):
    key, cid, ip, lp, mand = task
    rows = []
    try:
        im = nib.load(str(ip)); z, hp, fg, ax = xd.prep(np.asarray(im.dataobj), im.header.get_zooms())
        lbl = np.squeeze(np.asarray(nib.load(str(lp)).dataobj)).round().astype(int)
        for lab in ([1] if mand else [int(v) for v in np.unique(lbl) if v > 0]):
            o = xd.label_feats(z, hp, fg, lbl == lab, ax)
            if o:
                rows.append((key, cid, lab, o[1], o[0][0]))
    except Exception as e:  # noqa: BLE001
        print(f"SKIP {key} {cid}: {e}", flush=True)
    return rows


def extract():
    M = manifest(); tasks = []
    for k, (mand, fn) in M.items():
        out = DATA / f"vis_label_{k.replace(':', '_').replace('/', '_')}.csv"
        if out.exists():
            continue
        cs = fn()[:MAXC]
        if not cs:
            print(f"{k}: no cases found, skipped", flush=True); continue
        print(f"{k}: {len(cs)} cases", flush=True); tasks += [(k, c, i, l, mand) for c, i, l in cs]
    if tasks:
        with Pool(8) as pool:
            res = [r for rs in pool.imap_unordered(case_task, tasks, chunksize=1) for r in rs]
        df = pd.DataFrame(res, columns=["key", "case", "label", "n_vox", "vis"])
        for k, g in df.groupby("key"):
            g.to_csv(DATA / f"vis_label_{k.replace(':', '_').replace('/', '_')}.csv", index=False)
    # brats from bigfeat (per region, same AUC definition), label ids per BREG
    b = pd.concat([pd.read_csv(f, usecols=["patient", "contrast", "region", "n_R", "con_auc"]) for f in glob.glob(str(DATA / "bigfeat_shard*.csv"))]).drop_duplicates(["patient", "contrast", "region"])
    b = b.rename(columns={"patient": "case", "n_R": "n_vox", "con_auc": "vis"}); b["key"] = "brats:" + b.contrast; b["label"] = b.region.map(xd.BREG)
    for k, g in b.groupby("key"):
        g[["key", "case", "label", "n_vox", "vis"]].to_csv(DATA / f"vis_label_{k.replace(':', '_')}.csv", index=False)


# ---------------- ladders: per-label deltas ----------------
# ladder json glob -> (dataset alias for the TRAIN key, eval-key -> feature-key alias, dataset.json with label names)
LADDERS = [
    ("brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/{t}/ablations/ladder_series.json", "brats", "brats", "brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n/dataset.json"),
    ("brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/{t}/ablations/ladder_series.json", "open-ms", "open-ms", "brain_ms/open-ms/2_nnUNet_open-ms/raw/Dataset070_OpenMS_FLAIR/dataset.json"),
    ("abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/{t}/ablations/ladder_series.json", "chaos", "chaos", "abdomen_healthy/chaos/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in/dataset.json"),
    ("breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/{t}/ablations/ladder_series.json", "ispy2", "ispy2", "breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce/dataset.json"),
    ("breast_cancer/duke-breast-mri/8_results_duke-breast-mri/02_metrics/ispy2_model/{t}/ablations/*/ladder_series.json", "ispy2", "duke", "breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce/dataset.json"),
    ("breast_cancer/ispy1/8_results_ispy1/02_metrics/ispy2_model/{t}/ablations/*/ladder_series.json", "ispy2", "ispy1", "breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce/dataset.json"),
    ("breast_cancer/acrin6698/8_results_acrin6698/02_metrics/ispy2_model/{t}/ablations/*/ladder_series.json", "ispy2", "acrin", "breast_cancer/ispy2/2_nnUNet_ispy2/raw/Dataset100_ISPY2T1wce/dataset.json"),
    ("brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/{t}/ablations/ladder_series.json", "onharmony", "onharmony", "brain_healthy/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31/dataset.json"),
    ("mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/{t}/ablations/ladder_series.json", "toothfairy2", "", "mandible_healthy/toothfairy2/2_nnUNet_toothfairy2/raw/Dataset110_ToothFairy2CBCT/dataset.json"),
    ("pancreas_disease/pansegdata/8_results_pansegdata/02_metrics/pansegdata_model/{t}/ablations/ladder_series.json", "pansegdata", "pansegdata", "pancreas_disease/pansegdata/2_nnUNet_pansegdata/raw/Dataset150_PanSegData_T1WCE/dataset.json"),
    ("pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/{t}/ablations/ladder_series.json", "totalseg-pelvic", "totalseg-pelvic", "pelvis_healthy/totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT/dataset.json"),
]
SKIP_EVAL = {"duke": {"t1wce": ("t1wce_uniap",)}, "ispy1": {"t1wce": ("t1wce",)}}   # same-contrast cross-dataset, not OOD


def label_names(path):
    if path is None:
        return {}
    d = json.load(open(REPO / "benchmark/02_tasks" / path))["labels"]
    return {v: k for k, v in d.items()} if isinstance(list(d.values())[0], int) else {}


def ladder_cells():
    rows = []
    for pat, train_alias, eval_alias, dsjson in LADDERS:
        for f in sorted(glob.glob(str(T2 / pat.replace("{t}", "*")))):
            d = json.load(open(f)); ir = next(i for i, l in enumerate(d["labels"]) if "real fill" in l)
            mroot = Path(f.split("/ablations")[0]); train = mroot.name
            src = f.split("/ablations/")[1].replace("/ladder_series.json", "") if not f.endswith("ablations/ladder_series.json") else ""
            if eval_alias in ("duke", "ispy1") and src in SKIP_EVAL.get(eval_alias, {}).get(train, ()):
                continue
            dirs = {}
            for rung, key in (("noise", d["run_keys"][ir - 1]), ("real", d["run_keys"][ir])):
                name = Path(key).name
                c = [p for p in mroot.rglob(f"*{name}") if p.is_dir() and (p / "fold0/eval_all.csv").exists() and (not src or f"/{src}/" in str(p) + "/")]
                if rung == "noise" and RUNG4 == "lblvor":   # 2026-10-08: retrained rung 4 (label_voronoi) wherever it exists, else drop the cell
                    c = [p for p in mroot.rglob("*baseline_kmeans_label_remap_voronoi*lblvor*") if p.is_dir() and (p / "fold0/eval_all.csv").exists()
                         and len(list(p.glob("fold[0-2]/eval_all.csv"))) == 3 and (not src or f"/{src}/" in str(p) + "/")]
                    if not c:
                        print("no lblvor rung 4 yet:", f); dirs = None; break
                if not c:
                    print("missing rung dir", f, name); dirs = None; break
                dirs[rung] = c[0]
            if dirs is None:
                continue
            per = {}
            for rung, p in dirs.items():
                df = pd.concat([pd.read_csv(x) for x in sorted(p.glob("fold[0-2]/eval_all.csv"))])
                df = df[df.dice.notna()]
                per[rung] = df.groupby(["group", "case", "label"]).dice.mean()
            m = pd.concat({"noise": per["noise"], "real": per["real"]}, axis=1).dropna().reset_index()
            names = label_names(dsjson)
            for (grp, lab), g in m.groupby(["group", "label"]):
                if grp.endswith("(in-domain)") or grp == train or (src == "" and grp not in d.get("ood_contrasts", list(d["per_contrast"]["dice"]))):
                    continue
                if src and grp != src:
                    continue
                if len(g) < 5:
                    continue
                ev_key = (f"{eval_alias}:{grp}" if eval_alias else grp.replace("/", ":"))
                if eval_alias == "duke":
                    ev_key = f"duke:{grp}"
                rows.append(dict(dataset=Path(f).parts[-8] if src else Path(f).parts[-7], train_key=f"{train_alias}:{train}", eval_key=ev_key, label=str(lab), n=len(g),
                                 delta=100 * (g.real - g.noise).mean(), p=wilcoxon(g.real, g.noise).pvalue if (g.real != g.noise).any() else 1.0))
            # pooled (labels averaged per case) per eval group
            mp = m.groupby(["group", "case"])[["noise", "real"]].mean().reset_index()
            for grp, g in mp.groupby("group"):
                if grp.endswith("(in-domain)") or grp == train or (src and grp != src) or (src == "" and grp not in d.get("ood_contrasts", list(d["per_contrast"]["dice"]))):
                    continue
                ev_key = f"{eval_alias}:{grp}" if eval_alias else grp.replace("/", ":")
                rows.append(dict(dataset=Path(f).parts[-8] if src else Path(f).parts[-7], train_key=f"{train_alias}:{train}", eval_key=ev_key, label="POOLED", n=len(g),
                                 delta=100 * (g.real - g.noise).mean(), p=wilcoxon(g.real, g.noise).pvalue if (g.real != g.noise).any() else 1.0))
    # external mandible sources (no per-source ladder json): toothfairy2 cbct model on hanseg ct / mrt1 and pddca ct
    for ds, root in (("hanseg", T2 / "mandible_healthy/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct/ablations"),
                     ("pddca", T2 / "mandible_healthy/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct/ablations")):
        r4 = sorted(root.glob("*baseline_kmeans_label_remap_voronoi" + ("*lblvor*" if RUNG4 == "lblvor" else "_2*"))); r5 = sorted(root.glob("*_v26_6_2_train050_val000_*"))
        if not r4 or not r5:
            print("mandible source missing", ds, RUNG4); continue
        per = {}
        for rung, pth in (("noise", r4[-1]), ("real", r5[-1])):
            df = pd.concat([pd.read_csv(x) for x in sorted(pth.glob("fold[0-2]/eval_all.csv"))]); df = df[df.dice.notna()]
            per[rung] = df.groupby(["group", "case", "label"]).dice.mean()
        m = pd.concat({"noise": per["noise"], "real": per["real"]}, axis=1).dropna().reset_index()
        for (grp, lab), g in m.groupby(["group", "label"]):
            for lbl in (str(lab), "POOLED"):
                rows.append(dict(dataset=ds, train_key="toothfairy2:cbct", eval_key=f"{ds}:{grp}", label=lbl, n=len(g), delta=100 * (g.real - g.noise).mean(),
                                 p=wilcoxon(g.real, g.noise).pvalue if (g.real != g.noise).any() else 1.0))
    c = pd.DataFrame(rows)
    c["p_holm"] = np.nan
    for (ds, pooled), idx in c.groupby(["dataset", c.label == "POOLED"]).groups.items():
        c.loc[idx, "p_holm"] = holm(c.loc[idx, "p"].tolist())
    return c


def vis_table():
    v = pd.concat([pd.read_csv(f) for f in glob.glob(str(DATA / "vis_label_*.csv"))])
    v["label"] = v.label.astype(str)
    per_label = v.groupby(["key", "label"]).agg(vis=("vis", "mean"), n_vox=("n_vox", "mean"), n_cases=("case", "nunique")).reset_index()
    pooled = v.groupby(["key", "case"]).apply(lambda g: np.average(g.vis, weights=g.n_vox)).groupby("key").mean().rename("vis").reset_index(); pooled["label"] = "POOLED"
    return pd.concat([per_label, pooled[["key", "label", "vis"]]], ignore_index=True)


def score():
    c = ladder_cells(); v = vis_table()
    # brats labels in eval_all are names (SNFH...) -> ids used in vis files
    c["label_id"] = c.label.map(lambda s: str(xd.BREG.get(s, s)))
    # pelvic / chaos / on-harmony eval_all labels may be names: map via the visibility keys' dataset.json where needed
    names = {}
    for pat, train_alias, eval_alias, dsjson in LADDERS:
        names.update({k: str(i) for i, k in label_names(dsjson).items()})
    c["label_id"] = c.label_id.map(lambda s: names.get(s, s)).replace({"mandible": "1", "tumour": "1", "lesion": "1", "pancreas": "1"})
    v_tr = v.rename(columns={"key": "train_key", "label": "label_id", "vis": "vis_train"}); v_ev = v.rename(columns={"key": "eval_key", "label": "label_id", "vis": "vis_eval"})
    x = c.merge(v_tr[["train_key", "label_id", "vis_train"]], on=["train_key", "label_id"], how="left").merge(v_ev[["eval_key", "label_id", "vis_eval"]], on=["eval_key", "label_id"], how="left")
    x["gap"] = x.vis_eval - x.vis_train; x["pred"] = np.where(x.gap >= 0, 1, -1); x["actual"] = np.sign(x.delta); x["hit"] = x.pred == x.actual; x["sig"] = x.p_holm < 0.05
    x.to_csv(DATA / f"visibility_sign_rule_cells{TAG}.csv", index=False)
    L = [f"# Visibility sign rule across all ladders, per label (val000 rung 5{', RETRAINED rung 4 (lblvor)' if RUNG4 else ''})", "",
         "Rule (frozen): real-fill helps iff vis_eval >= vis_train (label-vs-ring AUC, mean over <=40 cases). Per-label Δ from each ladder's own rung 4/5 metrics "
         "(own-source test items; breast companions via their own per-source ladders; chaos/pansegdata external pools not included), Holm within dataset. "
         f"Cells: {len(x)}; with both visibilities: {int(x.gap.notna().sum())}.", ""]
    for title, sel in (("Per-label cells", x[(x.label != "POOLED") & x.gap.notna()]), ("Pooled (train, eval) cells", x[(x.label == "POOLED") & x.gap.notna()])):
        s = sel[sel.sig]; rho = spearmanr(sel.gap, sel.delta) if len(sel) > 3 else (np.nan, np.nan)
        dz = sel[sel.gap.abs() >= 0.02]; dzs = dz[dz.sig]
        L += [f"## {title}: {len(sel)} cells, {len(s)} significant", "",
              f"- excluding near-zero gaps (|gap| < 0.02, no call): rule {int(dz.hit.sum())}/{len(dz)} vs always-helps {int((dz.actual > 0).sum())}/{len(dz)}; significant {int(dzs.hit.sum())}/{len(dzs)} vs {int((dzs.actual > 0).sum())}/{len(dzs)}",
              f"- all cells: rule {int(sel.hit.sum())}/{len(sel)}, always-helps {int((sel.actual > 0).sum())}/{len(sel)}; Spearman(gap, Δ) = {rho[0]:+.2f} (p={rho[1]:.2g})",
              f"- significant cells: rule {int(s.hit.sum())}/{len(s)}, always-helps {int((s.actual > 0).sum())}/{len(s)}; losses caught {int(((s.pred < 0) & (s.actual < 0)).sum())}/{int((s.actual < 0).sum())}, false alarms {int(((s.pred < 0) & (s.actual > 0)).sum())}",
              "- per dataset (rule hits / always-helps / n, all cells): " + "; ".join(f"{ds} {int(g.hit.sum())}/{int((g.actual > 0).sum())}/{len(g)}" for ds, g in sel.groupby("dataset")),
              "- per dataset (significant only): " + "; ".join(f"{ds} {int(g.hit.sum())}/{int((g.actual > 0).sum())}/{len(g)}" for ds, g in s.groupby("dataset")),
              "", "| dataset | train | eval | label | n | vis train | vis eval | gap | pred | Δ | p Holm | hit |", "|---|---|---|---|--:|--:|--:|--:|:--:|--:|--:|:--:|"]
        for r in sel.sort_values(["dataset", "train_key", "eval_key", "label"]).itertuples():
            L.append(f"| {r.dataset} | {r.train_key} | {r.eval_key} | {r.label} | {r.n} | {r.vis_train:.3f} | {r.vis_eval:.3f} | {r.gap:+.3f} | {'real' if r.pred > 0 else 'noise'} | {r.delta:+.2f} | {r.p_holm:.2g}{'*' if r.sig else ''} | {'YES' if r.hit else 'no'} |")
        L.append("")
    if RUNG4 == "lblvor" and (DATA / "visibility_sign_rule_cells.csv").exists():
        o = pd.read_csv(DATA / "visibility_sign_rule_cells.csv"); o = o[o.label == "POOLED"][["train_key", "eval_key", "delta", "p_holm", "hit"]].rename(columns={"delta": "delta_old", "p_holm": "p_old", "hit": "hit_old"})
        cmp_ = x[x.label == "POOLED"].merge(o, on=["train_key", "eval_key"], how="left")
        L += ["## Before / after the rung-4 retrain (pooled cells with a visibility value)", "",
              "| dataset | train | eval | gap | pred | Δ old rung4 | p old | hit old | Δ new rung4 | p new | hit new |", "|---|---|---|--:|:--:|--:|--:|:--:|--:|--:|:--:|"]
        for r in cmp_.sort_values(["dataset", "train_key", "eval_key"]).itertuples():
            L.append(f"| {r.dataset} | {r.train_key} | {r.eval_key} | {r.gap:+.3f} | {'real' if r.pred > 0 else 'noise'} | {r.delta_old:+.2f} | {r.p_old:.2g} | {'YES' if r.hit_old else 'no'} | {r.delta:+.2f} | {r.p_holm:.2g}{'*' if r.sig else ''} | {'YES' if r.hit else 'no'} |")
        L.append("")
    miss = x[x.gap.isna()]
    if len(miss):
        L += ["## Cells without a visibility value (extraction missing)", "", ", ".join(sorted(set(miss.train_key + "->" + miss.eval_key + ":" + miss.label))), ""]
    L += ["## Visibility per contrast and label", "", "```", v[v.label != "POOLED"].pivot_table(index="key", columns="label", values="vis").round(3).to_string(), "```"]
    (TAB / f"visibility_sign_rule{TAG}.md").write_text("\n".join(L)); print("\n".join(L[:14]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("stage", choices=["extract", "score", "all"]); a = ap.parse_args()
    if a.stage in ("extract", "all"):
        extract()
    if a.stage in ("score", "all"):
        score()
